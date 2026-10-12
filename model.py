"""
CVSegNet -- a from-scratch instance-segmentation model.

Architecture
------------
A U-Net style network with a ResNet encoder (ImageNet-initialised) and an
FPN-style decoder. It has three output heads:

  * fg  : per-pixel foreground probability (binary mask)
  * ctr : per-pixel "instance centre" heat-map (a Gaussian blob at the centre
          of every object instance)
  * bnd : per-pixel object-boundary energy (the seams between touching objects)

At inference we threshold the foreground, pick peaks in the centre heat-map as
seed markers, and run a watershed whose energy is `-distance + weight * boundary`
so touching objects split **along the learned boundary** instead of merging.

Improvements over the original
------------------------------
* third **boundary** head -> the watershed no longer merges touching objects
* FPN-style decoder (lateral projections) that reaches **stride 2** -> sharper masks
* capacity is a knob: `backbone` (resnet18 / resnet50) and `decoder_width`
* low-VRAM engineering: frozen-BN encoder, GroupNorm decoder (safe at batch 1),
  optional gradient checkpointing, optional early-layer freezing
* focal + Dice + BCE on the foreground -> fewer spurious blobs

Presets:  CVSegNet.light()  (resnet18, original scale)
          CVSegNet.large()  (resnet50, 1.5x decoder, checkpointing)
          CVSegNet.xlarge() (resnet50, 2x decoder)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.checkpoint import checkpoint as _ckpt

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _gn(c: int) -> nn.GroupNorm:
    g = 8
    while c % g != 0:
        g //= 2
    return nn.GroupNorm(g, c)


class ConvBlock(nn.Module):
    """Two 3x3 convs with GroupNorm (batch-size independent, so batch 1 is fine)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), _gn(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), _gn(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


_BACKBONES = {
    "resnet18": (torchvision.models.resnet18,
                 torchvision.models.ResNet18_Weights.DEFAULT, (64, 128, 256, 512)),
    "resnet50": (torchvision.models.resnet50,
                 torchvision.models.ResNet50_Weights.DEFAULT, (256, 512, 1024, 2048)),
}


class CVSegNet(nn.Module):
    """Proposal-free instance segmentation with fg / centre / boundary heads."""

    def __init__(self, pretrained: bool = True, backbone: str = "resnet18",
                 decoder_width: float = 1.0, out_stride: int = 2,
                 checkpoint: bool = False, freeze_at: int = 0):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"backbone must be one of {list(_BACKBONES)}")
        ctor, weights, chans = _BACKBONES[backbone]
        self.backbone_name = backbone
        self.decoder_width = decoder_width
        self.out_stride = out_stride
        self.use_checkpoint = checkpoint
        self.freeze_at = freeze_at

        m = ctor(weights=weights if pretrained else None)
        c2, c3, c4, c5 = chans
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)   # stride 2
        self.pool = m.maxpool                               # stride 4
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            m.layer1, m.layer2, m.layer3, m.layer4)

        w = decoder_width
        w16, w8, w4, w2 = int(128 * w), int(64 * w), int(64 * w), int(48 * w)

        # lateral 1x1 projections keep skip channels small (memory-friendly).
        # chans = (layer1, layer2, layer3, layer4) output channels; the stem
        # output is always 64 channels for both ResNet-18 and ResNet-50.
        c2, c3, c4, c5 = chans
        self.lat16 = nn.Conv2d(c4, w16, 1)   # layer3 output (x16)
        self.lat8 = nn.Conv2d(c3, w8, 1)     # layer2 output (x8)
        self.lat4 = nn.Conv2d(c2, w4, 1)     # layer1 output (x4)
        self.lat2 = nn.Conv2d(64, w2, 1)     # stem output (stride 2)
        self.proj32 = nn.Conv2d(c5, w16, 1)  # layer4 output (x32)

        self.dec16 = ConvBlock(w16 + w16, w16)
        self.dec8 = ConvBlock(w16 + w8, w8)
        self.dec4 = ConvBlock(w8 + w4, w4)
        self.dec2 = ConvBlock(w4 + w2, w2) if out_stride == 2 else None

        head_in = w2 if out_stride == 2 else w4
        self.head_fg = nn.Conv2d(head_in, 1, 1)
        self.head_ctr = nn.Conv2d(head_in, 1, 1)
        self.head_bnd = nn.Conv2d(head_in, 1, 1)

        self._apply_freeze()

    def _apply_freeze(self):
        """freeze_at: 0 none, 1 stem, 2 +layer1, 3 +layer2, 4 +layer3."""
        if self.freeze_at <= 0:
            return
        for p in self.stem.parameters():
            p.requires_grad = False
        if self.freeze_at >= 2:
            for p in self.layer1.parameters():
                p.requires_grad = False
        if self.freeze_at >= 3:
            for p in self.layer2.parameters():
                p.requires_grad = False
        if self.freeze_at >= 4:
            for p in self.layer3.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        """Keep pretrained BatchNorm statistics frozen (safe at batch size 1)."""
        super().train(mode)
        for mod in self.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        return self

    def _run(self, fn, x):
        # Checkpoint when the block has trainable params and we're training.
        # Note: the *input* may not require grad (e.g. right after a frozen
        # stage), but the block's own weights still do -- so keying off
        # x.requires_grad alone would silently disable checkpointing for most
        # of the encoder whenever --freeze-at is used.
        if (self.use_checkpoint and self.training and torch.is_grad_enabled()
                and any(p.requires_grad for p in fn.parameters())):
            return _ckpt(fn, x, use_reentrant=False)
        return fn(x)

    def forward(self, x):
        h, w = x.shape[-2:]
        s = self.stem(x)                       # stride 2
        p = self.pool(s)                       # stride 4
        x4 = self._run(self.layer1, p)
        x8 = self._run(self.layer2, x4)
        x16 = self._run(self.layer3, x8)
        x32 = self._run(self.layer4, x16)

        u = self.proj32(x32)
        u = F.interpolate(u, size=x16.shape[-2:], mode="nearest")
        u = self.dec16(torch.cat([u, self.lat16(x16)], 1))
        u = F.interpolate(u, size=x8.shape[-2:], mode="nearest")
        u = self.dec8(torch.cat([u, self.lat8(x8)], 1))
        u = F.interpolate(u, size=x4.shape[-2:], mode="nearest")
        u = self.dec4(torch.cat([u, self.lat4(x4)], 1))
        if self.out_stride == 2:
            u = F.interpolate(u, size=s.shape[-2:], mode="nearest")
            u = self.dec2(torch.cat([u, self.lat2(s)], 1))

        def up(t):
            return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)

        return up(self.head_fg(u)), up(self.head_ctr(u)), up(self.head_bnd(u))

    # ---- presets ------------------------------------------------------
    @classmethod
    def light(cls) -> "CVSegNet":
        """ResNet-18, 1x decoder -- the original scale (~13.7 M params)."""
        return cls(backbone="resnet18", decoder_width=1.0)

    @classmethod
    def large(cls) -> "CVSegNet":
        """ResNet-50, 1.5x decoder, gradient checkpointing -- clearly bigger."""
        return cls(backbone="resnet50", decoder_width=1.5, checkpoint=True, freeze_at=2)

    @classmethod
    def xlarge(cls) -> "CVSegNet":
        """ResNet-50, 2x decoder -- biggest that is still plausibly 2 GB-able."""
        return cls(backbone="resnet50", decoder_width=2.0, checkpoint=True, freeze_at=2)


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def build_center_target(inst_mask: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Gaussian blob at the centroid of every instance id."""
    h, w = inst_mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    target = np.zeros((h, w), np.float32)
    for inst in np.unique(inst_mask):
        if inst == 0:
            continue
        ys, xs = np.where(inst_mask == inst)
        if len(ys) == 0:
            continue
        cy, cx = ys.mean(), xs.mean()
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        target = np.maximum(target, g)
    return target


def build_boundary_target(inst_mask: np.ndarray) -> np.ndarray:
    """1 where the instance id changes (outlines + seams between touching objects)."""
    from scipy import ndimage as ndi
    d = np.zeros(inst_mask.shape, bool)
    d[:, :-1] |= inst_mask[:, :-1] != inst_mask[:, 1:]
    d[:, 1:] |= inst_mask[:, :-1] != inst_mask[:, 1:]
    d[:-1, :] |= inst_mask[:-1, :] != inst_mask[1:, :]
    d[1:, :] |= inst_mask[:-1, :] != inst_mask[1:, :]
    d &= inst_mask > 0
    return ndi.binary_dilation(d, iterations=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    wgt = alpha * target + (1 - alpha) * (1 - target)
    return (wgt * (1 - pt) ** gamma * ce).mean()


def cvseg_loss(fg_l, ctr_l, bnd_l, fg, ctr, bnd):
    l_fg = (F.binary_cross_entropy_with_logits(fg_l, fg)
            + dice_loss(fg_l, fg) + focal_loss(fg_l, fg))
    l_ctr = F.binary_cross_entropy_with_logits(ctr_l, ctr)
    l_bnd = F.binary_cross_entropy_with_logits(bnd_l, bnd) + 0.5 * dice_loss(bnd_l, bnd)
    return {"fg": l_fg, "ctr": l_ctr, "bnd": l_bnd,
            "total": l_fg + 1.5 * l_ctr + 1.0 * l_bnd}


# --------------------------------------------------------------------------- #
# Inference / post-processing
# --------------------------------------------------------------------------- #
def _resize_hw(a: np.ndarray, hw) -> np.ndarray:
    return np.array(Image.fromarray(a).resize((hw[1], hw[0]), Image.BILINEAR))


def preprocess(image_rgb: np.ndarray, size: int = 320):
    """HWC uint8 RGB -> normalised 1x3xSxS float tensor."""
    img = np.array(Image.fromarray(image_rgb).resize((size, size), Image.BILINEAR))
    x = img.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()


@torch.no_grad()
def predict(model: nn.Module, image_rgb: np.ndarray, size: int = 320, tta: bool = False):
    """Return (fg_prob, ctr_prob, bnd_prob) at the *original* image resolution."""
    model.eval()
    dev = next(model.parameters()).device      # work on whatever device the model is on
    x = preprocess(image_rgb, size).to(dev)
    fg, ctr, bnd = model(x)
    if tta:
        fg2, ctr2, bnd2 = model(torch.flip(x, dims=[3]))
        fg = (fg + torch.flip(fg2, dims=[3])) / 2
        ctr = (ctr + torch.flip(ctr2, dims=[3])) / 2
        bnd = (bnd + torch.flip(bnd2, dims=[3])) / 2

    h, w = image_rgb.shape[:2]

    def to_np(t):
        return _resize_hw(torch.sigmoid(t)[0, 0].cpu().numpy(), (h, w))

    return to_np(fg), to_np(ctr), to_np(bnd)


def extract_instances(fg_prob: np.ndarray, ctr_prob: np.ndarray, bnd_prob: np.ndarray,
                      fg_thr: float = 0.5, ctr_thr: float = 0.35,
                      bnd_weight: float = 0.7, min_area: int = 40,
                      min_distance: int = 7):
    """Turn probability maps into a labelled instance map + instance records.

    Returns
    -------
    labels : int32 HxW array, 0 = background, 1..N = instance ids
    instances : list of dicts {id, area, bbox:[x,y,w,h], score}
    """
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    fg = fg_prob > fg_thr
    if fg.sum() == 0:
        return np.zeros_like(fg, dtype=np.int32), []
    fg = ndi.binary_closing(fg, structure=np.ones((3, 3)), iterations=1)
    fg = ndi.binary_fill_holes(fg)
    fg = ndi.binary_opening(fg, structure=np.ones((3, 3)), iterations=1)
    if fg.sum() == 0:
        return np.zeros_like(fg, dtype=np.int32), []

    dist = ndi.distance_transform_edt(fg)
    # watershed energy: low inside objects, high on the learned boundaries
    energy = -dist + bnd_weight * bnd_prob * dist.max()

    coords = peak_local_max(ctr_prob, min_distance=min_distance,
                            threshold_abs=ctr_thr, labels=fg, exclude_border=False)
    if len(coords) == 0:
        coords = peak_local_max(dist, min_distance=max(8, min_distance),
                                labels=fg, exclude_border=False)

    markers = np.zeros(fg.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if markers.max() == 0:
        labels, _ = ndi.label(fg)
    else:
        labels = watershed(energy, markers, mask=fg)

    out = np.zeros_like(labels, dtype=np.int32)
    instances, nxt = [], 1
    for lab in range(1, labels.max() + 1):
        ys, xs = np.where(labels == lab)
        if len(ys) < min_area:
            continue
        out[ys, xs] = nxt
        instances.append({
            "id": nxt, "area": int(len(ys)),
            "bbox": [int(xs.min()), int(ys.min()),
                     int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
            "score": round(float(ctr_prob[ys, xs].mean()), 4),
        })
        nxt += 1
    return out, instances


def instance_palette(n: int, seed: int = 7):
    """Deterministic bright RGB colour for instance ids 0..n (0 = background)."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(60, 256, size=(n + 1, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    return palette


def instance_color_map(labels: np.ndarray, seed: int = 7):
    """Deterministic bright colour per instance id -> HxWx3 uint8 RGB."""
    return instance_palette(int(labels.max()), seed)[labels]
