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

import json
import os
import warnings

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
                 checkpoint: bool = False, freeze_at: int = 0,
                 train_bn: bool | None = None, embed_dim: int = 0):
        """
        train_bn  : keep BatchNorm layers in training mode while training. ``None`` (default)
                    means "only when the backbone is NOT pretrained" - frozen BatchNorm
                    statistics only make sense for ImageNet-pretrained weights.
        embed_dim : >0 adds a pixel-embedding head (discriminative loss) that helps separate
                    touching / overlapping instances. 0 keeps the original 3-head model.
        """
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"backbone must be one of {list(_BACKBONES)}")
        ctor, weights, chans = _BACKBONES[backbone]
        self.backbone_name = backbone
        self.decoder_width = decoder_width
        self.out_stride = out_stride
        self.use_checkpoint = checkpoint
        self.freeze_at = freeze_at
        self.embed_dim = int(embed_dim)

        # `pretrained` may be True/False, or a path to a local ImageNet state-dict (.pth) - handy
        # when download.pytorch.org is blocked (see tools/get_backbone_weights.py). The
        # CVSEG_BACKBONE_WEIGHTS environment variable is an alternative to passing the path.
        local = pretrained if isinstance(pretrained, str) else (
            os.environ.get("CVSEG_BACKBONE_WEIGHTS") if pretrained else None)
        self.pretrained = bool(pretrained)
        m = None
        if local and os.path.exists(local):
            try:
                m = ctor(weights=None)
                sd = torch.load(local, map_location="cpu", weights_only=True)
                res = m.load_state_dict(sd.get("state_dict", sd), strict=False)
                if res.missing_keys:
                    raise ValueError(f"missing keys {res.missing_keys[:3]}")
            except Exception as e:                   # wrong file for this backbone, etc.
                warnings.warn(f"could not use backbone weights {local} ({e}); trying the "
                              f"standard download", RuntimeWarning)
                m = None
        if m is None:
            try:
                m = ctor(weights=weights if pretrained else None)
            except Exception as e:                   # offline / blocked download
                warnings.warn(f"could not load pretrained {backbone} weights "
                              f"({type(e).__name__}); using random initialisation instead",
                              RuntimeWarning)
                m = ctor(weights=None)
                self.pretrained = False
        self.train_bn = (not self.pretrained) if train_bn is None else bool(train_bn)
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
        self.head_emb = nn.Conv2d(head_in, self.embed_dim, 1) if self.embed_dim > 0 else None

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

    def _frozen_blocks(self):
        """Encoder stages whose weights are frozen by ``freeze_at``."""
        blocks = []
        if self.freeze_at >= 1:
            blocks.append(self.stem)
        if self.freeze_at >= 2:
            blocks.append(self.layer1)
        if self.freeze_at >= 3:
            blocks.append(self.layer2)
        if self.freeze_at >= 4:
            blocks.append(self.layer3)
        return blocks

    def train(self, mode: bool = True):
        """BatchNorm handling.

        Pretrained backbone (default): BatchNorm statistics stay frozen - safe at batch size 1.
        Random-init backbone (``train_bn=True``): BatchNorm trains normally, except inside
        frozen stages, whose statistics stay fixed with their weights.
        """
        super().train(mode)
        if not mode:
            return self
        if not self.train_bn:
            for mod in self.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.eval()
        else:
            for blk in self._frozen_blocks():
                for mod in blk.modules():
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

    def forward(self, x, return_embed: bool = False):
        """Returns ``(fg, ctr, bnd)`` logits at input resolution. With ``return_embed=True``
        (and an embedding head) a 4th tensor is returned: the pixel embedding at the decoder
        resolution (stride 2), shape B x embed_dim x h/2 x w/2."""
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

        out = (up(self.head_fg(u)), up(self.head_ctr(u)), up(self.head_bnd(u)))
        if return_embed and self.head_emb is not None:
            return out + (self.head_emb(u),)
        return out

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
def build_center_target(inst_mask: np.ndarray, sigma: float | None = 4.0,
                        adaptive: bool = False) -> np.ndarray:
    """Gaussian blob at the centre of every instance id.

    ``sigma`` is in pixels of ``inst_mask`` (NOT the image size - passing the image size here
    makes the target ~1 everywhere and destroys instance splitting).

    adaptive=True (or sigma=None): sigma scales with each object's size,
    ``clip(0.12 * sqrt(area), 2.5, 14)``, and the blob sits at the deepest point inside the
    object (so concave objects still get a centre inside themselves). adaptive=False keeps the
    original behaviour: a fixed sigma at the centroid.
    """
    from scipy import ndimage as ndi
    adaptive = adaptive or sigma is None
    h, w = inst_mask.shape
    target = np.zeros((h, w), np.float32)
    for inst in np.unique(inst_mask):
        if inst == 0:
            continue
        ys, xs = np.where(inst_mask == inst)
        if len(ys) == 0:
            continue
        if adaptive:
            s = float(np.clip(0.12 * np.sqrt(len(ys)), 2.5, 14.0))
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = np.pad(inst_mask[y0:y1, x0:x1] == inst, 1)
            dist = ndi.distance_transform_edt(crop)[1:-1, 1:-1]
            iy, ix = np.unravel_index(np.argmax(dist), dist.shape)
            cy, cx = y0 + iy, x0 + ix
        else:
            s, cy, cx = float(sigma), ys.mean(), xs.mean()
        # evaluate the Gaussian only in a +-4 sigma window (identical result, much cheaper)
        r = int(np.ceil(4 * s))
        wy0, wy1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
        wx0, wx1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
        yy, xx = np.mgrid[wy0:wy1, wx0:wx1]
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2))
        target[wy0:wy1, wx0:wx1] = np.maximum(target[wy0:wy1, wx0:wx1], g)
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
    # losses are always computed in float32, so mixed-precision (--amp) can't distort
    # sigmoid / Dice / focal terms
    fg_l, ctr_l, bnd_l = fg_l.float(), ctr_l.float(), bnd_l.float()
    fg, ctr, bnd = fg.float(), ctr.float(), bnd.float()
    l_fg = (F.binary_cross_entropy_with_logits(fg_l, fg)
            + dice_loss(fg_l, fg) + focal_loss(fg_l, fg))
    # the centre target is sparse (a few small blobs): weight pixels near a peak up so the
    # head does not collapse to "no centres"
    l_ctr = F.binary_cross_entropy_with_logits(ctr_l, ctr, weight=1.0 + 9.0 * ctr)
    l_bnd = F.binary_cross_entropy_with_logits(bnd_l, bnd) + 0.5 * dice_loss(bnd_l, bnd)
    return {"fg": l_fg, "ctr": l_ctr, "bnd": l_bnd,
            "total": l_fg + 1.5 * l_ctr + 1.0 * l_bnd}


def discriminative_loss(emb, inst, delta_v=0.5, delta_d=1.5, max_instances=40):
    """Pixel-embedding loss (De Brabandere et al. 2017): pull pixels towards their instance
    mean, push instance means apart. emb: B x E x h x w, inst: B x h x w (0 = background).
    Only foreground pixels are used. Returns a scalar float32 tensor."""
    emb = emb.float()
    B, E = emb.shape[:2]
    total, n_img = emb.new_zeros(()), 0
    for b in range(B):
        e = emb[b].reshape(E, -1).t()
        lab = inst[b].reshape(-1)
        ids = torch.unique(lab)
        ids = ids[ids > 0]
        if len(ids) == 0:
            continue
        if len(ids) > max_instances:
            ids = ids[torch.randperm(len(ids), device=ids.device)[:max_instances]]
        mus, l_var = [], emb.new_zeros(())
        for i in ids:
            ei = e[lab == i]
            mu = ei.mean(0)
            mus.append(mu)
            l_var = l_var + (torch.clamp((ei - mu).norm(dim=1) - delta_v, min=0) ** 2).mean()
        l_var = l_var / len(ids)
        mus = torch.stack(mus)
        l_dist = emb.new_zeros(())
        if len(ids) > 1:
            d = torch.cdist(mus, mus)
            off = ~torch.eye(len(ids), dtype=torch.bool, device=d.device)
            l_dist = (torch.clamp(2 * delta_d - d[off], min=0) ** 2).mean()
        total = total + l_var + l_dist + 0.001 * mus.norm(dim=1).mean()
        n_img += 1
    return total / max(n_img, 1)


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


_TTA_VIEWS = {   # name -> flip dims applied to the input (and undone on the output)
    "none": [[]],
    "h": [[], [3]],
    "hv": [[], [3], [2], [2, 3]],
}


def _tta_name(tta):
    if tta in (False, None, 0, "none", ""):
        return "none"
    if tta in (True, 1, "h"):
        return "h"
    if tta in (2, 4, "hv"):
        return "hv"
    raise ValueError(f"tta must be False/True/'none'/'h'/'hv', got {tta!r}")


@torch.no_grad()
def predict(model: nn.Module, image_rgb: np.ndarray, size: int = 320, tta=False,
            return_embed: bool = False):
    """Return (fg_prob, ctr_prob, bnd_prob) at the *original* image resolution.

    tta: False | True ('h': horizontal flip, 2 passes) | 'hv' (4 flips). Logits are averaged.
    return_embed: if the model has an embedding head, also return the embedding map
    (embed_dim x H x W, float32, from the un-flipped pass) as a 4th element, else None.
    """
    model.eval()
    dev = next(model.parameters()).device      # work on whatever device the model is on
    x = preprocess(image_rgb, size).to(dev)
    want_emb = bool(return_embed and getattr(model, "embed_dim", 0) > 0)
    h, w = image_rgb.shape[:2]

    acc, emb = None, None
    views = _TTA_VIEWS[_tta_name(tta)]
    for k, dims in enumerate(views):
        xv = torch.flip(x, dims=dims) if dims else x
        if k == 0 and want_emb:
            *outs, e = model(xv, return_embed=True)
            emb = F.interpolate(e, size=(h, w), mode="bilinear", align_corners=False)[0]
            emb = emb.cpu().numpy().astype(np.float32)
        else:
            outs = list(model(xv))
        if dims:
            outs = [torch.flip(o, dims=dims) for o in outs]
        acc = outs if acc is None else [a + o for a, o in zip(acc, outs)]
    acc = [a / len(views) for a in acc]

    def to_np(t):
        return _resize_hw(torch.sigmoid(t)[0, 0].cpu().numpy(), (h, w))

    res = (to_np(acc[0]), to_np(acc[1]), to_np(acc[2]))
    return res + (emb,) if return_embed else res


def _embedding_edges(embed: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Magnitude of the spatial gradient of the embedding map, scaled to [0, 1]
    (1 = a strong embedding discontinuity, i.e. a likely seam between two objects)."""
    g = np.zeros(fg.shape, np.float32)
    dx = np.abs(embed[:, :, 1:] - embed[:, :, :-1]).sum(0)
    dy = np.abs(embed[:, 1:, :] - embed[:, :-1, :]).sum(0)
    g[:, :-1] += dx
    g[:, 1:] += dx
    g[:-1, :] += dy
    g[1:, :] += dy
    ref = np.percentile(g[fg], 99) if fg.any() else 1.0
    return np.clip(g / max(float(ref), 1e-6), 0, 1)


def _embedding_assign(fg, embed, coords, spatial):
    """Label every foreground pixel with the nearest seed in embedding space (plus a small
    spatial prior), then keep only each label's largest connected component."""
    from scipy import ndimage as ndi
    E, H, W = embed.shape
    seeds = np.stack([embed[:, max(r - 2, 0):r + 3, max(c - 2, 0):c + 3].reshape(E, -1).mean(1)
                      for r, c in coords])
    ys, xs = np.nonzero(fg)
    pe = embed[:, ys, xs].T
    diag2 = float(H * H + W * W)
    best = np.full(len(ys), np.inf, np.float32)
    lab = np.zeros(len(ys), np.int32)
    for k, (r, c) in enumerate(coords, start=1):
        d = ((pe - seeds[k - 1]) ** 2).sum(1) + spatial * ((ys - r) ** 2 + (xs - c) ** 2) / diag2
        better = d < best
        best[better] = d[better]
        lab[better] = k
    labels = np.zeros(fg.shape, np.int32)
    labels[ys, xs] = lab
    out = np.zeros_like(labels)
    for k in range(1, len(coords) + 1):
        comp, n = ndi.label(labels == k)
        if n == 0:
            continue
        sizes = ndi.sum(np.ones_like(comp), comp, index=range(1, n + 1))
        out[comp == (1 + int(np.argmax(sizes)))] = k
    return out


def extract_instances(fg_prob: np.ndarray, ctr_prob: np.ndarray, bnd_prob: np.ndarray,
                      fg_thr: float = 0.5, ctr_thr: float = 0.35,
                      bnd_weight: float = 0.7, min_area: int = 40,
                      min_distance: int = 7, embed: np.ndarray | None = None,
                      mode: str = "watershed", emb_weight: float = 1.0,
                      emb_spatial: float = 5.0, score: str = "ctr"):
    """Turn probability maps into a labelled instance map + instance records.

    mode (needs ``embed`` from a model with an embedding head, else falls back to watershed):
      "watershed"  distance + learned-boundary watershed (original behaviour)
      "hybrid"     the same watershed, with embedding discontinuities added to its energy
                   (weight ``emb_weight``)
      "embedding"  nearest-seed assignment in embedding space (spatial prior ``emb_spatial``)

    score: how each instance is scored - "ctr" (mean centre confidence, the original),
    "fg" (mean foreground probability) or "mix" (their product).

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

    use_emb = embed is not None and mode in ("hybrid", "embedding")
    dist = ndi.distance_transform_edt(fg)
    # watershed energy: low inside objects, high on the learned boundaries
    energy = -dist + bnd_weight * bnd_prob * dist.max()
    if use_emb and mode == "hybrid":
        energy = energy + emb_weight * _embedding_edges(embed, fg) * dist.max()

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
    elif use_emb and mode == "embedding":
        labels = _embedding_assign(fg, embed, coords, emb_spatial)
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
            "score": round(float({"ctr": ctr_prob[ys, xs].mean(),
                                  "fg": fg_prob[ys, xs].mean(),
                                  "mix": ctr_prob[ys, xs].mean() * fg_prob[ys, xs].mean()}[score]), 4),
        })
        nxt += 1
    return out, instances


POSTPROC_DEFAULTS = {"size": 320, "tta": "none", "mode": "watershed", "fg_thr": 0.5,
                     "ctr_thr": 0.35, "bnd_weight": 0.7, "min_distance": 7, "min_area": 40,
                     "emb_weight": 1.0, "emb_spatial": 5.0, "score": "ctr"}


def load_postproc(path: str | None) -> dict:
    """Post-processing settings: the built-in defaults, overridden by a JSON file written by
    tools/eval_ap.py (``weights/postproc.json``) when one exists."""
    cfg = dict(POSTPROC_DEFAULTS)
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                cfg.update({k: v for k, v in json.load(f).items() if k in cfg})
        except (OSError, ValueError) as e:
            warnings.warn(f"ignoring unreadable post-processing file {path}: {e}")
    return cfg


def instance_palette(n: int, seed: int = 7):
    """Deterministic bright RGB colour for instance ids 0..n (0 = background)."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(60, 256, size=(n + 1, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    return palette


def instance_color_map(labels: np.ndarray, seed: int = 7):
    """Deterministic bright colour per instance id -> HxWx3 uint8 RGB."""
    return instance_palette(int(labels.max()), seed)[labels]
