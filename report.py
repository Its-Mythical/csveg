"""
report.py - generate a report on how well CVSegNet works, what was fixed, and
what is left to do.

It measures the model live on the held-out validation split (foreground IoU,
instance-count accuracy, merge rate) plus a couple of synthetic capability
checks, then writes a Markdown report.

Privacy: hardware-capacity numbers (RAM, VRAM, core count) are deliberately
**never** read or printed - the report only ever refers to "the compute device".

    python report.py                       # writes report.md
    python report.py --out my_report.md
    python report.py --checkpoint weights/cvseg_unet.pt --n-val 40
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from model import CVSegNet, extract_instances, predict
from train import InstanceMaskDataset, evaluate, gather_items

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "weights", "cvseg_unet.pt")

# --------------------------------------------------------------------------- #
# what changed, and what is still open (kept in sync with the README)
# --------------------------------------------------------------------------- #
FIXES = [
    ("Instance separation",
     "watershed over the distance transform only (touching objects merged)",
     "NEW boundary head; watershed energy is now -distance + weight x boundary, "
     "so it cuts along learned seams"),
    ("Mask sharpness", "decoder reached stride 4",
     "FPN-style decoder reaches stride 2 (sharper masks)"),
    ("False positives", "BCE + Dice on the foreground",
     "focal + Dice + BCE; instance score = mean centre confidence"),
    ("Capacity", "fixed ResNet-18 (~13.6 M params)",
     "configurable ResNet-18/50 + decoder width; default is the large config (~26 M)"),
    ("Small-batch stability", "BatchNorm everywhere (unreliable at batch 1)",
     "frozen-BN encoder + GroupNorm decoder"),
    ("Memory", "no saving mechanism",
     "gradient checkpointing + early-layer freezing; guard fixed so it actually "
     "engages when layers are frozen"),
    ("Checkpoints", "cvseg_unet.pt overwritten every epoch",
     "best kept at cvseg_best.pt and published to cvseg_unet.pt"),
    ("Data", "one hard-coded dataset",
     "multi-dataset auto-discovery + prepare_data.py (COCO / VOC / YOLO-seg)"),
    ("GPU correctness", "validation / predict did not move data to the model's device",
     "fixed - on a GPU, training previously crashed at the first validation"),
    ("Web app", "deprecated startup hook; model could fail to load",
     "lifespan + lazy load, so the API works regardless of startup timing"),
    ("Tooling", "manual pip installs",
     "install_requirements.py (venv-safe), test_all.py (27 checks), benchmark.py"),
]

WORK = [
    "Near-fully-overlapping objects can still merge - inherent to watershed; a "
    "learned instance-embedding + clustering head would push this further.",
    "Class-agnostic: it segments 'objects', not categories. The COCO subset is "
    "therefore treated as generic foreground, not 80 classes.",
    "Validation is noisy (a small held-out split) and uses foreground IoU - an "
    "official COCO-style AP would be a fairer, more comparable number.",
    "Trained from scratch on modest data; a longer schedule or a "
    "segmentation-pretrained initialisation would likely improve masks.",
    "Test-time augmentation exists but is off by default; enabling it should "
    "sharpen boundaries at some cost in latency.",
    "Several hyper-parameters are fixed defaults (centre-target sigma, "
    "peak min-distance, boundary weight) and could be tuned per dataset.",
    "Mixed precision is wired but could not be exercised on a GPU here, so its "
    "GPU behaviour is unverified in this environment.",
]


def _gt_count(mask_path: str) -> int:
    m = np.array(Image.open(mask_path))
    if m.ndim == 3:
        m = m[..., 0]
    u = np.unique(m)
    return int(len(u) - (1 if 0 in u else 0))


def _boundary_split_check() -> bool:
    """Two touching blobs with a marked boundary should split into 2 instances."""
    fg = np.zeros((80, 80), np.float32)
    fg[20:60, 15:40] = 1.0
    fg[20:60, 40:65] = 1.0
    bnd = np.zeros((80, 80), np.float32)
    bnd[:, 38:42] = 1.0
    ctr = np.zeros((80, 80), np.float32)
    ctr[40, 27] = 1.0
    ctr[40, 52] = 1.0
    _, inst = extract_instances(fg, ctr, bnd, ctr_thr=0.5, bnd_weight=1.0, min_area=20)
    return len(inst) == 2


def _distance_fallback_check() -> bool:
    """A blob with no centre peaks should still yield one instance (fallback)."""
    fg = np.zeros((80, 80), np.float32)
    fg[20:60, 20:60] = 1.0
    z = np.zeros((80, 80), np.float32)
    _, inst = extract_instances(fg, z, z, ctr_thr=0.5, min_area=20)
    return len(inst) == 1


def collect_metrics(model, val_items, size, ctr_thr=0.35, min_area=40):
    """Foreground IoU + instance-level stats on the validation split."""
    iou = evaluate(model, DataLoader(InstanceMaskDataset(val_items, size, False),
                                     batch_size=1))
    exact = 0
    within1 = 0
    abs_err = 0.0
    merges = 0
    multi_gt = 0
    scores = []
    model.eval()
    for it in val_items:
        img = np.array(Image.open(it[0]).convert("RGB"))
        fg, ctr, bnd = predict(model, img, size=size)
        _, inst = extract_instances(fg, ctr, bnd, ctr_thr=ctr_thr, min_area=min_area)
        gt = _gt_count(it[1])
        pred = len(inst)
        if pred == gt:
            exact += 1
        if abs(pred - gt) <= 1:
            within1 += 1
        abs_err += abs(pred - gt)
        if gt >= 2:
            multi_gt += 1
            if pred < gt:
                merges += 1
        scores += [i["score"] for i in inst]
        del fg, ctr, bnd
    n = max(1, len(val_items))
    return {
        "n_val": len(val_items),
        "fg_iou": iou,
        "count_exact": exact / n,
        "count_within1": within1 / n,
        "count_mae": abs_err / n,
        "merge_rate": (merges / multi_gt) if multi_gt else 0.0,
        "multi_gt": multi_gt,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
    }


def build_report(model, cfg, metrics, checks, ckpt_used, ckpt_meta, size, args):
    p = sum(x.numel() for x in model.parameters())
    ptrain = sum(x.numel() for x in model.parameters() if x.requires_grad)
    trained = ckpt_used is not None
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    L = []
    L.append("# CVSegNet - model report\n")
    L.append(f"_Generated {now}_\n")
    L.append("> Hardware-capacity details (RAM, VRAM, core count) are intentionally "
             "omitted; the model runs on CPU or CUDA and the report is device-agnostic.\n")

    L.append("## 1. Summary\n")
    if trained:
        extra = ""
        if ckpt_meta:
            vi = ckpt_meta.get("val_iou")
            vi = f"{vi:.3f}" if isinstance(vi, (int, float)) else str(vi)
            extra = (f" It was trained to epoch {ckpt_meta.get('epoch')} "
                     f"(its own val fg-IoU {vi}).")
        L.append(f"Evaluated the checkpoint `{os.path.basename(ckpt_used)}` on "
                 f"{metrics['n_val']} held-out images.{extra}\n")
        L.append(f"- **Foreground IoU: {metrics['fg_iou']:.3f}** "
                 f"(0 = nothing, 1 = perfect)")
        L.append(f"- **Instance count exactly right on {metrics['count_exact']*100:.0f}%** "
                 f"of images, within +/-1 on {metrics['count_within1']*100:.0f}% "
                 f"(mean absolute error {metrics['count_mae']:.2f})")
        L.append(f"- **Merge rate on multi-object images: {metrics['merge_rate']*100:.0f}%** "
                 f"({metrics['multi_gt']} such images)")
        L.append(f"- Mean instance confidence: {metrics['mean_score']:.3f}\n")
    else:
        L.append("**No trained checkpoint was found**, so these are *untrained* "
                 "baseline numbers. Run `python train.py` first for real results.\n")
        L.append(f"- Foreground IoU: {metrics['fg_iou']:.3f}")
        L.append(f"- Instance count exact: {metrics['count_exact']*100:.0f}%\n")

    L.append("## 2. Model\n")
    L.append("| property | value |")
    L.append("|---|---|")
    L.append(f"| backbone | {cfg.backbone} |")
    L.append(f"| decoder width | {cfg.decoder_width}x |")
    L.append(f"| output stride | {cfg.out_stride} |")
    L.append(f"| parameters | {p/1e6:.2f} M ({ptrain/1e6:.2f} M trainable) |")
    L.append(f"| heads | foreground + instance-centre + boundary |")
    L.append(f"| inference size used here | {size}px |")
    L.append("")

    L.append("## 3. Capability checks\n")
    L.append("| check | result |")
    L.append("|---|---|")
    L.append(f"| boundary-aware watershed splits touching objects | "
             f"{'PASS' if checks['split'] else 'FAIL'} |")
    L.append(f"| distance-transform fallback when no centre peak exists | "
             f"{'PASS' if checks['fallback'] else 'FAIL'} |")
    L.append("")

    L.append("## 4. What was fixed\n")
    L.append("| area | before | now |")
    L.append("|---|---|---|")
    for area, before, after in FIXES:
        L.append(f"| {area} | {before} | {after} |")
    L.append("")

    L.append("## 5. Areas of work\n")
    for i, w in enumerate(WORK, 1):
        L.append(f"{i}. {w}")
    L.append("")

    L.append("## 6. Reproduce\n")
    L.append("```bash")
    L.append("python install_requirements.py          # dependencies")
    L.append("python train.py --size 256 --batch 1 --grad-accum 8 --amp")
    L.append("python test_all.py                      # 27 self-checks")
    L.append("python benchmark.py --models light large --epochs 3")
    L.append("python report.py                        # this report")
    L.append("```")
    L.append("")
    L.append("---")
    L.append("_Metrics are measured live on the held-out split; a small split makes "
             "the numbers noisy, so read them alongside the loss trend and the "
             "masks themselves._")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Generate a CVSegNet quality report")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=os.path.join(HERE, "report.md"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n-val", type=int, default=30)
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    print(f"report: evaluating on device type '{device.type}' "
          f"(no hardware capacity is read)")

    items = gather_items(HERE, args.datasets)
    if not items:
        sys.exit("no data found - run prepare_data.py first")
    rng = np.random.default_rng(0)
    rng.shuffle(items)
    n_val = min(args.n_val, max(1, len(items) // 5))
    val_items = items[:n_val]

    ckpt_used = None
    ckpt_meta = None
    if os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        backbone = ck.get("backbone", "resnet18") if isinstance(ck, dict) else "resnet18"
        width = ck.get("decoder_width", 1.0) if isinstance(ck, dict) else 1.0
        model = CVSegNet(pretrained=False, backbone=backbone, decoder_width=width)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        ckpt_used = args.checkpoint
        ckpt_meta = {"epoch": ck.get("epoch"), "val_iou": ck.get("val_iou")}
        print(f"loaded {args.checkpoint} (epoch {ck.get('epoch')}, "
              f"val_fgIoU {ck.get('val_iou')})")
    else:
        model = CVSegNet.light()
        print(f"no checkpoint at {args.checkpoint} - using an untrained model")

    model.to(device)
    t0 = time.time()
    metrics = collect_metrics(model, val_items, args.size)
    checks = {"split": _boundary_split_check(), "fallback": _distance_fallback_check()}
    gc.collect()

    cfg = type("C", (), {"backbone": model.backbone_name,
                         "decoder_width": model.decoder_width,
                         "out_stride": model.out_stride})
    report = build_report(model, cfg, metrics, checks, ckpt_used, ckpt_meta, args.size, args)

    with open(args.out, "w") as fh:
        fh.write(report)
    print(f"\nwrote {args.out}  ({time.time()-t0:.0f}s)")
    print("-" * 60)
    print(report)


if __name__ == "__main__":
    main()
