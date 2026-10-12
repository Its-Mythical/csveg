"""
Benchmark model sizes on the bundled data.

Trains one or more CVSegNet configurations on the *same* split for the same
number of epochs and reports trainable params, wall-clock time and the final
validation foreground IoU, so you can pick a size by measurement rather than by
parameter count.

    python benchmark.py --models light large --epochs 3 --size 160
    python benchmark.py --models light --epochs 5 --size 256 --amp   # on a GPU

The project default is the *large* config (resnet50, 1.5x decoder, ~26 M). This
script does not change any default - it only measures.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import CVSegNet, cvseg_loss
from train import InstanceMaskDataset, evaluate, gather_items

CONFIGS = {
    "light": dict(backbone="resnet18", decoder_width=1.0, checkpoint=False, freeze_at=0),
    "large": dict(backbone="resnet50", decoder_width=1.5, checkpoint=True, freeze_at=2),
    "xlarge": dict(backbone="resnet50", decoder_width=2.0, checkpoint=True, freeze_at=2),
}


def run_one(name, cfg, train_items, val_items, args, device):
    torch.manual_seed(0)
    train_dl = DataLoader(InstanceMaskDataset(train_items, args.size, True),
                          batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(InstanceMaskDataset(val_items, args.size, False),
                        batch_size=args.batch, num_workers=0)

    model = CVSegNet(pretrained=True, **cfg).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    t0 = time.time()
    history = []
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, (x, fg, ctr, bnd) in enumerate(train_dl):
            x = x.to(device)
            fg, ctr, bnd = fg.to(device), ctr.to(device), bnd.to(device)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                fg_l, ctr_l, bnd_l = model(x)
                loss = cvseg_loss(fg_l, ctr_l, bnd_l, fg, ctr, bnd)["total"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        iou = evaluate(model, val_dl)
        history.append(iou)
        print(f"    [{name}] epoch {epoch+1}/{args.epochs}  val_fgIoU {iou:.4f}  "
              f"t={time.time()-t0:.0f}s", flush=True)

    return {"name": name, "params_M": params / 1e6, "iou": history[-1],
            "best": max(history), "secs": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["light", "large"],
                    choices=list(CONFIGS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--n-val", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    root = os.path.dirname(os.path.abspath(__file__))
    items = gather_items(root, args.datasets)
    if not items:
        raise SystemExit("no data found - run prepare_data.py first")
    rng = np.random.default_rng(0)
    rng.shuffle(items)
    n_val = min(args.n_val, max(1, len(items) // 10))
    val_items, train_items = items[:n_val], items[n_val:]
    print(f"device={device} size={args.size} epochs={args.epochs} "
          f"train={len(train_items)} val={len(val_items)}", flush=True)

    results = []
    for name in args.models:
        print(f"\n=== {name} ===", flush=True)
        results.append(run_one(name, CONFIGS[name], train_items, val_items, args, device))

    print("\n" + "=" * 62)
    print(f"{'model':8s} {'params':>9s} {'val_fgIoU':>10s} {'best':>8s} {'time':>8s}")
    print("-" * 62)
    for r in results:
        print(f"{r['name']:8s} {r['params_M']:8.2f}M {r['iou']:10.4f} "
              f"{r['best']:8.4f} {r['secs']:7.0f}s")
    print("=" * 62)


if __name__ == "__main__":
    main()
