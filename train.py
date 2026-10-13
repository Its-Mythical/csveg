"""
Train CVSegNet on one or more instance-segmentation datasets.

The network learns three things at once:
  * fg  head  -> is this pixel an object?
  * ctr head  -> a Gaussian heat-map peaked at each object's centre
  * bnd head  -> the seams between touching objects (so the watershed splits)

Resumable training
------------------
Checkpoints are written every `--save-every-steps` optimiser steps, at each
epoch end, and on Ctrl-C. A checkpoint stores the model, optimiser, AMP scaler,
LR-scheduler, epoch, how many batches of that epoch are done, the global step,
the best score and the RNG states - so a resumed run continues exactly where it
left off (same data order, same LR, same augmentation stream).

    --resume auto     resume from weights/cvseg_last.pt if it exists (default)
    --resume none     always start fresh
    --resume PATH     resume from a specific checkpoint

`--max-steps N` stops cleanly after N optimiser steps, which makes chunked
training (and testing resume) easy.

Usage
-----
    # bigger model (DEFAULT) -- resnet50, 1.5x decoder, ~26 M params
    python train.py --size 256 --batch 1 --grad-accum 8 --amp

    # original scale (fast, CPU-friendly)
    python train.py --backbone resnet18 --decoder-width 1.0 --no-checkpoint \
        --freeze-at 0 --size 192 --batch 4

    # pick datasets explicitly (default: all found)
    python train.py --datasets penn coco_seg

    # train in chunks
    python train.py --max-steps 500        # then run it again to continue
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model import (CVSegNet, build_boundary_target, build_center_target,
                   cvseg_loss)

torch.set_num_threads(2)

ROOT = os.path.dirname(os.path.abspath(__file__))
W_DIR = os.path.join(ROOT, "weights")
LAST = os.path.join(W_DIR, "cvseg_last.pt")
BEST = os.path.join(W_DIR, "cvseg_best.pt")
PUBLISH = os.path.join(W_DIR, "cvseg_unet.pt")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def _resize(img, size, nearest=False):
    return np.array(Image.fromarray(img).resize(size, Image.NEAREST if nearest else Image.BILINEAR))


# --------------------------------------------------------------------------- #
# data discovery
# --------------------------------------------------------------------------- #
def gather_items(root: str, datasets=None):
    """Find (image_path, mask_path, dataset_name) triples under data/."""
    items = []
    data_dir = os.path.join(root, "data")

    penn = os.path.join(data_dir, "penn", "PennFudanPed")
    if os.path.isdir(os.path.join(penn, "PNGImages")):
        for f in sorted(glob.glob(os.path.join(penn, "PNGImages", "*"))):
            if f.lower().endswith(IMG_EXT):
                stem = os.path.splitext(os.path.basename(f))[0]
                items.append((f, os.path.join(penn, "PedMasks", f"{stem}_mask.png"), "penn"))

    for d in sorted(glob.glob(os.path.join(data_dir, "*"))):
        img_dir = os.path.join(d, "images")
        msk_dir = os.path.join(d, "masks")
        if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
            continue
        name = os.path.basename(d)
        for f in sorted(glob.glob(os.path.join(img_dir, "*"))):
            if f.lower().endswith(IMG_EXT):
                stem = os.path.splitext(os.path.basename(f))[0]
                items.append((f, os.path.join(msk_dir, f"{stem}.png"), name))

    if datasets:
        items = [it for it in items if it[2] in set(datasets)]
    return items


class InstanceMaskDataset(Dataset):
    def __init__(self, items, size=192, train=True):
        self.items = items
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.items)

    def _augment(self, img, inst):
        S = self.size
        if self.train:
            s = np.random.uniform(0.7, 1.4)
            h, w = img.shape[:2]
            nh, nw = max(64, int(h * s)), max(64, int(w * s))
            img = _resize(img, (nw, nh))
            inst = _resize(inst, (nw, nh), nearest=True)
            h, w = img.shape[:2]
            if h >= S and w >= S:
                y0 = np.random.randint(0, h - S + 1)
                x0 = np.random.randint(0, w - S + 1)
                img = img[y0:y0 + S, x0:x0 + S]
                inst = inst[y0:y0 + S, x0:x0 + S]
            else:
                pad = ((0, max(0, S - h)), (0, max(0, S - w)), (0, 0))
                img = np.pad(img, pad, mode="reflect")
                inst = np.pad(inst, pad[:2], mode="reflect")
                img = _resize(img, (S, S))
                inst = _resize(inst, (S, S), nearest=True)
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
                inst = inst[:, ::-1].copy()
        else:
            img = _resize(img, (S, S))
            inst = _resize(inst, (S, S), nearest=True)
        return img, inst

    def __getitem__(self, i):
        img_path, mask_path, _ = self.items[i]
        img = np.array(Image.open(img_path).convert("RGB"))
        inst = np.array(Image.open(mask_path))
        if inst.ndim == 3:
            inst = inst[..., 0]
        img, inst = self._augment(img, inst)

        if self.train:
            img = img.astype(np.float32)
            img *= np.random.uniform(0.85, 1.15)
            img += np.random.uniform(-18, 18)
            img = np.clip(img, 0, 255).astype(np.uint8)

        x = img.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        fg = torch.from_numpy((inst > 0).astype(np.float32))[None]
        ctr = torch.from_numpy(build_center_target(inst, self.size))[None]
        bnd = torch.from_numpy(build_boundary_target(inst))[None]
        return x, fg, ctr, bnd


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    dev = next(model.parameters()).device          # move val data to the model's device
    inter = union = 0.0
    for x, fg, _, _ in loader:
        x = x.to(dev)
        fg = fg.to(dev)
        p = torch.sigmoid(model(x)[0]) > 0.5
        t = fg > 0.5
        inter += (p & t).sum().item()
        union += (p | t).sum().item()
    model.train()
    return inter / max(union, 1)


# --------------------------------------------------------------------------- #
# resumable-training helpers
# --------------------------------------------------------------------------- #
def _rng_state():
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(state):
    if not state:
        return
    try:
        torch.set_rng_state(state["torch"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
    except Exception as e:                       # noqa: BLE001
        print(f"  (could not restore RNG state: {e})")


def _config_meta(args):
    return {"backbone": args.backbone, "decoder_width": args.decoder_width,
            "size": args.size, "grad_accum": args.grad_accum,
            "freeze_at": args.freeze_at}


def save_checkpoint(path, model, opt, scaler, sched, epoch, batch_in_epoch,
                    global_step, best, args):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "sched": sched.state_dict(),
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "best": best,
        "rng": _rng_state(),
        "meta": _config_meta(args),
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--decoder-width", type=float, default=1.5)
    ap.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True,
                    help="gradient checkpointing (saves VRAM, slower)")
    ap.add_argument("--freeze-at", type=int, default=2,
                    help="freeze early encoder stages (0..4)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=1e9)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop cleanly after this many optimiser steps (0 = off)")
    ap.add_argument("--save-every-steps", type=int, default=200)
    ap.add_argument("--resume", default="auto",
                    help="'auto' | 'none' | path to a checkpoint")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(W_DIR, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    items = gather_items(ROOT, args.datasets)
    if not items:
        raise SystemExit("no data found - run prepare_data.py first")
    from collections import Counter
    counts = Counter(it[2] for it in items)
    print(f"data: {len(items)} images " +
          " ".join(f"[{k}={v}]" for k, v in counts.items()), flush=True)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(items)
    n_val = min(args.n_val, max(1, len(items) // 10))
    val_items, train_items = items[:n_val], items[n_val:]
    print(f"train={len(train_items)} val={len(val_items)} device={device}", flush=True)

    # a seeded generator makes the per-epoch data order reproducible, which is
    # what lets us skip already-done batches on resume
    gen = torch.Generator()
    train_ds = InstanceMaskDataset(train_items, args.size, True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=True, generator=gen)
    val_dl = DataLoader(InstanceMaskDataset(val_items, args.size, False),
                        batch_size=args.batch, num_workers=0)
    batches_per_epoch = len(train_dl)
    if batches_per_epoch == 0:
        raise SystemExit("no training batches - dataset smaller than --batch?")
    print(f"batches/epoch={batches_per_epoch} grad_accum={args.grad_accum}", flush=True)

    model = CVSegNet(pretrained=True, backbone=args.backbone,
                     decoder_width=args.decoder_width,
                     checkpoint=args.checkpoint, freeze_at=args.freeze_at).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {args.backbone} width={args.decoder_width} "
          f"| {n_params/1e6:.2f} M trainable params | amp={args.amp}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    total = args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)

    # ---- resume ----
    start_epoch, resume_batches, global_step, best = 0, 0, 0, -1.0
    resume_path = None
    if args.resume == "auto":
        resume_path = LAST if os.path.exists(LAST) else None
    elif args.resume not in ("none", "", None):
        resume_path = args.resume
        if not os.path.exists(resume_path):
            raise SystemExit(f"checkpoint not found: {resume_path}")

    if resume_path:
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        meta = ck.get("meta", {})
        if meta.get("backbone") and meta["backbone"] != args.backbone:
            raise SystemExit(
                f"checkpoint was trained with backbone={meta['backbone']} but you "
                f"asked for {args.backbone}; weights would not load. Use "
                f"--backbone {meta['backbone']} or --resume none.")
        if meta.get("decoder_width") and abs(meta["decoder_width"] - args.decoder_width) > 1e-9:
            raise SystemExit(
                f"checkpoint decoder_width={meta['decoder_width']} != "
                f"{args.decoder_width}; use --decoder-width {meta['decoder_width']} "
                f"or --resume none.")
        if meta.get("size") and meta["size"] != args.size:
            print(f"  note: resuming with --size {args.size} "
                  f"(checkpoint used {meta['size']})", flush=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck.get("epoch", 0)
        resume_batches = ck.get("batch_in_epoch", 0)
        best = ck.get("best", 0.0)
        _restore_rng(ck.get("rng"))

        if ck.get("sched"):
            sched.load_state_dict(ck["sched"])
        else:
            # Older checkpoints (written before this field existed) carry no
            # scheduler state. The optimiser state restores the *current* LR, but
            # the scheduler's own clock is still at 0, so the next sched.step()
            # would jump the LR back up to its starting value. Advance the
            # scheduler to the checkpoint's epoch to stay on the same curve.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(start_epoch):
                    sched.step()
            print(f"  note: checkpoint had no scheduler state; LR schedule "
                  f"advanced to epoch {start_epoch}", flush=True)

        global_step = ck.get("global_step")
        if not global_step:
            global_step = start_epoch * batches_per_epoch // max(1, args.grad_accum)
            if global_step:
                print(f"  note: checkpoint had no global_step; estimated "
                      f"{global_step}", flush=True)
        if not ck.get("meta"):
            print("  note: checkpoint has no config metadata (older format)",
                  flush=True)

        print(f"resumed {os.path.basename(resume_path)}: epoch {start_epoch}, "
              f"{resume_batches}/{batches_per_epoch} batches done, "
              f"global_step {global_step}, best {best:.3f}, "
              f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)

    t0 = time.time()
    stop = False
    epoch, step = start_epoch, -1          # so an early Ctrl-C is safe
    try:
        for epoch in range(start_epoch, total):
            gen.manual_seed(args.seed + epoch)       # reproducible order per epoch
            model.train()
            ep_loss = 0.0
            ep_batches = 0
            opt.zero_grad(set_to_none=True)
            for step, (x, fg, ctr, bnd) in enumerate(train_dl):
                if epoch == start_epoch and step < resume_batches:
                    continue                          # already trained this epoch
                x = x.to(device)
                fg, ctr, bnd = fg.to(device), ctr.to(device), bnd.to(device)
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    fg_l, ctr_l, bnd_l = model(x)
                    losses = cvseg_loss(fg_l, ctr_l, bnd_l, fg, ctr, bnd)
                    loss = losses["total"] / args.grad_accum
                scaler.scale(loss).backward()
                ep_loss += float(losses["total"].detach())
                ep_batches += 1

                if (step + 1) % args.grad_accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 5.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % args.save_every_steps == 0:
                        save_checkpoint(LAST, model, opt, scaler, sched, epoch,
                                        step + 1, global_step, best, args)
                        print(f"  [ckpt] step {global_step} saved", flush=True)
                    if args.max_steps and global_step >= args.max_steps:
                        save_checkpoint(LAST, model, opt, scaler, sched, epoch,
                                        step + 1, global_step, best, args)
                        print(f"reached --max-steps {args.max_steps}; "
                              f"checkpoint saved", flush=True)
                        # quick eval so chunked runs still produce usable weights
                        iou = evaluate(model, val_dl)
                        if iou > best:
                            best = iou
                            meta = {"model": model.state_dict(), "epoch": epoch,
                                    "best": best, "val_iou": iou,
                                    "backbone": args.backbone,
                                    "decoder_width": args.decoder_width}
                            torch.save(meta, BEST)
                            torch.save(meta, PUBLISH)
                            print(f"  [best] val_fgIoU {iou:.4f} -> {PUBLISH}", flush=True)
                        stop = True
                        break
            if stop:
                break

            # flush any leftover accumulated gradients (when batches_per_epoch
            # is not a multiple of grad_accum)
            if (step + 1) % args.grad_accum != 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            sched.step()
            iou = evaluate(model, val_dl)
            is_best = iou > best
            best = max(best, iou)
            print(f"epoch {epoch+1}/{total} loss {ep_loss/max(1,ep_batches):.4f} "
                  f"val_fgIoU {iou:.4f}{' *' if is_best else ''} "
                  f"t={time.time()-t0:.0f}s", flush=True)
            # epoch-end checkpoint (batch_in_epoch=0 -> next epoch starts clean)
            save_checkpoint(LAST, model, opt, scaler, sched, epoch + 1, 0,
                            global_step, best, args)
            if is_best:
                meta = {"model": model.state_dict(), "epoch": epoch + 1,
                        "best": best, "val_iou": iou, "backbone": args.backbone,
                        "decoder_width": args.decoder_width}
                torch.save(meta, BEST)
                torch.save(meta, PUBLISH)
            if time.time() - t0 > args.max_seconds:
                print(f"time budget reached after epoch {epoch+1}; checkpoint saved",
                      flush=True)
                break
    except KeyboardInterrupt:
        # Ctrl-C: save where we are so the run is resumable
        print("\ninterrupted - saving checkpoint", flush=True)
        save_checkpoint(LAST, model, opt, scaler, sched, epoch,
                        step + 1, global_step, best, args)
        print(f"checkpoint saved to {LAST}; re-run to continue", flush=True)
        return

    print(f"done. best val fg IoU {best:.4f} -> {PUBLISH}", flush=True)


if __name__ == "__main__":
    main()
