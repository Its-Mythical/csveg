"""COCO-style instance-segmentation evaluation + post-processing tuner for CVSegNet.

Why: foreground IoU (what train.py prints) ignores whether objects are split correctly. This
tool reports class-agnostic mask AP (AP@[.5:.95], AP50, AP75, AR) with pycocotools on the same
held-out split train.py used (same seed / --n-val), so numbers are comparable between models.

What it does
  1. input size x test-time-augmentation comparison (accuracy and latency)
  2. tunes the post-processing parameters (thresholds, watershed / embedding settings, instance
     score) by coordinate descent on validation AP, with 2-fold cross-fitting so the headline
     number is NOT measured on the images the parameters were tuned on
  3. optionally writes the chosen settings to weights/postproc.json, which app.py loads

    pip install pycocotools
    python tools/eval_ap.py --checkpoint weights/cvseg_unet.pt --n-val 60 \
        --write-postproc weights/postproc.json --out-json weights/eval_ap.json

Caveat: with ~60 validation images, AP differences of ~0.02 are within noise.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from model import CVSegNet, POSTPROC_DEFAULTS, extract_instances, predict  # noqa: E402
from train import gather_items  # noqa: E402

torch.set_num_threads(2)


# --------------------------------------------------------------------------- #
def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = CVSegNet(pretrained=False, backbone=ck.get("backbone", "resnet18"),
                     decoder_width=ck.get("decoder_width", 1.0), embed_dim=ck.get("embed_dim", 0))
    model.load_state_dict(ck["model"])
    return model.eval(), ck


def load_val(root, n_val, seed):
    """Same split as train.py: all items, shuffled with default_rng(seed), first n_val."""
    items = gather_items(root, None)
    np.random.default_rng(seed).shuffle(items)
    out = []
    for img_path, mask_path, ds in items[:n_val]:
        img = np.array(Image.open(img_path).convert("RGB"))
        m = np.array(Image.open(mask_path))
        m = m[..., 0] if m.ndim == 3 else m
        out.append({"ds": ds, "img": img, "inst": m, "name": os.path.basename(img_path)})
    return out


def build_gt(val):
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    images, anns = [], []
    for i, v in enumerate(val):
        h, w = v["inst"].shape
        images.append({"id": i, "height": h, "width": w})
        for k in np.unique(v["inst"]):
            if k == 0:
                continue
            m = np.asfortranarray((v["inst"] == k).astype(np.uint8))
            if m.sum() == 0:
                continue
            rle = mu.encode(m)
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1, "iscrowd": 0,
                         "segmentation": rle, "area": float(mu.area(rle)),
                         "bbox": mu.toBbox(rle).tolist()})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": 1, "name": "object"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def compute_maps(model, val, size, tta):
    """Run the network once per image; returns (maps, mean seconds per image)."""
    maps, t = [], 0.0
    predict(model, val[0]["img"], size=size, tta=tta, return_embed=True)   # warm-up (not timed)
    for v in val:
        t0 = time.perf_counter()
        out = predict(model, v["img"], size=size, tta=tta, return_embed=True)
        t += time.perf_counter() - t0
        maps.append(out)
    return maps, t / max(len(val), 1)


def detections(maps, idx, params):
    from pycocotools import mask as mu
    dets = []
    for i in idx:
        fg, ctr, bnd, emb = maps[i]
        labels, inst = extract_instances(
            fg, ctr, bnd, fg_thr=params["fg_thr"], ctr_thr=params["ctr_thr"],
            bnd_weight=params["bnd_weight"], min_area=params["min_area"],
            min_distance=params["min_distance"], embed=emb, mode=params["mode"],
            emb_weight=params["emb_weight"], emb_spatial=params["emb_spatial"],
            score=params["score"])
        for r in inst:
            rle = mu.encode(np.asfortranarray((labels == r["id"]).astype(np.uint8)))
            dets.append({"image_id": int(i), "category_id": 1, "segmentation": rle,
                         "score": float(r["score"])})
    return dets


def coco_eval(gt, dets, idx):
    from pycocotools.cocoeval import COCOeval
    if not dets:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        e = COCOeval(gt, dt, "segm")
        e.params.imgIds = [int(i) for i in idx]
        e.params.maxDets = [1, 10, 300]
        e.evaluate(); e.accumulate(); e.summarize()
    s = [max(float(x), 0.0) for x in e.stats]
    return {"AP": s[0], "AP50": s[1], "AP75": s[2], "AR": s[8]}


def score(gt, maps, idx, params):
    return coco_eval(gt, detections(maps, idx, params), idx)


# --------------------------------------------------------------------------- #
def make_grid(has_embed):
    g = {"fg_thr": [0.35, 0.5, 0.65], "ctr_thr": [0.15, 0.25, 0.35, 0.5],
         "bnd_weight": [0.3, 0.7, 1.2], "min_distance": [4, 7, 11],
         "min_area": [20, 40, 80], "score": ["ctr", "fg", "mix"]}
    if has_embed:
        g = {"mode": ["watershed", "hybrid", "embedding"], "emb_weight": [0.5, 1.0, 2.0],
             "emb_spatial": [0.0, 5.0, 20.0], **g}
    return g


def tune(gt, maps, idx, start, grid, objective="AP50"):
    """One pass of coordinate descent on ``objective`` over the images ``idx``."""
    best = dict(start)
    best_ap = score(gt, maps, idx, best)[objective]
    for key, values in grid.items():
        for v in values:
            if v == best[key]:
                continue
            trial = {**best, key: v}
            ap = score(gt, maps, idx, trial)[objective]
            if ap > best_ap + 1e-6:
                best, best_ap = trial, ap
    return best, best_ap


def fmt(m):
    return f"AP {m['AP']*100:5.1f} | AP50 {m['AP50']*100:5.1f} | AP75 {m['AP75']*100:5.1f} | AR {m['AR']*100:5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(REPO, "weights", "cvseg_unet.pt"))
    ap.add_argument("--root", default=REPO, help="folder containing data/ (default: repo)")
    ap.add_argument("--n-val", type=int, default=60, help="must equal the --n-val used in training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", type=int, nargs="*", default=None,
                    help="input sizes to compare (default: train size and 320)")
    ap.add_argument("--tta", nargs="*", default=["none", "h", "hv"])
    ap.add_argument("--objective", default="AP50", choices=["AP", "AP50", "AP75", "AR"],
                    help="metric the size/TTA choice and the tuner maximise. AP50 by default: "
                         "for a weak model strict AP@[.5:.95] is ~0 and gives the tuner nothing "
                         "to climb; use AP once masks are good")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--write-postproc", default=None)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    model, ck = load_model(a.checkpoint)
    has_embed = model.embed_dim > 0
    train_size = int(ck.get("train_size", 192))
    sizes = a.sizes or sorted({train_size, 320})
    val = load_val(a.root, a.n_val, a.seed)
    gt = build_gt(val)
    allidx = list(range(len(val)))
    print(f"checkpoint {os.path.basename(a.checkpoint)} (epoch {ck.get('epoch')}, "
          f"embed_dim {model.embed_dim}) | {len(val)} val images, "
          f"{len(gt.dataset['annotations'])} instances", flush=True)

    # ---- 1. input size x TTA, default post-processing ----------------------
    report = {"objective": a.objective, "checkpoint": os.path.basename(a.checkpoint), "n_val": len(val),
              "n_instances": len(gt.dataset["annotations"]), "size_tta": []}
    base = dict(POSTPROC_DEFAULTS)
    print("\n== input size x TTA (default post-processing, mode=watershed) ==")
    for size in sizes:
        for tta in a.tta:
            # maps are large (embedding at full resolution): keep only one combination in
            # memory at a time, and recompute the chosen one afterwards
            maps, sec = compute_maps(model, val, size, "none" if tta == "none" else tta)
            m = score(gt, maps, allidx, {**base, "size": size})
            del maps
            report["size_tta"].append({"size": size, "tta": tta, "ms_per_image": round(sec * 1000),
                                       **{k: round(v, 4) for k, v in m.items()}})
            print(f"size {size:3d} tta {tta:4s} | {fmt(m)} | {sec*1000:5.0f} ms/img", flush=True)

    obj = a.objective

    def pick():
        rows = report["size_tta"]
        best = max(rows, key=lambda r: r[obj])
        none_same = next((r for r in rows if r["size"] == best["size"] and r["tta"] == "none"),
                         best)
        if best["tta"] != "none" and best[obj] - none_same[obj] < 0.005:
            best = none_same                      # TTA must buy >= 0.5 points to be worth it
        return best["size"], best["tta"]

    size, tta = pick()
    maps, _ = compute_maps(model, val, size, tta)
    print(f"\nchosen: size {size}, tta {tta}")
    report["chosen"] = {"size": size, "tta": tta}
    params0 = {**base, "size": size, "tta": tta}
    default_m = score(gt, maps, allidx, params0)
    report["default_postproc"] = {k: round(v, 4) for k, v in default_m.items()}

    if has_embed:
        print("\n== instance mode (default settings) ==")
        report["modes"] = {}
        for mode in ("watershed", "hybrid", "embedding"):
            m = score(gt, maps, allidx, {**params0, "mode": mode})
            report["modes"][mode] = {k: round(v, 4) for k, v in m.items()}
            print(f"{mode:9s} | {fmt(m)}", flush=True)

    if a.no_tune:
        print("\ndefault post-processing:", fmt(default_m))
        final = params0
    else:
        # ---- 2. tuning with 2-fold cross-fitting ---------------------------
        grid = make_grid(has_embed)
        folds = [allidx[0::2], allidx[1::2]]
        held_dets = []
        print("\n== tuning post-processing (2-fold cross-fit) ==", flush=True)
        for f in (0, 1):
            tr, te = folds[1 - f], folds[f]           # tune on one half, test on the other
            p, _ = tune(gt, maps, tr, params0, grid, obj)
            held_dets += detections(maps, te, p)
        held = coco_eval(gt, held_dets, allidx)
        final, tuned_ap = tune(gt, maps, allidx, params0, grid, obj)
        tuned_m = score(gt, maps, allidx, final)
        print(f"default settings           | {fmt(default_m)}")
        print(f"tuned, in-sample (optimist.)| {fmt(tuned_m)}")
        print(f"tuned, cross-fitted (honest)| {fmt(held)}")
        print("final settings:", json.dumps({k: final[k] for k in final}))
        report.update({"tuned_in_sample": {k: round(v, 4) for k, v in tuned_m.items()},
                       "tuned_cross_fitted": {k: round(v, 4) for k, v in held.items()},
                       "final_params": final})

    # ---- 3. per-dataset AP (final settings; in-sample) ------------------------
    print("\n== per-dataset AP (final settings; few images each, noisy) ==")
    report["per_dataset"] = {}
    for ds in sorted({v["ds"] for v in val}):
        idx = [i for i, v in enumerate(val) if v["ds"] == ds]
        m = score(gt, maps, idx, final)
        report["per_dataset"][ds] = {"images": len(idx), **{k: round(v, 4) for k, v in m.items()}}
        print(f"{ds:9s} ({len(idx):2d} imgs) | {fmt(m)}")

    if a.write_postproc:
        os.makedirs(os.path.dirname(os.path.abspath(a.write_postproc)), exist_ok=True)
        with open(a.write_postproc, "w") as f:
            json.dump({k: final[k] for k in POSTPROC_DEFAULTS}, f, indent=2)
        print("wrote", a.write_postproc)
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print("wrote", a.out_json)


if __name__ == "__main__":
    main()
