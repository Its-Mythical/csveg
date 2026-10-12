"""
Prepare training data for CVSegNet.

CVSegNet is a *binary* (class-agnostic) instance segmenter, so every dataset is
normalised to one simple layout:

    <out>/images/<stem>.<ext>     the photo
    <out>/masks/<stem>.png        an 8-bit PNG where each object instance is a
                                  distinct integer id (0 = background)

Then `train.py` just reads those two folders.

Subcommands
-----------
  coco-seg   convert a YOLO-seg dataset (e.g. Ultralytics coco128-seg) to the layout
  coco       download COCO 2017 val + annotations and convert the first N images
  voc        download VOC 2012 and convert the first N images

Examples
--------
  python prepare_data.py coco-seg --src data/coco_seg/coco128-seg --out data/coco_seg
  python prepare_data.py coco --out data/coco --limit 500
  python prepare_data.py voc  --out data/voc  --limit 1000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageDraw

COCO_VAL_IMG = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VOC_TAR = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"


def _progress(count, block, total):
    if total > 0:
        print(f"\r  {int(min(100, count*block*100/total)):3d}%", end="", flush=True)


def _download(url, dest):
    if os.path.exists(dest):
        print(f"  have {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _out_dirs(out):
    img = os.path.join(out, "images")
    msk = os.path.join(out, "masks")
    os.makedirs(img, exist_ok=True)
    os.makedirs(msk, exist_ok=True)
    return img, msk


def _draw_polygons(inst_mask_img, polys, value):
    d = ImageDraw.Draw(inst_mask_img)
    for poly in polys:
        if len(poly) >= 6:
            d.polygon(list(zip(poly[0::2], poly[1::2])), fill=value)


# --------------------------------------------------------------------------- #
def convert_yolo_seg(src, out):
    """Ultralytics coco*-seg layout: images/train2017 + labels/train2017/*.txt."""
    img_in = os.path.join(src, "images", "train2017")
    lbl_in = os.path.join(src, "labels", "train2017")
    img_out, msk_out = _out_dirs(out)
    n_inst = 0
    files = sorted(glob.glob(os.path.join(img_in, "*.jpg")))
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        im = Image.open(f)
        W, H = im.size
        im.save(os.path.join(img_out, f"{stem}.jpg"), quality=95)
        mask = Image.new("L", (W, H), 0)
        lbl = os.path.join(lbl_in, stem + ".txt")
        value = 1
        if os.path.exists(lbl):
            for line in open(lbl):
                p = line.split()
                if len(p) < 7:
                    continue
                coords = [float(v) for v in p[1:]]
                poly = []
                for x, y in zip(coords[0::2], coords[1::2]):
                    poly += [x * W, y * H]
                _draw_polygons(mask, [poly], value)
                value += 1
                n_inst += 1
        mask.save(os.path.join(msk_out, f"{stem}.png"))
    print(f"coco-seg: {len(files)} images, {n_inst} instances -> {out}")


# --------------------------------------------------------------------------- #
def download_coco(out, limit=500):
    root = os.path.join(out, "_raw")
    os.makedirs(root, exist_ok=True)
    ann_zip = os.path.join(root, "annotations_trainval2017.zip")
    img_zip = os.path.join(root, "val2017.zip")
    _download(COCO_ANN, ann_zip)
    _download(COCO_VAL_IMG, img_zip)
    for z in (ann_zip, img_zip):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)

    with open(os.path.join(root, "annotations", "instances_val2017.json")) as fh:
        coco = json.load(fh)
    by_img = {}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) or not isinstance(a["segmentation"], list):
            continue
        by_img.setdefault(a["image_id"], []).append(a)

    img_out, msk_out = _out_dirs(out)
    n = 0
    for info in coco["images"]:
        if n >= limit:
            break
        anns = by_img.get(info["id"])
        if not anns:
            continue
        src = os.path.join(root, "val2017", info["file_name"])
        if not os.path.exists(src):
            continue
        im = Image.open(src).convert("RGB")
        W, H = im.size
        stem = os.path.splitext(info["file_name"])[0]
        im.save(os.path.join(img_out, f"{stem}.jpg"), quality=95)
        mask = Image.new("L", (W, H), 0)
        for i, a in enumerate(anns, start=1):
            _draw_polygons(mask, a["segmentation"], i)
        mask.save(os.path.join(msk_out, f"{stem}.png"))
        n += 1
    print(f"coco: {n} images -> {out}")


# --------------------------------------------------------------------------- #
def download_voc(out, limit=1000):
    root = os.path.join(out, "_raw")
    os.makedirs(root, exist_ok=True)
    tar = os.path.join(root, "VOCtrainval_11-May-2012.tar")
    _download(VOC_TAR, tar)
    if not os.path.isdir(os.path.join(root, "VOCdevkit")):
        print("  extracting...")
        with tarfile.open(tar) as t:
            t.extractall(root)

    voc = os.path.join(root, "VOCdevkit", "VOC2012")
    obj_dir = os.path.join(voc, "SegmentationObject")   # instance ids per pixel
    cls_dir = os.path.join(voc, "SegmentationClass")    # class ids per pixel
    sets = os.path.join(voc, "ImageSets", "Segmentation", "trainval.txt")
    ids = [l.strip() for l in open(sets) if l.strip()]

    img_out, msk_out = _out_dirs(out)
    n = 0
    for vid in ids:
        if n >= limit:
            break
        jpg = os.path.join(voc, "JPEGImages", f"{vid}.jpg")
        obj = os.path.join(obj_dir, f"{vid}.png")
        if not (os.path.exists(jpg) and os.path.exists(obj)):
            continue
        Image.open(jpg).convert("RGB").save(os.path.join(img_out, f"{vid}.jpg"), quality=95)
        m = np.array(Image.open(obj))          # palette index == instance id
        if m.ndim == 3:
            m = m[..., 0]
        m = m.astype(np.uint8)
        m[m == 255] = 0                        # void -> background
        Image.fromarray(m).save(os.path.join(msk_out, f"{vid}.png"))
        n += 1
    print(f"voc: {n} images -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coco-seg"); c.add_argument("--src", required=True); c.add_argument("--out", required=True)
    c = sub.add_parser("coco"); c.add_argument("--out", required=True); c.add_argument("--limit", type=int, default=500)
    c = sub.add_parser("voc"); c.add_argument("--out", required=True); c.add_argument("--limit", type=int, default=1000)
    a = ap.parse_args()
    if a.cmd == "coco-seg":
        convert_yolo_seg(a.src, a.out)
    elif a.cmd == "coco":
        download_coco(a.out, a.limit)
    elif a.cmd == "voc":
        download_voc(a.out, a.limit)


if __name__ == "__main__":
    main()
