"""Download extra instance-segmentation datasets from GitHub releases and convert them to the
CVSegNet layout (data/<name>/images/*.jpg + data/<name>/masks/*.png, one integer id per object).

Datasets (all class-agnostic here; every polygon becomes one instance):
  package   Ultralytics package-seg   (YOLO-seg polygons)  - sampled from the train split
  carparts  Ultralytics carparts-seg  (YOLO-seg polygons)  - sampled from the train split
  balloon   Matterport balloon dataset (VIA json polygons) - train + val, all images

The COCO / VOC downloads in prepare_data.py are on hosts some networks block; these come from
github.com release assets. Check each dataset's licence before redistributing it.

    python tools/add_datasets.py                       # all three, 250 images from the big ones
    python tools/add_datasets.py --datasets balloon    # just one
    python tools/add_datasets.py --per-dataset 500     # larger samples (slower training)
"""
import argparse
import glob
import io
import json
import os
import random
import urllib.request
import zipfile

from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URLS = {
    "package": "https://github.com/ultralytics/assets/releases/download/v0.0.0/package-seg.zip",
    "carparts": "https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip",
    "balloon": "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip",
}
MAX_SIDE = 640          # downscale so loading is fast; train.py resizes again anyway
MAX_INSTANCES = 255     # masks are 8-bit PNGs


def _fetch(name, cache):
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, os.path.basename(URLS[name]))
    if not os.path.exists(dest):
        print(f"downloading {URLS[name]}")
        urllib.request.urlretrieve(URLS[name], dest)
    return dest


def _save(img, polys_px, scale, out, stem):
    """polys_px: list of flat [x0,y0,x1,y1,...] in ORIGINAL pixel coordinates."""
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    sx, sy = img.size[0] / w, img.size[1] / h
    mask = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(mask)
    n = 0
    for poly in polys_px[:MAX_INSTANCES]:
        if len(poly) >= 6:
            n += 1
            d.polygon([(x * sx, y * sy) for x, y in zip(poly[0::2], poly[1::2])], fill=n)
    if n == 0:
        return 0                                   # skip images with no usable object
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)
    img.convert("RGB").save(os.path.join(out, "images", stem + ".jpg"), quality=92)
    mask.save(os.path.join(out, "masks", stem + ".png"))
    return n


def convert_yolo(zip_path, out, limit, seed, tag):
    z = zipfile.ZipFile(zip_path)
    names = z.namelist()
    imgs = sorted(n for n in names if n.startswith("images/train/") and n.lower().endswith((".jpg", ".jpeg", ".png")))
    random.Random(seed).shuffle(imgs)
    done = inst = 0
    for n in imgs:
        if done >= limit:
            break
        lbl = n.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
        if lbl not in names:
            continue
        img = Image.open(io.BytesIO(z.read(n)))
        W, H = img.size
        polys = []
        for line in z.read(lbl).decode().splitlines():
            p = line.split()
            if len(p) < 7:
                continue
            c = [float(v) for v in p[1:]]
            polys.append([v * (W if i % 2 == 0 else H) for i, v in enumerate(c)])
        k = _save(img, polys, 1.0, out, f"{tag}_{done:05d}")
        if k:
            done += 1
            inst += k
    print(f"{tag}: {done} images, {inst} instances -> {out}")


def convert_balloon(zip_path, out):
    z = zipfile.ZipFile(zip_path)
    done = inst = 0
    for split in ("train", "val"):
        meta = json.loads(z.read(f"balloon/{split}/via_region_data.json"))
        for v in meta.values():
            fn = v["filename"]
            polys = []
            regions = v["regions"]
            regions = regions.values() if isinstance(regions, dict) else regions
            for r in regions:
                sa = r["shape_attributes"]
                if "all_points_x" in sa:
                    xs, ys = sa["all_points_x"], sa["all_points_y"]
                    polys.append([c for xy in zip(xs, ys) for c in xy])
            img = Image.open(io.BytesIO(z.read(f"balloon/{split}/{fn}")))
            k = _save(img, polys, 1.0, out, f"balloon_{done:05d}")
            if k:
                done += 1
                inst += k
    print(f"balloon: {done} images, {inst} instances -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(URLS), choices=list(URLS))
    ap.add_argument("--per-dataset", type=int, default=250, help="images sampled from package / carparts")
    ap.add_argument("--cache", default=os.path.join(REPO, ".dataset_cache"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for name in a.datasets:
        zp = _fetch(name, a.cache)
        out = os.path.join(REPO, "data", name)
        if name == "balloon":
            convert_balloon(zp, out)
        else:
            convert_yolo(zp, out, a.per_dataset, a.seed, name)


if __name__ == "__main__":
    main()
