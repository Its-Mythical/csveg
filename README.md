# CVSegNet — instance segmentation (fixed + improved)

A self-contained instance-segmentation project: a model, training data, a
training script and a web app. Upload an image and the app finds each distinct
object, giving it its own mask, colour, bounding box and confidence.

```
cvseg/
├── app.py            FastAPI backend (loads the model, serves the API + UI)
├── model.py          CVSegNet architecture + inference / instance post-processing
├── train.py          Training script (trains on the bundled datasets)
├── benchmark.py      Compare model sizes on the same split (measures only)
├── prepare_data.py   Convert / download datasets into the expected layout
├── install_requirements.py  Auto-install requirements.txt (CPU/GPU torch)
├── test_all.py       Self-test: model, data, post-processing, app (28 checks)
├── requirements.txt
├── static/           Front-end (index.html, app.js, style.css)
├── weights/          Trained weights land here (cvseg_best.pt, cvseg_unet.pt)
└── data/             Bundled training data
    ├── penn/PennFudanPed/   PNGImages + PedMasks   (170 images, pedestrians)
    └── coco_seg/            images  + masks        (128 images, varied objects)
```

---

## Data

CVSegNet is a **binary (class-agnostic)** instance segmenter, so every dataset is
normalised to one simple layout:

```
data/<name>/images/<stem>.<ext>    the photo
data/<name>/masks/<stem>.png       8-bit PNG, one integer id per object (0 = bg)
```

`train.py` auto-discovers every `data/<name>/{images,masks}` folder and trains on
all of them (or pick with `--datasets`). Two datasets are **bundled**:

| dataset | images | instances | subjects |
|---|---|---|---|
| Penn-Fudan (`penn`) | 170 | 345 | pedestrians |
| COCO-Seg subset (`coco_seg`) | 128 | 929 | varied objects |

### Adding more

```bash
# convert an Ultralytics YOLO-seg dataset (e.g. coco128-seg) to the layout
python prepare_data.py coco-seg --src /path/to/coco128-seg --out data/coco_seg

# download COCO 2017 val + annotations and convert the first 500 images
python prepare_data.py coco --out data/coco --limit 500

# download VOC 2012 and convert the first 1000 images
python prepare_data.py voc --out data/voc --limit 1000
```

The `coco`/`voc` downloads are **not** bundled (COCO val is ~1 GB + 241 MB of
annotations; VOC is ~2 GB). Note that some networks block `cocodataset.org`; any
COCO-format or YOLO-seg mirror works via `prepare_data.py coco-seg`.

---

## What was fixed and improved

| area | before | now |
|---|---|---|
| **instance split** | watershed on distance transform only | **NEW boundary head** — energy is `-distance + weight × boundary`, so touching objects split along learned seams |
| **mask sharpness** | decoder to stride 4 | FPN-style decoder to **stride 2** |
| **false positives** | BCE + Dice | **focal + Dice + BCE**; instance score = mean centre confidence |
| **capacity** | fixed ResNet-18, 13.6 M | configurable **ResNet-18/50** + decoder width |
| **batch-size safety** | BatchNorm everywhere | **frozen-BN encoder + GroupNorm decoder** |
| **VRAM** | — | optional **gradient checkpointing** + early-layer freezing |
| **checkpoints** | overwrote `cvseg_unet.pt` every epoch | keeps **best** at `cvseg_best.pt`, publishes to `cvseg_unet.pt` |
| **data** | one hard-coded dataset | **multi-dataset**, auto-discovered, plus download/conversion tooling |
| **app/train mismatch** | app 160px, training 192px | both default to the same size |

---

## Model sizes

| preset | backbone | decoder | params |
|---|---|---|---|
| `CVSegNet.light()` | resnet18 | 1.0x | ~12 M |
| **default in `train.py`** | **resnet50** | **1.5x** | **~26 M** |
| `CVSegNet.xlarge()` | resnet50 | 2.0x | ~28 M |

The default is the **large** config. Use `--backbone resnet18 --decoder-width
1.0 --no-checkpoint --freeze-at 0` for the fast scale.

---

## Install

```bash
python install_requirements.py        # auto-installs everything in requirements.txt
```

It reads `requirements.txt`, installs the CPU torch wheels by default (`--gpu`
for CUDA, `--upgrade` to update, `--dry-run` to preview), then imports each
package and prints its version. Prefer pip directly?

```bash
pip install -r requirements.txt
```

## Train

```bash
# default: resnet50, 1.5x decoder, ~26 M params, tuned for a ~2 GB GPU
python train.py --size 256 --batch 1 --grad-accum 8 --amp

# choose datasets explicitly
python train.py --datasets penn coco_seg

# fast / CPU
python train.py --backbone resnet18 --decoder-width 1.0 --no-checkpoint \
    --freeze-at 0 --size 192 --batch 4
```

### Resumable training

Checkpoints are written every `--save-every-steps` optimiser steps, at each
epoch end, and on **Ctrl-C**, so a run can always be continued:

```bash
python train.py --max-steps 500          # train 500 steps, then stop cleanly
python train.py --max-steps 1000         # re-run: resumes and continues
python train.py --resume none ...        # force a fresh start
python train.py --resume weights/cvseg_best.pt ...   # resume a specific file
```

A checkpoint stores the model, optimiser, AMP scaler, LR scheduler, epoch, how
many batches of that epoch are done, the global step, the best score and the RNG
states - so a resumed run continues exactly where it left off (same data order,
same LR, same augmentation stream). The best model is kept at
`weights/cvseg_best.pt` and published as `weights/cvseg_unet.pt` (the file the app
loads); chunked runs also evaluate at `--max-steps` so they still produce usable
weights. `--max-seconds` additionally stops a run after a time budget.

### Fitting a 2 GB GPU

| lever | flag | effect |
|---|---|---|
| gradient checkpointing | `--checkpoint` (default on) | biggest activation saving |
| freeze early layers | `--freeze-at 2` (default) | less optimiser memory |
| smaller inputs | `--size 192` | linear saving |
| mixed precision | `--amp` | ~2x activation saving |
| gradient accumulation | `--batch 1 --grad-accum 8` | simulate a bigger batch |
| allocator | `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` | less fragmentation |

If you still OOM, drop `--size` first, then `--freeze-at 3`.

## Benchmark sizes

```bash
python benchmark.py --models light large --epochs 3 --size 160
```

Trains both on the same split and prints params / val foreground IoU / time. It
changes no defaults — it only measures.

## Web app

```bash
python app.py            # http://127.0.0.1:8000
```

---

## API

`POST /api/segment` — multipart form

| field | default | meaning |
|---|---|---|
| `file` | — | the image |
| `fg_thr` | 0.5 | foreground probability threshold |
| `ctr_thr` | 0.35 | centre-heatmap peak threshold |
| `bnd_weight` | 0.7 | how strongly boundaries split the watershed |
| `min_area` | 40 | drop instances smaller than this |

Returns `{count, instances:[{id, area, bbox, score, color}], overlay}`.
`GET /api/health` — model status.

---

## Tests

```bash
python test_all.py        # 28 checks: model, data, post-processing, app
```

Covers: forward on odd / non-square sizes (light + large), that gradient
checkpointing actually engages when `--freeze-at` is used, dataset discovery,
real-image inference at the original resolution, edge cases (blank image, no
centre peaks, touching blobs), training + checkpoint round-trip, and the
web-app endpoints.

## Verification

The default (resnet50, ~26 M trainable) config was run on the bundled data on a
2 GB / 2-core machine: it trains (loss 3.00, val foreground IoU 0.29 after one
epoch — versus ~0.14 for the old untrained path) and the full predict →
boundary-watershed → instances path runs. The multi-dataset loader was verified
to find all 298 bundled images (penn 170 + coco_seg 128) and train a batch. That
host is too small to train to convergence, so **train it on your machine**.

No weights are shipped (the architecture changed, so an old checkpoint would not
load) — run `train.py` once and the app comes alive.

---

## Notes & limitations

- Class-agnostic: it segments "objects", not categories. The COCO-Seg subset is
  therefore treated as generic foreground, not 80 classes.
- Trained on pedestrians + varied objects; retrain for a specific domain.
- The boundary head fixes most touching-object merges, but **almost fully
  overlapping** objects can still merge — inherent to watershed.
- `val_IoU` is foreground IoU on a held-out slice; it is noisy, so trust the
  loss trend and eyeball the masks.
