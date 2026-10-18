# Trained weights

Two full-precision (float32) CVSegNet checkpoints, both ResNet-18 / 1.0x decoder (12.06 M trainable
params), trained on **all five bundled datasets (872 images; 812 train / 60 val)**, input 192 px,
40 epochs, AdamW lr 1e-3 cosine, seed 0, on a 2-core CPU, starting from a pretrained ResNet-18
(timm `resnet18_a1`, fetched by `tools/get_backbone_weights.py`).

| file | recipe | best epoch | val fg-IoU |
|---|---|---|---|
| `cvseg_unet.pt` (**default**, loaded by `app.py`) | *original* code: frozen BN, centre-target sigma bug, no embedding head | 28 | 0.375 |
| `cvseg_improved.pt` | *current* `train.py`: adaptive centre sigma, BN trained, 0.1x encoder LR, 8-d embedding head | 36 | 0.436 |

`postproc.json` / `postproc_improved.json` hold the post-processing settings tuned for each
(`app.py` reads `weights/postproc.json`; use `CVSEG_WEIGHTS=weights/cvseg_improved.pt
CVSEG_POSTPROC=weights/postproc_improved.json python app.py` for the other one).

## Results — read these before trusting either model

Class-agnostic **instance** AP from `tools/eval_ap.py` (pycocotools; in **percent**) on the same
60 held-out images (263 instances). Both models are weak at instance level.

| | `cvseg_unet.pt` | `cvseg_improved.pt` |
|---|---|---|
| val fg-IoU (foreground mask quality) | 0.375 | **0.436** |
| AP @ [.5:.95] | 0.0 | 0.0 |
| AP50, default post-processing (320 px, watershed) | 0.8 | 2.0 |
| AP50, tuned, **cross-fitted** (honest; tuned on one half, tested on the other) | **6.2** | 3.8 |
| AP50, tuned, in-sample (optimistic) | 5.5 | 6.0 |
| AR (max 300 dets), cross-fitted | **7.9** | 4.0 |
| latency, 320 px, no TTA, 2 CPU cores | 97 ms | 102 ms |

What this says, plainly:

* The improved recipe gives a clearly **better foreground mask** (IoU 0.375 -> 0.436), but that did
  **not** turn into better **instances**: on the honest cross-fitted AP50 the original recipe is
  ahead, and its recall is about double. With 60 images / 263 instances the AP50 gap is within
  noise (the in-sample and cross-fitted numbers differ by 1-2 points on their own), but the
  recall gap is not. That is why the original-recipe checkpoint stays the default.
* Strict AP@[.5:.95] is ~0 for both: masks overlap the ground truth, but not tightly enough for
  high-IoU matches. Do not read these as a production-quality instance segmenter.
* The tuned settings were fit on the same 60 images used to pick the checkpoint, so the numbers
  above are mildly optimistic. Per-dataset numbers (in `eval_*.json`) are 3-17 images each.
* Per-dataset AP is dominated by the datasets whose "object" definition matches the model's idea
  of one: `package` (AP50 ~21-27) and `penn` (~9). `coco_seg` and `carparts` are near 2-4.

Things measured in `tools/eval_ap.py` for the improved model (details in `eval_cvseg_improved.json`):

* **Test-time augmentation:** horizontal-flip TTA gained +0.1 AP50 for ~1.7x latency; 4-way flip
  TTA was *worse* and ~3.4x slower. Left **off** by default.
* **Input size:** larger helps (AP50 0.9 / 1.3 / 2.0 at 192 / 256 / 320 px) at 58 / 68 / 102 ms.
* **Instance mode:** `hybrid` (embedding edges added to the watershed energy) 2.8 > `watershed` 2.0
  > pure `embedding` clustering 0.3 (default post-processing).

Not changed: the model is still class-agnostic (item "segments objects, not categories").
Mixed precision: the losses are now computed in float32 and a CPU bf16 autocast check matched fp32
to ~1e-3 with finite gradients, but the real GPU fp16 + GradScaler path has **not** been run.

## Reproduce

```bash
python tools/get_backbone_weights.py            # -> weights/backbones/resnet18.pth

# cvseg_improved.pt  (current code)
python train.py --backbone resnet18 --decoder-width 1.0 --no-checkpoint --freeze-at 0 \
    --size 192 --batch 8 --epochs 40 --lr 1e-3 --encoder-lr-mult 0.1 --embed-dim 8 \
    --n-val 60 --seed 0 --backbone-weights weights/backbones/resnet18.pth

# cvseg_unet.pt  (needs the old code, see the script header)
python tools/train_baseline_old_recipe.py <old-checkout> weights/backbones/resnet18.pth

# score a checkpoint (also tunes post-processing; add --write-postproc weights/postproc.json)
python tools/eval_ap.py --checkpoint weights/cvseg_improved.pt --root . --n-val 60 \
    --sizes 192 256 320 --tta none h hv
```

Training logs: `tools/logs/`. Datasets can be re-created with `tools/add_datasets.py`.

## Older files

`weights/resume/cvseg_last.pt.part*` is the resume state of the **first** model (trained from random
weights on 298 images, val IoU 0.38 on 20 images; its weights are in git history, commit
`044c3cf`). It only works with the original code and is kept for completeness. Rejoin and verify:

```bash
cat weights/resume/cvseg_last.pt.part* > cvseg_last.pt
sha256sum cvseg_last.pt
# expect 060f03de7cee99dde02db436dc85a0b45f0bd53ad34c65364c280a6da9d7c924
```

For a better result with a GPU, run `python train.py` (default resnet50 config, `--amp`).
