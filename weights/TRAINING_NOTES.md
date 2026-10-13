# Trained weights: `cvseg_unet.pt`

Full-precision (float32) CVSegNet checkpoint, the file `app.py` loads.

| item | value |
|---|---|
| backbone / decoder width | resnet18 / 1.0x (12.06 M trainable params) |
| data | bundled `penn` (170) + `coco_seg` (128) = 298 images, 278 train / 20 val |
| input size | 192 |
| schedule | 40 epochs, batch 4, grad-accum 1, AdamW lr 1e-3, cosine, seed 0 |
| best checkpoint | epoch 36, **val foreground IoU 0.3816** (20 val images, so noisy) |
| hardware | 2-core CPU, about 25 minutes |

## Important caveat: trained from random weights

The training machine could not download the ImageNet weights, so the backbone was
**not pretrained** (`CVSegNet(pretrained=False)`). The model also keeps BatchNorm
frozen (`train()` forces BN to eval), which suits pretrained statistics but is a poor
fit for random initialisation. Treat this checkpoint as a working baseline, not a
good model. No code in the repo was changed to produce it.

## Better result: retrain with pretrained weights on a GPU

```bash
pip install -r requirements.txt
python train.py --size 256 --batch 1 --grad-accum 8 --amp   # default resnet50 config
```

The result is written to `weights/cvseg_unet.pt`.

## Resume checkpoint (`weights/resume/`)

`cvseg_last.pt` is the epoch-40 training state (model + optimizer + scheduler + RNG, 145 MB).
It is over GitHub's 100 MB file limit, so it is stored as two parts. Rejoin and verify:

```bash
cat weights/resume/cvseg_last.pt.part* > weights/cvseg_last.pt
sha256sum weights/cvseg_last.pt
# expect 060f03de7cee99dde02db436dc85a0b45f0bd53ad34c65364c280a6da9d7c924
```

Use it with `train.py` only if you keep the same config (`--backbone resnet18 --decoder-width 1.0`).
Note `weights/cvseg_best.pt` is not included: it holds the same weights as `cvseg_unet.pt`.
