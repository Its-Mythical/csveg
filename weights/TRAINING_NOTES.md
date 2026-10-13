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
