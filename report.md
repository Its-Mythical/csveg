# CVSegNet - model report

_Generated 2026-10-13 16:01_

> Hardware-capacity details (RAM, VRAM, core count) are intentionally omitted; the model runs on CPU or CUDA and the report is device-agnostic.

## 1. Summary

Evaluated the checkpoint `cvseg_unet.pt` on 30 held-out images. It was trained to epoch 36 (its own val fg-IoU 0.382).

- **Foreground IoU: 0.480** (0 = nothing, 1 = perfect)
- **Instance count exactly right on 0%** of images, within +/-1 on 7% (mean absolute error 9.50)
- **Merge rate on multi-object images: 0%** (21 such images)
- Mean instance confidence: 0.903
- **Tends to over-segment**: 12.0 instances predicted vs 2.5 true on average - raise `ctr_thr` / `min_area`, or train longer

## 2. Model

| property | value |
|---|---|
| backbone | resnet18 |
| decoder width | 1.0x |
| output stride | 2 |
| parameters | 12.06 M (12.06 M trainable) |
| heads | foreground + instance-centre + boundary |
| inference size used here | 192px |

## 3. Capability checks

| check | result |
|---|---|
| boundary-aware watershed splits touching objects | PASS |
| distance-transform fallback when no centre peak exists | PASS |

## 4. What was fixed

| area | before | now |
|---|---|---|
| Instance separation | watershed over the distance transform only (touching objects merged) | NEW boundary head; watershed energy is now -distance + weight x boundary, so it cuts along learned seams |
| Mask sharpness | decoder reached stride 4 | FPN-style decoder reaches stride 2 (sharper masks) |
| False positives | BCE + Dice on the foreground | focal + Dice + BCE; instance score = mean centre confidence |
| Capacity | fixed ResNet-18 (~13.6 M params) | configurable ResNet-18/50 + decoder width; default is the large config (~26 M) |
| Small-batch stability | BatchNorm everywhere (unreliable at batch 1) | frozen-BN encoder + GroupNorm decoder |
| Memory | no saving mechanism | gradient checkpointing + early-layer freezing; guard fixed so it actually engages when layers are frozen |
| Checkpoints | cvseg_unet.pt overwritten every epoch | best kept at cvseg_best.pt and published to cvseg_unet.pt |
| Data | one hard-coded dataset | multi-dataset auto-discovery + prepare_data.py (COCO / VOC / YOLO-seg) |
| GPU correctness | validation / predict did not move data to the model's device | fixed - on a GPU, training previously crashed at the first validation |
| Web app | deprecated startup hook; model could fail to load | lifespan + lazy load, so the API works regardless of startup timing |
| Tooling | manual pip installs | install_requirements.py (venv-safe), test_all.py (27 checks), benchmark.py |

## 5. Areas of work

1. Near-fully-overlapping objects can still merge - inherent to watershed; a learned instance-embedding + clustering head would push this further.
2. Class-agnostic: it segments 'objects', not categories. The COCO subset is therefore treated as generic foreground, not 80 classes.
3. Validation is noisy (a small held-out split) and uses foreground IoU - an official COCO-style AP would be a fairer, more comparable number.
4. Trained from scratch on modest data; a longer schedule or a segmentation-pretrained initialisation would likely improve masks.
5. Test-time augmentation exists but is off by default; enabling it should sharpen boundaries at some cost in latency.
6. Several hyper-parameters are fixed defaults (centre-target sigma, peak min-distance, boundary weight) and could be tuned per dataset.
7. Mixed precision is wired but could not be exercised on a GPU here, so its GPU behaviour is unverified in this environment.

## 6. Reproduce

```bash
python install_requirements.py          # dependencies
python train.py --size 256 --batch 1 --grad-accum 8 --amp
python test_all.py                      # 27 self-checks
python benchmark.py --models light large --epochs 3
python report.py                        # this report
```

---
_Metrics are measured live on the held-out split; a small split makes the numbers noisy, so read them alongside the loss trend and the masks themselves._