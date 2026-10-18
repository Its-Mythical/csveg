# CVSegNet - model report

_Generated 2026-10-16 21:11_

> Hardware-capacity details (RAM, VRAM, core count) are intentionally omitted; the model runs on CPU or CUDA and the report is device-agnostic.

## 1. Summary

**No trained checkpoint was found**, so these are *untrained* baseline numbers. Run `python train.py` first for real results.

- Foreground IoU: 0.217
- Instance count exact: 0%

## 2. Model

| property | value |
|---|---|
| backbone | resnet18 |
| decoder width | 1.0x |
| output stride | 2 |
| parameters | 12.06 M (12.06 M trainable) |
| heads | foreground + instance-centre + boundary |
| inference size used here | 256px |

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
| Tooling | manual pip installs | install_requirements.py (venv-safe), test_all.py (30 checks), benchmark.py, tools/eval_ap.py |
| Centre target | sigma was the image size, so the target was ~flat | adaptive per-object sigma |

## 5. Areas of work

1. Near-fully-overlapping objects can still merge. An instance-embedding head now exists (--embed-dim, 'hybrid' mode) but after tuning it did not beat plain watershed - still open.
2. Class-agnostic: it segments 'objects', not categories (unchanged by design).
3. Evaluation: tools/eval_ap.py now reports COCO-style AP (cross-fitted). Result: AP50 ~4-6 %, strict AP ~0 - instance quality is low. The 60-image val split is still small/noisy.
4. Training: pretrained init + a 40-epoch schedule on 872 images raised foreground IoU to 0.44, but did not raise instance AP; more consistent data / a larger GPU run is next.
5. Test-time augmentation measured: 1-flip ~+0.1 AP50 for ~1.7x latency, 4-flip worse; left off by default (selectable per request).
6. Post-processing hyper-parameters are tuned per checkpoint (weights/postproc*.json); the tuner is fit on the same small val split, so the gains are approximate.
7. Mixed precision: losses are float32 and a CPU bf16 check passes, but the GPU fp16 GradScaler path is still unverified here.

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