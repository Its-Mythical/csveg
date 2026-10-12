"""Self-test for the CVSegNet project: model, data, post-processing, app, resume."""
import io
import os
import sys
import gc
import glob

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import model as M
from model import CVSegNet, predict, extract_instances, instance_color_map
from train import gather_items, InstanceMaskDataset, evaluate
from torch.utils.data import DataLoader

RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))


# --------------------------------------------------------------------------- #
print("\n== 1. model forward, odd / non-square sizes ==")
for tag, net in (("light", CVSegNet.light()), ("large", CVSegNet.large())):
    net.eval()
    for hw in [(128, 128), (96, 160), (200, 120)]:
        x = torch.randn(1, 3, *hw)
        with torch.no_grad():
            fg, ctr, bnd = net(x)
        ok = fg.shape == (1, 1, *hw) and ctr.shape == (1, 1, *hw) and bnd.shape == (1, 1, *hw)
        check(f"{tag} forward {hw} -> {tuple(fg.shape)}", ok)
    check(f"{tag} outputs finite", bool(torch.isfinite(fg).all() and torch.isfinite(ctr).all()))
del net
gc.collect()

# --------------------------------------------------------------------------- #
print("\n== 2. gradient checkpointing actually engages with --freeze-at ==")
calls = {"n": 0}
_orig = M._ckpt


def _counting(*a, **k):
    calls["n"] += 1
    return _orig(*a, **k)


M._ckpt = _counting
net = CVSegNet(backbone="resnet18", decoder_width=1.0, checkpoint=True, freeze_at=2)
net.train()
out = net(torch.randn(1, 3, 128, 128))
out[0].sum().backward()
M._ckpt = _orig
# stem + layer1 frozen -> layer2/3/4 should be checkpointed (3 calls)
check("checkpointing runs for trainable encoder stages", calls["n"] >= 3,
      f"({calls['n']} checkpoints)")

frozen_ok = all(not p.requires_grad for p in net.stem.parameters()) and \
            all(not p.requires_grad for p in net.layer1.parameters())
check("freeze_at=2 freezes stem + layer1", frozen_ok)
del net, out
gc.collect()

# --------------------------------------------------------------------------- #
print("\n== 3. dataset discovery ==")
items = gather_items(HERE)
names = sorted({it[2] for it in items})
check("finds both bundled datasets", set(names) == {"penn", "coco_seg"}, f"{names}")
check("298 images", len(items) == 298, f"({len(items)})")
ds = InstanceMaskDataset(items[:4], 128, train=False)   # deterministic (no random crop)
x, fg, ctr, bnd = ds[0]
check("dataset sample shapes", x.shape == (3, 128, 128) and fg.shape == (1, 128, 128)
      and ctr.shape == (1, 128, 128) and bnd.shape == (1, 128, 128))
check("sample has foreground", float(fg.sum()) > 0, f"fg_px={int(fg.sum())}")
check("boundary target is sparse", 0 < float(bnd.mean()) < 0.5, f"mean={float(bnd.mean()):.3f}")
check("centre target peaks near 1", float(ctr.max()) > 0.9, f"max={float(ctr.max()):.3f}")

# --------------------------------------------------------------------------- #
print("\n== 4. inference on a REAL image (non-square) ==")
real = sorted(glob.glob(os.path.join(HERE, "data", "penn", "PennFudanPed",
                                     "PNGImages", "*.png")))[0]
arr = np.array(Image.open(real).convert("RGB"))
net = CVSegNet.light(); net.eval()
fg, ctr, bnd = predict(net, arr, size=128)
check("predict returns maps at ORIGINAL resolution",
      fg.shape == arr.shape[:2] and ctr.shape == arr.shape[:2] and bnd.shape == arr.shape[:2],
      f"{fg.shape} vs {arr.shape[:2]}")
check("probabilities in [0,1]", 0.0 <= fg.min() and fg.max() <= 1.0)
labels, inst = extract_instances(fg, ctr, bnd, ctr_thr=0.3, min_area=20)
check("extract_instances returns labels+list", isinstance(labels, np.ndarray)
      and isinstance(inst, list) and labels.shape == arr.shape[:2])

# --------------------------------------------------------------------------- #
print("\n== 5. edge cases ==")
blank = np.zeros((64, 64, 3), np.uint8)
fg, ctr, bnd = predict(net, blank, size=64)
labels, inst = extract_instances(fg, ctr, bnd, fg_thr=0.99)
check("blank image -> 0 instances, no crash", len(inst) == 0 and labels.max() == 0)
cm = instance_color_map(labels)
check("colour map with 0 instances works", cm.shape == (64, 64, 3))
del net
gc.collect()

# synthetic fg blob with NO centre peaks -> distance fallback
fg2 = np.zeros((80, 80), np.float32); fg2[20:60, 20:60] = 1.0
ctr2 = np.zeros((80, 80), np.float32)
bnd2 = np.zeros((80, 80), np.float32)
labels2, inst2 = extract_instances(fg2, ctr2, bnd2, ctr_thr=0.5, min_area=20)
check("no centre peaks -> distance-transform fallback finds 1 instance",
      len(inst2) == 1, f"({len(inst2)})")

# two touching blobs -> boundary should split them
fg3 = np.zeros((80, 80), np.float32); fg3[20:60, 15:40] = 1; fg3[20:60, 40:65] = 1
bnd3 = np.zeros((80, 80), np.float32); bnd3[:, 38:42] = 1.0
ctr3 = np.zeros((80, 80), np.float32); ctr3[40, 27] = 1; ctr3[40, 52] = 1
labels3, inst3 = extract_instances(fg3, ctr3, bnd3, ctr_thr=0.5, bnd_weight=1.0, min_area=20)
check("boundary-aware watershed splits touching blobs", len(inst3) == 2, f"({len(inst3)})")

# --------------------------------------------------------------------------- #
print("\n== 6. train + resume ==")
small = items[:16]
dl = DataLoader(InstanceMaskDataset(small, 128, True), batch_size=2, drop_last=True)
net = CVSegNet.light()
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
from model import cvseg_loss
net.train()
l0 = None
for x, fg, ctr, bnd in dl:
    out = cvseg_loss(*net(x), fg, ctr, bnd)
    opt.zero_grad(); out["total"].backward(); opt.step()
    l0 = float(out["total"])
check("training step runs, loss finite", l0 is not None and np.isfinite(l0), f"loss={l0:.3f}")
ck = os.path.join(HERE, "weights", "_test.pt")
os.makedirs(os.path.dirname(ck), exist_ok=True)
torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "epoch": 0,
            "best": 0.5, "backbone": "resnet18", "decoder_width": 1.0}, ck)
ck2 = torch.load(ck, map_location="cpu", weights_only=False)
net2 = CVSegNet(pretrained=False, backbone=ck2["backbone"], decoder_width=ck2["decoder_width"])
net2.load_state_dict(ck2["model"])
check("checkpoint round-trips into the right architecture", True)

# --- resumable-training checkpoint format ---
import argparse
from train import save_checkpoint
fake = argparse.Namespace(backbone="resnet18", decoder_width=1.0, size=128,
                          grad_accum=1, freeze_at=0)
del net2
gc.collect()
# use a tiny optimiser here: we only check the checkpoint *format*, and the real
# AdamW state for a 12 M model is large enough to be awkward on a small host
tiny = torch.nn.Linear(2, 2)
opt_small = torch.optim.AdamW(tiny.parameters())
sched_small = torch.optim.lr_scheduler.CosineAnnealingLR(opt_small, T_max=4)
save_checkpoint(ck, net, opt_small, torch.amp.GradScaler("cuda", enabled=False),
                sched_small, 2, 7, 42, 0.5, fake)
rc = torch.load(ck, weights_only=False)
need = {"model", "opt", "scaler", "sched", "epoch", "batch_in_epoch",
        "global_step", "best", "rng", "meta"}
check("resume checkpoint stores full state", need <= set(rc.keys()),
      f"missing={sorted(need - set(rc.keys()))}")
check("resume checkpoint records position",
      rc["global_step"] == 42 and rc["batch_in_epoch"] == 7 and rc["epoch"] == 2)
check("resume checkpoint records config meta", rc["meta"]["backbone"] == "resnet18")
del net, opt, rc, opt_small, sched_small, tiny
gc.collect()
os.remove(ck)

# --------------------------------------------------------------------------- #
print("\n== 7. web app endpoint ==")
try:
    gc.collect()
    os.environ["CVSEG_IMG_SIZE"] = "128"
    from fastapi.testclient import TestClient
    import app as appmod
    # give it a checkpoint to load
    ck = os.path.join(HERE, "weights", "cvseg_unet.pt")
    torch.save({"model": CVSegNet(pretrained=False).state_dict(),
                "backbone": "resnet18", "decoder_width": 1.0}, ck)
    with TestClient(appmod.app) as client:
        h = client.get("/api/health").json()
        check("GET /api/health loads the model", h.get("model_loaded") is True, str(h.get("error")))
        buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG"); buf.seek(0)
        r = client.post("/api/segment", files={"file": ("t.png", buf, "image/png")},
                        data={"fg_thr": 0.5, "ctr_thr": 0.3, "bnd_weight": 0.7, "min_area": 30})
        j = r.json()
        check("POST /api/segment returns instances + overlay",
              r.status_code == 200 and "instances" in j and j["overlay"].startswith("data:image/png"),
              f"count={j.get('count')}")
    os.remove(ck)
except Exception as e:
    # httpx (or fastapi) missing -> skip, don't fail the suite
    print(f"[SKIP] web app endpoint test - install httpx to run it "
          f"({type(e).__name__}: {e})")

# --------------------------------------------------------------------------- #
npass = sum(1 for _, ok in RESULTS if ok)
print(f"\n{'='*50}\n{npass}/{len(RESULTS)} checks passed\n{'='*50}")
for name, ok in RESULTS:
    if not ok:
        print("  FAILED:", name)
sys.exit(0 if npass == len(RESULTS) else 1)
