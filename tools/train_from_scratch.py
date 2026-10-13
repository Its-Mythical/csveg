"""Train CVSegNet WITHOUT pretrained ImageNet weights (for machines that cannot download them).

This is the exact recipe used for weights/cvseg_unet.pt. It wraps train.py unchanged: it only
forces CVSegNet(pretrained=False). Run from anywhere:

    python tools/train_from_scratch.py

Roughly 25 minutes on a 2-core CPU. With internet access to download.pytorch.org, run
`python train.py ...` directly instead (pretrained weights give a much better model).
"""
import functools
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import model as M  # noqa: E402

_orig = M.CVSegNet.__init__


@functools.wraps(_orig)
def _no_pretrained(self, *a, **kw):
    kw["pretrained"] = False
    return _orig(self, *a, **kw)


M.CVSegNet.__init__ = _no_pretrained

import train  # noqa: E402

sys.argv = ["train.py", "--backbone", "resnet18", "--decoder-width", "1.0", "--no-checkpoint",
            "--freeze-at", "0", "--size", "192", "--batch", "4", "--grad-accum", "1",
            "--epochs", "40", "--resume", "none", "--lr", "1e-3"]
train.main()
