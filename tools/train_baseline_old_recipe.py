"""Re-create the training recipe behind weights/cvseg_unet.pt (the default checkpoint).

That checkpoint was trained with the ORIGINAL training code (frozen BatchNorm, centre-target
sigma bug, no embedding head) but from a pretrained ResNet-18, on all five bundled datasets.
The current train.py deliberately differs (it fixes the sigma bug, trains BN when needed, ...),
so to reproduce it you need the code as it was at commit e7c1d7b:

    git worktree add ../cvseg_old e7c1d7b
    cp -r data ../cvseg_old/                       # the five datasets
    python tools/get_backbone_weights.py           # -> weights/backbones/resnet18.pth
    cd ../cvseg_old
    python /path/to/this/script.py . /path/to/weights/backbones/resnet18.pth

About 1 h on a 2-core CPU.  The improved recipe is just `train.py` in this repo
(see weights/TRAINING_NOTES.md for the exact command).
"""
import sys

if len(sys.argv) != 3:
    sys.exit(__doc__)
OLD_REPO, R18 = sys.argv[1], sys.argv[2]
sys.path.insert(0, OLD_REPO)

import torch                     # noqa: E402
import torchvision               # noqa: E402
import model as M                # noqa: E402

_orig = M.CVSegNet.__init__


def _patched(self, *a, **kw):
    kw["pretrained"] = False
    _orig(self, *a, **kw)
    tv = torchvision.models.resnet18()
    tv.load_state_dict(torch.load(R18, map_location="cpu", weights_only=True))
    self.stem[0].load_state_dict(tv.conv1.state_dict())
    self.stem[1].load_state_dict(tv.bn1.state_dict())
    for n in ("layer1", "layer2", "layer3", "layer4"):
        getattr(self, n).load_state_dict(getattr(tv, n).state_dict())


M.CVSegNet.__init__ = _patched

import train                     # noqa: E402

sys.argv = ["train.py", "--backbone", "resnet18", "--decoder-width", "1.0", "--no-checkpoint",
            "--freeze-at", "0", "--size", "192", "--batch", "4", "--grad-accum", "1",
            "--epochs", "40", "--lr", "1e-3", "--n-val", "60", "--resume", "auto"]
train.main()
