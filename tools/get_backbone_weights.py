"""Download ImageNet-pretrained ResNet weights for the CVSegNet encoder from GitHub releases.

Use this when download.pytorch.org (torchvision's default weight host) is blocked but GitHub is
reachable. The files are timm's ResNet "a1" ImageNet checkpoints (Apache-2.0 licence,
https://github.com/huggingface/pytorch-image-models); their key names match torchvision's ResNet.

    python tools/get_backbone_weights.py resnet18            # -> weights/backbones/resnet18.pth
    python train.py --backbone resnet18 --backbone-weights weights/backbones/resnet18.pth ...
"""
import os
import sys
import urllib.request

BASE = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/"
FILES = {"resnet18": "resnet18_a1_0-d63eafa0.pth", "resnet50": "resnet50_a1_0-14fe96d1.pth"}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
    if name not in FILES:
        raise SystemExit(f"choose one of {list(FILES)}")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "weights", "backbones")
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{name}.pth")
    if not os.path.exists(dest):
        print("downloading", BASE + FILES[name])
        urllib.request.urlretrieve(BASE + FILES[name], dest)
    print(dest)


if __name__ == "__main__":
    main()
