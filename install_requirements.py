"""
Auto-install every dependency listed in requirements.txt.

Reads requirements.txt, strips comments / blank lines / pip options, and
installs the packages. torch and torchvision get the CPU wheel index by default
(smaller, no CUDA needed); pass --gpu to install the default CUDA wheels
instead. Finally it imports each package and prints the version so you can see
at a glance that everything is in place.

Usage
-----
    python install_requirements.py              # CPU torch (default)
    python install_requirements.py --gpu        # CUDA torch wheels
    python install_requirements.py --upgrade    # upgrade to latest
    python install_requirements.py --dry-run    # show what would be installed
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REQ = os.path.join(HERE, "requirements.txt")
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# packages that have a separate CPU wheel index
TORCH_RE = re.compile(r"^(torch|torchvision|torchaudio)([=<>!~].*)?$")
# import name -> pip name (when they differ)
IMPORT_NAMES = {
    "scikit-image": "skimage",
    "python-multipart": "multipart",
    "pillow": "PIL",
}


def parse_requirements(path):
    """Return the list of requirement strings, comments/options removed."""
    pkgs = []
    with open(path) as fh:
        for raw in fh:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            if line.startswith("-"):        # pip options like -r / --index-url
                continue
            pkgs.append(line)
    return pkgs


def pip(args, env=None):
    cmd = [sys.executable, "-m", "pip"] + args
    print("  $ " + " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        sys.exit(f"\npip failed (exit {e.returncode}) - fix the error above and "
                 f"re-run; this script is safe to run again.")


def _clean_env(user: bool):
    """A venv cannot do --user installs, but some shells export PIP_USER=1,
    which pip honours and then fails. Neutralise it unless --user was asked for."""
    env = os.environ.copy()
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv and not user:
        env["PIP_USER"] = "0"
        env.pop("PYTHONUSERBASE", None)
    return env


def main():
    ap = argparse.ArgumentParser(description="Install everything in requirements.txt")
    ap.add_argument("--requirements", default=DEFAULT_REQ,
                    help="path to requirements.txt")
    ap.add_argument("--gpu", action="store_true",
                    help="install CUDA torch wheels instead of the CPU build")
    ap.add_argument("--upgrade", action="store_true", help="pass --upgrade to pip")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be installed, install nothing")
    ap.add_argument("--user", action="store_true",
                    help="pip install --user (no virtualenv)")
    args = ap.parse_args()

    if not os.path.exists(args.requirements):
        sys.exit(f"requirements file not found: {args.requirements}")

    pkgs = parse_requirements(args.requirements)
    if not pkgs:
        sys.exit(f"no packages found in {args.requirements}")

    torch_pkgs = [p for p in pkgs if TORCH_RE.match(p)]
    other = [p for p in pkgs if p not in torch_pkgs]

    print(f"requirements : {args.requirements}")
    print(f"found        : {len(pkgs)} packages")
    print(f"  torch set  : {torch_pkgs or '-'}")
    print(f"  other      : {other}")
    print(f"  torch build: {'CUDA (default index)' if args.gpu else 'CPU wheel index'}\n")

    if args.dry_run:
        print("dry-run: nothing installed")
        return

    # make sure pip exists
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"],
                              stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("pip missing - bootstrapping with ensurepip")
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])

    base = (["install"] + (["--upgrade"] if args.upgrade else [])
            + (["--user"] if args.user else []))
    env = _clean_env(args.user)

    if torch_pkgs:
        print("\n[1/2] torch / torchvision")
        extra = [] if args.gpu else ["--index-url", CPU_INDEX]
        pip(base + torch_pkgs + extra, env)

    if other:
        print("\n[2/2] other packages")
        pip(base + other, env)

    # Verify in a FRESH subprocess. Packages installed during this process can
    # land in a user-site directory that this interpreter never added to
    # sys.path (site.py only adds it if it exists at startup), so an in-process
    # import reports false failures.
    import json
    pairs = []
    for pkg in pkgs:
        name = pkg.split("=")[0].split("<")[0].split(">")[0].split("!")[0].strip()
        pairs.append((name, IMPORT_NAMES.get(name, name)))
    probe = (
        "import importlib, json\n"
        f"pairs = {pairs!r}\n"
        "out = {}\n"
        "for name, mod in pairs:\n"
        "    try:\n"
        "        m = importlib.import_module(mod)\n"
        "        out[name] = getattr(m, '__version__', '?')\n"
        "    except Exception as e:\n"
        "        out[name] = 'FAIL: ' + str(e)\n"
        "print('RESULT' + json.dumps(out))\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe],
                          capture_output=True, text=True)
    line = next((l for l in reversed(proc.stdout.splitlines())
                 if l.startswith("RESULT")), None)
    print("\nverify imports:")
    ok = True
    if line is None:
        ok = False
        print("  could not run the verification probe")
        print(proc.stderr.strip()[-500:])
    else:
        for name, val in json.loads(line[len("RESULT"):]).items():
            if str(val).startswith("FAIL"):
                ok = False
                print(f"  FAIL {name:18s} {val}")
            else:
                print(f"  ok   {name:18s} {val}")

    print("\nAll requirements installed." if ok else
          "\nSome packages failed to import - see above.")
    if ok:
        print("Next:  python test_all.py   then   python train.py")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
