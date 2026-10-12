"""
CVSegNet web app -- FastAPI backend.

Serves a small front-end and exposes POST /api/segment, which takes an image
and returns the object instances the model finds, plus a colour overlay.

Run:
    pip install -r requirements.txt
    python app.py            # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model import (CVSegNet, extract_instances, instance_color_map,
                   instance_palette, predict)

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")
WEIGHTS = os.environ.get("CVSEG_WEIGHTS", os.path.join(ROOT, "weights", "cvseg_unet.pt"))
IMG_SIZE = int(os.environ.get("CVSEG_IMG_SIZE", "320"))

torch.set_num_threads(2)

_state = {"model": None, "error": None}


def load_model():
    """Load weights/cvseg_unet.pt, rebuilding the architecture it was trained with."""
    if _state["model"] is not None:
        return
    if not os.path.exists(WEIGHTS):
        _state["error"] = ("No trained weights found at weights/cvseg_unet.pt. "
                           "Run `python train.py` first (see README).")
        return
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    backbone = ck.get("backbone", "resnet18") if isinstance(ck, dict) else "resnet18"
    width = ck.get("decoder_width", 1.0) if isinstance(ck, dict) else 1.0
    model = CVSegNet(pretrained=False, backbone=backbone, decoder_width=width)
    model.load_state_dict(sd)
    model.eval()
    _state["model"] = model
    _state["error"] = None


def _ensure_loaded():
    """Lazy fallback so the API works even if the startup event didn't fire."""
    if _state["model"] is None and _state["error"] is None:
        load_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="CVSegNet",
              description="Instance segmentation from a model trained on your data",
              lifespan=lifespan)


@app.get("/api/health")
def health():
    _ensure_loaded()
    m = _state["model"]
    return {"model_loaded": m is not None, "error": _state["error"],
            "img_size": IMG_SIZE,
            "params_millions": round(sum(p.numel() for p in m.parameters()) / 1e6, 2) if m else None}


def _overlay_rgba(rgb: np.ndarray, labels: np.ndarray, alpha: int = 200) -> str:
    """Colour each instance, transparent everywhere else -> base64 PNG."""
    colors = instance_color_map(labels)
    h, w = labels.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = colors
    rgba[..., 3] = np.where(labels > 0, alpha, 0)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/api/segment")
async def segment(file: UploadFile = File(...), fg_thr: float = Form(0.5),
                  ctr_thr: float = Form(0.35), bnd_weight: float = Form(0.7),
                  min_area: int = Form(40)):
    _ensure_loaded()
    if _state["model"] is None:
        return JSONResponse(status_code=503, content={"error": _state["error"]})

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Could not read that image."})

    arr = np.array(img)
    fg, ctr, bnd = predict(_state["model"], arr, size=IMG_SIZE)
    labels, instances = extract_instances(fg, ctr, bnd, fg_thr=fg_thr,
                                          ctr_thr=ctr_thr, bnd_weight=bnd_weight,
                                          min_area=min_area)

    palette = instance_palette(int(labels.max()))
    for inst in instances:
        r, g, b = (int(x) for x in palette[inst["id"]])
        inst["color"] = f"#{r:02x}{g:02x}{b:02x}"

    return {"count": len(instances), "instances": instances,
            "overlay": _overlay_rgba(arr, labels),
            "width": int(arr.shape[1]), "height": int(arr.shape[0])}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
