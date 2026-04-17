"""Local CPU demo: eggplant 10-class classifier (CE vs SupCon pipeline) with Grad-CAM / occlusion.

Run from the repo root:  .venv/bin/streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eggplant.data import eval_geometry  # noqa: E402
from eggplant.model import build_classifier  # noqa: E402
from eggplant.xai import gradcam, mean_fill, normalize, occlusion  # noqa: E402

DEMO = ROOT / "exports" / "demo"
st.set_page_config(page_title="Eggplant condition classifier (research demo)", layout="wide")
st.title("Eggplant leaf/fruit condition classifier — research demo")
st.caption("EfficientNet-B0 trained on 20% of the training pool of the Mendeley eggplant dataset "
           "(DOI 10.17632/5drkk544k8.1, CC BY 4.0). Portfolio experiment, **not a diagnostic tool**.")


@st.cache_resource
def load_models():
    meta_path = DEMO / "demo_models.json"
    if not meta_path.exists():
        return None
    models = {}
    for m in json.loads(meta_path.read_text()):
        p = DEMO / m["file"]
        if not p.exists():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        net = build_classifier(len(ck["class_map"]), pretrained=False)
        net.load_state_dict(ck["model"])
        models[m["method"]] = (net.eval(), ck)
    return models or None


models = load_models()
if not models:
    st.error("No exported checkpoints found in exports/demo/. Run `bash scripts/run.sh export` after training.")
    st.stop()

method = st.sidebar.radio("Model", list(models), format_func=lambda m: {"ce": "CE baseline", "supcon": "SupCon → CE"}[m])
net, ck = models[method]
names = [c["display_name"] for c in ck["class_map"]]
size = ck["image_size"]
st.sidebar.markdown("**Supported labels (the only possible outputs):**\n" + "\n".join(f"- {n}" for n in names))
st.sidebar.markdown(f"run: `{ck['run_id']}`  \nvalidation macro-F1: {ck['best']['val_macro_f1']:.3f}")
st.sidebar.warning("Known limitations: trained on one small dataset from one source; the model always picks one of "
                   "these 10 labels and cannot reliably reject unrelated or out-of-distribution images; scores are "
                   "uncalibrated; heatmaps are not validated lesion localisation.")

up = st.file_uploader("Upload an eggplant leaf or fruit image (JPG/PNG)", type=["jpg", "jpeg", "png", "bmp", "webp"])
if up is None:
    st.info("Upload an image to classify.")
    st.stop()
try:
    with Image.open(up) as im:
        im.load()
        img = ImageOps.exif_transpose(im).convert("RGB")
except (UnidentifiedImageError, OSError, ValueError) as e:
    st.error(f"Could not read this file as an image ({type(e).__name__}).")
    st.stop()
if min(img.size) < 32:
    st.error(f"Image too small ({img.size[0]}×{img.size[1]}).")
    st.stop()

crop = eval_geometry(size)(img)
x01 = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255.0).permute(2, 0, 1)
with torch.no_grad():
    probs = net(normalize(x01[None])).softmax(1)[0].numpy()
order = np.argsort(-probs)

c1, c2 = st.columns([1, 1])
with c1:
    st.image(crop, caption=f"model input (EXIF-corrected, resized + center-cropped to {size}px)", width="stretch")
with c2:
    st.subheader("Model scores (uncalibrated softmax — not probabilities of being correct)")
    for k in order[:5]:
        st.progress(float(probs[k]), text=f"{names[k]}: {probs[k]:.3f}")

target = st.selectbox("Explain class", order, format_func=lambda k: f"{names[k]} ({probs[k]:.3f})")


def overlay(m: np.ndarray) -> np.ndarray:
    heat = cm.jet(m)[..., :3]
    return (0.55 * np.asarray(crop, dtype=np.float32) / 255 + 0.45 * heat).clip(0, 1)


cam, const = gradcam(net, x01, int(target))
e1, e2 = st.columns(2)
with e1:
    st.image(overlay(cam), caption="Grad-CAM (last conv block)" + (" — constant/zero map" if const else ""),
             width="stretch")
with e2:
    if st.checkbox("Compute occlusion sensitivity (slower, 169 forward passes on CPU)"):
        with st.spinner("occluding 32×32 patches, stride 16 ..."):
            occ = occlusion(net, x01, int(target), 32, 16, mean_fill(x01))
        rng = occ.max() - occ.min()
        st.image(overlay((occ - occ.min()) / rng if rng > 0 else np.zeros_like(occ)),
                 caption="occlusion: drop in score when a patch is replaced by the channel mean", width="stretch")
st.caption("Explanations show what influenced this model's score; they do not prove biological correctness.")
