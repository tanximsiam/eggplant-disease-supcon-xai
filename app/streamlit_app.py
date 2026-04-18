"""Eggplant 10-class condition classifier (CE vs SupCon pipeline) with Grad-CAM / occlusion — CPU demo.

Local:            .venv/bin/streamlit run app/streamlit_app.py
Streamlit Cloud:  entrypoint app/streamlit_app.py (dependencies from app/requirements.txt)
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

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT))
from eggplant.data import eval_geometry  # noqa: E402
from eggplant.model import build_classifier  # noqa: E402
from eggplant.xai import gradcam, mean_fill, normalize, occlusion  # noqa: E402

MODEL_DIRS = [APP / "models", ROOT / "exports" / "demo"]  # committed copies first, local export as fallback
EXAMPLES = APP / "examples"
LABEL = {"ce": "CE baseline", "supcon": "SupCon → CE"}
torch.set_num_threads(2)

st.set_page_config(page_title="Eggplant condition classifier", page_icon="🍆", layout="wide")
st.title("Eggplant leaf & fruit condition classifier")
st.caption("Research demo: EfficientNet-B0 trained on 20% of the training pool (412 images) of the Mendeley eggplant "
           "dataset ([DOI 10.17632/5drkk544k8.1](https://data.mendeley.com/datasets/5drkk544k8/1), CC BY 4.0). "
           "Compares a cross-entropy baseline with supervised-contrastive pretraining. **Not a diagnostic tool.**")


@st.cache_resource(show_spinner="Loading models ...")
def load_models():
    for d in MODEL_DIRS:
        meta = d / "demo_models.json"
        if not meta.exists():
            continue
        models = {}
        for m in json.loads(meta.read_text()):
            p = d / m["file"]
            if not p.exists():
                continue
            ck = torch.load(p, map_location="cpu", weights_only=False)
            net = build_classifier(len(ck["class_map"]), pretrained=False)
            net.load_state_dict(ck["model"])
            models[m["method"]] = (net.eval(), ck)
        if models:
            return models
    return None


def read_image(src) -> Image.Image | None:
    try:
        with Image.open(src) as im:
            im.load()
            return ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        st.error(f"Could not read this file as an image ({type(e).__name__}).")
        return None


models = load_models()
if not models:
    st.error("No model checkpoints found (expected app/models/ or exports/demo/).")
    st.stop()

# ---------------------------------------------------------------- sidebar
method = st.sidebar.radio("Model to explain", list(models), format_func=LABEL.get)
net, ck = models[method]
names = [c["display_name"] for c in ck["class_map"]]
size = ck["image_size"]
st.sidebar.markdown("**The only possible outputs:**\n" + "\n".join(f"- {n}" for n in names))
st.sidebar.markdown(f"run `{ck['run_id']}`  \nvalidation macro-F1 {ck['best']['val_macro_f1']:.3f}")
st.sidebar.warning("**Limitations.** Trained on one small dataset from one source. The model always picks one of "
                   "these 10 labels and cannot reject unrelated images. Scores are uncalibrated. Heatmaps show "
                   "what drove the score, not validated lesion locations.")
st.sidebar.markdown("[Code & report on GitHub](https://github.com/tanximsiam/eggplant-disease-supcon-xai)")

# ---------------------------------------------------------------- input
examples = sorted(EXAMPLES.glob("*.jpg"))
src_kind = st.radio("Image source", ["Example from the test split", "Upload your own"], horizontal=True)
img = None
if src_kind.startswith("Example"):
    if not examples:
        st.warning("No example images bundled.")
        st.stop()
    ex = st.selectbox("Example (true label in the name)", examples,
                      format_func=lambda p: p.stem.split("_", 1)[1].replace("_", " "))
    img = read_image(ex)
else:
    up = st.file_uploader("Eggplant leaf or fruit photo", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up is None:
        st.info("Upload an image to classify.")
        st.stop()
    img = read_image(up)
if img is None:
    st.stop()
if min(img.size) < 32:
    st.error(f"Image too small ({img.size[0]}×{img.size[1]}).")
    st.stop()

crop = eval_geometry(size)(img)
x01 = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255.0).permute(2, 0, 1)
with torch.no_grad():
    all_probs = {m: n(normalize(x01[None])).softmax(1)[0].numpy() for m, (n, _) in models.items()}
probs = all_probs[method]
order = np.argsort(-probs)

# ---------------------------------------------------------------- prediction
c1, c2 = st.columns([1, 1.3])
with c1:
    st.image(crop, caption=f"Model input: EXIF-corrected, resized and centre-cropped to {size}px", width="stretch")
with c2:
    st.subheader("Top predictions")
    st.caption("Uncalibrated softmax scores, not probabilities of being correct.")
    cols = st.columns(len(models))
    for col, (m, p) in zip(cols, all_probs.items()):
        with col:
            st.markdown(f"**{LABEL[m]}**")
            for k in np.argsort(-p)[:5]:
                st.progress(float(p[k]), text=f"{names[k]}: {p[k]:.3f}")

# ---------------------------------------------------------------- explanations
st.divider()
st.subheader(f"Why? Explanations for the {LABEL[method]} model")
target = int(st.selectbox("Class to explain", order, format_func=lambda k: f"{names[k]} (score {probs[k]:.3f})"))


def overlay(m: np.ndarray) -> np.ndarray:
    return (0.55 * np.asarray(crop, dtype=np.float32) / 255 + 0.45 * cm.jet(m)[..., :3]).clip(0, 1)


cam, const = gradcam(net, x01, target)
e1, e2 = st.columns(2)
with e1:
    st.image(overlay(cam), width="stretch",
             caption="Grad-CAM on the last convolutional block" + (" (constant/zero map)" if const else ""))
with e2:
    if st.toggle("Compute occlusion sensitivity (169 forward passes, a few seconds on CPU)"):
        with st.spinner("Occluding 32×32 patches, stride 16 ..."):
            occ = occlusion(net, x01, target, 32, 16, mean_fill(x01))
        rng = occ.max() - occ.min()
        st.image(overlay((occ - occ.min()) / rng if rng > 0 else np.zeros_like(occ)), width="stretch",
                 caption="Occlusion: drop in the class score when a patch is replaced by the image's mean colour")
st.caption("Explanations show what influenced this model's score. They do not prove biological correctness. "
           "Example images: Nirob et al., Mendeley Data, CC BY 4.0 (downscaled).")
