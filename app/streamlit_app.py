"""Eggplant 10-class condition classifier (CE vs SupCon pipeline) with Grad-CAM / occlusion — CPU demo.

Local:            .venv/bin/streamlit run app/streamlit_app.py
Streamlit Cloud:  entrypoint app/streamlit_app.py (dependencies from app/requirements.txt)
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from html import escape
from pathlib import Path

import matplotlib
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
REPO = "https://github.com/tanximsiam/eggplant-disease-supcon-xai"
DATASET = "https://data.mendeley.com/datasets/5drkk544k8/1"
LABEL = {"ce": "CE baseline", "supcon": "SupCon + CE"}
HEAT = matplotlib.colormaps["inferno"]
torch.set_num_threads(2)

st.set_page_config(page_title="Eggplant condition classifier", page_icon=":material/eco:", layout="wide")
ACCENT = "#8A63C2"  # aubergine; >=4:1 against both theme backgrounds, so it reads in light and dark mode
st.html(f"""<style>
  .block-container {{ max-width: 1120px; padding-top: 2.5rem; padding-bottom: 3rem; }}
  .block-container h1 {{ font-size: clamp(1.75rem, 1.1rem + 2.2vw, 2.5rem); letter-spacing: -0.02em;
                         text-wrap: balance; }}
  .block-container h3 {{ font-size: 1.2rem; font-weight: 600; margin-top: 1.75rem; }}
  ::selection {{ background: color-mix(in srgb, {ACCENT} 32%, transparent); }}
  [data-testid="stMetricValue"] {{ font-size: 1.45rem; line-height: 1.25; }}
  [data-testid="stImageCaption"] {{ text-align: left; }}
  .cmp {{ font-variant-numeric: tabular-nums; font-size: 0.925rem; }}
  .cmp-head, .cmp-row {{ display: grid; grid-template-columns: minmax(9rem, 1.15fr) 1fr 1fr; gap: 0.25rem 1.25rem;
                         align-items: center; padding: 0.55rem 0;
                         border-bottom: 1px solid color-mix(in srgb, currentColor 12%, transparent); }}
  .cmp-head {{ font-size: 0.8rem; font-weight: 600; color: color-mix(in srgb, currentColor 72%, transparent); }}
  .cmp-cell {{ display: grid; grid-template-columns: 1fr 3rem; gap: 0.6rem; align-items: center; }}
  .cmp-track {{ height: 6px; border-radius: 3px; background: color-mix(in srgb, currentColor 12%, transparent); }}
  .cmp-fill {{ height: 100%; border-radius: 3px; background: {ACCENT}; }}
  .cmp-val {{ text-align: right; }}
  .cmp-tag {{ margin-left: 0.4rem; font-size: 0.75rem; padding: 0.05rem 0.4rem; border-radius: 0.6rem;
              border: 1px solid color-mix(in srgb, currentColor 30%, transparent); white-space: nowrap; }}
  .cmp-cell::before {{ display: none; }}
  @media (max-width: 560px) {{
    .block-container {{ padding-top: 1.25rem; }}
    [data-testid="stImage"] img {{ max-height: 38vh; width: auto !important; max-width: 100%; }}
    .cmp-head {{ display: none; }}
    .cmp-row {{ grid-template-columns: 1fr; gap: 0.35rem; }}
    .cmp-cell {{ grid-template-columns: 6.5rem 1fr 3rem; }}
    .cmp-cell::before {{ display: block; content: attr(data-label); font-size: 0.8rem;
                         color: color-mix(in srgb, currentColor 72%, transparent); }}
  }}
</style>""")


# ---------------------------------------------------------------- data + model helpers
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


def read_image(data: bytes) -> Image.Image | None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


@st.cache_data(show_spinner=False, max_entries=64)
def explain_gradcam(_net, method: str, img_key: str, target: int, _x01) -> tuple[np.ndarray, bool]:
    return gradcam(_net, _x01, target)


@st.cache_data(show_spinner=False, max_entries=32)
def explain_occlusion(_net, method: str, img_key: str, target: int, _x01) -> np.ndarray:
    occ = occlusion(_net, _x01, target, 32, 16, mean_fill(_x01))
    rng = occ.max() - occ.min()
    return (occ - occ.min()) / rng if rng > 0 else np.zeros_like(occ)


def overlay(base: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Heat colour with opacity proportional to the map, so low-influence regions keep the photo's own colours."""
    a = (0.75 * m)[..., None]
    return (base * (1 - a) + HEAT(m)[..., :3] * a).clip(0, 1)


def score_table(order, truth: int | None) -> str:
    """Paired CE/SupCon score rows; wide screens get columns, phones get stacked labelled bars."""
    head = "".join(f"<span>{escape(LABEL[m])}</span>" for m in models)
    rows = []
    for k in order:
        tag = '<span class="cmp-tag">true label</span>' if k == truth else ""
        cells = "".join(
            f'<span class="cmp-cell" data-label="{escape(LABEL[m])}">'
            f'<span class="cmp-track" role="img" aria-label="{escape(LABEL[m])} score {probs[m][k]:.3f}">'
            f'<span class="cmp-fill" style="display:block;width:{probs[m][k] * 100:.1f}%"></span></span>'
            f'<span class="cmp-val">{probs[m][k]:.3f}</span></span>' for m in models)
        rows.append(f'<div class="cmp-row"><span>{escape(names[k])}{tag}</span>{cells}</div>')
    return f'<div class="cmp"><div class="cmp-head"><span>Class</span>{head}</div>{"".join(rows)}</div>'


models = load_models()
if not models:
    st.error("The model files are missing. Expected `app/models/` with `demo_models.json`.", icon=":material/error:")
    st.stop()
any_ck = next(iter(models.values()))[1]
names = [c["display_name"] for c in any_ck["class_map"]]
size = any_ck["image_size"]

# ---------------------------------------------------------------- header
st.title("Eggplant condition classifier", anchor=False)
st.markdown(
    "Classifies a photo of an eggplant leaf or fruit into one of 10 conditions and shows what each model looked at. "
    "Two EfficientNet-B0 models trained on the same 412 labelled images are compared: a cross-entropy baseline and "
    "supervised contrastive pretraining followed by the same fine-tuning.")
st.markdown(f":material/code: [Code and full report]({REPO}) &nbsp;·&nbsp; "
            f":material/dataset: [Dataset (Mendeley, CC BY 4.0)]({DATASET}) &nbsp;·&nbsp; "
            ":material/info: Research demo, not a diagnostic tool")

# ---------------------------------------------------------------- input
st.subheader("Image", anchor=False)
EXAMPLE, UPLOAD = "Example images", "Upload your photo"
def _keep_one_source():
    """Clicking the active segment would deselect it; keep exactly one source selected."""
    if st.session_state.source is None:
        st.session_state.source = st.session_state.get("source_last", EXAMPLE)
    st.session_state.source_last = st.session_state.source


st.session_state.setdefault("source", EXAMPLE)
source = st.segmented_control(
    "Choose an image source", [EXAMPLE, UPLOAD], key="source", on_change=_keep_one_source,
    format_func=lambda s: f":material/{'photo_library' if s == EXAMPLE else 'upload'}: {s}") or EXAMPLE
if source == EXAMPLE:
    st.warning(f"You are viewing a bundled example from the test set, so its true label is known. "
               f"To check your own photo, switch to **{UPLOAD}** above.", icon=":material/priority_high:")
else:
    st.warning(f"You are classifying your own photo. There is no true label to compare against. "
               f"Switch back to **{EXAMPLE}** to see the models on known test images.",
               icon=":material/priority_high:")
examples = {p.stem.split("_", 1)[1].replace("_", " "): p for p in sorted(EXAMPLES.glob("*.jpg"))}
truth = None
if source == UPLOAD:
    up = st.file_uploader("Photo of an eggplant leaf or fruit", type=["jpg", "jpeg", "png", "bmp", "webp"],
                          help="JPG, PNG, BMP or WEBP, up to 15 MB. Processed in memory, not stored.")
    if up is None:
        st.caption("Results appear here once a photo is uploaded.")
        st.stop()
    data = up.getvalue()
else:
    pick = st.pills("Example images from the held-out test set, one per class", list(examples),
                    default=next(iter(examples)), selection_mode="single")
    if pick is None:
        st.caption("Choose an example to classify.")
        st.stop()
    data = examples[pick].read_bytes()
    truth = names.index(pick)

img = read_image(data)
if img is None:
    st.error("This file could not be read as an image. Try a JPG or PNG photo.", icon=":material/broken_image:")
    st.stop()
if min(img.size) < 32:
    st.error(f"This image is too small ({img.size[0]}×{img.size[1]} px). Use a photo at least 32 px on each side.",
             icon=":material/photo_size_select_small:")
    st.stop()

img_key = hashlib.sha1(data).hexdigest()
crop = eval_geometry(size)(img)
base = np.asarray(crop, dtype=np.float32) / 255.0
x01 = torch.from_numpy(base).permute(2, 0, 1)
with torch.no_grad():
    probs = {m: n(normalize(x01[None])).softmax(1)[0].numpy() for m, (n, _) in models.items()}
preds = {m: int(p.argmax()) for m, p in probs.items()}

# ---------------------------------------------------------------- prediction
st.subheader("Prediction", anchor=False)
left, right = st.columns([5, 7], gap="large")
with left:
    st.image(crop, width="stretch",
             caption=f"Model input, resized and centre-cropped to {size} px"
                     + (f". True label: {names[truth]}." if truth is not None else "."))
with right:
    with st.container(border=True):
        cols = st.columns(len(models))
        for col, m in zip(cols, models):
            k = preds[m]
            with col:
                st.metric(LABEL[m], names[k], f"score {probs[m][k]:.2f}", delta_color="off", delta_arrow="off")
                if truth is not None:
                    st.markdown(":material/check_circle: Matches the true label" if k == truth
                                else ":material/cancel: Does not match the true label")
        if len(set(preds.values())) == 1:
            st.caption("Both models give the same top class.")
        else:
            st.caption("The two models disagree on this image.")

    order = np.argsort(-np.max(np.stack(list(probs.values())), axis=0))[:5]
    st.html(score_table(order, truth))
    st.caption("Top 5 classes by either model. Softmax scores are uncalibrated: they rank classes, they are not "
               "the probability of being correct. The model always picks one of its 10 classes, even for photos "
               "that are not eggplants.")

# ---------------------------------------------------------------- explanation
st.subheader("Explanation", anchor=False)
c1, c2 = st.columns(2, gap="medium")
with c1:
    method = st.segmented_control("Model", list(models), format_func=LABEL.get, default=next(iter(models)))
    method = method or next(iter(models))
with c2:
    target = int(st.selectbox("Class to explain", np.argsort(-probs[method]),
                              format_func=lambda k: f"{names[k]} · {probs[method][k]:.3f}"))
net = models[method][0]

e1, e2 = st.columns(2, gap="medium")
with e1:
    cam, const = explain_gradcam(net, method, img_key, target, x01)
    st.image(overlay(base, cam), width="stretch",
             caption="Grad-CAM: regions whose features raised this class score most"
                     + (" (no positive evidence found)" if const else ""))
with e2:
    occ_key = f"occ-{img_key}-{method}-{target}"
    if st.session_state.get(occ_key) or st.button("Run occlusion analysis", icon=":material/grid_on:",
                                                  help="Masks 169 patches of 32×32 px one at a time. "
                                                       "Takes a few seconds on CPU."):
        st.session_state[occ_key] = True
        with st.spinner("Masking patches ..."):
            occ = explain_occlusion(net, method, img_key, target, x01)
        st.image(overlay(base, occ), width="stretch",
                 caption="Occlusion: drop in the class score when each patch is greyed out")
    else:
        st.caption("Occlusion is a second, slower check: it hides each part of the image in turn and measures "
                   "how much the score falls.")
st.caption("Brighter means more influence on the score. These maps show what the model used, not where the disease "
           "is: the dataset has no lesion annotations to check them against.")

# ---------------------------------------------------------------- about
with st.expander("About the models and their limits", icon=":material/info:"):
    rows = [f"- **{LABEL[m]}**: run `{ck['run_id']}`, validation macro-F1 {ck['best']['val_macro_f1']:.3f}"
            for m, (_, ck) in models.items()]
    st.markdown("\n".join(rows))
    st.markdown(
        "- Trained on 20% of the training pool (412 images), seed 42, from one dataset collected in Bangladesh.\n"
        "- On the 450-image test set, the mean macro-F1 at this budget was 0.843 for CE and 0.845 for SupCon "
        "(3 seeds), so the two are effectively tied. SupCon was ahead at 10% of the data.\n"
        "- Not validated on other farms, cameras or regions. Not a diagnostic tool.\n"
        f"- Method, audit, all results and failure cases are in the [README]({REPO}#readme).")
st.caption("Example images: Nirob et al., Eggplant Dataset, Mendeley Data v1, CC BY 4.0, downscaled.")
