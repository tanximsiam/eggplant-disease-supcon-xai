# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Primary: recruiters and technical reviewers evaluating Tanzim Siam's ML portfolio. They arrive from the GitHub README or a CV link, skim, try an example or their own photo, and judge whether the work is careful and whether the model actually works. They are often on a laptop, sometimes on a phone.

## Product Purpose
A hosted demo of a low-resource eggplant leaf and fruit condition classifier. It covers 10 classes and uses EfficientNet-B0 trained on 20% of the training pool. It compares a cross-entropy baseline with supervised-contrastive pretraining followed by the same CE fine-tuning, and explains predictions with Grad-CAM and optional occlusion. Success means a visitor gets a working prediction on an example or their own photo within seconds, and leaves with the impression of careful, honest ML work. The two goals carry roughly equal weight. The full methodology lives in the README, not the app.

## Positioning
The demo shows two paired models side by side on the same image, with explanations and stated limitations. It is a research comparison, not a single "AI diagnosis" result.

## Operating Context
Streamlit app, CPU only, hosted on Streamlit Community Cloud (entrypoint `app/streamlit_app.py`, dependencies in `app/requirements.txt`). Inference plus Grad-CAM takes about a second. Occlusion takes several seconds on CPU and is opt-in.

## Capabilities and Constraints
- Inputs: 10 bundled example images (one per class, from the test split) or an upload (JPG/PNG/BMP/WEBP, up to 15 MB, EXIF-corrected).
- Outputs: top-5 uncalibrated softmax scores for both models, Grad-CAM for any chosen class on the selected model, and optional occlusion.
- The model always outputs one of the 10 labels and cannot reject images that are not eggplants.
- Must stay within Streamlit's component model plus custom CSS. Use the fonts Streamlit already ships (Source Sans for UI, Source Code for monospace).
- Must support light and dark mode, and phone, tablet and desktop widths.

## Brand Commitments
Minimal, professional and clean. No flashy "AI-generated" styling (user directive).

## Evidence on Hand
- Models: `app/models/` (CE and SupCon, 20% fraction, seed 42; validation macro-F1 0.866 and 0.879).
- Examples: `app/examples/` (CC BY 4.0, Nirob et al., Mendeley Data, DOI 10.17632/5drkk544k8.1).
- Results and methodology: README.md and `results/`.
- No user testimonials, field validation or expert review exist, so none may be claimed.

## Product Principles
1. A working prediction comes first: an example is preselected and results show without any setup.
2. Always show both models together; the comparison is the point.
3. Honesty is visible but brief: limitations sit next to the results, and the long form stays in the README.
4. Explanations are labelled as model evidence, never as lesion localisation or diagnosis.

## Accessibility & Inclusion
No product-specific standard was set. Meet WCAG AA contrast in both themes, and never use colour as the only signal.
