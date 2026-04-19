# Design

Scope: the Streamlit demo (`app/streamlit_app.py`, theme in `.streamlit/config.toml`). An Operate surface: a tool screen, not a landing page.

## Principles
- Minimal and professional: Streamlit's standard components, one accent colour, no gradients, glass or decorative chrome.
- The CE vs SupCon comparison is the signature: one verdict card with both models, then one paired score table.
- Honesty sits next to the result. Short captions explain uncalibrated scores and what the heatmaps do not show; longer context goes in the "About" expander and the README.

## Type
Streamlit defaults only: Source Sans for UI and Source Code for code. No custom fonts.
- Title: `clamp(1.75rem, 1.1rem + 2.2vw, 2.5rem)`, tracking -0.02em, balanced wrapping.
- Section headings (`st.subheader`, no anchors): 1.2rem / 600, 1.75rem above.
- Metric values: 1.45rem. The score table: 0.925rem with tabular numerals.

## Colour
Restrained neutrals plus one aubergine accent, defined for both themes. Viewers switch theme under ⋮ (System / Light / Dark).

| token | light | dark |
|---|---|---|
| primary | `#6B3FA0` | `#B89BE3` |
| background | `#FBFBFA` | `#131315` |
| secondary background | `#F1F0EE` | `#1D1D21` |
| text | `#1D1B20` | `#ECEAF0` |
| link | `#5E3591` | `#C7B0EA` |
| border | `#DEDBD6` | `#34333A` |

Custom HTML (the score table) never hard-codes text colours. It derives them from `currentColor` with `color-mix`, so it follows either theme. The bar fill is `#8A63C2`, at least 4:1 against both backgrounds. State is never shown by colour alone: correctness uses a check or cross icon plus text, and the true class gets a text tag.

Heatmaps use matplotlib `inferno`, with opacity proportional to the map value (max 0.75), so low-influence areas keep the photo's own colours.

## Shape and spacing
`baseRadius` 0.5rem, widget borders on, 1px hairline row dividers in the table. Content width is capped at 1120px. Bars are 6px tall with 3px radius.

## Layout and responsiveness
Vertical order: header → Image (segmented Example/Upload, pills for the 10 examples) → Prediction (image 5/12 | verdict card + score table 7/12) → Explanation (model + class controls, Grad-CAM | occlusion) → About expander → credit.
At 560px and below:
- columns stack;
- the score table turns into per-class blocks with a labelled bar per model;
- images are capped at 38vh so the verdict stays near the top.

## Components
- Segmented controls for exclusive choices (source, model). The image-source switch always keeps one option selected and has a visible label with icons.
- A guide note directly under the source switch. It is a caption with a violet circled `error` icon that says which mode is active and how to switch. It is an instruction, not an alert, so it uses no coloured box.
- Pills for examples.
- A selectbox for the class to explain.
- A button for slow work (occlusion), which names its action and gives a timing tooltip.
- Material icons for meaning, never emoji.
- Errors use `st.error` with an icon, name the problem and give the fix.
