"""
dashboard/app.py  -  ResistAI robustness demo
=============================================
Tells the experiment's story in three parts:

  A. AI detection accuracy         - clean-test performance
  B. Robustness under degradation  - pick a corruption, compare Baseline vs Robust
  C. Why this matters              - distribution shift, and how augmentation helps

All numbers come from reports/benchmark_results.csv (the authoritative
10,000-image frozen-test benchmark). An optional live demo at the bottom loads
outputs/baseline_v1/best_model.pt and outputs/robust_v1/best_model.pt if present.

Run:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV = ROOT / "reports" / "benchmark_results.csv"
FIGS = ROOT / "reports" / "figures"
BASELINE_CKPT = ROOT / "outputs" / "baseline_v1" / "best_model.pt"
ROBUST_CKPT = ROOT / "outputs" / "robust_v1" / "best_model.pt"

st.set_page_config(page_title="ResistAI - robustness demo", layout="wide")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_benchmark():
    df = pd.read_csv(CSV)
    df["label"] = df.apply(
        lambda r: r["transformation"] if r["parameter"] == "-"
        else f"{r['transformation']}:{r['parameter']}", axis=1
    )
    clean = df[df.transformation == "clean"].iloc[0]
    for m in ("baseline", "robust"):
        df[f"{m}_acc_retention"] = df[f"{m}_acc"] / clean[f"{m}_acc"]
    df["acc_gap"] = df["robust_acc"] - df["baseline_acc"]
    df["f1_gap"] = df["robust_f1"] - df["baseline_f1"]
    df["auc_gap"] = df["robust_auc"] - df["baseline_auc"]
    return df, clean


df, clean = load_benchmark()
trans = df[df.transformation != "clean"]

# authoritative clean-test headline (single 10k eval, from evaluate.py)
CLEAN_HEADLINE = {
    "baseline": {"acc": 0.9152, "precision": 0.9126, "recall": 0.9184, "f1": 0.9155,
                 "auc": 0.9694, "fp": 440, "fn": 408},
    "robust": {"acc": 0.8966, "precision": 0.8808, "recall": 0.9174, "f1": 0.8987,
               "auc": 0.9573, "fp": 621, "fn": 413},
}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("ResistAI - robust detection of AI-generated images")
st.markdown(
    "Same EfficientNet-B0, same data, same training budget and seed. **The only "
    "difference:** *Robust v1* saw real-world transformations (JPEG, blur, resize, "
    "noise, colour jitter, crop) applied on the fly during training - *Baseline v1* "
    "did not. Every number below is from the same 10,000-image held-out test set."
)

# ---------------------------------------------------------------------------
# A. AI detection accuracy (clean)
# ---------------------------------------------------------------------------
st.header("A. AI detection accuracy (clean images)")
ca, cb = st.columns(2)
for col, name, title in ((ca, "baseline", "Baseline v1"), (cb, "robust", "Robust v1")):
    h = CLEAN_HEADLINE[name]
    with col:
        st.subheader(title)
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{h['acc']:.3f}")
        m2.metric("F1", f"{h['f1']:.3f}")
        m3.metric("ROC-AUC", f"{h['auc']:.3f}")
        st.caption(
            f"precision {h['precision']:.3f} - recall {h['recall']:.3f} - "
            f"false positives {h['fp']} - false negatives {h['fn']}"
        )
st.info(
    "On clean images the Robust model is **slightly worse** "
    f"(accuracy {CLEAN_HEADLINE['robust']['acc']:.3f} vs "
    f"{CLEAN_HEADLINE['baseline']['acc']:.3f}, i.e. -1.9 pp). That is the price of "
    "transformation-aware training - see part C for what it buys."
)

# ---------------------------------------------------------------------------
# B. Robustness under real-world degradation
# ---------------------------------------------------------------------------
st.header("B. Robustness under real-world degradation")

pick = st.selectbox(
    "Choose a degradation applied to *every* test image:",
    list(trans["label"]),
    index=list(trans["label"]).index("blur:sigma1.0"),
)
row = trans[trans.label == pick].iloc[0]

g1, g2, g3 = st.columns(3)
g1.metric("Baseline accuracy", f"{row.baseline_acc:.3f}",
          f"{row.baseline_acc - clean.baseline_acc:+.3f} vs its clean")
g2.metric("Robust accuracy", f"{row.robust_acc:.3f}",
          f"{row.robust_acc - clean.robust_acc:+.3f} vs its clean")
g3.metric("Robust - Baseline", f"{row.acc_gap:+.3f}",
          "accuracy gap (green = augmentation helps)")

t = pd.DataFrame({
    "metric": ["accuracy", "F1", "ROC-AUC"],
    "Baseline v1": [row.baseline_acc, row.baseline_f1, row.baseline_auc],
    "Robust v1": [row.robust_acc, row.robust_f1, row.robust_auc],
    "Robust - Baseline": [row.acc_gap, row.f1_gap, row.auc_gap],
}).set_index("metric")
st.dataframe(t.style.format("{:.3f}"), use_container_width=True)

if row.acc_gap > 0.03:
    st.success(f"Under **{pick}**, transformation-aware training recovers "
               f"**{row.acc_gap*100:+.1f} accuracy points** and "
               f"**{row.f1_gap*100:+.1f} F1 points**.")
elif row.acc_gap < -0.005:
    st.warning(f"Under **{pick}** the Robust model is **{row.acc_gap*100:.1f} pp** "
               f"worse - this is a mild transform the Baseline already handles.")
else:
    st.info(f"Under **{pick}** the two models are within {abs(row.acc_gap)*100:.1f} pp.")

st.markdown("#### All conditions at a glance")
c1, c2 = st.columns(2)
for col, fig in ((c1, "accuracy_by_transformation.png"),
                 (c2, "f1_by_transformation.png")):
    p = FIGS / fig
    if p.exists():
        col.image(str(p), use_container_width=True)
p = FIGS / "accuracy_gap.png"
if p.exists():
    st.image(str(p), use_container_width=True)

# ---------------------------------------------------------------------------
# C. Why this matters
# ---------------------------------------------------------------------------
st.header("C. Why this matters")
st.markdown(
    """
Real AI images almost never reach a detector untouched. They are **screenshotted,
re-compressed, resized for social media, denoised, cropped**. Each step shifts the
input distribution away from the clean images the model trained on.

**A model can look excellent on clean data and fail badly after degradation.**
Baseline v1 scores **0.915** clean accuracy - but under a moderate Gaussian blur
(sigma 1.0) it drops to **0.569** (F1 **0.259**): it has effectively stopped
detecting AI images and now labels almost everything "real". The same collapse
happens under downscaling and cropping.

**Transformation-aware training closes most of that gap.** Showing the model
degraded images *during training* keeps it working when degraded images arrive at
test time:
"""
)
agg = {
    "baseline": {"acc": 0.754, "f1": 0.676, "auc": 0.836},
    "robust": {"acc": 0.833, "f1": 0.821, "auc": 0.905},
}
w1, w2, w3 = st.columns(3)
w1.metric("Mean accuracy (16 transformed)", f"{agg['robust']['acc']:.3f}",
          f"{(agg['robust']['acc']-agg['baseline']['acc'])*100:+.1f} pp vs baseline")
w2.metric("Mean F1 (16 transformed)", f"{agg['robust']['f1']:.3f}",
          f"{(agg['robust']['f1']-agg['baseline']['f1'])*100:+.1f} pp vs baseline")
w3.metric("Mean ROC-AUC (16 transformed)", f"{agg['robust']['auc']:.3f}",
          f"{(agg['robust']['auc']-agg['baseline']['auc'])*100:+.1f} pp vs baseline")
st.caption("Accuracy retention under corruption rises from ~0.82 (Baseline) to "
           "~0.93 (Robust); F1 retention from ~0.74 to ~0.91.")

c1, c2 = st.columns(2)
for col, fig in ((c1, "aggregate_clean_vs_transformed.png"),
                 (c2, "retention_heatmap.png")):
    p = FIGS / fig
    if p.exists():
        col.image(str(p), use_container_width=True)

st.markdown(
    "**Honest summary:** Robust v1 is *not* universally better. It trades ~1.8 pp "
    "of clean accuracy (and <= 2 pp on mild JPEG / colour shifts) for large gains "
    "wherever the image is meaningfully degraded. That trade is what the "
    "\"robust detection under real-world transformations\" brief asks for."
)

# ---------------------------------------------------------------------------
# Optional: live demo (loads the checkpoints if available)
# ---------------------------------------------------------------------------
with st.expander("Live demo - run both models on your own image (optional)"):
    if not (BASELINE_CKPT.exists() and ROBUST_CKPT.exists()):
        st.info(
            f"Checkpoints not found. Expected:\n\n"
            f"- `{BASELINE_CKPT.relative_to(ROOT)}`\n- `{ROBUST_CKPT.relative_to(ROOT)}`\n\n"
            "Train them (see README) or copy them in, then reload this page."
        )
    else:
        try:
            import torch
            from PIL import Image
            from src.model import build_model
            from src.transforms import build_eval_transform
            from scripts.evaluate_robustness import CONDITIONS, apply_condition

            @st.cache_resource
            def _load(ckpt_path):
                m = build_model(backbone="efficientnet_b0", pretrained=False, num_classes=2)
                m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                m.eval()
                return m

            base_m = _load(str(BASELINE_CKPT))
            rob_m = _load(str(ROBUST_CKPT))
            tfm = build_eval_transform(64)

            up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
            cond_labels = [c[0] if c[1] == "-" else f"{c[0]}:{c[1]}" for c in CONDITIONS]
            cond_pick = st.selectbox("Apply a transformation", cond_labels, index=0)
            cname, plabel, pvalue = CONDITIONS[cond_labels.index(cond_pick)]

            if up is not None:
                img = Image.open(up).convert("RGB")
                shown = apply_condition(img, cname, pvalue, base_seed=42, idx=0)
                cc1, cc2 = st.columns([1, 2])
                cc1.image(shown, caption=f"input after: {cond_pick}", width=200)
                with torch.no_grad():
                    x = tfm(shown).unsqueeze(0)
                    pb = torch.softmax(base_m(x), 1)[0, 1].item()
                    pr = torch.softmax(rob_m(x), 1)[0, 1].item()
                cc2.write("**P(AI-generated)** - 0 = real, 1 = AI")
                cc2.progress(pb, text=f"Baseline v1: {pb:.3f} -> "
                                      f"{'AI' if pb >= 0.5 else 'REAL'}")
                cc2.progress(pr, text=f"Robust v1: {pr:.3f} -> "
                                      f"{'AI' if pr >= 0.5 else 'REAL'}")
                cc2.caption("Note: models were trained on 32x32 CIFAKE upscaled to "
                            "64px; predictions on unrelated images are illustrative.")
        except Exception as e:  # keep the dashboard usable if torch is missing
            st.warning(f"Live demo unavailable: {e}")
