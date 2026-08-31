"""
reports/make_figures.py
=======================
Build the presentation figures for the ResistAI robustness experiment.

Reads reports/benchmark_results.csv (the authoritative per-condition
accuracy / F1 / ROC-AUC for Baseline v1 and Robust v1, from the 10,000-image
frozen-test benchmark run on the Colab Tesla T4). Nothing is fabricated - every
number is read from that CSV; gap / retention / reconstructed error counts are
computed from it.

Run:  python reports/make_figures.py
Out:  reports/figures/*.png  +  reports/benchmark_derived.csv
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV_IN = HERE / "benchmark_results.csv"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Colour-blind-safe categorical pair (blue / orange - the standard safe pair).
C_BASE = "#4E79A7"
C_ROBUST = "#F28E2B"
C_POS = "#59A14F"   # robust better
C_NEG = "#E15759"   # robust worse
INK = "#222222"
GRID = "#D9D9D9"

# fixed test composition (data/splits/test.csv): 5000 real + 5000 AI
N = 10000
N_POS = 5000        # actual AI (positive class)
N_NEG = 5000        # actual real

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
})


def load_rows():
    with open(CSV_IN) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in list(r):
            if k not in ("transformation", "parameter"):
                r[k] = float(r[k])
        r["label"] = r["transformation"] if r["parameter"] == "-" else f"{r['transformation']}:{r['parameter']}"
    return rows


def reconstruct_counts(acc, f1):
    """
    Reconstruct (TP, FP, FN, TN) from accuracy + F1 on the fixed 5000/5000
    test split. This is exact algebra, not an estimate:
        acc = (TP + TN) / N,  TN = N_NEG - FP,  FN = N_POS - TP
        F1  = 2TP / (2TP + FP + FN)
    -> TP = N_POS * F1 * (1 - acc) / (1 - F1)
    Values are rounded to whole images; 3-decimal inputs give +/-~1 image slack.
    """
    if f1 >= 1.0:
        tp = N_POS
    else:
        tp = N_POS * f1 * (1.0 - acc) / (1.0 - f1)
    tp = round(tp)
    fp = round(tp - (N * acc - N_NEG))
    fp = max(0, min(N_NEG, fp))
    fn = N_POS - tp
    tn = N_NEG - fp
    return tp, fp, fn, tn


def derive(rows):
    clean = next(r for r in rows if r["transformation"] == "clean")
    out = []
    for r in rows:
        d = dict(r)
        d["acc_gap"] = r["robust_acc"] - r["baseline_acc"]
        d["f1_gap"] = r["robust_f1"] - r["baseline_f1"]
        d["auc_gap"] = r["robust_auc"] - r["baseline_auc"]
        d["baseline_acc_retention"] = r["baseline_acc"] / clean["baseline_acc"]
        d["robust_acc_retention"] = r["robust_acc"] / clean["robust_acc"]
        d["baseline_f1_retention"] = r["baseline_f1"] / clean["baseline_f1"]
        d["robust_f1_retention"] = r["robust_f1"] / clean["robust_f1"]
        for m in ("baseline", "robust"):
            tp, fp, fn, tn = reconstruct_counts(r[f"{m}_acc"], r[f"{m}_f1"])
            d[f"{m}_fp"], d[f"{m}_fn"] = fp, fn
        out.append(d)
    return out, clean


def mean(xs):
    return sum(xs) / len(xs)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_grouped_metric(rows, metric, title, fname):
    labels = [r["label"] for r in rows]
    base = [r[f"baseline_{metric}"] for r in rows]
    rob = [r[f"robust_{metric}"] for r in rows]
    x = range(len(rows))
    w = 0.4

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar([i - w / 2 for i in x], base, w, label="Baseline v1", color=C_BASE)
    ax.bar([i + w / 2 for i in x], rob, w, label="Robust v1", color=C_ROBUST)
    # mark the clean reference
    ax.axvline(0.5, color=GRID, lw=1.0, ls="--")
    ax.text(0, 1.02, "clean", ha="center", va="bottom", fontsize=8, color="#666")
    ax.text(len(rows) / 2 + 0.5, 1.02, "transformed (stress test)", ha="center",
            va="bottom", fontsize=8, color="#666")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel(metric.upper() if metric == "f1" else metric.capitalize())
    ax.set_ylim(0, 1.08)
    ax.set_title(title, fontsize=12, pad=16, loc="left")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def fig_gap(rows, fname):
    # sort by accuracy gap so the story reads top-to-bottom
    rs = sorted(rows, key=lambda r: r["acc_gap"])
    labels = [r["label"] for r in rs]
    gaps = [r["acc_gap"] * 100 for r in rs]
    colors = [C_POS if g >= 0 else C_NEG for g in gaps]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, gaps, color=colors)
    ax.axvline(0, color=INK, lw=1.0)
    for i, g in enumerate(gaps):
        ax.text(g + (0.6 if g >= 0 else -0.6), i, f"{g:+.1f}",
                va="center", ha="left" if g >= 0 else "right", fontsize=8.5)
    ax.set_xlabel("Robust v1 accuracy - Baseline v1 accuracy  (percentage points)")
    ax.set_title("Robustness gap by transformation\n"
                 "green = transformation-aware training helps,  red = it hurts",
                 fontsize=12, pad=12, loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(gaps) - 6, max(gaps) + 6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def fig_aggregate(rows, fname):
    trans = [r for r in rows if r["transformation"] != "clean"]
    clean = next(r for r in rows if r["transformation"] == "clean")

    groups = ["CLEAN", "TRANSFORMED\n(mean of 16)"]
    metrics = ["acc", "f1", "auc"]
    titles = {"acc": "Accuracy", "f1": "F1", "auc": "ROC-AUC"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    for ax, m in zip(axes, metrics):
        base_vals = [clean[f"baseline_{m}"], mean([r[f"baseline_{m}"] for r in trans])]
        rob_vals = [clean[f"robust_{m}"], mean([r[f"robust_{m}"] for r in trans])]
        x = range(len(groups)); w = 0.36
        ax.bar([i - w / 2 for i in x], base_vals, w, color=C_BASE, label="Baseline v1")
        ax.bar([i + w / 2 for i in x], rob_vals, w, color=C_ROBUST, label="Robust v1")
        for i, (b, r) in enumerate(zip(base_vals, rob_vals)):
            ax.text(i - w / 2, b + 0.015, f"{b:.3f}", ha="center", fontsize=8.5)
            ax.text(i + w / 2, r + 0.015, f"{r:.3f}", ha="center", fontsize=8.5)
        ax.set_xticks(list(x)); ax.set_xticklabels(groups, fontsize=9)
        ax.set_title(titles[m], fontsize=11)
        ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("score")
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle("Aggregate performance: clean vs mean over transformed conditions",
                 fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def fig_retention_heatmap(rows, fname):
    trans = [r for r in rows if r["transformation"] != "clean"]
    labels = [r["label"] for r in trans]
    data = [[r["baseline_acc_retention"], r["robust_acc_retention"]] for r in trans]

    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    ax.grid(False)
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline v1", "Robust v1"])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5)
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            # YlGnBu: high retention -> dark cell -> needs white text
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v >= 0.8 else INK)
    ax.set_title("Accuracy retention under transformation\n"
                 "(transformed accuracy / that model's clean accuracy)",
                 fontsize=11, pad=12, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.045, label="retention")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def write_derived(rows):
    cols = ["label", "transformation", "parameter",
            "baseline_acc", "robust_acc", "acc_gap",
            "baseline_f1", "robust_f1", "f1_gap",
            "baseline_auc", "robust_auc", "auc_gap",
            "baseline_acc_retention", "robust_acc_retention",
            "baseline_f1_retention", "robust_f1_retention",
            "baseline_fp", "baseline_fn", "robust_fp", "robust_fn"]
    with open(HERE / "benchmark_derived.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) if not isinstance(r.get(c), float) else round(r[c], 6)
                        for c in cols])


def main():
    rows = load_rows()
    derived, clean = derive(rows)
    trans = [r for r in derived if r["transformation"] != "clean"]

    fig_grouped_metric(derived, "acc",
                       "Detection accuracy by transformation - Baseline v1 vs Robust v1\n"
                       "10,000-image frozen test set, every image transformed at the listed severity",
                       "accuracy_by_transformation.png")
    fig_grouped_metric(derived, "f1",
                       "Detection F1 by transformation - Baseline v1 vs Robust v1",
                       "f1_by_transformation.png")
    fig_gap(trans, "accuracy_gap.png")
    fig_aggregate(derived, "aggregate_clean_vs_transformed.png")
    fig_retention_heatmap(derived, "retention_heatmap.png")
    write_derived(derived)

    # ---- aggregate cross-check against the numbers provided --------------
    def agg(key):
        return mean([r[key] for r in trans])

    print("figures written to", FIG_DIR)
    print("\nTransformed-only aggregates (mean of 16 conditions), computed from CSV:")
    print(f"  baseline: acc {agg('baseline_acc'):.3f}  f1 {agg('baseline_f1'):.3f}  auc {agg('baseline_auc'):.3f}")
    print(f"  robust  : acc {agg('robust_acc'):.3f}  f1 {agg('robust_f1'):.3f}  auc {agg('robust_auc'):.3f}")
    print("  (provided authoritative: baseline 0.754/0.676/0.836,  robust 0.833/0.821/0.905)")
    print(f"\n  mean accuracy gap (robust - baseline), transformed: "
          f"{agg('acc_gap')*100:+.1f} pp")
    print(f"  mean F1 gap, transformed: {agg('f1_gap')*100:+.1f} pp")
    print(f"  mean acc retention  baseline {agg('baseline_acc_retention'):.3f}  "
          f"robust {agg('robust_acc_retention'):.3f}")
    print(f"  mean F1  retention  baseline {agg('baseline_f1_retention'):.3f}  "
          f"robust {agg('robust_f1_retention'):.3f}")

    # reconstructed error-count sums (clearly a sum across evaluations)
    b_fp = sum(r["baseline_fp"] for r in trans); b_fn = sum(r["baseline_fn"] for r in trans)
    r_fp = sum(r["robust_fp"] for r in trans); r_fn = sum(r["robust_fn"] for r in trans)
    print(f"\n  reconstructed FP/FN summed across the 16 transformed evaluations:")
    print(f"    baseline  FP {b_fp}   FN {b_fn}")
    print(f"    robust    FP {r_fp}   FN {r_fn}")


if __name__ == "__main__":
    main()
