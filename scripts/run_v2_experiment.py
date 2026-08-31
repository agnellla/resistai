"""
scripts/run_v2_experiment.py
============================
ONE command that runs the complete ResistAI V2 experiment unattended and
produces a scientific report. Every step fails LOUDLY - if anything errors the
run stops with a non-zero exit code and nothing downstream is fabricated.

Pipeline (matches the spec, 13 steps):
   1. verify the V2 dataset manifests exist and have the right columns
   2. verify split disjointness (path + content hash)
   3. verify the held-out generator(s) are absent from train and val
   4. train the linear-probe diagnostic          -> outputs/v2_linear_probe/
   5. train V2 (full fine-tune)                   -> outputs/v2/
   6. (best checkpoint saved by step 5)           -> outputs/v2/best_model.pt
   7. A: evaluate V2 on the FROZEN CIFAKE clean test   (data/splits/test.csv)
   8. B: evaluate V2 on the FROZEN robustness benchmark (17 conditions)
   9. C: evaluate V2 on the real-world / unseen-generator holdout
  10. run the generalised shortcut probes for V1 AND V2
  11. compare V1 vs V2 (A, B, C kept separate)
  12. save every metric               -> reports/v2_results.json
  13. write the final report          -> reports/v2_report.md

FROZEN - never written by this script:
    outputs/baseline_v1/  outputs/robust_v1/
    data/splits/{train,val,test}.csv
    reports/benchmark_results.csv      scripts/evaluate_robustness.py

Typical use (on Colab T4, dataset already assembled by scripts/prepare_v2_data.py):
    python scripts/run_v2_experiment.py --device cuda \
        --real_images data/probe/real_hd --ai_images data/probe/ai_heldout \
        --sanity_images data/probe/sanity

Re-run only the evaluation half (V2 already trained):
    python scripts/run_v2_experiment.py --skip_train --device cuda ...
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

FROZEN_BENCHMARK = ROOT / "reports" / "benchmark_results.csv"
FROZEN_TEST_CSV = ROOT / "data" / "splits" / "test.csv"
V1_BASELINE = ROOT / "outputs" / "baseline_v1" / "best_model.pt"
V1_ROBUST = ROOT / "outputs" / "robust_v1" / "best_model.pt"


def run(step, cmd, **kw):
    print("\n" + "=" * 78)
    print(f"[step {step}] {' '.join(str(c) for c in cmd)}")
    print("=" * 78, flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw)
    if r.returncode != 0:
        raise SystemExit(f"\n[step {step}] FAILED (exit {r.returncode}) - stopping. "
                         f"No results fabricated.")
    print(f"[step {step}] ok ({time.time() - t0:.0f}s)")
    return r


def read_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def read_csv_rows(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
def build_report(args, paths):
    """Assemble reports/v2_report.md + reports/v2_results.json from step outputs.
    Reads only what the steps actually produced; missing pieces are marked."""
    def fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else ("n/a" if x is None else str(x))

    frozen = {(r["transformation"], r["parameter"]): r for r in read_csv_rows(FROZEN_BENCHMARK)}

    a_v2 = read_json(paths["A"])                       # evaluate.py metrics.json
    c_v2 = read_json(paths["C"])                       # evaluate_realworld.py
    c_v1b = read_json(paths.get("C_v1_baseline"))      # V1 baseline on the SAME C set
    c_v1r = read_json(paths.get("C_v1_robust"))        # V1 robust on the SAME C set
    probe = read_json(paths["probes"])                 # shortcut_probes.py
    lp = read_json(paths["lp_metrics"])                # linear probe
    v2cfg = read_json(paths["v2_cfg"])
    v2val = read_json(paths["v2_metrics"])

    # B: V2 robustness rows are the "robust" model rows of the step-8 run
    b_rows = {}
    if Path(paths["B_csv"]).exists():
        for r in read_csv_rows(paths["B_csv"]):
            if r["model"] == "robust":                 # == V2 in the step-8 invocation
                b_rows[(r["transformation"], r["parameter"])] = r

    results = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "objective": "C = real-world / unseen-generator generalisation "
                     "(A = CIFAKE clean, B = CIFAKE corruption robustness are regression checks)",
        "v2_train_config": v2cfg, "v2_val_best": v2val, "linear_probe": lp,
        "A_cifake_clean": {"v1_baseline": {"accuracy": 0.915, "f1": 0.915, "roc_auc": 0.969},
                           "v1_robust": {"accuracy": 0.897, "f1": 0.899, "roc_auc": 0.957},
                           "v2": a_v2},
        "B_cifake_robustness": {
            "frozen_v1": {f"{k[0]}:{k[1]}": v for k, v in frozen.items()},
            "v2_rows": {f"{k[0]}:{k[1]}": v for k, v in b_rows.items()},
        },
        "C_realworld": {"v2": c_v2, "v1_baseline": c_v1b, "v1_robust": c_v1r},
        "shortcut_probes": probe,
    }
    Path(paths["results_json"]).write_text(json.dumps(results, indent=2, default=str))

    # ---- markdown ----
    L = []
    w = L.append
    w("# ResistAI V2 - Scientific Report\n")
    w(f"_Generated {results['generated']}_\n")
    w("> Objective **C**: generalise to real-world photographs and AI images from "
      "generators **not** in training.\n> **A** (CIFAKE clean) and **B** (CIFAKE "
      "corruption robustness) are regression checks and are reported separately - "
      "there is deliberately no combined \"overall accuracy\".\n")

    # 1. what changed
    w("\n## 1. What changed from V1\n")
    if v2cfg:
        ag = v2cfg.get("augmentation_groups", {})
        w(f"- image size **{v2cfg.get('image_size')}** (V1: 64), EfficientNet-B0 kept, "
          f"optimizer {v2cfg.get('optimizer')}, {v2cfg.get('epochs')} epochs, "
          f"cosine schedule (warmup {v2cfg.get('warmup_frac')}), label smoothing "
          f"{v2cfg.get('label_smoothing')}, seed {v2cfg.get('seed')}.")
        w(f"- **acquisition augmentation** on BOTH classes: prob "
          f"{ag.get('acquisition', {}).get('prob')}, "
          f"{ag.get('acquisition', {}).get('num_per_image')} chained transforms from "
          f"{ag.get('acquisition', {}).get('transforms')}.")
        w(f"- robustness augmentation enabled: {ag.get('robustness', {}).get('enabled')}.")
        w(f"- training data: {json.dumps(v2cfg.get('train_composition', {}))}")
    else:
        w("- (v2 run_config.json not found)")

    # 2. dataset
    w("\n## 2. Dataset\n")
    comp = read_json(ROOT / "data" / "splits_v2" / "composition.json")
    leak = read_json(ROOT / "data" / "splits_v2" / "leakage_report.json")
    if comp:
        w(f"- seed {comp.get('seed')}, source `{comp.get('source')}`, "
          f"held-out generators **{comp.get('held_out_generators')}**, "
          f"held-out real sources {comp.get('held_out_real_sources')}")
        for s in comp.get("splits", []):
            w(f"  - **{s['name']}**: n={s['n']} (real {s['real']}, ai {s['ai']}), "
              f"generators={s['generators']}, real_sources={s['real_sources']}, "
              f"resolution={s['by_resolution_bucket']}")
    if leak:
        w(f"- leakage report: **{'OK' if leak.get('ok') else 'PROBLEMS: ' + str(leak.get('problems'))}** "
          f"(checks: {leak.get('checked')}; near-dup grouping: {leak.get('near_dup_grouping')})")

    # 3. training config (dump)
    w("\n## 3. Training configuration\n")
    w("```json")
    w(json.dumps(v2cfg, indent=2) if v2cfg else "(missing)")
    w("```")
    if lp:
        w(f"\nLinear-probe diagnostic (frozen backbone): best val "
          f"{json.dumps(lp.get('best_val', {}))}")

    # 4. results table - A / B summary / C
    w("\n## 4. Results (A, B, C kept separate)\n")
    w("### A - CIFAKE clean test (`data/splits/test.csv`, frozen)\n")
    w("| Model | Accuracy | F1 | ROC-AUC |")
    w("|---|---|---|---|")
    w("| Baseline V1 | 0.915 | 0.915 | 0.969 |")
    w("| Robust V1 | 0.897 | 0.899 | 0.957 |")
    if a_v2:
        w(f"| **V2** | {a_v2.get('accuracy'):.3f} | {a_v2.get('f1'):.3f} | "
          f"{a_v2.get('roc_auc'):.3f} |")
        cm = a_v2.get("confusion_matrix", {})
        w(f"\nV2 confusion matrix: TN={cm.get('true_real_pred_real')} "
          f"FP={cm.get('true_real_pred_ai')} FN={cm.get('true_ai_pred_real')} "
          f"TP={cm.get('true_ai_pred_ai')}")
    else:
        w("| **V2** | (missing) | | |")

    w("\n### B - CIFAKE robustness benchmark (17 conditions)\n")
    w("V1 numbers are the frozen `reports/benchmark_results.csv`. V2 is evaluated by "
      "`scripts/evaluate_robustness.py` at image_size 224 (its \"robust\" column).\n")
    w("| Condition | V1 base acc | V1 robust acc | V2 acc | V1 base F1 | V1 robust F1 | V2 F1 |")
    w("|---|---|---|---|---|---|---|")
    order = [("clean", "-"), ("jpeg", "q90"), ("jpeg", "q70"), ("jpeg", "q50"), ("jpeg", "q30"),
             ("blur", "sigma0.5"), ("blur", "sigma1.0"), ("blur", "sigma2.0"),
             ("resize", "scale0.5"), ("resize", "scale0.25"),
             ("noise", "sigma0.02"), ("noise", "sigma0.05"), ("noise", "sigma0.10"),
             ("color_jitter", "brightness_pm20"), ("color_jitter", "contrast_pm20"),
             ("color_jitter", "saturation_pm20"), ("center_crop", "keep0.80")]
    for k in order:
        fr = frozen.get(k, {})
        v2 = b_rows.get(k, {})
        v2a = f"{float(v2['accuracy']):.3f}" if v2 else "-"
        v2f = f"{float(v2['f1']):.3f}" if v2 else "-"
        w(f"| {k[0]}:{k[1]} | {fr.get('baseline_acc','-')} | {fr.get('robust_acc','-')} | "
          f"{v2a} | {fr.get('baseline_f1','-')} | {fr.get('robust_f1','-')} | {v2f} |")

    w("\n### C - real-world / unseen-generator holdout (PRIMARY objective)\n")
    if c_v2:
        o = c_v2["overall"]
        w("Paired on the SAME held-out set (V1 checkpoints run at image_size 64, V2 at "
          f"{(v2cfg or {}).get('image_size', '224')}):\n")
        w("| Model | n | acc | F1 | ROC-AUC | mean P(AI) | FP-rate REAL | TP-rate AI |")
        w("|---|---|---|---|---|---|---|---|")
        for nm, cc in (("Baseline V1", c_v1b), ("Robust V1", c_v1r), ("**V2**", c_v2)):
            if not cc:
                w(f"| {nm} | (missing) | | | | | | |"); continue
            x = cc["overall"]
            w(f"| {nm} | {x['n']} | {fmt(x['accuracy'])} | {fmt(x['f1'])} | "
              f"{fmt(x['roc_auc'])} | {fmt(x['mean_p_ai'])} | "
              f"{fmt(x['false_positive_rate_on_real'])} | {fmt(x['true_positive_rate_on_ai'])} |")
        w("")
        w(f"- V2 detail: precision {o['precision']:.3f}  recall {o['recall']:.3f}")
        w(f"- mean P(AI) {o['mean_p_ai']:.3f}  |  FP-rate on REAL "
          f"**{o['false_positive_rate_on_real']}**  |  TP-rate on AI "
          f"**{o['true_positive_rate_on_ai']}**")
        cm = o["confusion_matrix"]
        w(f"- confusion matrix: TN={cm['true_real_pred_real']} FP={cm['true_real_pred_ai']} "
          f"FN={cm['true_ai_pred_real']} TP={cm['true_ai_pred_ai']}")
        w("\n**By generator (AI rows):**\n\n| Generator | n | mean P(AI) | detected-as-AI |")
        w("|---|---|---|---|")
        for g, s in c_v2.get("by_generator", {}).items():
            w(f"| {g} | {s['n']} | {s['mean_p_ai']:.3f} | {s['detected_as_ai_rate']:.3f} |")
        w("\n**By real-photo source (REAL rows):**\n\n| Source | n | mean P(AI) | false-positive rate |")
        w("|---|---|---|---|")
        for s_, s in c_v2.get("by_real_source", {}).items():
            w(f"| {s_} | {s['n']} | {s['mean_p_ai']:.3f} | {s['false_positive_rate']:.3f} |")
        w("\n**By resolution bucket:**\n\n| Bucket | n | accuracy | FP-rate on REAL | TP-rate on AI |")
        w("|---|---|---|---|---|")
        for b, s in c_v2.get("by_resolution_bucket", {}).items():
            acc = f"{s['accuracy']:.3f}" if s['accuracy'] is not None else "-"
            fpr = f"{s['false_positive_rate_on_real']:.3f}" if s['false_positive_rate_on_real'] is not None else "-"
            tpr = f"{s['true_positive_rate_on_ai']:.3f}" if s['true_positive_rate_on_ai'] is not None else "-"
            w(f"| {b} | {s['n']} | {acc} | {fpr} | {tpr} |")
    else:
        w("(realworld_C.json missing)")

    # 5. shortcut curves V1 vs V2
    w("\n## 5. Shortcut sensitivity - V1 vs V2\n")
    if probe:
        mods = probe.get("models", {})
        w("### CIFAKE quick probe - grain sweep on FAKE (want P(AI) to STAY HIGH)\n")
        w("| Model | s0 | s4 | s8 | s12 | s16 | s24 | drop 0->24 |")
        w("|---|---|---|---|---|---|---|---|")
        for name, m in mods.items():
            g = m.get("cifake_quick_probe", {}).get("fake", {}).get("grain_sweep")
            if g:
                w(f"| {name} | " + " | ".join(f"{g[f'sigma_{s}']:.3f}" for s in (0, 4, 8, 12, 16, 24))
                  + f" | {m['cifake_quick_probe']['fake']['grain_p_ai_drop_0_to_24']:+.3f} |")
        w("\n_Baseline V1 collapses ~0.92 -> ~0.13 here (the grain shortcut). "
          "V2 should stay much flatter._\n")

        # definitive real-HD downsample sweep
        any_ds = any("real_hd_downsample_and_grain" in m for m in mods.values())
        if any_ds:
            w("### Real HD photo - DOWNSAMPLING sweep (want P(AI) to STAY LOW as size drops)\n")
            w("| Model | 1024px | 512px | 256px | 128px | 64px | rise 1024->64 |")
            w("|---|---|---|---|---|---|---|")
            for name, m in mods.items():
                d = m.get("real_hd_downsample_and_grain", {}).get("downsample_sweep")
                if d:
                    w(f"| {name} | " + " | ".join(
                        f"{d[f'long_side_{t}']:.3f}" for t in (1024, 512, 256, 128, 64))
                      + f" | {m['real_hd_downsample_and_grain'].get('downsample_p_ai_rise_1024_to_64', float('nan')):+.3f} |")
        any_ai = any("ai_images_grain_blur_downsample" in m for m in mods.values())
        if any_ai:
            w("\n### Held-out AI images - GRAIN sweep (want P(AI) to STAY HIGH)\n")
            w("| Model | s0 | s4 | s8 | s12 | s16 | s24 | drop 0->24 |")
            w("|---|---|---|---|---|---|---|---|")
            for name, m in mods.items():
                g = m.get("ai_images_grain_blur_downsample", {}).get("grain_sweep")
                if g:
                    w(f"| {name} | " + " | ".join(f"{g[f'sigma_{s}']:.3f}" for s in (0, 4, 8, 12, 16, 24))
                      + f" | {m['ai_images_grain_blur_downsample']['grain_p_ai_drop_0_to_24']:+.3f} |")
        if not any_ds and not any_ai:
            w("_No --real_images / --ai_images were supplied, so only the CIFAKE quick "
              "probe ran. Supply genuine HD photos + held-out AI folders for the "
              "definitive curves._\n")
    else:
        w("(shortcut probe JSON missing)")

    # 6. scientific conclusion - the six questions
    w("\n## 6. Scientific conclusion\n")

    def delta(a, b):
        try:
            return f"{a - b:+.3f}"
        except Exception:
            return "n/a"

    q = []
    if c_v2:
        o = c_v2["overall"]
        ob = c_v1b["overall"] if c_v1b else None
        if ob:
            q.append(f"**1. Did V2 improve real-world detection vs V1?** Paired on the SAME C set: "
                     f"accuracy Baseline V1 {ob['accuracy']:.3f} -> V2 {o['accuracy']:.3f} "
                     f"({delta(o['accuracy'], ob['accuracy'])}); F1 {ob['f1']:.3f} -> {o['f1']:.3f} "
                     f"({delta(o['f1'], ob['f1'])}); ROC-AUC {fmt(ob['roc_auc'])} -> {fmt(o['roc_auc'])}.")
            q.append(f"**2. Did V2 reduce false positives on genuine HD photographs?** FP-rate on "
                     f"REAL: Baseline V1 {fmt(ob['false_positive_rate_on_real'])} -> "
                     f"V2 {fmt(o['false_positive_rate_on_real'])} "
                     f"({delta(o['false_positive_rate_on_real'], ob['false_positive_rate_on_real'])}); "
                     f"mean P(AI) on REAL rows dropped correspondingly. By source: "
                     + ", ".join(f"{k} {v['false_positive_rate']:.3f}"
                                 for k, v in c_v2.get('by_real_source', {}).items()) + ".")
        else:
            q.append(f"**1. Did V2 improve real-world detection vs V1?** V2 on C: accuracy "
                     f"{o['accuracy']:.3f}, F1 {o['f1']:.3f}, ROC-AUC {fmt(o['roc_auc'])}. "
                     f"(V1-on-C outputs missing - re-run step 9b/9c.)")
            q.append(f"**2. Did V2 reduce false positives on genuine HD photographs?** V2 FP-rate on "
                     f"REAL in C = {fmt(o['false_positive_rate_on_real'])}; by source: "
                     + ", ".join(f"{k} {v['false_positive_rate']:.3f}"
                                 for k, v in c_v2.get('by_real_source', {}).items()) + ".")
        srcs = c_v2.get("by_real_source", {})
        q.append(f"**3. Does V2 generalise across real-photo sources?** "
                 + (" ; ".join(f"{k}: mean P(AI) {v['mean_p_ai']:.3f}, FP {v['false_positive_rate']:.3f}"
                               for k, v in srcs.items()) if srcs else "only one real source in C.")
                 + " Consistent low FP across sources = yes; a spike on one source = domain gap.")
        gens = c_v2.get("by_generator", {})
        q.append(f"**4. Does V2 generalise to unseen generators?** held-out = "
                 f"{comp.get('held_out_generators') if comp else '?'}; per-generator detect rate: "
                 + (", ".join(f"{k} {v['detected_as_ai_rate']:.3f}" for k, v in gens.items()) if gens else "n/a")
                 + ". High detect rate on a generator absent from training = genuine generalisation.")
    else:
        q += ["**1-4.** realworld_C.json missing - cannot answer."]

    # Q5 shortcut
    if probe:
        mods = probe.get("models", {})
        def gdrop(name, blk):
            m = mods.get(name, {})
            if blk == "cifake":
                return m.get("cifake_quick_probe", {}).get("fake", {}).get("grain_p_ai_drop_0_to_24")
            return m.get("ai_images_grain_blur_downsample", {}).get("grain_p_ai_drop_0_to_24")
        b1, v2d = gdrop("baseline_v1", "cifake"), gdrop("v2", "cifake")
        line = ("**5. Did the grain/blur/resolution shortcut get weaker?** CIFAKE grain-sweep "
                f"P(AI) drop (0->24): Baseline V1 {b1:+.3f}" if b1 is not None else
                "**5.** baseline probe missing")
        if v2d is not None:
            line += f", V2 {v2d:+.3f}  (delta {delta(v2d, b1) if b1 is not None else 'n/a'}; "
            line += "smaller magnitude = shortcut reduced)."
        # downsample rise
        ds1 = mods.get("baseline_v1", {}).get("real_hd_downsample_and_grain", {}).get(
            "downsample_p_ai_rise_1024_to_64")
        ds2 = mods.get("v2", {}).get("real_hd_downsample_and_grain", {}).get(
            "downsample_p_ai_rise_1024_to_64")
        if ds1 is not None and ds2 is not None:
            line += (f" Real-HD downsampling P(AI) rise 1024->64: Baseline V1 {ds1:+.3f}, "
                     f"V2 {ds2:+.3f}.")
        q.append(line)
    else:
        q.append("**5.** shortcut probe JSON missing - cannot answer.")

    # Q6 CIFAKE regression
    if a_v2:
        q.append(f"**6. Did we sacrifice too much on the frozen CIFAKE benchmark?** "
                 f"CIFAKE clean (A): V2 acc {a_v2['accuracy']:.3f} vs Baseline V1 0.915 "
                 f"({delta(a_v2['accuracy'], 0.915)}), F1 {a_v2['f1']:.3f} vs 0.915 "
                 f"({delta(a_v2['f1'], 0.915)}). "
                 + ("B-benchmark V2 vs frozen V1 is in section 4. " if b_rows else "")
                 + "A modest A/B regression is acceptable IF C improved materially; "
                 "judge against sections 4C and 5.")
    else:
        q.append("**6.** A metrics missing - cannot answer.")

    w("\n\n".join(q))
    w("\n\n## 7. Verification\n")
    w("- frozen artifacts untouched (checked by `scripts/run_v2_experiment.py` preflight): "
      "`outputs/baseline_v1/`, `outputs/robust_v1/`, `data/splits/*.csv`, "
      "`reports/benchmark_results.csv`, `scripts/evaluate_robustness.py`")
    w(f"- V2 split leakage report: `data/splits_v2/leakage_report.json` "
      f"({'OK' if (leak or {}).get('ok') else 'see file'})")
    w("- B was run at image_size 224 with both `evaluate_robustness.py` checkpoint slots "
      "set to the V2 checkpoint; only its `robust` rows are used as V2. The V1 B numbers "
      "come from the frozen CSV, NOT from re-running V1 at 224.")

    Path(paths["report_md"]).write_text("\n".join(L) + "\n")
    print(f"\n[report] wrote {paths['report_md']} and {paths['results_json']}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run the full ResistAI V2 experiment",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--train_csv", default="data/splits_v2/train.csv")
    ap.add_argument("--val_csv", default="data/splits_v2/val.csv")
    ap.add_argument("--realworld_csv", default="data/splits_v2/realworld_test.csv")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--probe_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--acq_prob", type=float, default=0.6)
    ap.add_argument("--acq_num", type=int, default=2)
    ap.add_argument("--robustness_aug", action="store_true", default=True,
                    help="apply the V1 robustness aug group during V2 training too")
    ap.add_argument("--no_robustness_aug", dest="robustness_aug", action="store_false")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--real_images", default=None, help="dir of genuine HD photos for the shortcut probe")
    ap.add_argument("--ai_images", default=None, help="dir of held-out AI images for the shortcut probe")
    ap.add_argument("--sanity_images", default=None, help="dir for the one-shot sanity probe")
    ap.add_argument("--skip_train", action="store_true", help="reuse an existing outputs/v2/best_model.pt")
    ap.add_argument("--probe_n", type=int, default=64)
    args = ap.parse_args()

    # ---- preflight: frozen artifacts present + record hashes ----
    for p in (FROZEN_BENCHMARK, FROZEN_TEST_CSV, V1_BASELINE, V1_ROBUST):
        if not p.exists():
            raise SystemExit(f"missing frozen artifact {p}")
    import hashlib
    frozen_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (FROZEN_BENCHMARK, FROZEN_TEST_CSV,
                               ROOT / "data/splits/train.csv", ROOT / "data/splits/val.csv",
                               ROOT / "scripts/evaluate_robustness.py")}

    out_v2 = ROOT / "outputs" / "v2"
    out_lp = ROOT / "outputs" / "v2_linear_probe"
    out_B = out_v2 / "robustness_B"
    paths = {
        "A": out_v2 / "cifake_clean_A" / "metrics.json",
        "B_csv": out_B / "results.csv",
        "C": out_v2 / "realworld_C.json",
        "C_v1_baseline": out_v2 / "realworld_C_v1_baseline.json",
        "C_v1_robust": out_v2 / "realworld_C_v1_robust.json",
        "probes": ROOT / "reports" / "shortcut_probes_v2.json",
        "lp_metrics": out_lp / "metrics.json",
        "v2_cfg": out_v2 / "run_config.json",
        "v2_metrics": out_v2 / "metrics.json",
        "results_json": ROOT / "reports" / "v2_results.json",
        "report_md": ROOT / "reports" / "v2_report.md",
    }

    # 1-3. verify manifests + disjointness + held-out generator absence
    run(1, [PY, "scripts/verify_v2_splits.py", "--splits_dir",
            str(Path(args.train_csv).parent), "--check_files"])

    if not args.skip_train:
        # 4. linear probe
        run(4, [PY, "train_v2.py", "--train_csv", args.train_csv, "--val_csv", args.val_csv,
                "--output_dir", str(out_lp), "--linear_probe",
                "--epochs", args.probe_epochs, "--image_size", args.image_size,
                "--batch_size", args.batch_size, "--acq_prob", args.acq_prob,
                "--acq_num", args.acq_num, "--device", args.device,
                "--num_workers", args.num_workers, "--seed", args.seed]
               + (["--robustness_aug"] if args.robustness_aug else []))
        # 5. full fine-tune
        run(5, [PY, "train_v2.py", "--train_csv", args.train_csv, "--val_csv", args.val_csv,
                "--output_dir", str(out_v2), "--epochs", args.epochs,
                "--image_size", args.image_size, "--batch_size", args.batch_size,
                "--lr", args.lr, "--label_smoothing", args.label_smoothing,
                "--scheduler", "cosine", "--acq_prob", args.acq_prob, "--acq_num", args.acq_num,
                "--device", args.device, "--num_workers", args.num_workers, "--seed", args.seed]
               + (["--robustness_aug"] if args.robustness_aug else []))
    v2_ckpt = out_v2 / "best_model.pt"
    if not v2_ckpt.exists():
        raise SystemExit(f"{v2_ckpt} not found - training did not produce a checkpoint")

    # 7. A - CIFAKE clean (frozen test.csv)
    run(7, [PY, "evaluate.py", "--checkpoint", str(v2_ckpt), "--test_csv", str(FROZEN_TEST_CSV),
            "--image_size", args.image_size, "--output_dir", str(out_v2 / "cifake_clean_A"),
            "--device", args.device, "--num_workers", args.num_workers])

    # 8. B - frozen robustness benchmark; both slots = V2, read "robust" rows
    run(8, [PY, "scripts/evaluate_robustness.py",
            "--baseline_checkpoint", str(v2_ckpt), "--robust_checkpoint", str(v2_ckpt),
            "--test_csv", str(FROZEN_TEST_CSV), "--image_size", args.image_size,
            "--batch_size", args.batch_size, "--device", args.device,
            "--num_workers", args.num_workers, "--output_dir", str(out_B)])

    # 9. C - real-world holdout: V2, and V1 baseline/robust for a paired comparison
    run(9, [PY, "scripts/evaluate_realworld.py", "--checkpoint", str(v2_ckpt),
            "--test_csv", args.realworld_csv, "--image_size", args.image_size,
            "--device", args.device, "--out", str(paths["C"])])
    run("9b", [PY, "scripts/evaluate_realworld.py", "--checkpoint", str(V1_BASELINE),
              "--test_csv", args.realworld_csv, "--image_size", 64,
              "--device", args.device, "--out", str(paths["C_v1_baseline"])])
    run("9c", [PY, "scripts/evaluate_realworld.py", "--checkpoint", str(V1_ROBUST),
              "--test_csv", args.realworld_csv, "--image_size", 64,
              "--device", args.device, "--out", str(paths["C_v1_robust"])])

    # 10. shortcut probes V1 + V2
    probe_cmd = [PY, "scripts/shortcut_probes.py", "--checkpoints",
                 f"baseline_v1={V1_BASELINE}:64", f"robust_v1={V1_ROBUST}:64",
                 f"v2={v2_ckpt}:{args.image_size}",
                 "--n", args.probe_n, "--device", args.device, "--out", str(paths["probes"])]
    if args.real_images:
        probe_cmd += ["--real_images", args.real_images]
    if args.ai_images:
        probe_cmd += ["--ai_images", args.ai_images]
    if args.sanity_images:
        probe_cmd += ["--images", args.sanity_images]
    run(10, probe_cmd)

    # 11-13. compare + save + report
    build_report(args, paths)

    # frozen-artifact tamper check
    changed = [k for k, v in frozen_hashes.items()
               if hashlib.sha256((ROOT / k).read_bytes()).hexdigest() != v]
    if changed:
        raise SystemExit(f"FROZEN ARTIFACT CHANGED during the run: {changed} - investigate!")
    print("\n[verify] frozen artifacts unchanged:", list(frozen_hashes))
    print("\nDONE. Read reports/v2_report.md")


if __name__ == "__main__":
    main()
