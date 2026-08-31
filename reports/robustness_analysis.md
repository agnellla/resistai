# ResistAI - Robustness Analysis (Baseline v1 vs Robust v1)

All numbers in this document are the **authoritative benchmark results** from the
10,000-image frozen-test run on the Colab Tesla T4. They are transcribed into
`reports/benchmark_results.csv`; `reports/make_figures.py` reads that file and
computes every gap / retention / figure. Nothing here is estimated except the
per-condition error counts, which are explicitly labelled as reconstructed.

---

## 1. Experiment objective

Test one hypothesis:

> **Transformation-aware training improves robustness to realistic image
> post-processing (JPEG, blur, resize, noise, colour shift, crop) while keeping
> competitive clean-image detection performance.**

Two models are compared on identical data: a plain **Baseline** and a **Robust**
model whose *only* difference is that real-world transforms are applied to
training images on the fly.

## 2. Controlled variables (identical for both models)

| Held constant | Value |
|---|---|
| Architecture | EfficientNet-B0, ImageNet-pretrained, 2-class head (~4.0M trainable params) |
| Optimiser / LR / weight decay | Adam, 3e-4, 1e-4 |
| Dataset | CIFAKE (CIFAR-10 reals + Stable-Diffusion-v1.4 fakes) |
| Split | Frozen 80/10/10 manifests (`data/splits/*.csv`), seed 42, stratified, disjoint |
| Train / val budget | 8,000 train / 2,000 val (class-balanced), 3 epochs |
| Image size | 64 px |
| Batch size | 64 |
| Seed | 42 |
| Test set | The full 10,000-image `data/splits/test.csv` (5,000 real + 5,000 AI), never seen in training or validation |

**Only difference:** Robust v1 adds on-the-fly augmentation during training -
augmentation probability **0.7**, exactly **one** randomly chosen transform per
transformed image, drawn from {JPEG, blur, resize, noise, colour jitter, centre
crop}. Validation and test are always **clean**. Labels never change.

## 3. Baseline vs Robust methodology

- **Training:** identical pipeline; Robust wraps each training image in
  `RandomTransform(prob=0.7, num=1)` (`src/augmentations.py`). Nothing is written
  to disk.
- **Benchmark** (`scripts/evaluate_robustness.py`): a **stress test**, not random
  augmentation. For each of 17 conditions (clean + 16 transformed), **every** test
  image is transformed at the fixed listed severity, then **both models are scored
  on the identical transformed tensor**. Deterministic (seed 42; each image's
  transform seeded from `(seed, condition, parameter, index)`). Models run in
  `eval()` under `torch.no_grad()`.
- Only `data/splits/test.csv` images are read (asserted; 0 outside).

## 4. Clean-test comparison

| Metric | Baseline v1 | Robust v1 | Δ (Robust - Baseline) |
|---|---|---|---|
| Accuracy | 0.9152 | 0.8966 | **-1.86 pp** |
| Precision | 0.9126 | 0.8808 | -3.18 pp |
| Recall | 0.9184 | 0.9174 | -0.10 pp |
| F1 | 0.9155 | 0.8987 | **-1.68 pp** |
| ROC-AUC | 0.9694 | 0.9573 | **-1.21 pp** |
| False positives (real - > AI) | 440 | 621 | +181 |
| False negatives (AI - > real) | 408 | 413 | +5 |

On clean images the Robust model is **slightly worse**, mostly through more false
positives (it flags more genuine photos as AI). This is the cost of spending
model capacity on degraded inputs it will not see at clean test time.

## 5. Robustness benchmark (10,000-image test set, per condition)

`acc / F1 / ROC-AUC`. See `reports/figures/accuracy_by_transformation.png` and
`f1_by_transformation.png`.

| Condition | Parameter | Baseline acc | Robust acc | Acc gap (pp) | Baseline F1 | Robust F1 | F1 gap (pp) | Baseline acc-retention | Robust acc-retention |
|---|---|---|---|---|---|---|---|---|---|
| clean | - | 0.915 | 0.897 | -1.8 | 0.915 | 0.899 | -1.6 | 1.00 | 1.00 |
| jpeg | q90 | 0.912 | 0.896 | -1.6 | 0.912 | 0.897 | -1.5 | 1.00 | 1.00 |
| jpeg | q70 | 0.908 | 0.895 | -1.3 | 0.911 | 0.897 | -1.4 | 0.99 | 1.00 |
| jpeg | q50 | 0.864 | 0.862 | -0.2 | 0.868 | 0.860 | -0.8 | 0.94 | 0.96 |
| jpeg | q30 | 0.862 | 0.857 | -0.5 | 0.852 | 0.857 | +0.5 | 0.94 | 0.96 |
| blur | sigma 0.5 | 0.786 | 0.875 | **+8.9** | 0.734 | 0.867 | **+13.3** | 0.86 | 0.98 |
| blur | sigma 1.0 | 0.569 | 0.819 | **+25.0** | 0.259 | 0.815 | **+55.6** | 0.62 | 0.91 |
| blur | sigma 2.0 | 0.531 | 0.728 | **+19.7** | 0.359 | 0.683 | **+32.4** | 0.58 | 0.81 |
| resize | scale 0.5 | 0.554 | 0.803 | **+24.9** | 0.236 | 0.794 | **+55.8** | 0.61 | 0.90 |
| resize | scale 0.25 | 0.524 | 0.721 | **+19.7** | 0.372 | 0.677 | **+30.5** | 0.57 | 0.80 |
| noise | sigma 0.02 | 0.887 | 0.875 | -1.2 | 0.887 | 0.875 | -1.2 | 0.97 | 0.98 |
| noise | sigma 0.05 | 0.744 | 0.822 | **+7.8** | 0.736 | 0.827 | **+9.1** | 0.81 | 0.92 |
| noise | sigma 0.10 | 0.533 | 0.660 | **+12.7** | 0.454 | 0.573 | **+11.9** | 0.58 | 0.74 |
| color_jitter | brightness +/-20% | 0.911 | 0.891 | -2.0 | 0.911 | 0.893 | -1.8 | 1.00 | 0.99 |
| color_jitter | contrast +/-20% | 0.906 | 0.891 | -1.5 | 0.906 | 0.893 | -1.3 | 0.99 | 0.99 |
| color_jitter | saturation +/-20% | 0.914 | 0.895 | -1.9 | 0.915 | 0.897 | -1.8 | 1.00 | 1.00 |
| center_crop | keep 80% | 0.664 | 0.833 | **+16.9** | 0.506 | 0.827 | **+32.1** | 0.73 | 0.93 |

### Aggregates

| | Baseline acc | Baseline F1 | Baseline AUC | Robust acc | Robust F1 | Robust AUC |
|---|---|---|---|---|---|---|
| All 17 conditions (incl. clean) | 0.764 | 0.690 | 0.844 | 0.836 | 0.825 | 0.908 |
| **Transformed only (16 conditions)** | **0.754** | **0.676** | **0.836** | **0.833** | **0.821** | **0.905** |

Derived over the 16 transformed conditions:

- Mean accuracy improvement (Robust - Baseline): **+7.9 pp**
- Mean F1 improvement: **+14.5 pp**
- Mean ROC-AUC improvement: **+6.9 pp**
- Mean accuracy retention (transformed acc / that model's clean acc):
  Baseline **0.82**, Robust **0.93**
- Mean F1 retention: Baseline **0.74**, Robust **0.91**

## 6. Strongest gains (where transformation-aware training helps most)

Ranked by accuracy gap (see `reports/figures/accuracy_gap.png`):

1. **blur sigma 1.0**: 0.569 -> 0.819 (**+25.0 pp**); F1 0.259 -> 0.815 (+55.6 pp)
2. **resize 0.5x**: 0.554 -> 0.803 (**+24.9 pp**); F1 0.236 -> 0.794 (+55.8 pp)
3. **blur sigma 2.0**: 0.531 -> 0.728 (**+19.7 pp**)
4. **resize 0.25x**: 0.524 -> 0.721 (**+19.7 pp**)
5. **centre crop 80%**: 0.664 -> 0.833 (**+16.9 pp**)
6. **noise sigma 0.10**: 0.533 -> 0.660 (**+12.7 pp**)

Pattern: the baseline **collapses** on geometric / low-pass degradations (blur,
resize, crop) - its F1 falls to 0.24-0.36, i.e. it stops detecting AI images and
defaults to "real". The robust model keeps F1 in the 0.68-0.83 range on the same
inputs.

## 7. Weakest / no-gain / negative transformations (honest)

The robust model is **worse** on 9 of 17 conditions, all mild:

- **clean**: -1.8 pp accuracy
- **all four JPEG levels** (q90 -1.6, q70 -1.3, q50 -0.2, q30 -0.5 pp)
- **light noise** (sigma 0.02): -1.2 pp
- **all three colour-jitter conditions** (brightness -2.0, saturation -1.9,
  contrast -1.5 pp)

These are exactly the transforms that change the image *least* - JPEG at readable
quality, +/-20% colour, faint noise. The baseline already handles them, and the
robust model pays its clean-data tax here for no benefit. The losses are small
(<= ~2 pp) and never catastrophic.

## 8. Trade-off analysis

| | Clean | Mean transformed |
|---|---|---|
| Baseline accuracy | 0.915 | 0.754 |
| Robust accuracy | 0.897 | 0.833 |
| **Robust - Baseline** | **-1.8 pp** | **+7.9 pp** |

The trade is asymmetric and favourable: give up **~1.8 pp** of clean accuracy to
gain **~7.9 pp** of mean transformed accuracy (and **+14.5 pp** mean F1, driven by
the blur/resize collapse the baseline suffers). Accuracy retention under
corruption rises from ~0.82 to ~0.93. See
`reports/figures/aggregate_clean_vs_transformed.png` and
`reports/figures/retention_heatmap.png`.

### Error counts

**Clean single 10,000-image evaluation** (directly measured):

| | FP (real->AI) | FN (AI->real) |
|---|---|---|
| Baseline v1 | 440 | 408 |
| Robust v1 | 621 | 413 |

**Summed across the 16 transformed evaluations** - these are **sums over 16
separate 10,000-image tests (160,000 image-evaluations), NOT one test set**, and
are **reconstructed** from the reported accuracy + F1 on the fixed 5,000/5,000
class balance (3-decimal inputs give a few-image slack per condition):

| | Sum FP | Sum FN |
|---|---|---|
| Baseline v1 | ~9,500 | **~29,800** |
| Robust v1 | ~10,700 | **~16,100** |

Direction is unambiguous: under transformation the **baseline's false negatives
roughly double the robust model's** - it misses AI images because blur/resize
push its predictions toward "real". False positives are similar between the two
(robust slightly higher, consistent with its clean-data behaviour).

## 9. Limitations

- **CIFAKE is 32x32**. Images are upscaled to 64 px for the model, so it never
  sees native high-frequency detail. Absolute numbers would differ on a
  native-resolution dataset.
- **Synthetic fakes only**: Stable Diffusion v1.4. Nothing here shows transfer to
  other generators (Midjourney, SDXL, FLUX, GANs, ...).
- **Simulated corruptions**: the six transform families are a useful proxy for
  "screenshot / re-upload / compression" pipelines, not the full space of
  real-world distribution shift.
- **Small v1 budget**: 3 epochs, 8,000 training images. Both models were still
  improving (validation loss decreasing at epoch 3). Longer training would move
  both curves and could change the size of the trade-off.
- **Clean-vs-robust trade-off is real**: this is not a free lunch. Robust v1 is
  measurably worse on clean and lightly-processed images.
- **Single seed, single run** per model. No confidence intervals.
- Per-condition FP/FN are reconstructed, not logged (see section 8).

## 10. Final conclusion

The experiment **supports the hypothesis, with a stated cost.**
Transformation-aware training makes the detector substantially more robust to
realistic degradation - mean transformed accuracy +7.9 pp, mean F1 +14.5 pp,
accuracy retention 0.82 -> 0.93 - with the largest gains (up to +25 pp accuracy,
+56 pp F1) exactly where the baseline collapses: blur, downscaling, and cropping.
The price is a ~1.8 pp drop in clean accuracy and small (<= 2 pp) regressions on
mild JPEG and colour changes. The robust model is **not universally better**; it
is better *under distribution shift*, which is the property the competition asks
for.

## 11. Reproducibility - exact commands

```bash
# 0. environment + data (once)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/cifake
curl -sL -o data/train.zip https://huggingface.co/datasets/yanbax/CIFAKE_autotrain_compatible/resolve/main/train.zip
unzip -q data/train.zip -d data/cifake

# 1. frozen split (committed; verify only)
python -m scripts.verify_splits

# 2. Baseline v1
python train.py --output_dir outputs/baseline_v1 \
    --epochs 3 --batch_size 64 --max_train_samples 8000 --max_val_samples 2000 \
    --image_size 64 --seed 42 --device cuda
python evaluate.py --test_csv data/splits/test.csv \
    --checkpoint outputs/baseline_v1/best_model.pt \
    --image_size 64 --batch_size 64 --device cuda

# 3. Robust v1  (only change: --augment ...)
python train.py --output_dir outputs/robust_v1 \
    --epochs 3 --batch_size 64 --max_train_samples 8000 --max_val_samples 2000 \
    --image_size 64 --seed 42 --device cuda \
    --augment --aug_prob 0.7 --aug_num 1
python evaluate.py --test_csv data/splits/test.csv \
    --checkpoint outputs/robust_v1/best_model.pt \
    --image_size 64 --batch_size 64 --device cuda

# 4. Robustness benchmark (17 conditions, both models, same test set)
python scripts/evaluate_robustness.py \
    --baseline_checkpoint outputs/baseline_v1/best_model.pt \
    --robust_checkpoint outputs/robust_v1/best_model.pt \
    --test_csv data/splits/test.csv --output_dir outputs/robustness_benchmark \
    --image_size 64 --batch_size 64 --device cuda --num_workers 2 --seed 42

# 5. Figures for this report
python reports/make_figures.py
```
