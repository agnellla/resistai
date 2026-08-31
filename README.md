# ResistAI

Robust detection of AI-generated images under real-world transformations.

Image-level AIGC detection: given an image, predict **real (0)** or **AI-generated (1)**,
and stay accurate after JPEG compression, Gaussian blur, resize down-then-up,
Gaussian noise, colour jitter, and centre crop.

Hackathon prototype. Backbone: pretrained **EfficientNet-B0** (~5.3M params, well
under the 2B limit), fine-tuned for binary classification.

## Approach

1. Baseline model: fine-tune EfficientNet-B0 on real vs AI, clean images only.
2. Transformation-aware model: same training, but real-world corruptions are
   randomly applied to training images.
3. Evaluate both under the six required corruptions and compare.
4. Extra robustness tricks only if time permits. Do not over-engineer.

## Repository layout

```
resistai/
├── train.py               # BASELINE training pipeline (CLI-driven)
├── evaluate.py             # score a checkpoint: acc / precision / recall / F1 / ROC-AUC / confusion matrix
├── inference.py            # deliverable: image dir -> predictions.json {image_path, pred}
├── requirements.txt
├── configs/
│   └── default.yaml        # model settings for inference.py only (train/evaluate use CLI flags)
├── src/
│   ├── transforms.py       # model input pipeline (resize -> tensor -> normalise)
│   ├── augmentations.py    # on-the-fly real-world transforms for training (Robust v1)
│   ├── datasets.py         # dataset class + loaders from the split manifests
│   ├── splits.py           # build / read / verify the frozen train/val/test split
│   ├── model.py            # EfficientNet-B0 via timm, 2-class head
│   └── utils.py            # seed, device select (CUDA/MPS/CPU), checkpoint save/load
├── scripts/
│   ├── make_splits.py      # ONE-OFF: create data/splits/{train,val,test}.csv
│   ├── verify_splits.py    # check the splits are disjoint / balanced / correctly sized
│   └── prepare_data.py     # PLACEHOLDER - dataset download, not implemented yet
├── dashboard/
│   └── app.py              # PLACEHOLDER - Streamlit demo, not built yet
├── data/
│   ├── cifake/             # the images (gitignored)
│   └── splits/             # train.csv / val.csv / test.csv - COMMITTED, frozen
├── outputs/                # train.py / evaluate.py write results here (gitignored)
├── checkpoints/            # saved model weights - gitignored
└── reports/                # generated CSVs and plots - gitignored
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # on Colab: just `pip install -r requirements.txt`
```

Runs on **CUDA (Colab Tesla T4)**, **Apple MPS**, or **CPU** with no code change.
Device is auto-detected (CUDA > MPS > CPU); override with `--device cuda|mps|cpu`.

## Dataset

**CIFAKE** - real CIFAR-10 photos + Stable-Diffusion-v1.4 fakes.
This repo pulls the HuggingFace mirror `yanbax/CIFAKE_autotrain_compatible`
(MIT license, single `train.zip`, no login needed), extracted to:

```
data/cifake/
├── real/   *.jpg   (50000)  -> label 0
└── fake/   *.jpg   (50000)  -> label 1
```

Images are **32x32**. The Kaggle version (`birdy654/cifake-...`, has `train/` +
`test/`) also works - point `make_splits.py --data_dir` at the folder that
contains the class sub-folders.

## Frozen train / val / test split

The HF mirror has no train/test split, so we define one **once** and commit it:

```bash
python -m scripts.make_splits --data_dir data/cifake     # writes data/splits/*.csv
python -m scripts.verify_splits                          # disjoint? balanced? sized right?
```

- stratified per class -> real/fake balanced in every split
- 80% train / 10% val / 10% test, seeded (`--seed 42`), no image in two splits
- `make_splits.py` refuses to overwrite existing manifests (use `--force` only to
  deliberately start a new split - it invalidates past results)
- images on disk are never copied or modified

`train.py` trains on `train.csv` + validates on `val.csv`; `evaluate.py` scores
**only** `test.csv`. Neither script ever re-splits.

## Usage

```bash
# 0. one-off: create + verify the split
python -m scripts.make_splits --data_dir data/cifake
python -m scripts.verify_splits

# 1. Baseline training - small subset first, to check everything runs
python train.py --output_dir outputs/baseline \
    --epochs 3 --max_train_samples 8000 --max_val_samples 2000

# 2. Full baseline run (drop the caps, more epochs) - do this on Colab GPU
python train.py --output_dir outputs/baseline_full \
    --epochs 5 --batch_size 128 --device cuda

# 3. Score the checkpoint on the held-out test split
python evaluate.py --checkpoint outputs/baseline/best_model.pt

# 4. Produce the competition JSON
python inference.py --images path/to/test_dir \
    --checkpoint outputs/baseline/best_model.pt --out predictions.json
```

`python train.py --help` / `python evaluate.py --help` list every flag.
`train.py` defaults to `data/splits/train.csv` + `val.csv`; `evaluate.py` to
`data/splits/test.csv`.

## Transformation-aware training (Robust v1)

Same model, optimiser, and frozen split as the baseline. The difference: during
training only, each image has probability `--aug_prob` of getting ONE randomly
chosen real-world transform applied on the fly (`src/augmentations.py`):

| transform | parameters |
|---|---|
| `jpeg` | quality 90 / 70 / 50 / 30 |
| `blur` | Gaussian sigma 0.5 / 1.0 / 2.0 |
| `resize` | downscale 0.5x / 0.25x then upscale back |
| `noise` | Gaussian sigma 0.02 / 0.05 / 0.10 (fraction of 255) |
| `color_jitter` | brightness / contrast / saturation +/-20% |
| `center_crop` | keep 80%, resize back to model input |

Validation and test stay **clean**. The label never changes. Without `--augment`
`train.py` is byte-identical to Baseline v1.

```bash
# Robust v1 - first experiment (mirror the Baseline v1 settings + --augment)
python train.py --output_dir outputs/robust_v1 \
    --epochs 3 --batch_size 64 \
    --max_train_samples 8000 --max_val_samples 2000 \
    --image_size 64 --seed 42 --device mps \
    --augment --aug_prob 0.7 --aug_num 1
```

`--aug_transforms` (comma list) restricts which transforms are drawn from;
`--aug_num` chains more than one per transformed image (spec: 1).
`outputs/robust_v1/run_config.json` records the full augmentation config.

Trained model -> `<output_dir>/best_model.pt`
Per-epoch metrics -> `<output_dir>/metrics.csv`
Eval metrics -> `<checkpoint_dir>/metrics.json` + `confusion_matrix.csv`
Exact settings used -> `<output_dir>/run_config.json`

## Experiment results: Baseline v1 vs Robust v1

Full analysis in [`reports/robustness_analysis.md`](reports/robustness_analysis.md);
figures in `reports/figures/`; per-condition numbers in
`reports/benchmark_results.csv`. All figures come from the **same 10,000-image
held-out test set** (`data/splits/test.csv`), scored by
`scripts/evaluate_robustness.py` under 17 conditions (clean + 16 transformed, every
image transformed at a fixed severity).

**The honest result: Robust v1 is NOT universally more accurate.** It trades a
small amount of clean-image performance for a large improvement under realistic
degradation.

| | Baseline v1 | Robust v1 | Δ |
|---|---|---|---|
| **Clean** accuracy / F1 / ROC-AUC | 0.915 / 0.915 / 0.969 | 0.897 / 0.899 / 0.957 | **-1.8 / -1.6 / -1.2 pp** |
| **Transformed** (mean of 16) accuracy | 0.754 | **0.833** | **+7.9 pp** |
| **Transformed** (mean of 16) F1 | 0.676 | **0.821** | **+14.5 pp** |
| **Transformed** (mean of 16) ROC-AUC | 0.836 | **0.905** | **+6.9 pp** |
| Accuracy retention under corruption | ~0.82 | **~0.93** | |

Largest gains (where the baseline collapses to predicting "real"):

| Condition | Baseline acc | Robust acc | Gain |
|---|---|---|---|
| blur sigma 1.0 | 0.569 | 0.819 | **+25.0 pp** (F1 +55.6) |
| resize 0.5x | 0.554 | 0.803 | **+24.9 pp** (F1 +55.8) |
| blur sigma 2.0 | 0.531 | 0.728 | +19.7 pp |
| resize 0.25x | 0.524 | 0.721 | +19.7 pp |
| centre crop 80% | 0.664 | 0.833 | +16.9 pp |
| noise sigma 0.10 | 0.533 | 0.660 | +12.7 pp |

Where Robust v1 is **worse** (all mild, <= ~2 pp): clean, all four JPEG levels,
light noise (sigma 0.02), and all three colour-jitter conditions - transforms that
barely change the image and that the baseline already handles.

Rebuild the figures with `python reports/make_figures.py`.

## Google Colab (Tesla T4 GPU)

Runtime -> Change runtime type -> **T4 GPU**, then:

```python
# 1. get the code
!git clone https://github.com/<your-user>/resistai.git
%cd resistai

# 2. deps (Colab already has a CUDA torch; this adds timm etc.)
!pip install -q -r requirements.txt

# 3. get the dataset (same HF mirror, ~86 MB)
!mkdir -p data/cifake
!curl -sL -o data/train.zip https://huggingface.co/datasets/yanbax/CIFAKE_autotrain_compatible/resolve/main/train.zip
!unzip -q data/train.zip -d data/cifake
!ls data/cifake            # -> real  fake

# 4. sanity check the GPU is seen
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))

# 5. use the COMMITTED split if present; otherwise create it once
!test -f data/splits/train.csv || python -m scripts.make_splits --data_dir data/cifake
!python -m scripts.verify_splits

# 6. full baseline training on GPU (reads data/splits/train.csv + val.csv)
!python train.py --output_dir outputs/baseline_full \
    --epochs 5 --batch_size 128 --device cuda

# 7. evaluate on the held-out test split (data/splits/test.csv)
!python evaluate.py --checkpoint outputs/baseline_full/best_model.pt --device cuda

# 8. download the checkpoint so a Mac / CPU can load it (map_location handles the rest)
from google.colab import files; files.download('outputs/baseline_full/best_model.pt')
```

## Limitations

- **CIFAKE is 32x32.** Images are upscaled to 64 px for the model, which never
  sees native high-frequency detail; absolute accuracy would differ on a
  native-resolution dataset. (`--image_size` is configurable; keep train and
  evaluate equal.)
- **Synthetic fakes only** - Stable Diffusion v1.4. These results do **not** prove
  generalisation to other generators (SDXL, Midjourney, FLUX, GANs, ...).
- **Simulated corruptions.** The six transform families approximate
  screenshot / re-upload / compression pipelines; they are not the full space of
  real-world distribution shift.
- **Small v1 budget** - 3 epochs, 8,000 training images. Both models were still
  improving (validation loss decreasing at epoch 3); longer training would shift
  both curves and could change the size of the trade-off.
- **The clean-vs-robust trade-off is real** - Robust v1 is measurably worse on
  clean and lightly-processed images. This is not a free lunch.
- **Single seed, single run** per model; no confidence intervals. Per-condition
  false-positive / false-negative counts in the analysis are reconstructed from
  accuracy + F1 on the fixed 5,000/5,000 test balance, not separately logged.

## Status

- [x] Repo structure + configs
- [x] Frozen 80/10/10 split (`data/splits/*.csv`)
- [x] Baseline v1 trained + evaluated
- [x] Transformation-aware training (`--augment`, `src/augmentations.py`)
- [x] Robust v1 trained + evaluated
- [x] Full robustness benchmark, 17 conditions (`scripts/evaluate_robustness.py`)
- [x] Analysis + figures (`reports/robustness_analysis.md`, `reports/figures/`)
- [x] Streamlit demo (`dashboard/app.py`)
- [ ] Demo video
- [ ] v2 experiments (longer training, larger `--image_size`, more transforms)
