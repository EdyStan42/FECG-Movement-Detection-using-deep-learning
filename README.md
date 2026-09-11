Detecting maternal/fetal movement artefacts in fetal ECG (fECG) signals using 1D Attention U-Net models.

## Problem

Fetal ECG recordings (500 Hz) are corrupted by movement artefacts that distort the signal morphology. This project trains deep learning models to flag, per-sample, which parts of a recording are affected by movement, so that downstream analysis (e.g. fetal heart rate extraction) can discount those segments.

Two task formulations are explored:

- **Binary detection** — movement vs. no movement at every sample.
- **Multiclass detection** — no movement vs. 3 distinct movement categories.

## Approach

1. **QRS peak detection** on the raw signal (used by earlier model variants for peak-relative features).
2. **Feature extraction** — combinations of the normalised raw signal, QRS-relative residuals, rolling RMS energy, instantaneous beat rate, and first-derivative transients. Later models (e.g. 21+) drop the QRS-dependent features so they also work on datasets without peak annotations.
3. **Sliding-window inference** (3840-sample / 7.68 s windows) fed into a 1D Attention U-Net.
4. **Attention U-Net** — a ResNet-style encoder/decoder with attention gates on the skip connections, plus a bottleneck self-attention block for global temporal context.
5. **Overlap-add averaging** at inference time to reconcile predictions across overlapping windows before thresholding.

See [documentation.md](documentation.md) for a detailed write-up of the pipeline and a comparison across model versions.

## Repository layout

Model iterations are numbered scripts (`14.py`, `15.py`, ... `23.py`), each a self-contained experiment: feature extraction, model architecture, data loader, and training loop. Later numbers generally supersede earlier ones — see `documentation.md` for what changed between versions. Supporting scripts:

- `eval*.py`, `infer_*.py`, `predict_script.py` — evaluation and inference on trained checkpoints.
- `plot*.py`, `analyze_classes_*.py` — result visualization and analysis.
- `*_demo.py`, `dataloader*.py`, `data_verify.py`, `npy_maker.py`, `mask_npy.py` — data preparation and sanity-check utilities.
- `documentation.md` / `documentation.tex` — detailed method notes.

Trained checkpoints (`*.pth`) and generated plots (`*.png`) are experiment artifacts, not source — see below.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install torch numpy scipy matplotlib
```

## Usage

Each numbered script is a standalone training run, pointed at a dataset of paired signal/mask `.npy` files:

```bash
python 21.py
```

Update the `DATA_ROOT` path at the top of the script's `main()` to your dataset location before running.
