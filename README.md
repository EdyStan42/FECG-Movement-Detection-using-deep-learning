# Fetal ECG Movement Detection — Method Documentation

## 1. Problem Statement

The goal is to detect maternal and fetal movement artefacts in fetal ECG (fECG) signals recorded at 500 Hz. A movement corrupts the ECG morphology and must be identified so downstream analysis (e.g. heart rate extraction) can discount those segments.

Two formulations are studied:

- **Binary detection** (models 14, 18): predict 0 (no movement) or 1 (movement) at every sample.
- **Multiclass detection** (model 15): predict one of 4 classes at every sample — class 0 (no movement) and classes 1–3 representing distinct movement categories.

- 
## Approach

1. **QRS peak detection** on the raw signal (used by earlier model variants for peak-relative features).
2. **Feature extraction** — combinations of the normalised raw signal, QRS-relative residuals, rolling RMS energy, instantaneous beat rate, and first-derivative transients. Later models (e.g. 21+) drop the QRS-dependent features so they also work on datasets without peak annotations.
3. **Sliding-window inference** (3840-sample / 7.68 s windows) fed into a 1D Attention U-Net.
4. **Attention U-Net** — a ResNet-style encoder/decoder with attention gates on the skip connections, plus a bottleneck self-attention block for global temporal context.
5. **Overlap-add averaging** at inference time to reconcile predictions across overlapping windows before thresholding.
---

## 2. General Pipeline

```
Raw fECG signal (500 Hz)
        │
        ▼
  QRS peak detection
  (scipy find_peaks, min distance 200 ms)
        │
        ▼
  Feature extraction
  (whole signal or per-window depending on model)
        │
        ▼
  Sliding window  ──────────────────────────────────────────────
  WINDOW = 3840 samples (7.68 s)                               │
  STRIDE = 250–500 samples                                     │
        │                                                       │
        ▼                                                  Ground truth
  AttentionUNet                                           movement mask
  (binary or multiclass output)                               (npy)
        │                                                       │
        ▼                                                       │
  Sigmoid / Softmax                                             │
        │                                                       │
        ▼                                                       │
  Threshold at 0.5  ◄────────────────────────────── Compare ───┘
        │
        ▼
  Predicted movement mask
```

During **inference**, overlapping windows are averaged (overlap-add) before thresholding:

```python
pred_prob = pred_sum / pred_count   # average over overlapping windows
pred_binary = (pred_prob > 0.5)
```

---

## 3. Model 14 — Binary Detection, 5 Features, Per-Window Normalisation

### 3.1 Feature Extraction

Features are computed **inside each sliding window** on the 3840-sample slice. Normalisation statistics (mean, std) are local to that window.

| # | Feature | Description |
|---|---------|-------------|
| 0 | `linear_env` | Linear interpolation of the signal amplitude at QRS peak locations — tracks the slow amplitude envelope |
| 1 | `hilbert_env` | Absolute value of the Hilbert transform — instantaneous amplitude, normalised to zero mean / unit variance |
| 2 | `residual` | `sig_norm − linear_env` — deviation of the signal from the QRS amplitude trend |
| 3 | `local_rms` | Rolling RMS energy over a 50-sample (100 ms) window |
| 4 | `qrs_rate` | Instantaneous beat rate in Hz, linearly interpolated between QRS peaks, normalised |

If fewer than 2 QRS peaks are detected in a window, features 0, 1 and 4 are set to zero and the raw normalised signal is used as a fallback.

Input tensor shape per window: **(5, 3840)**

### 3.2 Training Target — Step Function

Rather than using the raw sample-level mask, the target is smoothed into a **step function**: for each inter-QRS interval, the entire interval takes the binary label of the majority sample within it. This makes the target consistent with cardiac cycle boundaries, which are the natural unit of fECG analysis.

### 3.3 Architecture — Attention U-Net

```
Input (5, 3840)
  enc1: ResBlock(5→32)    ──────────────────────── skip s1 (32, 3840)
  pool → (32, 1920)
  enc2: ResBlock(32→64)   ──────────────────────── skip s2 (64, 1920)
  pool → (64, 960)
  enc3: ResBlock(64→128)  ──────────────────────── skip s3 (128, 960)
  pool → (128, 480)
  enc4: ResBlock(128→256) ──────────────────────── skip s4 (256, 480)
  pool → (256, 240)
  bottleneck: ResBlock(256→256)
  up4 → (256, 480)
  AttGate(g=up4, x=s4) → dec4: ResBlock(512→128)
  up3 → (128, 960)
  AttGate(g=up3, x=s3) → dec3: ResBlock(256→64)
  up2 → (64, 1920)
  AttGate(g=up2, x=s2) → dec2: ResBlock(128→32)
  up1 → (32, 3840)
  AttGate(g=up1, x=s1) → dec1: ResBlock(64→32)
  final: Conv1d(32→1, kernel=1)
Output (1, 3840) — raw logits → sigmoid → movement probability
```

**ResidualBlock**: two Conv1d(kernel=15, padding=7) + GroupNorm(4) + ReLU layers with a 1×1 shortcut.

**AttentionGate**: cross-attention gate. The decoder signal `g` acts as a query that gates the encoder skip connection `x`:
```
att = sigmoid( ReLU( W_g·g + W_x·x ) )
output = x * att
```
This suppresses encoder features that are inconsistent with the global movement context already built by the decoder.

### 3.4 Loss

Binary cross-entropy with positive class weight = 3.0 to compensate for class imbalance (movement epochs are less frequent than non-movement).

```python
F.binary_cross_entropy_with_logits(pred, target, pos_weight=tensor([3.0]))
```

### 3.5 Training

- Optimiser: Adam, lr = 1e-5
- Batch size: 32
- Epochs: 100, best checkpoint saved by validation F1

---

## 4. Model 15 — Multiclass Detection, 4 Classes, Per-Window Normalisation

### 4.1 Differences from Model 14

Model 15 is architecturally identical to model 14 except:

1. **Output**: `final` layer outputs 4 channels instead of 1 → `(4, 3840)` logits. Predicted class per sample = `argmax(softmax(logits), dim=1)`.
2. **Loss**: `nn.CrossEntropyLoss` (no pos_weight — class imbalance handled implicitly).
3. **Target**: integer class labels 0–3 per sample. The step function target uses `dominant_class()` — for each inter-QRS interval, the most frequent non-zero class wins; if all samples are class 0, the interval is labelled 0.
4. **Feature extraction**: identical 5-feature per-window extraction as model 14.

### 4.2 Class Definitions

| Class | Meaning |
|-------|---------|
| 0 | No movement |
| 1 | Movement type 1 |
| 2 | Movement type 2 |
| 3 | Movement type 3 |

### 4.3 Evaluation

Per-class precision, recall and F1 computed at each epoch. Macro-F1 (mean over 4 classes) used as the best-model criterion.

---

## 5. Model 18 — Binary Detection, 3 Features, Signal-Level Normalisation, Bottleneck Self-Attention

Model 18 incorporates two improvements motivated by analysis of models 14–17:

### 5.1 Signal-Level Normalisation

In models 14 and 15, each window is normalised independently. In model 18, normalisation is applied **once to the full recording** before windowing:

```python
sig_norm = (sig - sig.mean()) / (sig.std() + 1e-8)   # full signal
feats = extract_features_signal(sig, qrs_locs)        # (3, N)
# training then slices: feats[:, start:end]
```

This means the mean and standard deviation reflect the entire recording, preserving slow amplitude drift information that per-window normalisation would remove.

### 5.2 Feature Selection

An ablation study (model 16) trained six variants — all 5 features and five leave-one-out variants. Results after 30 epochs:

| Removed feature | Best F1 | Change vs baseline |
|---|---|---|
| none (all 5) | 0.926 | — |
| `linear_env` | 0.933 | **+0.007** |
| `hilbert_env` | 0.937 | **+0.011** |
| `residual` | 0.923 | −0.003 |
| `local_rms` | 0.918 | −0.008 |
| `qrs_rate` | 0.918 | −0.008 |

`linear_env` and `hilbert_env` were **redundant** under signal-level normalisation — removing them improved performance. `local_rms` and `qrs_rate` caused the largest drops and are the most informative features. Model 18 uses only the three features that contributed positively:

| # | Feature | Why kept |
|---|---------|---------|
| 0 | `residual` | Captures signal deviation from QRS amplitude trend |
| 1 | `local_rms` | Largest F1 drop when removed; captures local energy changes |
| 2 | `qrs_rate` | Second largest drop; beat rate variation is a direct movement indicator |

Input tensor shape per window: **(3, 3840)**

### 5.3 Bottleneck Self-Attention

A multi-head self-attention block is inserted **after** the bottleneck ResidualBlock. By this point the sequence has been compressed to `3840 / 2⁴ = 240` timesteps, making full self-attention computationally feasible (240² ≈ 57K vs 3840² ≈ 15M at input level).

```python
class BottleneckSelfAttention(nn.Module):
    def __init__(self, channels=256, heads=4):
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):          # (B, 256, 240)
        x_t = x.permute(0, 2, 1)  # (B, 240, 256)
        out, _ = self.attn(x_t, x_t, x_t)
        return self.norm(x_t + out).permute(0, 2, 1)
```

Each of the 240 positions attends to all other positions using learned Q, K, V projections. The residual connection and LayerNorm ensure the block cannot hurt performance relative to skipping it. This provides **global temporal context** — the model can learn that a sustained amplitude drop over 5 seconds is a movement, not a transient artefact, by comparing distant positions within the same window.

**Relationship to the existing cross-attention gates**: the two mechanisms are complementary. Self-attention enriches the bottleneck representation with global context. The cross-attention gates then use that enriched representation to more accurately filter the encoder skip connections during decoding.

### 5.4 Architecture Summary

```
Input (3, 3840)
  Encoder (same as model 14, enc1 starts with 3 channels)
  Bottleneck: ResBlock(256→256) → BottleneckSelfAttention(256, heads=4)
  Decoder + 4 AttentionGates (same as model 14)
Output (1, 3840) → sigmoid → movement probability
```

### 5.5 Training

Identical to model 14: Adam lr=1e-5, batch=32, BCE with pos_weight=3.0, 100 epochs, best by F1.

---

## 6. Comparison Summary

| | Model 14 | Model 15 | Model 18 |
|---|---|---|---|
| Task | Binary | 4-class | Binary |
| Features | 5 | 5 | 3 |
| Normalisation | Per-window | Per-window | Whole signal |
| Bottleneck | ResBlock | ResBlock | ResBlock + Self-Attention |
| Output channels | 1 | 4 | 1 |
| Loss | BCE (w=3) | CrossEntropy | BCE (w=3) |
| Target | Step (binary) | Step (dominant class) | Step (binary) |
