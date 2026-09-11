# Fetal ECG Movement Segmentation with Attention Res-UNet

A PyTorch-based deep learning pipeline designed to detect and segment fetal movement events from raw ECG signals using advanced physiological feature engineering, sequential stride inference, and hybrid temporal-spatial architectures.

---

## Overview

Fetal movement detection in ECG signals suffers from non-stationary noise, class imbalance, and baseline drift. This project implements a **1D Residual U-Net (Res-UNet)** augmented with **Atrous Spatial Pyramid Pooling (ASPP)**, **Transformer Self-Attention Bottlenecks**, and **Attention Skip Gates** to accurately locate movement masks in continuous 1D physiological time-series.

### Key Features
* **Engineered Multi-Channel Inputs**:
  * Raw normalized ECG signal ($z$-score standardization).
  * QRS outline envelope tracking extracted via linear interpolation.
  * Low-frequency baseline wander extraction using moving-average convolutions.
* **Architecture**:
  * **Residual Blocks**: 1D convolutions with Group Normalization (`GroupNorm(4, C)`) for stable training across varying batch sizes.
  * **Multi-Scale Context**: ASPP module using parallel dilated convolutions ($d \in [1, 2, 4, 8]$) to capture localized heartbeats alongside wide-window arrhythmia patterns.
  * **Global Attention**: Transformer encoder bottleneck modeling long-range temporal dependencies across 7.68-second windows.
  * **Attention Gates**: Skip-connection gating to filter noise passed from encoder to decoder.
* **Clinical Sequential Data Loading**:
  * Strict patient-by-patient, sequential sliding-window extraction (Stride: 250 samples / ~0.5s) to preserve physiological temporal continuity and prevent data leakage.
* **Class-Balanced Loss**:
  * Weighted Binary Cross-Entropy (`pos_weight=5.0`) combined with continuous soft-Dice loss for continuous boundary segmentation.

---

## Model Architecture

```text
Input Signal [B, 3, 3840] (7.68s @ 500Hz)
       │
       ▼
 [ResBlock 1] ──── (Attention Gate) ────┐
       │ MaxPool1d(2)                   │
       ▼                                │
 [ResBlock 2] ──── (Attention Gate) ──┐ │
       │ MaxPool1d(2)                 │ │
       ▼                              │ │
 [ResBlock 3] ──── (Attention Gate) ┐ │ │
       │ MaxPool1d(2)               │ │ │
       ▼                            │ │ │
 [ResBlock 4] ───┐                  │ │ │
       │ MaxPool1d(2)               │ │ │
       ▼                            │ │ │
  [ASPP Block] (d=1, 2, 4, 8)       │ │ │
       │                            │ │ │
       ▼                            │ │ │
 [Transformer Bottleneck]           │ │ │
       │                            │ │ │
       ▼                            │ │ │
  [Decoder 4] ◄── Concatenate ◄─────┘ │ │
       │ Upsample1d(2)                │ │
       ▼                              │ │
  [Decoder 3] ◄── Concatenate ◄───────┘ │
       │ Upsample1d(2)                  │
       ▼                                │
  [Decoder 2] ◄── Concatenate ◄─────────┘
       │ Upsample1d(2)
       ▼
  [Decoder 1]
       │ Conv1d(1x1)
       ▼
 Output Mask Logits [B, 1, 3840]


