import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import os

# Model 15_3: same architecture/loss as 15_2 (which mirrors model 17's
# no-self-attention AttentionUNet, whole-signal normalization), but with a
# reworked `curvature` feature.
#
# curvature_demo.py showed that the pointwise second derivative of the
# smoothed QRS-peak envelope (linear_env) is mostly zero/noise except near
# array boundaries (convolution edge artifacts), while a *windowed average*
# of that second derivative cleanly separates a straight (linear) envelope
# (~0) from a bending (spline/curved) envelope (consistently nonzero,
# same sign across the bend). So here `curvature` is computed once per
# training window as the mean of the (edge-clamped) pointwise second
# derivative over that window, and broadcast as a constant across the
# window's curvature channel.

NUM_CLASSES  = 4
NUM_FEATURES = 4   # residual, local_rms, qrs_rate, curvature
SMOOTH_WINDOW = 50


# --- 1. WHOLE-SIGNAL FEATURE EXTRACTION (4 features) ---
def extract_features_signal(sig, qrs_locs):
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    n = len(sig_norm)
    x = np.arange(n)

    if len(qrs_locs) < 2:
        return np.stack([
            sig_norm,
            np.sqrt(np.convolve(sig_norm ** 2, np.ones(50) / 50, mode='same')),
            np.zeros(n),
            np.zeros(n)
        ], axis=0)

    linear_env   = np.interp(x, qrs_locs, sig_norm[qrs_locs])
    residual     = sig_norm - linear_env

    local_rms    = np.sqrt(np.convolve(sig_norm ** 2, np.ones(50) / 50, mode='same'))

    rr_intervals = np.diff(qrs_locs).astype(float)
    beat_rates   = 500.0 / rr_intervals
    rate_at_qrs  = np.concatenate([[beat_rates[0]], beat_rates])
    qrs_rate     = np.interp(x, qrs_locs, rate_at_qrs)
    qrs_rate     = (qrs_rate - np.mean(qrs_rate)) / (np.std(qrs_rate) + 1e-8)

    # Pointwise second derivative of the smoothed QRS-peak envelope. Edge
    # samples (within SMOOTH_WINDOW of either end) are clamped to the
    # nearest valid value to avoid convolution boundary artifacts. Per-window
    # averaging happens in the data loader.
    smooth_env = np.convolve(linear_env, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode='same')
    curvature_point = np.gradient(np.gradient(smooth_env))
    curvature_point[:SMOOTH_WINDOW]  = curvature_point[SMOOTH_WINDOW]
    curvature_point[-SMOOTH_WINDOW:] = curvature_point[-SMOOTH_WINDOW - 1]
    curvature_point = (curvature_point - np.mean(curvature_point)) / (np.std(curvature_point) + 1e-8)

    return np.stack([residual, local_rms, qrs_rate, curvature_point], axis=0)


# --- 2. STEP FUNCTION TARGET (multiclass) ---
def dominant_class(segment):
    if len(segment) == 0:
        return 0
    counts = np.bincount(segment.astype(int), minlength=NUM_CLASSES)
    non_zero = counts[1:]
    if non_zero.sum() > 0:
        return int(np.argmax(non_zero)) + 1
    return 0


def build_step_target(mask_window, qrs_indices):
    n = len(mask_window)
    target = np.zeros(n, dtype=np.int64)

    if len(qrs_indices) < 2:
        return mask_window.astype(np.int64)

    if qrs_indices[0] > 0:
        target[:qrs_indices[0]] = dominant_class(mask_window[:qrs_indices[0]])

    for k in range(len(qrs_indices) - 1):
        s, e = qrs_indices[k], qrs_indices[k + 1]
        target[s:e] = dominant_class(mask_window[s:e])

    target[qrs_indices[-1]:] = dominant_class(mask_window[qrs_indices[-1]:])

    return target


# --- 3. COMPONENTS ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=15, padding=7),
            nn.GroupNorm(4, out_c), nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=15, padding=7),
            nn.GroupNorm(4, out_c)
        )
        self.shortcut = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class AttentionGate(nn.Module):
    def __init__(self, f_g, f_x, f_int):
        super().__init__()
        self.W_g = nn.Conv1d(f_g, f_int, kernel_size=1)
        self.W_x = nn.Conv1d(f_x, f_int, kernel_size=1)
        self.psi = nn.Sequential(nn.Conv1d(f_int, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(F.relu(self.W_g(g) + self.W_x(x)))


# --- 4. ARCHITECTURE ---
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=NUM_FEATURES):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, 32)
        self.enc2 = ResidualBlock(32, 64)
        self.enc3 = ResidualBlock(64, 128)
        self.enc4 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = ResidualBlock(256, 256)
        self.att4 = AttentionGate(256, 256, 128)
        self.att3 = AttentionGate(128, 128, 64)
        self.att2 = AttentionGate(64,  64,  32)
        self.att1 = AttentionGate(32,  32,  16)
        self.up4  = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec4 = ResidualBlock(512, 128)
        self.up3  = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec3 = ResidualBlock(256, 64)
        self.up2  = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec2 = ResidualBlock(128, 32)
        self.up1  = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec1 = ResidualBlock(64, 32)
        self.final = nn.Conv1d(32, NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x);  p1 = self.pool(s1)
        s2 = self.enc2(p1); p2 = self.pool(s2)
        s3 = self.enc3(p2); p3 = self.pool(s3)
        s4 = self.enc4(p3); p4 = self.pool(s4)
        b  = self.bottleneck(p4)
        g4 = self.up4(b);  d4 = self.dec4(torch.cat([g4, self.att4(g4, s4)], dim=1))
        g3 = self.up3(d4); d3 = self.dec3(torch.cat([g3, self.att3(g3, s3)], dim=1))
        g2 = self.up2(d3); d2 = self.dec2(torch.cat([g2, self.att2(g2, s2)], dim=1))
        g1 = self.up1(d2); d1 = self.dec1(torch.cat([g1, self.att1(g1, s1)], dim=1))
        return self.final(d1)  # (batch, NUM_CLASSES, seq_len)


# --- 5. DATA LOADER ---
CURVATURE_CHANNEL = 3


class SignalLevelLoader:
    def __init__(self, sig_paths, mask_paths, qrs_paths):
        self.sig_paths  = sig_paths
        self.mask_paths = mask_paths
        self.qrs_paths  = qrs_paths

    def get_batches(self, window_size, stride, batch_size):
        indices = np.arange(len(self.sig_paths))
        np.random.shuffle(indices)
        x_batch, y_batch = [], []

        for idx in indices:
            sig      = np.load(self.sig_paths[idx]).flatten().astype(np.float64)
            mask     = np.load(self.mask_paths[idx]).flatten()
            qrs_locs = np.load(self.qrs_paths[idx]).flatten().astype(int)

            feats = extract_features_signal(sig, qrs_locs)   # (4, N), whole-signal normalization

            for start in range(0, sig.shape[0] - window_size, stride):
                end = start + window_size
                v_qrs = qrs_locs[(qrs_locs >= start) & (qrs_locs < end)] - start

                window_feats = feats[:, start:end].copy()
                # curvature: replace pointwise values with their per-window
                # average, broadcast as a constant across the window.
                window_feats[CURVATURE_CHANNEL, :] = window_feats[CURVATURE_CHANNEL, :].mean()

                x_batch.append(window_feats)
                y_batch.append(build_step_target(mask[start:end], v_qrs))

                if len(x_batch) == batch_size:
                    yield (torch.from_numpy(np.array(x_batch)).float(),
                           torch.from_numpy(np.array(y_batch)).long())
                    x_batch, y_batch = [], []


# --- 6. EVALUATION ---
def evaluate(model, val_loader, criterion, device, window_size, stride, batch_size):
    model.eval()
    v_loss, v_steps = 0, 0
    correct = total = 0
    per_class_tp = np.zeros(NUM_CLASSES)
    per_class_fp = np.zeros(NUM_CLASSES)
    per_class_fn = np.zeros(NUM_CLASSES)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)  # rows=true, cols=pred

    with torch.no_grad():
        for x, y in val_loader.get_batches(window_size, stride, batch_size):
            x, y = x.to(device), y.to(device)
            pred  = model(x)
            v_loss += criterion(pred, y).item()
            v_steps += 1
            preds = pred.argmax(dim=1)  # (batch, seq_len)
            correct += (preds == y).sum().item()
            total   += y.numel()
            for c in range(NUM_CLASSES):
                per_class_tp[c] += ((preds == c) & (y == c)).sum().item()
                per_class_fp[c] += ((preds == c) & (y != c)).sum().item()
                per_class_fn[c] += ((preds != c) & (y == c)).sum().item()

            y_flat = y.flatten()
            p_flat = preds.flatten()
            for t in range(NUM_CLASSES):
                t_mask = (y_flat == t)
                if t_mask.any():
                    for p in range(NUM_CLASSES):
                        confusion[t, p] += ((p_flat == p) & t_mask).sum().item()

    accuracy  = correct / (total + 1e-8)
    precision = per_class_tp / (per_class_tp + per_class_fp + 1e-8)
    recall    = per_class_tp / (per_class_tp + per_class_fn + 1e-8)
    f1_per    = 2 * precision * recall / (precision + recall + 1e-8)
    macro_f1  = f1_per.mean()

    return v_loss / v_steps, accuracy, f1_per, macro_f1, confusion


def print_confusion(confusion):
    print("    Confusion matrix (rows=true, cols=pred):")
    header = "        " + "".join(f"C{p:<9d}" for p in range(NUM_CLASSES))
    print(header)
    for t in range(NUM_CLASSES):
        row = "".join(f"{confusion[t, p]:<10d}" for p in range(NUM_CLASSES))
        print(f"    C{t}  {row}")


# --- 7. MAIN ---
def main():
    WINDOW_SIZE   = 3840
    STRIDE        = 250
    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-5
    EPOCHS        = 100

    DATA_ROOT = "/home/20251020/ECG/Npy_DB2"
    sig_p  = sorted(glob.glob(os.path.join(DATA_ROOT, "signals",  "*.npy")))
    mask_p = sorted(glob.glob(os.path.join(DATA_ROOT, "mc_masks", "*.npy")))
    qrs_p  = sorted(glob.glob(os.path.join(DATA_ROOT, "qrs_locs", "*.npy")))

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model       = AttentionUNet().to(device)
    optimizer   = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion   = nn.CrossEntropyLoss()
    CHECKPOINT  = "15_3_AttUNet_multiclass_best.pth"

    split        = int(0.8 * len(sig_p))
    train_loader = SignalLevelLoader(sig_p[:split], mask_p[:split], qrs_p[:split])
    val_loader   = SignalLevelLoader(sig_p[split:], mask_p[split:], qrs_p[split:])

    print(f"Model 15_3 | Multiclass (4) | residual+rms+qrs_rate+windowed_curvature | signal-level norm | LR: {LEARNING_RATE}")
    print(f"Device: {device}  |  Train signals: {split}  |  Val signals: {len(sig_p)-split}\n")

    best_f1 = 0.0

    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
        _, _, _, best_f1, _ = evaluate(model, val_loader, criterion, device, WINDOW_SIZE, STRIDE, BATCH_SIZE)
        print(f"Resumed weights from {CHECKPOINT}  (MacroF1: {best_f1:.3f})")

    for epoch in range(EPOCHS):
        model.train()
        t_loss, t_steps = 0, 0
        for x, y in train_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item(); t_steps += 1

        v_loss, accuracy, f1_per, macro_f1, confusion = evaluate(
            model, val_loader, criterion, device, WINDOW_SIZE, STRIDE, BATCH_SIZE)

        cls_str = " | ".join(f"C{c} F1:{f1_per[c]:.3f}" for c in range(NUM_CLASSES))
        print(f"Epoch {epoch + 1:03d} | Train: {t_loss / t_steps:.4f} | Val: {v_loss:.4f} | "
              f"Acc: {accuracy:.3f} | MacroF1: {macro_f1:.3f} | [{cls_str}]")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"--> Saved New Best Model (MacroF1: {macro_f1:.3f})")
            print_confusion(confusion)


if __name__ == "__main__":
    main()
