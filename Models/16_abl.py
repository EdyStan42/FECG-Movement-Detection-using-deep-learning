import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import os
from scipy.signal import hilbert

# ── Feature names ─────────────────────────────────────────────────────────────
FEATURE_NAMES = ['linear_env', 'hilbert_env', 'residual', 'local_rms', 'qrs_rate']


# --- 1. WHOLE-SIGNAL FEATURE EXTRACTION ---
# Normalization and all feature statistics use the full recording.
# The data loader then slices the precomputed (5, N) array into windows.
def extract_features_signal(sig, qrs_locs):
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    n = len(sig_norm)
    x = np.arange(n)

    if len(qrs_locs) < 2:
        return np.stack([
            sig_norm,
            np.zeros(n),
            sig_norm,
            np.sqrt(np.convolve(sig_norm ** 2, np.ones(50) / 50, mode='same')),
            np.zeros(n)
        ], axis=0)

    linear_env   = np.interp(x, qrs_locs, sig_norm[qrs_locs])

    hilbert_env  = np.abs(hilbert(sig_norm))
    hilbert_env  = (hilbert_env - np.mean(hilbert_env)) / (np.std(hilbert_env) + 1e-8)

    residual     = sig_norm - linear_env

    local_rms    = np.sqrt(np.convolve(sig_norm ** 2, np.ones(50) / 50, mode='same'))

    rr_intervals = np.diff(qrs_locs).astype(float)
    beat_rates   = 500.0 / rr_intervals
    rate_at_qrs  = np.concatenate([[beat_rates[0]], beat_rates])
    qrs_rate     = np.interp(x, qrs_locs, rate_at_qrs)
    qrs_rate     = (qrs_rate - np.mean(qrs_rate)) / (np.std(qrs_rate) + 1e-8)

    return np.stack([linear_env, hilbert_env, residual, local_rms, qrs_rate], axis=0)


# --- 2. STEP FUNCTION TARGET (unchanged) ---
def build_step_target(mask_window, qrs_indices):
    n = len(mask_window)
    target = np.zeros(n, dtype=np.float32)

    if len(qrs_indices) < 2:
        return mask_window.astype(np.float32)

    if qrs_indices[0] > 0:
        target[:qrs_indices[0]] = float(mask_window[:qrs_indices[0]].max() > 0)

    for k in range(len(qrs_indices) - 1):
        s, e = qrs_indices[k], qrs_indices[k + 1]
        target[s:e] = float(mask_window[s:e].max() > 0)

    target[qrs_indices[-1]:] = float(mask_window[qrs_indices[-1]:].max() > 0)
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


# --- 4. ARCHITECTURE — parametric input channels for ablation ---
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=5):
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
        self.final = nn.Conv1d(32, 1, kernel_size=1)

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
        return self.final(d1)


# --- 5. DATA LOADER — features computed once per signal, then sliced ---
class SignalLevelLoader:
    def __init__(self, sig_paths, mask_paths, qrs_paths, active_features):
        self.sig_paths      = sig_paths
        self.mask_paths     = mask_paths
        self.qrs_paths      = qrs_paths
        self.active_features = active_features   # e.g. [0, 1, 2, 3, 4]

    def get_batches(self, window_size, stride, batch_size):
        indices = np.arange(len(self.sig_paths))
        np.random.shuffle(indices)
        x_batch, y_batch = [], []

        for idx in indices:
            sig      = np.load(self.sig_paths[idx]).flatten().astype(np.float64)
            mask     = np.load(self.mask_paths[idx]).flatten()
            qrs_locs = np.load(self.qrs_paths[idx]).flatten().astype(int)

            # Compute all features on the whole signal, then select active subset
            all_feats = extract_features_signal(sig, qrs_locs)   # (5, N)
            feats     = all_feats[self.active_features]           # (k, N)

            for start in range(0, sig.shape[0] - window_size, stride):
                end   = start + window_size
                v_qrs = qrs_locs[(qrs_locs >= start) & (qrs_locs < end)] - start

                x_batch.append(feats[:, start:end])              # (k, W)
                y_batch.append(build_step_target(mask[start:end], v_qrs))

                if len(x_batch) == batch_size:
                    yield (torch.from_numpy(np.array(x_batch)).float(),
                           torch.from_numpy(np.array(y_batch)).float().unsqueeze(1))
                    x_batch, y_batch = [], []


# --- 6. MAIN ---
def run_ablation(label, active_features, sig_p, mask_p, qrs_p,
                 device, window_size, stride, batch_size, lr, epochs):
    n_feat    = len(active_features)
    feat_str  = '+'.join(FEATURE_NAMES[i] for i in active_features)
    print(f"\n{'#'*70}")
    print(f"  ABLATION: {label}  ({n_feat} features: {feat_str})")
    print(f"{'#'*70}")

    split        = int(0.8 * len(sig_p))
    train_loader = SignalLevelLoader(sig_p[:split], mask_p[:split], qrs_p[:split], active_features)
    val_loader   = SignalLevelLoader(sig_p[split:], mask_p[split:], qrs_p[split:], active_features)

    model     = AttentionUNet(in_channels=n_feat).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def criterion(pred, target):
        return F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=torch.tensor([3.0]).to(pred.device))

    best_f1   = 0.0
    save_path = f"16_abl_{label}.pth"

    for epoch in range(epochs):
        model.train()
        t_loss, t_steps = 0, 0
        for x, y in train_loader.get_batches(window_size, stride, batch_size):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item(); t_steps += 1

        model.eval()
        v_loss, v_steps = 0, 0
        TP = FP = FN = TN = 0
        with torch.no_grad():
            for x, y in val_loader.get_batches(window_size, stride, batch_size):
                x, y = x.to(device), y.to(device)
                pred   = model(x)
                v_loss += criterion(pred, y).item(); v_steps += 1
                binary  = (torch.sigmoid(pred) > 0.5).float()
                TP += ((binary == 1) & (y == 1)).sum().item()
                FP += ((binary == 1) & (y == 0)).sum().item()
                FN += ((binary == 0) & (y == 1)).sum().item()
                TN += ((binary == 0) & (y == 0)).sum().item()

        prec = TP / (TP + FP + 1e-8)
        rec  = TP / (TP + FN + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        acc  = (TP + TN) / (TP + FP + FN + TN + 1e-8)

        print(f"  Ep {epoch+1:03d} | Train: {t_loss/t_steps:.4f} | Val: {v_loss/v_steps:.4f} | "
              f"Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"  --> New best (F1={f1:.3f})  saved: {save_path}")

    return best_f1


def main():
    WINDOW_SIZE   = 3840
    STRIDE        = 250
    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-5
    EPOCHS        = 100

    DATA_ROOT = "/home/20251020/ECG/Npy_DB"
    sig_p  = sorted(glob.glob(os.path.join(DATA_ROOT, "signals",  "*.npy")))
    mask_p = sorted(glob.glob(os.path.join(DATA_ROOT, "masks",    "*.npy")))
    qrs_p  = sorted(glob.glob(os.path.join(DATA_ROOT, "qrs_locs", "*.npy")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Signals: {len(sig_p)}")

    # Ablation configs: (label, active feature indices)
    # All-5 is the baseline; each subsequent run drops one feature to measure its contribution.
    ablation_configs = [
        ("all5",          [0, 1, 2, 3, 4]),
        ("no_linear_env", [   1, 2, 3, 4]),
        ("no_hilbert",    [0,    2, 3, 4]),
        ("no_residual",   [0, 1,    3, 4]),
        ("no_local_rms",  [0, 1, 2,    4]),
        ("no_qrs_rate",   [0, 1, 2, 3   ]),
    ]

    results = {}
    for label, active in ablation_configs:
        best_f1 = run_ablation(
            label, active, sig_p, mask_p, qrs_p,
            device, WINDOW_SIZE, STRIDE, BATCH_SIZE, LEARNING_RATE, EPOCHS
        )
        results[label] = best_f1

    # ── Summary table ──────────────────────────────────────────────────────────
    baseline = results.get("all5", 0.0)
    print(f"\n{'='*55}")
    print(f"  {'Config':<20}  {'Best F1':>8}  {'Drop vs all5':>12}")
    print(f"{'='*55}")
    for label, f1 in results.items():
        drop = baseline - f1
        marker = f"  ▲ baseline" if label == "all5" else f"  -{drop:.4f}"
        print(f"  {label:<20}  {f1:>8.4f}  {marker}")
    print(f"{'='*55}")
    print("\nLarger drop = that feature was more important.")


if __name__ == "__main__":
    main()
