import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import os

# Single-feature ablation for model 17's 3 features.
# Trains one model per feature (in_channels=1) for 50 epochs each.
# Answers: how much can each feature alone achieve?
# Compare results against model 17 (all 3 together) to see the combined benefit.
#
# Features:
#   0 — residual  (sig - linear amplitude envelope at QRS peaks)
#   1 — local_rms (rolling RMS energy, 50-sample window)
#   2 — qrs_rate  (instantaneous beat rate, normalised)

FEATURE_NAMES = ["residual", "local_rms", "qrs_rate"]
EPOCHS        = 50


# --- 1. WHOLE-SIGNAL FEATURE EXTRACTION ---
def extract_features_signal(sig, qrs_locs):
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    n = len(sig_norm)
    x = np.arange(n)

    if len(qrs_locs) < 2:
        return np.stack([
            sig_norm,
            np.sqrt(np.convolve(sig_norm ** 2, np.ones(50) / 50, mode='same')),
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

    return np.stack([residual, local_rms, qrs_rate], axis=0)


# --- 2. STEP FUNCTION TARGET ---
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


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1):
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


# --- 4. DATA LOADER ---
class SignalLevelLoader:
    """
    Operates on pre-loaded, pre-extracted data (feats_list, masks_list, qrs_list)
    so the expensive extract_features_signal() call happens only once per signal,
    not once per signal per epoch per feature run.
    """
    def __init__(self, feats_list, masks_list, qrs_list, feature_idx):
        self.feats_list  = feats_list
        self.masks_list  = masks_list
        self.qrs_list    = qrs_list
        self.feature_idx = feature_idx   # 0, 1, or 2

    def get_batches(self, window_size, stride, batch_size):
        indices = np.random.permutation(len(self.feats_list))
        x_batch, y_batch = [], []

        for idx in indices:
            feat     = self.feats_list[idx][self.feature_idx:self.feature_idx+1, :]  # (1, N)
            mask     = self.masks_list[idx]
            qrs_locs = self.qrs_list[idx]
            n        = min(feat.shape[1], len(mask))

            for start in range(0, n - window_size, stride):
                end   = start + window_size
                v_qrs = qrs_locs[(qrs_locs >= start) & (qrs_locs < end)] - start
                x_batch.append(feat[:, start:end])
                y_batch.append(build_step_target(mask[start:end], v_qrs))

                if len(x_batch) == batch_size:
                    yield (torch.from_numpy(np.array(x_batch)).float(),
                           torch.from_numpy(np.array(y_batch)).float().unsqueeze(1))
                    x_batch, y_batch = [], []


# --- 5. TRAINING RUN ---
def run(feature_idx, feats_list, masks_list, qrs_list, device):
    name        = FEATURE_NAMES[feature_idx]
    WINDOW_SIZE = 3840
    STRIDE      = 250
    BATCH_SIZE  = 32
    LR          = 1e-5

    model     = AttentionUNet(in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    def criterion(pred, target):
        return F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=torch.tensor([3.0]).to(pred.device))

    split        = int(0.8 * len(feats_list))
    train_loader = SignalLevelLoader(feats_list[:split], masks_list[:split], qrs_list[:split], feature_idx)
    val_loader   = SignalLevelLoader(feats_list[split:], masks_list[split:], qrs_list[split:], feature_idx)

    print(f"\n{'='*60}")
    print(f"Feature: {name}  (index {feature_idx})")
    print(f"{'='*60}")

    best_f1 = 0.0

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

        model.eval()
        v_loss, v_steps = 0, 0
        TP = FP = FN = TN = 0
        with torch.no_grad():
            for x, y in val_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE):
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

        print(f"Epoch {epoch+1:02d} | Train: {t_loss/t_steps:.4f} | Val: {v_loss/v_steps:.4f} | "
              f"Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"17_feat_{name}_best.pth")

    return best_f1


# --- 6. MAIN ---
def main():
    DATA_ROOT = "/home/20251020/ECG/Npy_DB2"
    sig_p  = sorted(glob.glob(os.path.join(DATA_ROOT, "signals",  "*.npy")))
    mask_p = sorted(glob.glob(os.path.join(DATA_ROOT, "masks",    "*.npy")))
    qrs_p  = sorted(glob.glob(os.path.join(DATA_ROOT, "qrs_locs", "*.npy")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"17_features | single-feature models | {EPOCHS} epochs each | device: {device}")

    print("Precomputing features for all signals (once)...")
    feats_list, masks_list, qrs_list = [], [], []
    for sp, mp, qp in zip(sig_p, mask_p, qrs_p):
        sig      = np.load(sp).flatten().astype(np.float64)
        mask     = np.load(mp).flatten()
        qrs_locs = np.load(qp).flatten().astype(int)
        feats_list.append(extract_features_signal(sig, qrs_locs))   # (3, N)
        masks_list.append(mask)
        qrs_list.append(qrs_locs)
    print(f"  {len(feats_list)} signals cached.\n")

    results = {}
    for i in range(3):
        results[FEATURE_NAMES[i]] = run(i, feats_list, masks_list, qrs_list, device)

    print(f"\n{'='*60}")
    print(f"SUMMARY — best validation F1 per feature")
    print(f"{'='*60}")
    for name, f1 in results.items():
        print(f"  {name:<12}  F1 = {f1:.4f}")
    print(f"\n  (model 17 all-3 combined for reference)")


if __name__ == "__main__":
    main()
