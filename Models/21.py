import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import os

# Model 21: QRS-free version of model 18.
# For datasets that provide only the cleaned fECG signal + binary movement mask
# with no QRS peak locations.
#
# Features dropped (required QRS locs):
#   residual  = sig - linear_interp(sig at QRS peaks)
#   qrs_rate  = instantaneous beat rate interpolated from RR intervals
#
# Replacement features (signal only):
#   sig_norm   — signal-level normalised raw waveform
#   local_rms  — rolling RMS energy (50-sample window)
#   first_diff — first derivative; highlights sharp QRS-like transitions
#                without needing explicit peak locations
#
# Target: raw binary mask directly (no QRS-interval step function).

NUM_FEATURES = 3


# --- 1. FEATURE EXTRACTION (no QRS needed) ---
def extract_features_signal(sig):
    sig_norm  = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    local_rms = np.sqrt(np.convolve(sig_norm ** 2, np.ones(50) / 50, mode='same'))
    diff      = np.diff(sig_norm, prepend=sig_norm[0])
    diff_norm = (diff - np.mean(diff)) / (np.std(diff) + 1e-8)
    return np.stack([sig_norm, local_rms, diff_norm], axis=0).astype(np.float32)


# --- 2. COMPONENTS ---
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


class BottleneckSelfAttention(nn.Module):
    def __init__(self, channels=256, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        return self.norm(x_t + attn_out).permute(0, 2, 1)


# --- 3. ARCHITECTURE ---
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=NUM_FEATURES):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, 32)
        self.enc2 = ResidualBlock(32, 64)
        self.enc3 = ResidualBlock(64, 128)
        self.enc4 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = ResidualBlock(256, 256)
        self.self_attn  = BottleneckSelfAttention(channels=256, heads=4)
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
        b  = self.self_attn(b)
        g4 = self.up4(b);  d4 = self.dec4(torch.cat([g4, self.att4(g4, s4)], dim=1))
        g3 = self.up3(d4); d3 = self.dec3(torch.cat([g3, self.att3(g3, s3)], dim=1))
        g2 = self.up2(d3); d2 = self.dec2(torch.cat([g2, self.att2(g2, s2)], dim=1))
        g1 = self.up1(d2); d1 = self.dec1(torch.cat([g1, self.att1(g1, s1)], dim=1))
        return self.final(d1)


# --- 4. DATA LOADER ---
class SignalLevelLoader:
    """
    Each recording's signal file is (6, N) — 6 electrode channels of the same
    20-minute recording. The movement mask is shared across all 6 channels
    (movement is a global event, not per-electrode). We treat each
    (recording, channel) pair as its own training signal, giving 6x the
    samples (122 recordings -> 732 entries).

    Features are precomputed once per channel at startup. Each epoch shuffles
    the flat list of 732 entries with np.random.permutation, so consecutive
    batches don't come from 6 channels of the same recording in a row
    (which would reproduce the correlated-batch jitter problem).
    """
    def __init__(self, sig_paths, mask_paths):
        print("  Precomputing features...")
        self.feats_list = []
        self.masks_list = []
        for sp, mp in zip(sig_paths, mask_paths):
            sig  = np.load(sp).astype(np.float64)
            mask = np.load(mp).flatten()

            if sig.ndim == 1:
                if len(sig) % len(mask) == 0:
                    n_channels = len(sig) // len(mask)
                    sig = sig.reshape(n_channels, len(mask))
                else:
                    sig = sig[np.newaxis, :]   # single channel

            for ch in range(sig.shape[0]):
                self.feats_list.append(extract_features_signal(sig[ch]))   # (3, N)
                self.masks_list.append(mask)

        print(f"  {len(sig_paths)} recordings -> {len(self.feats_list)} channel-signals loaded.")

    def get_batches(self, window_size, stride, batch_size):
        indices = np.random.permutation(len(self.feats_list))
        x_batch, y_batch = [], []

        for idx in indices:
            feats = self.feats_list[idx]
            mask  = self.masks_list[idx]
            n     = min(feats.shape[1], len(mask))

            for start in range(0, n - window_size, stride):
                end = start + window_size
                x_batch.append(feats[:, start:end])
                y_batch.append(mask[start:end].astype(np.float32))

                if len(x_batch) == batch_size:
                    yield (torch.from_numpy(np.stack(x_batch)),
                           torch.from_numpy(np.stack(y_batch)).unsqueeze(1))
                    x_batch, y_batch = [], []


# --- 5. MAIN ---
def main():
    WINDOW_SIZE   = 3840
    STRIDE        = 250
    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-5
    EPOCHS        = 100

    DATA_ROOT = "/home/20251020/ECG/Npy_DB_2"   # update to QRS-free dataset path
    sig_p  = sorted(glob.glob(os.path.join(DATA_ROOT, "signals", "*.npy")))
    mask_p = sorted(glob.glob(os.path.join(DATA_ROOT, "masks",   "*.npy")))

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = AttentionUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.95, 0.999))
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def criterion(pred, target):
        return F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=torch.tensor([3.0]).to(pred.device))

    split = int(0.8 * len(sig_p))

    print(f"Model 21 | 3 features (no QRS) | signal-level norm | self-attention")
    print(f"Device: {device}  |  Params: {n_params:,}  |  LR: {LEARNING_RATE}")
    train_loader = SignalLevelLoader(sig_p[:split], mask_p[:split])
    val_loader   = SignalLevelLoader(sig_p[split:], mask_p[split:])
    print()

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

        print(f"Epoch {epoch+1:03d} | Train: {t_loss/t_steps:.4f} | Val: {v_loss/v_steps:.4f} | "
              f"Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "21_AttUNet_3feat_noqrs_best.pth")
            print(f"--> New best (F1={f1:.3f})  saved: 21_AttUNet_3feat_noqrs_best.pth")


if __name__ == "__main__":
    main()
