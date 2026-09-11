import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import os
from scipy.signal import find_peaks


# --- 1. FEATURE EXTRACTION (30s Context) ---
def extract_clinical_features(sig):
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

    # CH1: R-Peak Amplitudes
    peaks, _ = find_peaks(sig_norm, height=0.5, distance=150)
    amp_map = np.zeros_like(sig_norm)
    amp_map[peaks] = sig_norm[peaks]

    # CH2: Rolling Energy (2-second window)
    energy_map = np.zeros_like(sig_norm)
    res = np.array([np.std(sig_norm[max(0, i - 500):min(len(sig_norm), i + 500)])
                    for i in range(0, len(sig_norm), 100)])
    energy_map = np.interp(np.arange(len(sig_norm)), np.arange(0, len(sig_norm), 100), res)

    return np.stack([sig_norm, amp_map, energy_map], axis=0)


# --- 2. ARCHITECTURE ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        padding = (15 - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=padding, dilation=dilation),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, padding=7),
            nn.GroupNorm(4, out_channels)
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class TransformerBottleneck(nn.Module):
    def __init__(self, channels, seq_len):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=4, dim_feedforward=channels * 4,
            dropout=0.2, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, channels))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x + self.pos_emb
        x = self.transformer(x)
        return x.permute(0, 2, 1)


class BigWindowTransUNet(nn.Module):
    def __init__(self, window_size=15000):
        super().__init__()
        bottleneck_len = window_size // 64

        self.enc1 = ResidualBlock(3, 16, dilation=1)
        self.pool1 = nn.MaxPool1d(4)
        self.enc2 = ResidualBlock(16, 32, dilation=2)
        self.pool2 = nn.MaxPool1d(4)
        self.enc3 = ResidualBlock(32, 64, dilation=4)
        self.pool3 = nn.MaxPool1d(4)

        self.bottleneck = TransformerBottleneck(64, seq_len=bottleneck_len)

        self.dec3 = ResidualBlock(64 + 64, 64)
        self.dec2 = ResidualBlock(64 + 32, 32)
        self.dec1 = ResidualBlock(32 + 16, 16)
        self.final = nn.Conv1d(16, 1, kernel_size=15, padding=7)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = F.interpolate(b, size=e3.shape[2], mode='linear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[2], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[2], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)


# --- 3. LOADER ---
class ClinicalPatientLoader:
    def __init__(self, sig_paths, mask_paths):
        self.sig_paths = sig_paths
        self.mask_paths = mask_paths

    def get_batches(self, window_size, stride, batch_size, shuffle=True):
        indices = np.arange(len(self.sig_paths))
        if shuffle: np.random.shuffle(indices)

        for idx in indices:
            sig = np.load(self.sig_paths[idx]).flatten()
            mask = np.load(self.mask_paths[idx]).flatten()
            full_feat = extract_clinical_features(sig)

            x_slices, y_slices = [], []
            for start in range(0, sig.shape[0] - window_size, stride):
                x_slices.append(full_feat[:, start:start + window_size])
                y_slices.append(mask[start:start + window_size])

                if len(x_slices) == batch_size:
                    yield (torch.from_numpy(np.array(x_slices)).float(),
                           torch.from_numpy(np.array(y_slices)).float().unsqueeze(1))
                    x_slices, y_slices = [], []


# --- 4. MAIN TRAINING LOOP ---
def main():
    DATA_ROOT = "/home/20251020/ECG/Npy_DB"
    # 30 seconds (15000), Step every 2 seconds (1000)
    WINDOW_SIZE, STRIDE, BATCH_SIZE = 15000, 1000, 8
    EPOCHS, LR = 400, 0.0001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sig_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "signals", "*.npy")))
    mask_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "masks", "*.npy")))
    split = int(0.8 * len(sig_paths))

    train_loader = ClinicalPatientLoader(sig_paths[:split], mask_paths[:split])
    val_loader = ClinicalPatientLoader(sig_paths[split:], mask_paths[split:])

    model = BigWindowTransUNet(window_size=WINDOW_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Weighted BCE Loss (pos_weight=5.0 to emphasize movement blocks)
    pos_weight = torch.tensor([5.0]).to(device)
    criterion = lambda out, y: F.binary_cross_entropy_with_logits(out, y, pos_weight=pos_weight)

    best_val_loss = float('inf')

    print(f"Starting 30s Window Training on {device}...")
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss, train_steps = 0, 0
        for x, y in train_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE, shuffle=True):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / train_steps

        # --- Validation Phase ---
        model.eval()
        val_loss, val_steps = 0, 0
        with torch.no_grad():
            for x, y in val_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE, shuffle=False):
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss = criterion(out, y)
                val_loss += v_loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps

        # --- Logging & Saving ---
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save last model
        torch.save(model.state_dict(), "big_window_last.pth")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "big_window_best.pth")
            print(f"--> Saved New Best Model (Val Loss: {best_val_loss:.4f})")


if __name__ == "__main__":
    main()