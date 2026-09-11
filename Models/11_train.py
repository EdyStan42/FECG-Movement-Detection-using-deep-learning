import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import os


# --- 1. FEATURE EXTRACTION ---
def extract_perfect_features(sig, qrs_indices):
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

    if len(qrs_indices) == 0:
        return np.stack([sig_norm, np.zeros_like(sig_norm)], axis=0)

    linear_outline = np.interp(np.arange(len(sig_norm)), qrs_indices, sig_norm[qrs_indices])

    return np.stack([sig_norm, linear_outline], axis=0)


# --- 2. COMPONENTS ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=15, padding=7),
            nn.GroupNorm(4, out_c),
            nn.ReLU(),
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
        # g: gating signal from decoder, x: skip connection from encoder
        return x * self.psi(F.relu(self.W_g(g) + self.W_x(x)))


class TransformerBottleneck(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        return x.permute(0, 2, 1)


# --- 3. ARCHITECTURE ---
class PrecisionResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualBlock(2, 32)
        self.enc2 = ResidualBlock(32, 64)
        self.enc3 = ResidualBlock(64, 128)
        self.enc4 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = TransformerBottleneck(256)
        self.att4 = AttentionGate(f_g=256, f_x=256, f_int=128)
        self.att3 = AttentionGate(f_g=128, f_x=128, f_int=64)
        self.att2 = AttentionGate(f_g=64,  f_x=64,  f_int=32)
        self.att1 = AttentionGate(f_g=32,  f_x=32,  f_int=16)
        self.up4 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec4 = ResidualBlock(512, 128)
        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec3 = ResidualBlock(256, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec2 = ResidualBlock(128, 32)
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
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
class TripleFolderLoader:
    def __init__(self, sig_paths, mask_paths, qrs_paths):
        self.sig_paths, self.mask_paths, self.qrs_paths = sig_paths, mask_paths, qrs_paths

    def get_batches(self, window_size, stride, batch_size):
        indices = np.arange(len(self.sig_paths))
        np.random.shuffle(indices)
        x_batch, y_batch = [], []

        for idx in indices:
            sig = np.load(self.sig_paths[idx]).flatten()
            mask = np.load(self.mask_paths[idx]).flatten()
            qrs_locs = np.load(self.qrs_paths[idx]).flatten()

            for start in range(0, sig.shape[0] - window_size, stride):
                end = start + window_size
                v_qrs = qrs_locs[(qrs_locs >= start) & (qrs_locs < end)] - start
                x_batch.append(extract_perfect_features(sig[start:end], v_qrs))
                y_batch.append(mask[start:end])

                if len(x_batch) == batch_size:
                    yield (torch.from_numpy(np.array(x_batch)).float(),
                           torch.from_numpy(np.array(y_batch)).float().unsqueeze(1))
                    x_batch, y_batch = [], []


# --- 5. MAIN ---
def main():
    WINDOW_SIZE = 3840
    STRIDE = 250
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4

    DATA_ROOT = "/home/20251020/ECG/Npy_DB"
    sig_p  = sorted(glob.glob(os.path.join(DATA_ROOT, "signals",  "*.npy")))
    mask_p = sorted(glob.glob(os.path.join(DATA_ROOT, "masks",    "*.npy")))
    qrs_p  = sorted(glob.glob(os.path.join(DATA_ROOT, "qrs_locs", "*.npy")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrecisionResUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def criterion(pred, target):
        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=torch.tensor([5.0]).to(pred.device))

    split = int(0.8 * len(sig_p))
    train_loader = TripleFolderLoader(sig_p[:split], mask_p[:split], qrs_p[:split])
    val_loader   = TripleFolderLoader(sig_p[split:], mask_p[split:], qrs_p[split:])

    print(f"Res-UNet + Transformer + Attention Gates | LR: {LEARNING_RATE}")
    best_loss = float('inf')

    for epoch in range(100):
        model.train()
        t_loss, t_steps = 0, 0
        for x, y in train_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_steps += 1

        model.eval()
        v_loss, v_steps = 0, 0
        TP = FP = FN = TN = 0
        with torch.no_grad():
            for x, y in val_loader.get_batches(WINDOW_SIZE, STRIDE, BATCH_SIZE):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                v_loss += criterion(pred, y).item()
                v_steps += 1
                binary = (torch.sigmoid(pred) > 0.5).float()
                t = y
                TP += ((binary == 1) & (t == 1)).sum().item()
                FP += ((binary == 1) & (t == 0)).sum().item()
                FN += ((binary == 0) & (t == 1)).sum().item()
                TN += ((binary == 0) & (t == 0)).sum().item()

        avg_v    = v_loss / v_steps
        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy  = (TP + TN) / (TP + FP + FN + TN + 1e-8)
        print(f"Epoch {epoch + 1:03d} | Train: {t_loss / t_steps:.4f} | Val: {avg_v:.4f} | "
              f"Acc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}")

        if avg_v < best_loss:
            best_loss = avg_v
            torch.save(model.state_dict(), "11_ResUNet_Trans_AttGates_best.pth")
            print("--> Saved New Best Model")


if __name__ == "__main__":
    main()
