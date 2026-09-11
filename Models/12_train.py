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


class TransformerBottleneck(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        return x.permute(0, 2, 1)


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (batch, channels, seq) → permute to (batch, seq, channels)
        x = x.permute(0, 2, 1)
        weights = torch.softmax(self.attn(x), dim=1)   # (batch, seq, 1)
        pooled  = (x * weights).sum(dim=1)              # (batch, channels)
        return pooled


# --- 3. ARCHITECTURE ---
class ResNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualBlock(2, 32)
        self.enc2 = ResidualBlock(32, 64)
        self.enc3 = ResidualBlock(64, 128)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = TransformerBottleneck(128, num_heads=4)
        self.attn_pool = AttentionPooling(128)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.bottleneck(x)
        x = self.attn_pool(x)
        return self.classifier(x)


# --- 4. DATA LOADER ---
class TripleFolderLoader:
    def __init__(self, sig_paths, mask_paths, qrs_paths):
        self.sig_paths, self.mask_paths, self.qrs_paths = sig_paths, mask_paths, qrs_paths

    def get_batches(self, window_size, stride, batch_size):
        indices = np.arange(len(self.sig_paths))
        np.random.shuffle(indices)
        x_batch, y_batch = [], []

        for idx in indices:
            sig      = np.load(self.sig_paths[idx]).flatten()
            mask     = np.load(self.mask_paths[idx]).flatten()
            qrs_locs = np.load(self.qrs_paths[idx]).flatten()

            for start in range(0, sig.shape[0] - window_size, stride):
                end   = start + window_size
                v_qrs = qrs_locs[(qrs_locs >= start) & (qrs_locs < end)] - start
                label = float(mask[start:end].max() > 0)   # 1 if any movement in window

                x_batch.append(extract_perfect_features(sig[start:end], v_qrs))
                y_batch.append(label)

                if len(x_batch) == batch_size:
                    yield (torch.from_numpy(np.array(x_batch)).float(),
                           torch.tensor(y_batch).float().unsqueeze(1))
                    x_batch, y_batch = [], []


# --- 5. MAIN ---
def main():
    WINDOW_SIZE = 500    # 1 second at 500Hz
    STRIDE      = 250    # 0.5s step
    BATCH_SIZE  = 64
    LEARNING_RATE = 1e-4

    DATA_ROOT = "/home/20251020/ECG/Npy_DB"
    sig_p  = sorted(glob.glob(os.path.join(DATA_ROOT, "signals",  "*.npy")))
    mask_p = sorted(glob.glob(os.path.join(DATA_ROOT, "masks",    "*.npy")))
    qrs_p  = sorted(glob.glob(os.path.join(DATA_ROOT, "qrs_locs", "*.npy")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ResNetClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def criterion(pred, target):
        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=torch.tensor([5.0]).to(pred.device))

    split        = int(0.8 * len(sig_p))
    train_loader = TripleFolderLoader(sig_p[:split], mask_p[:split], qrs_p[:split])
    val_loader   = TripleFolderLoader(sig_p[split:], mask_p[split:], qrs_p[split:])

    print(f"ResNet Classifier + Transformer + Attention Pooling | LR: {LEARNING_RATE}")
    best_f1 = 0.0

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
                pred   = model(x)
                v_loss += criterion(pred, y).item()
                v_steps += 1
                binary = (torch.sigmoid(pred) > 0.5).float()
                TP += ((binary == 1) & (y == 1)).sum().item()
                FP += ((binary == 1) & (y == 0)).sum().item()
                FN += ((binary == 0) & (y == 1)).sum().item()
                TN += ((binary == 0) & (y == 0)).sum().item()

        avg_v     = v_loss / v_steps
        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy  = (TP + TN) / (TP + FP + FN + TN + 1e-8)
        print(f"Epoch {epoch + 1:03d} | Train: {t_loss / t_steps:.4f} | Val: {avg_v:.4f} | "
              f"Acc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "(12)ResNet_Trans_AttPool_best.pth")
            print(f"--> Saved New Best Model (F1: {f1:.3f})")


if __name__ == "__main__":
    main()
