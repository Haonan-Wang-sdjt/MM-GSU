# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import csv

# =======================
# ✅ Positional Encoding (for Transformer use)
# =======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256, learnable: bool = False):
        super().__init__()
        if learnable:
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.trunc_normal_(self.pe, std=0.02)
            self.learnable = True
        else:
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe_buf", pe.unsqueeze(0))  # [1,T,D]
            self.learnable = False

    def forward(self, x):  # x: [B,T,D]
        if self.learnable:
            return x + self.pe[:, :x.size(1)]
        else:
            return x + self.pe_buf[:, :x.size(1)]

# =======================
# ✅ Model Structure Definition
# =======================

class TrafficEncoder(nn.Module):
    """
    Input:  [B, 120, 3] → (speed, occupancy, Δv)
    Output:  [B, 120, 64]  aligned with subsequent modules
    """
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        d_model: int = 64,     # Transformer channels
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
        use_depthwise_conv: bool = True,  # Optional: local pattern enhancement
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)

        self.use_depthwise_conv = use_depthwise_conv
        if use_depthwise_conv:
            # depthwise 1D conv for local pattern extraction, residual addition for stability
            self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
            self.bn = nn.BatchNorm1d(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # [B,T,D]
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pe = PositionalEncoding(d_model=d_model, max_len=256, learnable=False)
        self.out = nn.Linear(d_model, hidden_dim)

    def forward(self, x):           # x: [B,120,3]
        h = self.proj(x)            # [B,120,d]
        if self.use_depthwise_conv:
            h_conv = self.dwconv(h.permute(0, 2, 1))   # [B,d,120]
            h_conv = self.bn(h_conv).permute(0, 2, 1)  # [B,120,d]
            h = h + h_conv                              # residual enhancement for local patterns
        h = self.pe(h)
        h = self.encoder(h)          # [B,120,d]
        return self.out(h)           # [B,120,64]

class ImageEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, seq_len=120):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, 64))
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm([64, seq_len])  # Can replace with BatchNorm1d(64) if unstable
        )
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        self.project = nn.Linear(64, hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # B x C x T
        x = self.conv(x)        # B x C x T
        x = x.permute(0, 2, 1)  # B x T x C
        x = x + self.pos_embed[:, :x.size(1), :]
        attn_out, _ = self.attn(x, x, x)
        attn_out = attn_out + x
        ffn_out = self.ffn(attn_out) + attn_out
        weight = self.channel_gate(ffn_out)
        gated = ffn_out * weight
        return self.project(gated)   # [B,T,64]

class EnhancedGateFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim * 2, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, t_feat, i_feat):
        concat = torch.cat([t_feat, i_feat], dim=-1)
        weights = self.softmax(self.linear(concat))
        w_t = weights[..., 0:1]
        w_i = weights[..., 1:2]
        fused = w_t * t_feat + w_i * i_feat
        return fused, weights   # Return fused features and gating weights [B,T,2]

class StatePredictor(nn.Module):
    def __init__(self, input_dim=64, pred_len=30):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=input_dim, num_layers=2, dropout=0.1, batch_first=True)
        self.linear = nn.Linear(input_dim, 2)
        self.pred_len = pred_len

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear(out)
        return out[:, -self.pred_len:, :]

class MMGSU(nn.Module):
    def __init__(self, traffic_dim=3, image_dim=128, pred_len=30):
        super().__init__()
        self.traffic_encoder = TrafficEncoder(input_dim=traffic_dim)
        self.image_encoder = ImageEncoder(input_dim=image_dim)
        self.fusion = EnhancedGateFusion(dim=64)
        self.enhancer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.predictor = StatePredictor(input_dim=64, pred_len=pred_len)
        self.dropout_prob = 0.1  # Modality dropout probability

    def forward(self, x_traffic, x_image, return_weights: bool = False):
        t_feat = self.traffic_encoder(x_traffic)
        i_feat = self.image_encoder(x_image)

        # Modality dropout (only enabled during training)
        if self.training and self.dropout_prob > 0:
            drop_mask = torch.rand_like(i_feat[:, :, :1]) < self.dropout_prob
            i_feat = i_feat.masked_fill(drop_mask, 0.0)

        fused, weights = self.fusion(t_feat, i_feat)   # [B,120,64], [B,120,2]
        enhanced = self.enhancer(fused)                # [B,120,64]
        out = self.predictor(enhanced + fused)         # Prediction input is fused residual

        if return_weights:
            return out, weights
        return out

# =======================
# ✅ Dataset
# =======================
class NPYDataset(Dataset):
    def __init__(self, traffic_npy_dir, image_npy_dir):
        self.samples = []
        self.speed_max = 0.0
        self.occ_max = 0.0

        for subfolder in os.listdir(traffic_npy_dir):
            t_path = os.path.join(traffic_npy_dir, subfolder)
            i_path = os.path.join(image_npy_dir, subfolder)
            if not os.path.isdir(t_path): continue
            t_files = sorted(f for f in os.listdir(t_path) if f.endswith(".npy"))
            i_files = sorted(f for f in os.listdir(i_path) if f.endswith(".npy"))
            for tf, inf in zip(t_files, i_files):
                t_full = os.path.join(t_path, tf)
                t_arr = np.load(t_full)
                self.speed_max = max(self.speed_max, float(np.max(t_arr[:, 0])))
                self.occ_max = max(self.occ_max, float(np.max(t_arr[:, 1])))
                self.samples.append((t_full, os.path.join(i_path, inf)))

        # Prevent division by zero
        self.speed_max = self.speed_max if self.speed_max > 0 else 1.0
        self.occ_max = self.occ_max if self.occ_max > 0 else 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t_path, i_path = self.samples[idx]
        t_arr = np.load(t_path)
        i_arr = np.load(i_path)

        # Normalization
        t_arr[:, 0] /= self.speed_max
        t_arr[:, 1] /= self.occ_max

        # Δv
        delta_v = np.zeros_like(t_arr[:, 0])
        delta_v[1:] = t_arr[1:, 0] - t_arr[:-1, 0]
        delta_v = delta_v.reshape(-1, 1)
        t_arr_aug = np.concatenate([t_arr[:, :2], delta_v], axis=1)

        return (
            torch.tensor(t_arr_aug[:120], dtype=torch.float32),   # [120,3]
            torch.tensor(i_arr[:120], dtype=torch.float32),       # [120,128]
            torch.tensor(t_arr[120:150, :2], dtype=torch.float32) # [30,2]
        )

# =======================
# ✅ Training function (prints per-epoch and total time)
# =======================
def train_model(traffic_npy_dir, image_npy_dir, epochs=200, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    start_time = time.time()

    dataset = NPYDataset(traffic_npy_dir, image_npy_dir)
    print(f"✅ Loaded {len(dataset)} samples")

    # Dataset splitting
    train_len = int(0.7 * len(dataset))
    val_len = int(0.2 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_set, val_set, _ = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Initialize model and optimizer
    model = MMGSU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_start = time.time()

        # ===== Training =====
        model.train()
        total_loss = 0
        for x_t, x_i, y in train_loader:
            x_t, x_i, y = x_t.to(device), x_i.to(device), y.to(device)
            pred = model(x_t, x_i)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_t.size(0)
        avg_train = total_loss / train_len

        # ===== Validation =====
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_t, x_i, y in val_loader:
                x_t, x_i, y = x_t.to(device), x_i.to(device), y.to(device)
                pred = model(x_t, x_i)
                val_loss += criterion(pred, y).item() * x_t.size(0)
        avg_val = val_loss / val_len

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f} | Time: {epoch_time:.2f}s")

        # ===== Save Best =====
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_model.pt")
            print("📦 Saved best model")

    total_time = time.time() - start_time
    print(f"\n✅ Training completed, total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# =======================
# ✅ Evaluation (including inference time statistics)
# =======================
def evaluate_model_fixed(model, fixed_test_indices, dataset, device, speed_max, occ_max, save_dir=None):
    model.eval()
    results = []
    max_mae_speed = -float('inf')
    min_mae_speed = float('inf')
    max_mae_occ = -float('inf')
    min_mae_occ = float('inf')

    mae_speed_list, mae_occ_list = [], []
    rmse_speed_list, rmse_occ_list = [], []

    total_infer_time = 0.0
    infer_count = 0

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in fixed_test_indices:
        x_t, x_i, y = dataset[i]
        x_t = x_t.unsqueeze(0).to(device)
        x_i = x_i.unsqueeze(0).to(device)
        y = y.numpy()

        # Inference time
        start_time = time.time()
        with torch.no_grad():
            pred = model(x_t, x_i).squeeze(0).cpu().numpy()
        end_time = time.time()
        total_infer_time += (end_time - start_time)
        infer_count += 1

        # Denormalization
        pred[:, 0] *= speed_max
        pred[:, 1] *= occ_max
        y[:, 0] *= speed_max
        y[:, 1] *= occ_max

        # Metrics
        mae_speed = np.mean(np.abs(pred[:, 0] - y[:, 0]))
        mae_occ = np.mean(np.abs(pred[:, 1] - y[:, 1]))
        rmse_speed = np.sqrt(np.mean((pred[:, 0] - y[:, 0])**2))
        rmse_occ = np.sqrt(np.mean((pred[:, 1] - y[:, 1])** 2))

        mae_speed_list.append(mae_speed)
        rmse_speed_list.append(rmse_speed)
        mae_occ_list.append(mae_occ)
        rmse_occ_list.append(rmse_occ)

        max_mae_speed = max(max_mae_speed, mae_speed)
        min_mae_speed = min(min_mae_speed, mae_speed)
        max_mae_occ = max(max_mae_occ, mae_occ)
        min_mae_occ = min(min_mae_occ, mae_occ)

        results.append((pred, y))

        if save_dir:
            df = pd.DataFrame({
                "Frame": np.arange(pred.shape[0]),
                "True Speed": y[:, 0],
                "Pred Speed": pred[:, 0],
                "True Occ": y[:, 1],
                "Pred Occ": pred[:, 1]
            })
            df = df.replace([np.inf, -np.inf], np.nan)
            df.to_csv(os.path.join(save_dir, f"sample_{i}.csv"),
                      index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    # Error statistics
    print("\n📊 Prediction error results:")
    print(f"Speed MAE: {np.mean(mae_speed_list):.4f}, Occupancy MAE: {np.mean(mae_occ_list):.4f}")
    print(f"Speed RMSE: {np.mean(rmse_speed_list):.4f}, Occupancy RMSE: {np.mean(rmse_occ_list):.4f}")
    print(f"Max speed MAE: {max_mae_speed:.4f}, Min speed MAE: {min_mae_speed:.4f}")
    print(f"Max occupancy MAE: {max_mae_occ:.4f}, Min occupancy MAE: {min_mae_occ:.4f}")

    # Inference time information
    print(f"\n🕒 Total inference time ({infer_count} samples): {total_infer_time:.2f} seconds")
    print(f"🕒 Average inference time (per sample): {total_infer_time / infer_count * 1000:.2f} milliseconds")

    summary = pd.DataFrame({
        "MAE Speed": mae_speed_list,
        "MAE Occ": mae_occ_list,
        "RMSE Speed": rmse_speed_list,
        "RMSE Occ": rmse_occ_list
    })
    if save_dir:
        summary = summary.replace([np.inf, -np.inf], np.nan)
        summary.to_csv(os.path.join(save_dir, "evaluation_summary.csv"),
                       index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    return results

# =======================
# ✅ Main program entry
# =======================
if __name__ == '__main__':
    train = True  # Set to True for retraining
    traffic_npy_dir = r"C:\Users\czx43\Desktop\MM-GSU\datasets\traffic_npy"
    image_npy_dir = r"C:\Users\czx43\Desktop\MM-GSU\datasets\image_npy"
    output_dir = "prediction_200"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train:
        train_model(traffic_npy_dir, image_npy_dir)

    # Load dataset
    dataset = NPYDataset(traffic_npy_dir, image_npy_dir)
    train_len = int(0.7 * len(dataset))
    val_len = int(0.2 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    # Indices for three splits
    test_indices = list(range(train_len + val_len, len(dataset)))

    # Build and print model
    model = MMGSU().to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model Parameter Statistics] Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")

    # Load weights (if available)
    if os.path.exists("best_model.pt"):
        model.load_state_dict(torch.load("best_model.pt", map_location=device))

    # Evaluate
    results = evaluate_model_fixed(model, test_indices, dataset, device,
                                   dataset.speed_max, dataset.occ_max, save_dir=output_dir)