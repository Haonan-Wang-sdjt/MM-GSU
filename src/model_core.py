import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Base positional encoding module for sequence modeling"""
    def __init__(self, d_model, max_len=256, learnable=False):
        super().__init__()
        # Core logic: positional encoding implementation (specific calculations simplified)
        self.d_model = d_model
        self.learnable = learnable
        # ... (key parameters retained, specific initialization details simplified)

    def forward(self, x):
        # x: [B, T, D]
        return x  # Core logic: inject positional information (implementation simplified)


class TrafficEncoder(nn.Module):
    """Encoder for traffic time-series (speed/occupancy/Δv)"""
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        self.pe = PositionalEncoding(d_model=hidden_dim)

    def forward(self, x):
        # x: [B, T, 3] -> traffic features
        x = self.proj(x)  # Dimension mapping
        x = x + self.dwconv(x.permute(0,2,1)).permute(0,2,1)  # Local feature enhancement
        x = self.pe(x)
        return self.transformer(x)  # [B, T, 64]


class ImageFeatureEncoder(nn.Module):
    """Encoder for image-derived features"""
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        # Core structure: convolution for dimension reduction + attention mechanism (details simplified)

    def forward(self, x):
        # x: [B, T, 128] -> image features
        x = self.conv(x.permute(0,2,1)).permute(0,2,1)  # Dimension mapping
        attn_out, _ = self.attn(x, x, x)
        return attn_out  # [B, T, 64]


class GatedFusion(nn.Module):
    """Gated fusion for traffic and image features"""
    def __init__(self, dim=64):
        super().__init__()
        self.gate = nn.Linear(2*dim, dim)  # Gating weight calculation

    def forward(self, traffic_feat, image_feat):
        # Core logic: dynamically compute fusion weights (details simplified)
        fuse_weight = torch.sigmoid(self.gate(torch.cat([traffic_feat, image_feat], dim=-1)))
        return fuse_weight * traffic_feat + (1 - fuse_weight) * image_feat


class MMGSU(nn.Module):
    """Multi-modal gated fusion model for state prediction"""
    def __init__(self, pred_len=30):
        super().__init__()
        self.traffic_encoder = TrafficEncoder()
        self.image_encoder = ImageFeatureEncoder()
        self.fusion = GatedFusion()
        self.predictor = nn.GRU(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(64, 2)  # Predict speed and occupancy
        self.pred_len = pred_len

    def forward(self, traffic_data, image_data):
        # Core pipeline: encode -> fuse -> predict
        t_feat = self.traffic_encoder(traffic_data)
        i_feat = self.image_encoder(image_data)
        fused = self.fusion(t_feat, i_feat)
        pred_seq, _ = self.predictor(fused)
        return self.output_layer(pred_seq[:, -self.pred_len:, :])  # Take last pred_len steps