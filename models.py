"""
models.py
----------
The actual Objective 1 architecture:

  Stream A  -- CNN pileup encoder      (local, per-base evidence)
  Stream B  -- Transformer context encoder (long-range sequence context)
  Fusion    -- cross-attention between the two streams
  Head      -- VAF-aware classification head

Plus two ablation baselines (CNNOnlyModel, TransformerOnlyModel) so we can
empirically show the fusion is actually doing something, not just assert it.
"""

import math
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Stream A: CNN Pileup Encoder (DeepVariant-style local evidence encoder)
# ----------------------------------------------------------------------
class CNNPileupEncoder(nn.Module):
    def __init__(self, in_channels=5, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, out_dim)

    def forward(self, pileup):
        # pileup: (batch, channels=5, reads=64, width=21)
        feat_map = self.conv(pileup)          # (batch, 64, 64, 21)  -- this is what Grad-CAM hooks into
        self.last_feature_map = feat_map       # stashed for Grad-CAM
        pooled = self.pool(feat_map).flatten(1)  # (batch, 64)
        return self.proj(pooled), feat_map     # pooled_vector: (batch, out_dim)


# ----------------------------------------------------------------------
# Stream B: Transformer Context Encoder (+-150bp sequence context)
# ----------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerContextEncoder(nn.Module):
    """6-layer, 8-head self-attention encoder over a +-150bp k-mer window,
    per Objective 1's spec. (num_layers is configurable so we can trade
    fidelity for CPU training speed during prototyping -- see README.)"""
    def __init__(self, vocab_size=4, d_model=64, nhead=8, num_layers=6, ctx_len=301):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.posenc = PositionalEncoding(d_model, max_len=ctx_len + 1)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                            dim_feedforward=d_model * 4,
                                            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.center_idx = ctx_len // 2

    def forward(self, ctx_tokens):
        # ctx_tokens: (batch, ctx_len) integer base indices 0-3
        x = self.embed(ctx_tokens)
        x = self.posenc(x)
        tokens = self.encoder(x)                       # (batch, ctx_len, d_model)  -- all positions
        center_token = tokens[:, self.center_idx, :]    # embedding at the candidate site itself
        return tokens, center_token


# ----------------------------------------------------------------------
# Cross-Attention Fusion
# ----------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    """Stream A's pooled pileup vector acts as the QUERY; Stream B's full
    sequence of per-position embeddings act as KEY/VALUE. This lets the
    pileup evidence 'ask' the surrounding sequence context which positions
    are relevant to it, instead of averaging both streams blindly."""
    def __init__(self, a_dim=128, b_dim=64, fused_dim=64, nhead=4):
        super().__init__()
        self.query_proj = nn.Linear(a_dim, fused_dim)
        self.attn = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=nhead, batch_first=True)

    def forward(self, a_vec, b_tokens):
        # a_vec: (batch, a_dim)         b_tokens: (batch, ctx_len, b_dim)
        q = self.query_proj(a_vec).unsqueeze(1)             # (batch, 1, fused_dim)
        fused, attn_weights = self.attn(q, b_tokens, b_tokens)  # fused: (batch, 1, fused_dim)
        self.last_attn_weights = attn_weights                # (batch, 1, ctx_len) -- for inspection/plots
        return fused.squeeze(1), attn_weights


# ----------------------------------------------------------------------
# Full proposed model: Stream A + Stream B + Cross-Attention Fusion + head
# ----------------------------------------------------------------------
class ExplainSomaticModel(nn.Module):
    def __init__(self, transformer_layers=6):
        super().__init__()
        self.stream_a = CNNPileupEncoder(out_dim=128)
        self.stream_b = TransformerContextEncoder(d_model=64, nhead=8, num_layers=transformer_layers)
        self.fusion = CrossAttentionFusion(a_dim=128, b_dim=64, fused_dim=64, nhead=4)
        fused_input_dim = 128 + 64 + 64  # [stream_a_vec, fused_context, stream_b_center_token]
        self.head = nn.Sequential(
            nn.Linear(fused_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, pileup, ctx_tokens):
        a_vec, feat_map = self.stream_a(pileup)
        b_tokens, b_center = self.stream_b(ctx_tokens)
        fused_ctx, attn_w = self.fusion(a_vec, b_tokens)
        combined = torch.cat([a_vec, fused_ctx, b_center], dim=1)
        logit = self.head(combined).squeeze(1)
        return logit


# ----------------------------------------------------------------------
# Ablation baselines -- used ONLY to prove the fusion adds value
# ----------------------------------------------------------------------
class CNNOnlyModel(nn.Module):
    """Stream A alone -- mimics the DeepVariant / NeuSomatic family: local
    pileup evidence, no sequence context at all."""
    def __init__(self):
        super().__init__()
        self.stream_a = CNNPileupEncoder(out_dim=128)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, pileup, ctx_tokens=None):
        a_vec, _ = self.stream_a(pileup)
        return self.head(a_vec).squeeze(1)


class TransformerOnlyModel(nn.Module):
    """Stream B alone -- mimics TransSSVs: sequence context only, no
    pileup evidence at all."""
    def __init__(self, transformer_layers=6):
        super().__init__()
        self.stream_b = TransformerContextEncoder(d_model=64, nhead=8, num_layers=transformer_layers)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, pileup, ctx_tokens):
        _, b_center = self.stream_b(ctx_tokens)
        return self.head(b_center).squeeze(1)


if __name__ == "__main__":
    from data_sim import SomaticSimDataset
    from torch.utils.data import DataLoader

    ds = SomaticSimDataset(8, seed=0)
    dl = DataLoader(ds, batch_size=4)
    pileup, ctx, label, vaf, hp = next(iter(dl))

    model = ExplainSomaticModel(transformer_layers=2)  # small for a quick smoke test
    out = model(pileup, ctx)
    print("fusion model output shape:", out.shape)

    cnn_only = CNNOnlyModel()
    print("cnn-only output shape:", cnn_only(pileup, ctx).shape)

    trans_only = TransformerOnlyModel(transformer_layers=2)
    print("transformer-only output shape:", trans_only(pileup, ctx).shape)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"total trainable params (fusion model): {n_params:,}")