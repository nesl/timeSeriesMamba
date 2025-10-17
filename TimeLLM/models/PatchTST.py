# models/PatchTST.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed1D(nn.Module):
    """
    Turn [B, L, D] into patch tokens [B, Np, d_model].
    """
    def __init__(self, in_dim, patch_len=16, stride=8, d_model=64):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim * patch_len, d_model)

    def forward(self, x):  # x: [B, L, D]
        B, L, D = x.shape
        Np = 1 + max(0, (L - self.patch_len) // self.stride)
        patches = []
        for i in range(Np):
            s = i * self.stride
            e = s + self.patch_len
            if e > L: break
            patches.append(x[:, s:e, :])   # [B, patch_len, D]
        if not patches:  # fallback: pad
            pad = self.patch_len - L
            xpad = F.pad(x, (0,0,0,pad))   # pad time
            patches = [xpad[:, :self.patch_len, :]]
        X = torch.stack(patches, dim=1)    # [B, Np, patch_len, D]
        X = X.reshape(B, X.size(1), -1)    # [B, Np, patch_len*D]
        return self.proj(X)                # [B, Np, d_model]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, max_len, d_model]

    def forward(self, x):  # [B, N, d_model]
        return x + self.pe[:, :x.size(1), :]

class TinyTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, n_heads=4, num_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)

class Model(nn.Module):
    """
    Proper PatchTST-style forecaster:
      forward(batch_x, batch_x_mark, dec_inp, batch_y_mark) -> [B, H, D]
    - Channel-shared encoder (simple & fast).
    - Mean-pool encoder tokens, linear head to (H*D).
    """
    def __init__(self, args):
        super().__init__()
        self.seq_len   = args.seq_len
        self.pred_len  = args.pred_len
        self.enc_in    = args.enc_in
        self.c_out     = args.c_out
        self.patch_len = getattr(args, 'patch_len', 16)
        self.stride    = getattr(args, 'stride', 8)

        d_model = getattr(args, 'd_model', 64)
        n_heads = max(1, min(getattr(args, 'n_heads', 4), d_model))
        e_layers= getattr(args, 'e_layers', 2)
        d_ff    = getattr(args, 'd_ff', 128)
        dropout = getattr(args, 'dropout', 0.1)

        self.patch = PatchEmbed1D(self.enc_in, self.patch_len, self.stride, d_model)
        self.pos   = PositionalEncoding(d_model)
        self.enc   = TinyTransformerEncoder(d_model, n_heads, e_layers, d_ff, dropout)
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, self.pred_len * self.c_out)

    def forward(self, batch_x, batch_x_mark=None, dec_inp=None, batch_y_mark=None):
        tok = self.patch(batch_x)          # [B, Np, d_model]
        tok = self.pos(tok)                # add positions
        h   = self.enc(tok)                # [B, Np, d_model]
        h   = self.norm(h)
        pooled = h.mean(dim=1)             # [B, d_model]
        out = self.head(pooled)            # [B, H*D]
        return out.view(-1, self.pred_len, self.c_out)
