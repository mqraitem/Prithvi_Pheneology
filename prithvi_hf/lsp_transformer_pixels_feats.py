import torch
import torch.nn as nn
import math
from typing import Optional

# ---------- Positional Encoding ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (N, T, d)
        return x + self.pe[:, :x.size(1), :]

# ---------- Cross-Attention (Q-Former-style) ----------
class CrossAttnBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.q_ln  = nn.LayerNorm(d_model)
        self.kv_ln = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * d_model), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, ctx):
        """
        x:   (N, Tq, d)
        ctx: (N, Tk, d)
        """
        q  = self.q_ln(x)
        kv = self.kv_ln(ctx)
        x2, _ = self.cross_attn(q, kv, kv)  # (N, Tq, d)
        x = x + self.drop1(x2)
        x = x + self.ffn(x)
        return x

# ---------- Fusion Transformer with optional Q-Former ----------
class TemporalQFormer(nn.Module):
    """
    Two input modes:

    1) processing_images=True
       - x_img: (B, C, T, H, W)
       - z_ctx: (B, T, K)   (tile-level features)
       - output: (B, num_classes, H, W)

    2) processing_images=False
       - x_img: (N, T, C)   (each row is a pixel time-series)
       - z_ctx: (N, T, K)   (context per pixel)
       - output: (N, num_classes)

    fusion: 'qformer' (cross-attn) | 'concat' (channel concat)
    """
    def __init__(self,
                 input_channels=6,
                 ctx_channels=32,
                 seq_len=12,
                 num_classes=4,
                 d_model=128,
                 nhead=4,
                 num_layers=3,
                 dropout=0.1,
                 fusion='qformer',
                 num_xattn=2,
                 mlp_ratio=4.0):
        super().__init__()
        self.seq_len = seq_len
        self.fusion  = fusion

        if fusion == 'concat':
            self.input_proj = nn.Linear(input_channels + ctx_channels, d_model, bias=False)
            self.ctx_proj   = None
            self.encoder_pe = PositionalEncoding(d_model, max_len=seq_len)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=int(mlp_ratio*d_model),
                dropout=dropout, batch_first=True, activation='relu'
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        elif fusion == 'qformer':
            self.input_proj = nn.Linear(input_channels, d_model, bias=False)
            self.ctx_proj   = nn.Linear(ctx_channels, d_model, bias=False)
            self.encoder_pe = PositionalEncoding(d_model, max_len=seq_len)
            self.ctx_pe     = PositionalEncoding(d_model, max_len=seq_len)

            self.self_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=int(mlp_ratio*d_model),
                    dropout=dropout, batch_first=True, activation='relu'
                ) for _ in range(num_layers)
            ])
            self.xattn_layers = nn.ModuleList([
                CrossAttnBlock(d_model=d_model, nhead=nhead, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(num_xattn)
            ])
        else:
            raise ValueError("fusion must be 'qformer' or 'concat'")

        self.dropout     = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, num_classes)

    def _broadcast_ctx_to_pixels(self, z_ctx, B, H, W):
        # z_ctx: (B, T, K) -> (B, 1, 1, T, K) -> (B,H,W,T,K) -> (N, T, K)
        return z_ctx[:, None, None, :, :].expand(B, H, W, -1, -1).reshape(B*H*W, self.seq_len, z_ctx.size(-1))

    def forward(self,
                x_img: torch.Tensor,
                z_ctx: Optional[torch.Tensor] = None,
                processing_images: bool = True,
                chunk_size: int = 2000):
        """
        Returns:
          if processing_images: (B, num_classes, H, W)
          else:                 (N, num_classes)
        """
        if processing_images:
            # x_img: (B, C, T, H, W)
            assert x_img.dim() == 5, f"Expected (B,C,T,H,W), got {tuple(x_img.shape)}"
            B, C, T, H, W = x_img.shape
            assert T == self.seq_len, f"seq_len mismatch: model {self.seq_len}, x {T}"
            assert z_ctx is not None and z_ctx.shape[:2] == (B, T), \
                f"z_ctx must be (B,T,K), got {tuple(z_ctx.shape) if z_ctx is not None else None}"

            # (B,H,W,T,C) -> (N,T,C)
            x = x_img.permute(0, 3, 4, 2, 1).reshape(B*H*W, T, C)
            N = x.size(0)

            if self.fusion == 'concat':
                # broadcast ctx and concat
                z = self._broadcast_ctx_to_pixels(z_ctx, B, H, W)  # (N,T,K)
                xin = torch.cat([x, z], dim=-1)                    # (N,T,C+K)
                xin = self.encoder_pe(self.input_proj(xin))        # (N,T,d)

                outs = []
                for i in range(0, N, chunk_size):
                    xi = self.encoder(xin[i:i+chunk_size])         # (n,T,d)
                    xi = xi.mean(dim=1)
                    xi = self.dropout(xi)
                    xi = self.output_proj(xi)                      # (n,num_classes)
                    outs.append(xi)
                logits = torch.cat(outs, dim=0).view(B, H, W, -1).permute(0, 3, 1, 2)

            else:  # qformer
                z = self._broadcast_ctx_to_pixels(z_ctx, B, H, W)  # (N,T,K)
                x_in = self.encoder_pe(self.input_proj(x))         # (N,T,d)
                z_in = self.ctx_pe(self.ctx_proj(z))               # (N,T,d)

                outs = []
                for i in range(0, N, chunk_size):
                    xi  = x_in[i:i+chunk_size]
                    zii = z_in[i:i+chunk_size]
                    for layer in self.self_layers:
                        xi = layer(xi)
                    for xlayer in self.xattn_layers:
                        xi = xlayer(xi, zii)
                    xi = xi.mean(dim=1)
                    xi = self.dropout(xi)
                    xi = self.output_proj(xi)
                    outs.append(xi)
                logits = torch.cat(outs, dim=0).view(B, H, W, -1).permute(0, 3, 1, 2)  # (B,C,H,W)

            return logits

        else:
            # Pixel-mode: x_img: (N, T, C); z_ctx: (N, T, K)
            assert x_img.dim() == 3, f"Expected (N,T,C), got {tuple(x_img.shape)}"
            N, T, C = x_img.shape
            assert T == self.seq_len, f"seq_len mismatch: model {self.seq_len}, x {T}"
            assert z_ctx is not None and z_ctx.shape[:2] == (N, T), \
                f"z_ctx must be (N,T,K), got {tuple(z_ctx.shape) if z_ctx is not None else None}"

            if self.fusion == 'concat':
                xin = torch.cat([x_img, z_ctx], dim=-1)           # (N,T,C+K)
                xin = self.encoder_pe(self.input_proj(xin))       # (N,T,d)
                xi = self.encoder(xin)                            # (N,T,d)
                xi = xi.mean(dim=1)
                xi = self.dropout(xi)
                logits = self.output_proj(xi)                     # (N,num_classes)


            else:
                x_in = self.encoder_pe(self.input_proj(x_img))    # (N,T,d)
                z_in = self.ctx_pe(self.ctx_proj(z_ctx))          # (N,T,d)
                xi = x_in
                for layer in self.self_layers:
                    xi = layer(xi)
                for xlayer in self.xattn_layers:
                    xi = xlayer(xi, z_in)
                xi = xi.mean(dim=1)
                xi = self.dropout(xi)
                logits = self.output_proj(xi)                     # (N,num_classes)

            return logits
