import torch
import torch.nn as nn

class GlobalTemporalConvNet(nn.Module):
    """
    Full-sequence temporal summarizer (k_t = T, no padding) + pointwise residual head.

    Input:  (B, C, T, H, W)  with T == t_steps
    Output: (B, num_classes, H, W)

    Steps:
      1) Depthwise Conv3d with kernel=(T,1,1), padding=0  -> (B, C, 1, H, W)
      2) 1x1x1 conv to hidden width + n pointwise residual blocks at T=1
      3) 1x1x1 conv to num_classes, squeeze time
    """
    def __init__(self,
                 in_ch: int = 6,
                 num_classes: int = 4,
                 t_steps: int = 12,      # <-- must equal input T
                 hidden: int = 96,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 use_groupnorm_if_small_batch: bool = False):
        super().__init__()
        self.t_steps = t_steps

        # Norm for input
        if use_groupnorm_if_small_batch:
            # One group per channel ~ LayerNorm over (T,H,W) per channel
            self.in_norm = nn.GroupNorm(num_groups=in_ch, num_channels=in_ch)
        else:
            self.in_norm = nn.BatchNorm3d(in_ch)

        # Global temporal summarizer: depthwise along T, no spatial mixing.
        # kernel size equals full sequence length; no padding.
        self.dw_temporal = nn.Conv3d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(t_steps, 1, 1),
            padding=(0, 0, 0),
            groups=in_ch,
            bias=False
        )

        # Stem to hidden width (still at T=1 after the global temporal conv)
        self.stem_pw = nn.Conv3d(in_ch, hidden, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # Pointwise residual blocks (operate at T=1, HxW per-pixel, no spatial mixing)
        blocks = []
        for _ in range(n_layers):
            blocks.append(PointwiseResidualBlock(hidden, dropout=dropout,
                                                 use_groupnorm_if_small_batch=use_groupnorm_if_small_batch))
        self.blocks = nn.Sequential(*blocks)

        # Head to logits
        if use_groupnorm_if_small_batch:
            self.head_norm = nn.GroupNorm(num_groups=hidden, num_channels=hidden)
        else:
            self.head_norm = nn.BatchNorm3d(hidden)

        self.out = nn.Conv3d(hidden, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        assert T == self.t_steps, f"Expected T={self.t_steps}, got T={T}."

        x = self.in_norm(x)
        x = self.dw_temporal(x)      # -> (B, C, 1, H, W)
        x = self.stem_pw(self.act(x))
        x = self.drop(x)

        x = self.blocks(x)           # stays (B, hidden, 1, H, W)

        x = self.out(self.act(self.head_norm(x)))  # (B, num_classes, 1, H, W)
        x = x.squeeze(dim=2)         # (B, num_classes, H, W)
        return x


class PointwiseResidualBlock(nn.Module):
    """
    Residual MLP-like block using only 1x1x1 convs (no temporal/spatial mixing).
    Operates at shape (B, C, 1, H, W).
    """
    def __init__(self, channels: int, dropout: float = 0.1, use_groupnorm_if_small_batch: bool = False):
        super().__init__()
        if use_groupnorm_if_small_batch:
            self.norm = nn.GroupNorm(num_groups=channels, num_channels=channels)
        else:
            self.norm = nn.BatchNorm3d(channels)
        self.pw1 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.pw2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(x))
        h = self.pw1(h)
        h = self.drop(self.act(h))
        h = self.pw2(h)
        h = self.drop(h)
        return x + h
