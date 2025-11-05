import torch
import torch.nn as nn

class PointwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1, use_groupnorm_if_small_batch: bool = False):
        super().__init__()
        if use_groupnorm_if_small_batch:
            # one group per channel (works like per-channel norm)
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


class GlobalTemporalConvNetFlexible(nn.Module):
    """
    Two-mode interface:

    - processing_image=True:
        Input:  (B, C, T, H, W)
        Output: (B, num_classes, H, W)

    - processing_image=False:
        Input:  (B, T, C)     # per-pixel/time series batches
        Output: (B, num_classes)

    Assumes k_t = T (full-sequence temporal conv) with NO padding.
    """
    def __init__(self,
                 in_ch: int = 6,
                 num_classes: int = 4,
                 t_steps: int = 12,      # must match input T
                 hidden: int = 96,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 use_groupnorm_if_small_batch: bool = False):
        super().__init__()
        self.t_steps = t_steps

        # Shared norms
        if use_groupnorm_if_small_batch:
            self.in_norm = nn.GroupNorm(num_groups=in_ch, num_channels=in_ch)
            self.head_norm = nn.GroupNorm(num_groups=hidden, num_channels=hidden)
        else:
            self.in_norm = nn.BatchNorm3d(in_ch)
            self.head_norm = nn.BatchNorm3d(hidden)

        # Global temporal summarizer: kernel=(T,1,1), padding=0, depthwise
        self.dw_temporal = nn.Conv3d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(t_steps, 1, 1),
            padding=(0, 0, 0),
            groups=in_ch,
            bias=False
        )

        # Stem + residual pointwise stack
        self.stem_pw = nn.Conv3d(in_ch, hidden, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            PointwiseResidualBlock(hidden, dropout=dropout,
                                   use_groupnorm_if_small_batch=use_groupnorm_if_small_batch)
            for _ in range(n_layers)
        ])

        self.out = nn.Conv3d(hidden, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, processing_image: bool = True) -> torch.Tensor:
        if processing_image:
            # x: (B, C, T, H, W) -> (B, num_classes, H, W)
            B, C, T, H, W = x.shape
            assert T == self.t_steps, f"Expected T={self.t_steps}, got T={T}."
            x = self.in_norm(x)
            x = self.dw_temporal(x)                         # (B, C, 1, H, W)
            x = self.stem_pw(self.act(x))
            x = self.drop(x)
            x = self.blocks(x)                              # (B, hidden, 1, H, W)
            x = self.out(self.act(self.head_norm(x)))       # (B, num_classes, 1, H, W)
            return x.squeeze(dim=2)                         # (B, num_classes, H, W)
        else:
            # x: (B, T, C) -> (B, num_classes)
            B, T, C = x.shape
            assert T == self.t_steps, f"Expected T={self.t_steps}, got T={T}."
            # Reorder to (B, C, T) then reshape to (B, C, T, 1, 1) to reuse Conv3d stack
            x = x.permute(0, 2, 1).contiguous().view(B, C, T, 1, 1)
            x = self.in_norm(x)
            x = self.dw_temporal(x)                         # (B, C, 1, 1, 1)
            x = self.stem_pw(self.act(x))
            x = self.drop(x)
            x = self.blocks(x)                              # (B, hidden, 1, 1, 1)
            x = self.out(self.act(self.head_norm(x)))       # (B, num_classes, 1, 1, 1)
            return x.view(B, -1)                            # (B, num_classes)
