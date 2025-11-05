import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Building blocks ----------

class DoubleConv3d(nn.Module):
    """
    Two 3D convs + Norm + GELU.
    Temporal kernel set by k_t; temporal stride always 1 to preserve T.
    """
    def __init__(self, in_ch, out_ch, k_t=3, norm='bn', dropout=0.0):
        super().__init__()
        padding = (k_t//2, 1, 1)
        Conv = nn.Conv3d
        Norm = {'bn': nn.BatchNorm3d, 'gn': lambda c: nn.GroupNorm(8, c), 'ln': lambda c: nn.GroupNorm(1, c)}[norm]

        self.block = nn.Sequential(
            Conv(in_ch, out_ch, kernel_size=(k_t,3,3), padding=padding, bias=False),
            Norm(out_ch),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            Conv(out_ch, out_ch, kernel_size=(k_t,3,3), padding=padding, bias=False),
            Norm(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Down3d(nn.Module):
    """
    Spatial downsample by 2x (H,W) with stride (1,2,2); keep T unchanged.
    """
    def __init__(self, in_ch, out_ch, k_t=3, norm='bn', dropout=0.0):
        super().__init__()
        self.pool = nn.Conv3d(in_ch, in_ch, kernel_size=(1,2,2), stride=(1,2,2), groups=in_ch, bias=False)
        self.conv = DoubleConv3d(in_ch, out_ch, k_t=k_t, norm=norm, dropout=dropout)

    def forward(self, x):
        x = self.pool(x)     # (B,C,T,H/2,W/2)
        x = self.conv(x)
        return x


class Up3d(nn.Module):
    """
    Spatial upsample by 2x (H,W). Concatenate skip and refine.
    """
    def __init__(self, in_ch, skip_ch, out_ch, k_t=3, norm='bn', dropout=0.0, mode='tr'):
        super().__init__()
        if mode == 'tr':
            self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=(1,2,2), stride=(1,2,2), bias=False)
        else:
            self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        self.conv = DoubleConv3d(in_ch + skip_ch, out_ch, k_t=k_t, norm=norm, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # pad spatially if odd dims
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, max(dw,0), 0, max(dh,0)))
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# ---------- The model ----------

class UNet3DTimeAware(nn.Module):
    """
    Simple time-aware U-Net:
      - Input:  (B, C, T, H, W)
      - 3D convolutions mix across time (kernel k_t) but never downsample time (temporal stride = 1)
      - Spatial pyramid: /2, /4, /8, /16 (configurable depth)
      - Output: (B, num_classes, H, W), via time_pool ('mean'/'max'/'conv')
    """
    def __init__(
        self,
        in_ch=6,
        num_classes=4,
        base_ch=32,
        depth=4,            # number of downs (=> /2^depth)
        k_t=3,              # temporal kernel size (1=no temporal mixing; 3=light mixing)
        norm='bn',
        dropout=0.0,
        time_pool='mean',   # 'mean' | 'max' | 'conv' (learned 1x1 temporal collapse)
    ):
        super().__init__()
        assert time_pool in ['mean', 'max', 'conv']

        chs = [base_ch * (2**i) for i in range(depth+1)]  # encoder widths
        # Encoder
        self.inc = DoubleConv3d(in_ch, chs[0], k_t=k_t, norm=norm, dropout=dropout)
        self.downs = nn.ModuleList([
            Down3d(chs[i], chs[i+1], k_t=k_t, norm=norm, dropout=dropout)
            for i in range(depth)
        ])

        # Decoder
        self.ups = nn.ModuleList([
            Up3d(chs[i+1], chs[i], chs[i], k_t=k_t, norm=norm, dropout=dropout)
            for i in reversed(range(depth))
        ])

        # Temporal collapse head (produces class logits over (H,W))
        self.time_pool = time_pool
        if time_pool == 'conv':
            # learn a 1x1x1 Conv on channels after collapsing time via 1x1xk_t (conv along T only)
            self.tproj = nn.Conv3d(chs[0], chs[0], kernel_size=(k_t,1,1), padding=(k_t//2,0,0), bias=False)
            self.head = nn.Conv2d(chs[0], num_classes, kernel_size=1)
        else:
            self.head = nn.Conv2d(chs[0], num_classes, kernel_size=1)

        # final 1x1x1 to prepare for 2D head if needed
        self.fin = nn.Identity()

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        returns: (B, num_classes, H, W)
        """
        # Encoder
        x0 = self.inc(x)            # (B, ch0, T, H, W)
        skips = [x0]
        xi = x0
        for down in self.downs:
            xi = down(xi)
            skips.append(xi)

        # Decoder
        x = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-(i+2)]
            x = up(x, skip)

        # Temporal collapse
        # x is (B, ch0, T, H, W)
        if self.time_pool == 'mean':
            x2d = x.mean(dim=2)                   # (B, ch0, H, W)
        elif self.time_pool == 'max':
            x2d, _ = x.max(dim=2)                 # (B, ch0, H, W)
        else:  # 'conv'
            x = self.tproj(x)                     # (B, ch0, T, H, W)
            x2d = x.mean(dim=2)                   # (B, ch0, H, W)

        logits = self.head(x2d)                   # (B, num_classes, H, W)
        return logits
