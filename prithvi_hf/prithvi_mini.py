import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

def get_3d_sincos_pos_embed(embed_dim, grid_t, grid_h, grid_w):
    assert embed_dim % 6 == 0
    def pe_1d(D, N):
        omega = 1.0 / (10000 ** (np.arange(D//2)/ (D/2)))
        pos = np.arange(N)[:, None] * omega[None, :]
        return np.concatenate([np.sin(pos), np.cos(pos)], axis=1)  # (N, D)
    Dt = Dh = Dw = embed_dim // 3
    pt = pe_1d(Dt, grid_t); ph = pe_1d(Dh, grid_h); pw = pe_1d(Dw, grid_w)
    # combine
    pos = []
    for t in range(grid_t):
        for h in range(grid_h):
            for w in range(grid_w):
                pos.append(np.concatenate([pt[t], ph[h], pw[w]], axis=0))
    return torch.tensor(np.stack(pos), dtype=torch.float32)  # (L, E)

class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch, embed_dim, patch=(1,4,4)):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch, stride=patch, bias=False)
    def forward(self, x):
        # x: (B,C,T,H,W)
        x = self.proj(x)                 # (B, E, T', H', W')
        B,E,Tp,Hp,Wp = x.shape
        x = x.flatten(2).transpose(1,2)  # (B, L, E)  where L=Tp*Hp*Wp
        return x, (Tp, Hp, Wp)

class TinyPrithviEncoder(nn.Module):
    def __init__(self, in_ch=6, T=4, img_size=256, patch=(1,4,4),
                 d_model=128, depth=3, nhead=4, dropout=0.1):
        super().__init__()
        self.patch = patch
        self.embed = PatchEmbed3D(in_ch, d_model, patch)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, dropout,
                                         batch_first=True, activation='relu')
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        self.dropout = nn.Dropout(dropout)
        self.img_size = img_size
        self.T = T
        self.d_model = d_model
        self.pos_cache = None
        self.mask_cache = None  # <-- add

    def _temporal_only_mask(self, Tp, Hp, Wp, device, dtype=torch.float32):
        L = Tp*Hp*Wp
        # If cached, reuse
        if (self.mask_cache is not None and
            self.mask_cache.shape == (L, L) and
            self.mask_cache.device == device and
            self.mask_cache.dtype == dtype):
            return self.mask_cache

        # Build additive mask: 0 where allowed, -inf where blocked
        mask = torch.full((L, L), float('-inf'), device=device, dtype=dtype)
        hw = Hp * Wp
        # For each spatial location, allow all t<->t' attentions within that location
        # (loop over spatial positions once; this happens only when grid changes)
        for s in range(hw):
            idxs = s + torch.arange(Tp, device=device) * hw   # positions for this (h,w) across time
            mask[idxs[:, None], idxs[None, :]] = 0.0

        self.mask_cache = mask
        return mask

    def forward(self, x, use_temporal_only_attn=False):
        # x: (B,C,T,H,W)
        B,C,T,H,W = x.shape
        tok, (Tp,Hp,Wp) = self.embed(x)   # (B, L, E), L=Tp*Hp*Wp

        # Positional encoding (unchanged)
        L = Tp*Hp*Wp
        if (self.pos_cache is None) or (self.pos_cache.shape[0] != L or self.pos_cache.shape[1] != self.d_model):
            self.pos_cache = get_3d_sincos_pos_embed(self.d_model, Tp, Hp, Wp).to(tok.device)  # (L,E)
        tok = tok + self.pos_cache.unsqueeze(0)

        mask = None
        if use_temporal_only_attn:
            mask = self._temporal_only_mask(Tp, Hp, Wp, tok.device, tok.dtype)  # (L, L)

        tok = self.encoder(tok, mask=mask)   # <- key change
        tok = self.dropout(tok)
        return tok, (Tp,Hp,Wp)


class TinyPrithviSeg(nn.Module):
    """
    Patchify + full spatiotemporal attention; tiny; trained from scratch.
    """
    def __init__(self, in_ch=6, T=4, img_size=256, patch=(1,4,4),
                 d_model=128, depth=3, nhead=4, num_classes=4, up_depth=3, use_temporal_only_attn=False):
        super().__init__()
        self.enc = TinyPrithviEncoder(in_ch, T, img_size, patch, d_model, depth, nhead)
        self.patch = patch
        self.num_classes = num_classes
        # Light upsampler
        # reconstruct to (B, E*T', H', W') then ConvTranspose2d stack
        self.proj = nn.Identity()  # (optionally a 1x1 conv on channels after reshape)
        ch = d_model * (T // patch[0])
        blocks = []
        c = ch
        for _ in range(up_depth):   # e.g., 3 steps: /8 -> /4 -> /2 -> /1 depending on patch
            blocks += [
                nn.ConvTranspose2d(c, c//2, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(c//2),
                nn.GELU(),
                nn.Conv2d(c//2, c//2, kernel_size=3, padding=1),
                nn.BatchNorm2d(c//2),
                nn.ReLU(inplace=True),
            ]
            c = c//2
        self.up = nn.Sequential(*blocks)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)
        self.use_temporal_only_attn = use_temporal_only_attn

    def forward(self, x):
        # x: (B,C,T,H,W)
        B,C,T,H,W = x.shape
        tok, (Tp,Hp,Wp) = self.enc(x, use_temporal_only_attn=self.use_temporal_only_attn)
        B = x.shape[0]; E = tok.shape[-1]
        feat = tok[:, :Tp*Hp*Wp, :].reshape(B, Tp, Hp, Wp, E)
        feat = rearrange(feat, "b t h w e -> b (t e) h w")
        feat = self.proj(feat)
        out  = self.up(feat)
        logits = self.head(out)
        return logits

