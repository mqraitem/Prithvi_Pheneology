import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (N, T, d_model)
        return x + self.pe[:, :x.size(1), :]


def _edge_cover_starts(L: int, p: int):
    """
    Non-overlap starts (0, p, 2p, ...) plus, if needed, one extra start at L - p
    so the last patch touches the edge (minimal overlap). If L <= p -> [0].
    """
    if L <= p:
        return [0]
    starts = list(range(0, L - p + 1, p))
    last = L - p
    if starts[-1] != last:
        starts.append(last)   # edge-only overlap
    return starts


class TemporalTransformerPerPatch(nn.Module):
    """
    Temporal-only transformer that operates per patch.
    - processing_images=True: (B, C, T, H, W) -> (B, num_classes, H, W)
      * Interior: stride=patch (no overlap)
      * Edge-only overlap when H or W not divisible by patch size (last start at L-p)
      * Edge patches overwrite any tiny overlap from previous patch
    - processing_images=False: (B, T, C, H, W) where (H,W)==patch -> (B, num_classes, H, W)
    """
    def __init__(
        self,
        input_channels=6,
        seq_len=4,
        num_classes=4,
        d_model=256,
        nhead=8,
        num_layers=3,
        dropout=0.1,
        patch_size=(32, 32),
        pad_mode="replicate",   # used only if image smaller than patch
        pad_value=0.0,
    ):
        super().__init__()
        self.C = input_channels
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.ph, self.pw = patch_size
        self.pad_mode = pad_mode
        self.pad_value = pad_value

        patch_feat = input_channels * self.ph * self.pw  # feature length per time step
        self.input_proj = nn.Linear(patch_feat, d_model, bias=False)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, num_classes)

    # ---- internals ----
    def _temporal_patch_encode(self, x_seq, chunk_size=2048):
        """
        x_seq: (Npatches, T, C*ph*pw) -> (Npatches, num_classes)
        """
        x_seq = self.input_proj(x_seq)         # (N, T, d_model)
        x_seq = self.pos_encoder(x_seq)        # (N, T, d_model)
        outs = []
        N = x_seq.size(0)
        for i in range(0, N, chunk_size):
            z = self.encoder(x_seq[i:i+chunk_size])  # (chunk, T, d_model)
            z = z.mean(dim=1)                        # temporal pooling
            z = self.dropout(z)
            outs.append(self.output_proj(z))         # (chunk, K)
        return torch.cat(outs, dim=0)

    def _pad_min_if_needed(self, x):
        """
        If image smaller than patch, minimally pad H/W to ph/pw.
        For replicate/reflect, pad per-time-slice to avoid F.pad limitation on 5D tensors.
        Returns x_pad, (H0, W0), padded_flag
        """
        B, C, T, H, W = x.shape
        need_h = max(0, self.ph - H)
        need_w = max(0, self.pw - W)
        if need_h == 0 and need_w == 0:
            return x, (H, W), False

        if self.pad_mode == "constant":
            x = F.pad(x, (0, need_w, 0, need_h), mode="constant", value=self.pad_value)
        else:
            # pad each time slice (B, C, H, W)
            slices = []
            for t in range(T):
                xt = x[:, :, t, :, :]  # (B, C, H, W)
                xt = F.pad(xt, (0, need_w, 0, need_h), mode=self.pad_mode)
                slices.append(xt.unsqueeze(2))
            x = torch.cat(slices, dim=2)
        return x, (H, W), True

    # ---- forward ----
    def forward(self, x, processing_images=True, chunk_size=2048):
        if processing_images:
            # x: (B, C, T, H, W)
            assert x.dim() == 5, "processing_images=True expects (B, C, T, H, W)"
            B, C, T, H, W = x.shape
            assert C == self.C, f"input_channels mismatch: {C} vs {self.C}"
            assert T == self.seq_len, f"T={T} vs seq_len={self.seq_len}"

            # Only pad if smaller than a single patch
            x, (H0, W0), padded_small = self._pad_min_if_needed(x)
            _, _, _, Huse, Wuse = x.shape

            # compute starts with edge-only overlap
            starts_h = _edge_cover_starts(Huse, self.ph)
            starts_w = _edge_cover_starts(Wuse, self.pw)

            # output canvas; we will write directly; edge patches overwrite overlaps
            out = x.new_zeros((B, self.num_classes, Huse, Wuse))

            # iterate patches; edge-only overlap ensured by starts_h/starts_w
            for top in starts_h:
                for left in starts_w:
                    patch = x[:, :, :, top:top+self.ph, left:left+self.pw]       # (B, C, T, ph, pw)
                    seq = patch.permute(0, 2, 1, 3, 4).reshape(B, T, -1)         # (B, T, C*ph*pw)
                    logits = self._temporal_patch_encode(seq, chunk_size=chunk_size)  # (B, K)
                    logits_map = logits.view(B, -1, 1, 1).expand(B, -1, self.ph, self.pw)
                    # overwrite assignment (edge patches naturally processed last in that axis)
                    out[:, :, top:top+self.ph, left:left+self.pw] = logits_map

            # crop back if we minimally padded tiny images
            if padded_small:
                out = out[:, :, :H0, :W0]
            return out  # (B, K, H, W)

        # processing_images == False
        # x: (B, T, C, H, W) where (H,W) == patch_size
        assert x.dim() == 5, "processing_images=False expects (B, T, C, H, W)"
        B, T, C, H, W = x.shape
        assert C == self.C, f"input_channels mismatch: {C} vs {self.C}"
        assert T == self.seq_len, f"T={T} vs seq_len={self.seq_len}"
        assert (H, W) == (self.ph, self.pw), f"Patch size mismatch: ({H},{W}) vs ({self.ph},{self.pw})"

        seq = x.reshape(B, T, C * H * W)                      # (B, T, C*H*W)
        logits = self._temporal_patch_encode(seq, chunk_size=chunk_size)  # (B, K)
        return logits.view(B, -1, 1, 1).expand(B, -1, H, W)   # (B, K, H, W)
