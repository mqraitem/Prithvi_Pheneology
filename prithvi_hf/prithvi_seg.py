# file: hybrid_prithvi_pixel_from_cfg.py
# -----------------------------------------------------------------------------
# Hybrid pixel-first + Prithvi-ViT segmentation model (config-driven)
# -----------------------------------------------------------------------------
# Usage:
#   from hybrid_prithvi_pixel_from_cfg import HybridPrithviPixelSeg, build_model_from_config
#
#   cfg = <your JSON dict>              # the one you pasted (python-loaded)
#   pretrained_cfg = cfg["pretrained_cfg"]
#   weights_path = pretrained_cfg.get("prithvi_model_new_weight", None)  # or pass your own
#
#   model = HybridPrithviPixelSeg(pretrained_cfg, weights_path, n_classes=4)
#   # or:
#   model = build_model_from_config(cfg, n_classes=4)
#
# Forward:
#   x: torch.Tensor of shape (B, C, T, H, W)   # match pretrained_cfg fields
#   logits = model(x)
#
# Notes:
# - We REPLACE the original Conv3d PatchEmbed with:
#     (A) a per-pixel Temporal Transformer → (B, D, T, H, W)
#     (B) a patchifier that packs to tokens matching Prithvi's expected L = T * (H/P) * (W/P)
# - We KEEP Prithvi's transformer encoder blocks + CLS + LayerNorm and load their weights.
# - We DROP the original patch_embed.* (and pos_embed if grid differs) when loading the checkpoint.
# - Set `encoder_only=True` in your config (as in your JSON) — we do not use the MAE decoder here.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the ORIGINAL Prithvi code pieces from your local file
# Assumes your original file is available as prithvi_mae.py in the PYTHONPATH.
from prithvi_hf.prithvi_mae import (
    PrithviMAE,                   # to load checkpoint & reuse encoder blocks
    get_3d_sincos_pos_embed,      # 3D sin/cos positional embeddings
    TemporalEncoder,              # optional temporal coords encoding
    LocationEncoder,              # optional location coords encoding
)

# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#

def _to_2tuple(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 2
        return (int(x[0]), int(x[1]))
    return (int(x), int(x))


# -----------------------------------------------------------------------------#
# 1) Pixel-wise temporal encoder (feature mode)
# -----------------------------------------------------------------------------#

class PositionalEncoding1D(nn.Module):
    """Standard 1D sinusoidal PE for sequences (T)."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, L, D)
        return x + self.pe[:, :x.size(1), :]


class PixelTemporalEncoder(nn.Module):
    """
    Per-pixel temporal encoder based on a Transformer encoder.
    Returns features (not logits). Can return per-time features for better temporal use downstream.
    """
    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        d_out: int = 256,
        dropout: float = 0.1,
        return_per_time: bool = True,
        chunk_size: int = 2000,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.return_per_time = return_per_time
        self.chunk_size = chunk_size

        self.input_proj = nn.Linear(in_channels, d_model, bias=False)
        self.pos_encoding = PositionalEncoding1D(d_model, max_len=seq_len)

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
        self.output_proj = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        returns:
          if return_per_time=True: (B, d_out, T, H, W)
          else:                    (B, d_out, H, W)
        """
        B, C, T, H, W = x.shape
        assert T == self.seq_len, f"PixelTemporalEncoder configured for seq_len={self.seq_len}, got T={T}"

        # (B, C, T, H, W) -> (B, H, W, T, C) -> (N, T, C) with N=B*H*W
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        N = B * H * W
        seq = x.view(N, T, C)

        seq = self.input_proj(seq)
        seq = self.pos_encoding(seq)


        outs = []
        for i in range(0, N, self.chunk_size):
            s = seq[i:i + self.chunk_size]  # (n, T, d_model)
            s = self.encoder(s)             # (n, T, d_model)

            if self.return_per_time:
                s = self.output_proj(self.dropout(s))  # (n, T, d_out)
            else:
                s = self.output_proj(self.dropout(s.mean(dim=1)))  # (n, d_out)
            outs.append(s)

        out = torch.cat(outs, dim=0)

        if self.return_per_time:
            # (N, T, d_out) -> (B, d_out, T, H, W)
            out = out.view(B, H, W, T, -1).permute(0, 4, 3, 1, 2).contiguous()
        else:
            # (N, d_out) -> (B, d_out, H, W)
            out = out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return out


# -----------------------------------------------------------------------------#
# 2) Pixel patchifier: pack per-time pixel features into Prithvi tokens
# -----------------------------------------------------------------------------#

class PixelPatchifier(nn.Module):
    """
    Converts high-res pixel features into Prithvi-style patch tokens.
    - Accepts features per-time: (B, D, T, H, W)   [preferred]
      or time-aggregated:        (B, D, H, W) with T provided.
    - Downsamples with exact spatial patch stride (P) and projects to embed_dim (E).
    - Returns tokens of shape (B, L, E) with L = T * (H/P) * (W/P).
    """
    def __init__(self, patch: int, d_in: int, embed_dim: int, use_conv_pool: bool = True):
        super().__init__()
        self.patch = int(patch)
        self.d_in = d_in
        self.embed_dim = embed_dim
        if use_conv_pool:
            self.pool = nn.Conv2d(d_in, d_in, kernel_size=self.patch, stride=self.patch, bias=False)
        else:
            self.pool = None
        self.proj = nn.Conv2d(d_in, embed_dim, kernel_size=1, bias=False)

    def _to_patch_grid(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W = x.shape
        assert H % self.patch == 0 and W % self.patch == 0, \
            f"H,W must be divisible by patch={self.patch}, got H={H}, W={W}"
        if self.pool is not None:
            y = self.pool(x)
        else:
            y = F.avg_pool2d(x, kernel_size=self.patch, stride=self.patch)
        return y  # (B, D, H', W')

    def patchify_one(self, f_t: torch.Tensor) -> torch.Tensor:
        """f_t: (B, D, H, W) -> tokens_t: (B, H'*W', E)"""
        grid = self._to_patch_grid(f_t)
        grid = self.proj(grid)  # (B, E, H', W')
        B, E, Hp, Wp = grid.shape
        tokens_t = grid.flatten(2).transpose(1, 2).contiguous()  # (B, H'*W', E)
        return tokens_t

    def forward(self, feats: torch.Tensor, T: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        feats:
          - (B, D, T, H, W)  OR
          - (B, D, H, W) with T provided
        returns:
          tokens: (B, L, E), where L = T * H'*W'
          (H', W') patch-grid size
        """
        if feats.dim() == 5:
            B, D, TT, H, W = feats.shape
            assert TT == T, f"Provided T={T} but feats carries T={TT}"
            seq = []
            for t in range(T):
                tokens_t = self.patchify_one(feats[:, :, t, :, :])  # (B, H'*W', E)
                seq.append(tokens_t)
            tokens = torch.cat(seq, dim=1)  # (B, T*H'*W', E)
            Hp, Wp = H // self.patch, W // self.patch
        else:
            B, D, H, W = feats.shape
            tokens_once = self.patchify_one(feats)
            tokens = tokens_once.repeat(1, T, 1)  # tile across time (weaker temporal cues)
            Hp, Wp = H // self.patch, W // self.patch

        return tokens, (Hp, Wp)


# -----------------------------------------------------------------------------#
# 3) Prithvi encoder core (reuse pretrained encoder blocks, no PatchEmbed)
# -----------------------------------------------------------------------------#

class PrithviEncoderCore(nn.Module):
    """
    Wraps Prithvi's pretrained encoder blocks, but accepts pre-built tokens instead
    of using the original Conv3d PatchEmbed.

    - Builds fresh 3D sin/cos pos_embed for the current (T, H', W') grid.
    - Optionally adds TemporalEncoder and LocationEncoder embeddings.
    - Returns encoded tokens including CLS: (B, 1+L, E)
    """
    def __init__(self, prithvi_params: Dict[str, Any], prithvi_ckpt_path: Optional[str] = None):
        super().__init__()
        self.params = dict(prithvi_params)  # shallow copy

        # Ensure required fields exist for constructing PrithviMAE
        defaults = {
            "img_size": 224,
            "patch_size": (1, 16, 16),
            "num_frames": 1,
            "in_chans": 3,
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "decoder_embed_dim": 512,
            "decoder_depth": 8,
            "decoder_num_heads": 16,
            "mlp_ratio": 4.0,
            "norm_layer": nn.LayerNorm,
            "norm_pix_loss": False,
            "coords_encoding": [],
            "coords_scale_learn": False,
            "encoder_only": True,
        }
        for k, v in defaults.items():
            self.params.setdefault(k, v)

        mae = PrithviMAE(**self.params)
        if prithvi_ckpt_path:
            ckpt = torch.load(prithvi_ckpt_path, weights_only=False)
            # Handle nested state dicts
            if "encoder.pos_embed" not in ckpt.keys():
                key = "model" if "model" in ckpt else "state_dict"
                ckpt = ckpt[key]

            # Drop patch_embed.* and pos_embed.* (we rebuild pos per grid)
            for k in list(ckpt.keys()):
                if "patch_embed" in k or "pos_embed" in k:
                    del ckpt[k]
            mae.load_state_dict(ckpt, strict=False)

        # Keep encoder parts
        self.blocks = mae.encoder.blocks
        self.cls_token = mae.encoder.cls_token
        self.norm = mae.encoder.norm

        # Coord encodings (optional)
        coords_encoding = self.params.get("coords_encoding", []) or []
        self.use_temporal = ("time" in coords_encoding)
        self.use_location = ("location" in coords_encoding)

        embed_dim = self.params["embed_dim"]
        coords_scale_learn = self.params.get("coords_scale_learn", False)
        self.embed_dim = embed_dim

        self.temporal_enc = TemporalEncoder(embed_dim, coords_scale_learn) if self.use_temporal else None
        self.location_enc = LocationEncoder(embed_dim, coords_scale_learn) if self.use_location else None

    @torch.no_grad()
    def _pos_embed_for_grid(self, grid_size: Tuple[int, int, int], device, dtype) -> torch.Tensor:
        """grid_size: (T, H', W') -> (1, 1+L, E)"""
        pos = get_3d_sincos_pos_embed(
            embed_dim=self.embed_dim,
            grid_size=grid_size,
            add_cls_token=True,
        )
        pos = torch.from_numpy(pos).to(device=device, dtype=dtype).unsqueeze(0)  # (1, 1+L, E)
        return pos

    def forward(
        self,
        tokens: torch.Tensor,                               # (B, L, E)
        grid_size: Tuple[int, int, int],                    # (T, H', W')
        temporal_coords: Optional[torch.Tensor] = None,     # (B, T, 2)
        location_coords: Optional[torch.Tensor] = None,     # (B, 2)
    ) -> torch.Tensor:
        B, L, E = tokens.shape
        T, Hp, Wp = grid_size
        assert E == self.embed_dim, f"Token dim E={E} must equal Prithvi embed_dim={self.embed_dim}"
        assert L == T * Hp * Wp, f"L={L} must equal T*H'*W'={T*Hp*Wp}"

        device, dtype = tokens.device, tokens.dtype

        # Position embeddings
        pos = self._pos_embed_for_grid(grid_size, device, dtype)  # (1, 1+L, E)

        x = tokens + pos[:, 1:, :]

        # Optional temporal & location encodings
        if self.use_temporal:
            tpf = Hp * Wp
            te = self.temporal_enc(temporal_coords, tokens_per_frame=tpf)  # (B, L, E)
            x = x + te
        if self.use_location:
            le = self.location_enc(location_coords)  # (B, 1, E)
            x = x + le.expand(-1, L, -1)

        # Prepend CLS with its pos emb
        cls = self.cls_token + pos[:, :1, :]
        cls = cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+L, E)

        # Transformer encoder
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # (B, 1+L, E)


# -----------------------------------------------------------------------------#
# 4) Token -> patch-grid features
# -----------------------------------------------------------------------------#

class TokenToFeatureMap(nn.Module):
    """
    Reshape encoded tokens back to a (B, C, H', W') grid.
    time_agg='mean'  → C = E
    time_agg='concat'→ C = E * T
    """
    def __init__(self, embed_dim: int, time_agg: str = "mean"):
        super().__init__()
        assert time_agg in ("mean", "concat")
        self.embed_dim = embed_dim
        self.time_agg = time_agg

    def forward(self, tokens_enc: torch.Tensor, T: int, Hp: int, Wp: int) -> torch.Tensor:
        """
        tokens_enc: (B, 1+L, E), L = T*Hp*Wp
        returns: (B, Cg, Hp, Wp)
        """
        B, Lp1, E = tokens_enc.shape
        assert E == self.embed_dim
        L = Lp1 - 1
        assert L == T * Hp * Wp, "Mismatched token length and grid size"
        x = tokens_enc[:, 1:, :]                     # drop CLS
        x = x.view(B, T, Hp, Wp, E)                  # (B, T, Hp, Wp, E)

        if self.time_agg == "mean":
            x = x.mean(dim=1)                        # (B, Hp, Wp, E)
            x = x.permute(0, 3, 1, 2).contiguous()   # (B, E, Hp, Wp)
        else:
            x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, E, T, Hp, Wp)
            x = x.reshape(B, E * T, Hp, Wp)            # (B, E*T, Hp, Wp)
        return x


# -----------------------------------------------------------------------------#
# 5) Lightweight decoder / upsampler
# -----------------------------------------------------------------------------#

class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout(0.1) if dropout else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        return self.block(x)


class LightDecoder(nn.Module):
    """
    N-stage 2x upsampler to go from (Hp, Wp) back to (H, W),
    where up_factor = 2^num_stages should equal the spatial patch size P.
    """
    def __init__(self, ch_in: int, n_classes: int, num_stages: int, base_out: int = 512, dropout: bool = True):
        super().__init__()
        chs = [ch_in]
        for i in range(num_stages):
            chs.append(max(base_out // (2 ** i), n_classes))

        self.up = nn.Sequential(*[UpBlock(chs[i], chs[i + 1], dropout=dropout) for i in range(num_stages)])
        self.head = nn.Conv2d(chs[-1], n_classes, kernel_size=1)

    def forward(self, x):
        x = self.up(x)
        x = self.head(x)
        return x


# -----------------------------------------------------------------------------#
# 6) Orchestrating model (CONFIG-DRIVEN)
# -----------------------------------------------------------------------------#

class HybridPrithviPixelSeg(nn.Module):
    """
    Config-driven hybrid segmentation model.

    Construct with:
      HybridPrithviPixelSeg(cfg["pretrained_cfg"], weights_path, n_classes=4)

    Required keys in pretrained_cfg (from your JSON):
      - img_size: int
      - num_frames: int
      - patch_size: [t, p, p] (we use spatial p)
      - in_chans, embed_dim, depth, num_heads, decoder_* (for instantiation)
      - coords_encoding ([], or include 'time'/'location' if you plan to use coords)
      - encoder_only: True   (we use only encoder blocks)
    """
    def __init__(
        self,
        pretrained_cfg: Dict[str, Any],
        weights_path: Optional[str],
        n_classes: int = 1,
        # Optional overrides / knobs:
        pixel_d_model: int = 128,
        pixel_d_out: int = 256,
        pixel_layers: int = 1,
        pixel_nhead: int = 4,
        return_per_time: bool = True,
        time_agg: str = "mean",                 # 'mean' or 'concat'
        decoder_base_out: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Pull core fields from config
        img_size = pretrained_cfg.get("img_size", 224)
        H, W = _to_2tuple(img_size)
        num_frames = int(pretrained_cfg.get("num_frames", 1))
        in_chans = int(pretrained_cfg.get("in_chans", 3))
        embed_dim = int(pretrained_cfg.get("embed_dim", 1024))

        patch_size_cfg = pretrained_cfg.get("patch_size", [1, 16, 16])
        assert isinstance(patch_size_cfg, (list, tuple)) and len(patch_size_cfg) == 3
        _, p_h, p_w = int(patch_size_cfg[0]), int(patch_size_cfg[1]), int(patch_size_cfg[2])
        assert p_h == p_w, "This implementation assumes square spatial patch; got patch_size={}".format(patch_size_cfg)
        patch = p_h

        assert (H % patch == 0) and (W % patch == 0), f"img_size {H}x{W} must be divisible by patch={patch}"

        # Build prithvi_params for PrithviMAE instantiation (we reuse encoder blocks)
        prithvi_params = dict(pretrained_cfg)
        # Ensure types match expected constructor
        prithvi_params["patch_size"] = tuple(patch_size_cfg)
        prithvi_params["img_size"] = H  # Prithvi accepts int or tuple; keep int if square
        prithvi_params["norm_layer"] = nn.LayerNorm  # ensure callable, not string

        # ---------------- 1) Pixel-wise temporal encoder ----------------
        self.pixel_encoder = PixelTemporalEncoder(
            in_channels=in_chans,
            seq_len=num_frames,
            d_model=pixel_d_model,
            nhead=pixel_nhead,
            num_layers=pixel_layers,
            d_out=pixel_d_out,
            return_per_time=return_per_time,
            dropout=dropout,
        )

        # ---------------- 2) Patchifier to Prithvi tokens ----------------
        self.patchifier = PixelPatchifier(patch=patch, d_in=pixel_d_out, embed_dim=embed_dim, use_conv_pool=True)

        # ---------------- 3) Prithvi encoder core (pretrained blocks) ----
        self.prithvi_core = PrithviEncoderCore(prithvi_params, weights_path)

        # ---------------- 4) Token -> grid features ----------------------
        self.grid_proj = TokenToFeatureMap(embed_dim=embed_dim, time_agg=time_agg)

        # ---------------- 5) Decoder to full res -------------------------
        # We expect (2 ** stages) == spatial patch size
        stages = int(math.ceil(math.log2(patch)))
        ch_in_grid = embed_dim if time_agg == "mean" else embed_dim * num_frames
        self.decoder = LightDecoder(
            ch_in=ch_in_grid,
            n_classes=n_classes,
            num_stages=stages,
            base_out=decoder_base_out,
            dropout=dropout,
        )

        # Save a few attrs for assertions
        self.n_classes = n_classes
        self.num_frames = num_frames
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.img_size = (H, W)
        self.patch = patch
        self.time_agg = time_agg

        # (Optional) register mean/std for reference (not used internally)
        mean = pretrained_cfg.get("mean", None)
        std = pretrained_cfg.get("std", None)
        if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)) and len(mean) == in_chans and len(std) == in_chans:
            self.register_buffer("data_mean", torch.tensor(mean).view(1, in_chans, 1, 1), persistent=False)
            self.register_buffer("data_std", torch.tensor(std).view(1, in_chans, 1, 1), persistent=False)
        else:
            self.data_mean = None
            self.data_std = None

    def forward(
        self,
        x: torch.Tensor,                                 # (B, C, T, H, W)
        temporal_coords: Optional[torch.Tensor] = None,  # (B, T, 2) optional
        location_coords: Optional[torch.Tensor] = None,  # (B, 2)   optional
    ) -> torch.Tensor:
        B, C, T, H, W = x.shape
        assert C == self.in_chans, f"in_chans={self.in_chans}, got C={C}"
        assert T == self.num_frames, f"num_frames={self.num_frames}, got T={T}"
        assert (H, W) == self.img_size, f"Expected img_size={self.img_size}, got {(H, W)}"

        # 1) Pixel-wise temporal features (keeps pixel fidelity)
        feats = self.pixel_encoder(x)  # (B, D, T, H, W) or (B, D, H, W)


        # 2) Patchify -> tokens
        tokens, (Hp, Wp) = self.patchifier(feats, T=T)  # (B, L, E), L = T*Hp*Wp


        # 3) Prithvi blocks
        tokens_enc = self.prithvi_core(
            tokens=tokens,
            grid_size=(T, Hp, Wp),
            temporal_coords=temporal_coords,
            location_coords=location_coords,
        )  # (B, 1+L, E)


        # 4) Tokens -> grid features
        grid_feats = self.grid_proj(tokens_enc, T=T, Hp=Hp, Wp=Wp)  # (B, Cg, Hp, Wp)

        # 5) Decode to full resolution
        logits = self.decoder(grid_feats)  # (B, n_classes, H, W)
        return logits


# -----------------------------------------------------------------------------#
# 7) Convenience builder from your full JSON config
# -----------------------------------------------------------------------------#

def build_model_from_config(cfg: Dict[str, Any], n_classes: int = 1) -> HybridPrithviPixelSeg:
    """
    Helper to build the model directly from your JSON-like dict.
      cfg: the entire JSON dict (the one you pasted)
      n_classes: override number of classes (default 1)

    Returns a HybridPrithviPixelSeg instance.
    """
    pretrained_cfg = dict(cfg["pretrained_cfg"])  # shallow copy
    weights_path = pretrained_cfg.get("prithvi_model_new_weight", None)

    # You can override pixel encoder sizes here if you want (optional):
    pixel_d_model = 128
    pixel_d_out = 256
    pixel_layers = 3
    pixel_nhead = 4

    model = HybridPrithviPixelSeg(
        pretrained_cfg=pretrained_cfg,
        weights_path=weights_path,
        n_classes=n_classes,
        pixel_d_model=pixel_d_model,
        pixel_d_out=pixel_d_out,
        pixel_layers=pixel_layers,
        pixel_nhead=pixel_nhead,
        return_per_time=True,
        time_agg="mean",         # try "concat" later for stronger temporal signal
        decoder_base_out=512,
        dropout=True,
    )
    return model


# -----------------------------------------------------------------------------#
# 8) Smoke test (run this file directly)
# -----------------------------------------------------------------------------#

# if __name__ == "__main__":
#     torch.set_grad_enabled(False)

#     # Minimal config resembling your JSON
#     cfg = {
#         "pretrained_cfg": {
#             "img_size": 224,
#             "num_frames": 12,
#             "patch_size": [1, 16, 16],
#             "in_chans": 6,
#             "embed_dim": 1024,
#             "depth": 24,
#             "num_heads": 16,
#             "decoder_embed_dim": 512,
#             "decoder_depth": 8,
#             "decoder_num_heads": 16,
#             "mlp_ratio": 4.0,
#             "coords_encoding": [],
#             "coords_scale_learn": False,
#             "mask_ratio": 0.75,
#             "norm_pix_loss": False,
#             "encoder_only": True,
#             # "prithvi_model_new_weight": "/path/to/Prithvi_EO_V2_300M.pt",
#         }
#     }

#     model = build_model_from_config(cfg, n_classes=4).eval()

#     B, C, T, H, W = 1, cfg["pretrained_cfg"]["in_chans"], cfg["pretrained_cfg"]["num_frames"], cfg["pretrained_cfg"]["img_size"], cfg["pretrained_cfg"]["img_size"]
#     x = torch.randn(B, C, T, H, W)

#     y = model(x)  # (B, n_classes, H, W)
#     print("Output shape:", tuple(y.shape))
