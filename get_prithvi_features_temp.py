#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

# Optional if you want to load rasters directly somewhere else
# import rasterio

# project imports
sys.path.append("../")
from prithvi_hf.prithvi import PrithviSeg
from utils import data_path_paper_all_12month_match
from dataloader_fullsize_all import cycle_dataset

# ----------------------- Defaults / knobs -----------------------
DEFAULT_OUT_DIR    = "/projectnb/hlsfm/applications/lsp/outputs/HLS_prithvi_pixel_timeseries_224tiling"
DEFAULT_MODEL_SIZE = "300m"
SAVE_DTYPE         = np.float16
# Prithvi pretrain size
PT_SIZE            = 224
# We tile 336 with stride 112 -> windows at {0,112}
TILE_STRIDE        = 112
# ---------------------------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

@torch.no_grad()
def prithvi_forward_features(model: PrithviSeg, x_224: torch.Tensor) -> torch.Tensor:
    """
    x_224: (1, C, T, 224, 224) on device
    return: (1, T*E, Hp, Wp) where Hp=Wp=224/patch (e.g., 14 for 16x16)
    """
    model.eval()
    return model.forward_features(x_224)

@torch.no_grad()
def upsample_TE_to_HW(out_TEHWp: torch.Tensor, T: int, E: int, target_HW: tuple[int, int]) -> torch.Tensor:
    """
    out_TEHWp: (1, T*E, Hp, Wp)  ->  (1, T*E, H, W)
    """
    _, TE, Hp, Wp = out_TEHWp.shape
    H, W = target_HW
    assert TE == T * E, f"Expected channels {T*E}, got {TE}"
    return torch.nn.functional.interpolate(out_TEHWp, size=(H, W), mode="bilinear", align_corners=False)

def build_dataloaders(cfg):
    path_val   = data_path_paper_all_12month_match("validation")
    path_test  = data_path_paper_all_12month_match("testing")
    path_train = data_path_paper_all_12month_match("training")

    ds_train = cycle_dataset(path_train, split="training",   means=cfg["pretrained_cfg"]["mean"], stds=cfg["pretrained_cfg"]["std"])
    ds_val   = cycle_dataset(path_val,   split="validation", means=cfg["pretrained_cfg"]["mean"], stds=cfg["pretrained_cfg"]["std"])
    ds_test  = cycle_dataset(path_test,  split="testing",    means=cfg["pretrained_cfg"]["mean"], stds=cfg["pretrained_cfg"]["std"])

    train_loader = DataLoader(ds_train, batch_size=cfg["training"]["batch_size"],
                              shuffle=cfg["training"]["shuffle"], num_workers=2)
    val_loader   = DataLoader(ds_val,   batch_size=cfg["validation"]["batch_size"],
                              shuffle=cfg["validation"]["shuffle"], num_workers=2)
    test_loader  = DataLoader(ds_test,  batch_size=cfg["test"]["batch_size"],
                              shuffle=cfg["validation"]["shuffle"], num_workers=2)
    return train_loader, val_loader, test_loader

def infer_valid_hw_from_padded_chip(x_chip: torch.Tensor):
    """
    x_chip: (1, C, T, 336, 336) (float) where right/bottom are zero-padded.
    Returns H0, W0 = largest non-zero extents along bottom/right.
    NOTE: This assumes real data are nonzero at least in one band/time.
    """
    with torch.no_grad():
        # reduce over C,T -> presence mask over H,W
        mask = (x_chip.abs().sum(dim=(1,2)) > 0).squeeze(0)  # (H,W) boolean
        rows = torch.where(mask.any(dim=1))[0]
        cols = torch.where(mask.any(dim=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return 0, 0
        H0 = int(rows.max().item() + 1)
        W0 = int(cols.max().item() + 1)
        return H0, W0

@torch.no_grad()
def prithvi_features_224_tiled(model: PrithviSeg, x_336: torch.Tensor, T: int, E: int):
    """
    Run Prithvi at 224 on 336 by tiling with stride 112 (2x2 tiles), then stitch with overlap-averaging.

    x_336: (1, C, T, 336, 336)
    return: stitched (1, T*E, 336, 336) torch tensor.
    """
    assert x_336.ndim == 5 and x_336.shape[-1] == 336 and x_336.shape[-2] == 336
    device = x_336.device

    canvas = torch.zeros((1, T*E, 336, 336), device=device, dtype=torch.float32)
    weight = torch.zeros((1, 1, 336, 336), device=device, dtype=torch.float32)

    # Tile positions: (0,0), (0,112), (112,0), (112,112)
    starts = [0, TILE_STRIDE]
    for y0 in starts:
        for x0 in starts:
            y1 = y0 + PT_SIZE
            x1 = x0 + PT_SIZE
            # crop tile
            x_tile = x_336[:, :, :, y0:y1, x0:x1]  # (1,C,T,224,224)
            assert x_tile.shape[-1] == PT_SIZE and x_tile.shape[-2] == PT_SIZE

            # forward

            out = prithvi_forward_features(model, x_tile)     # (1, T*E, Hp, Wp)
            out_up = upsample_TE_to_HW(out, T=T, E=E, target_HW=(PT_SIZE, PT_SIZE))  # (1, T*E, 224, 224)

            # place into canvas
            canvas[:, :, y0:y1, x0:x1] += out_up
            weight[:, :, y0:y1, x0:x1] += 1.0

    # avoid divide-by-zero just in case
    weight = torch.clamp(weight, min=1.0)
    canvas = canvas / weight
    return canvas  # (1, T*E, 336, 336)

@torch.no_grad()
def to_per_pixel_timeseries_from_TE(canvas_TEHW: torch.Tensor, T: int, E: int) -> torch.Tensor:
    """
    canvas_TEHW: (1, T*E, H, W) -> (H, W, T, E)
    """
    _, TE, H, W = canvas_TEHW.shape
    assert TE == T * E, f"Expected channels {T*E}, got {TE}"
    feat = canvas_TEHW.view(1, E, T, H, W).permute(0, 3, 4, 2, 1).contiguous()  # (1, H, W, T, E)
    return feat.squeeze(0)  # (H, W, T, E)

@torch.no_grad()
def transform_and_save_split_pixels_tiled224(dataloader,
                                             model,
                                             device,
                                             split_name: str,
                                             out_dir_base: str,
                                             T: int,
                                             E: int,
                                             cache_basename: str = None,
                                             save_every: int = 0):
    """
    For each 336x336 tile in split:
      - infer original H0xW0 (nonzero area)
      - run Prithvi via 224-tiling + stitching
      - convert to (H,W,T,E), crop to (H0,W0)
      - flatten to (H0*W0, T, E), keep ALL real pixels
      - accumulate and save ONE .npz per split with 'inputs' and 'meta'
    """
    out_dir = ensure_dir(out_dir_base)
    cache_path = os.path.join(out_dir, f"{cache_basename or split_name}_pixels_224tiled_T{T}_E{E}.npz")

    pixel_inputs = []  # list of (Ni, T, E) arrays
    pixel_meta   = []  # list of (idx, h, w, tile)
    running_idx  = 0
    items        = 0

    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Vectorize 224-tiles [{split_name}]"):
        x = batch["image"].to(device)[:, 0]  # (1, C, T, 336, 336)
        assert x.ndim == 5 and x.shape[-1] == 336 and x.shape[-2] == 336 and x.size(0) == 1

        # original (unpadded) size
        H0, W0 = infer_valid_hw_from_padded_chip(x)  # e.g., 330x330
        if H0 == 0 or W0 == 0:
            # nothing valid — skip
            continue

        # stitched TE @ 336x336
        stitched_TE = prithvi_features_224_tiled(model, x, T=T, E=E)  # (1, T*E, 336, 336)
        # to per-pixel sequences and crop to H0xW0
        pix = to_per_pixel_timeseries_from_TE(stitched_TE, T=T, E=E)  # (336,336,T,E)
        pix = pix[:H0, :W0]  # (H0, W0, T, E)

        # flatten spatial and stash
        Ni = H0 * W0
        pix_flat = pix.reshape(Ni, T, E).cpu().numpy().astype(SAVE_DTYPE)



        # meta for ALL real pixels
        h_coords, w_coords = np.meshgrid(np.arange(H0), np.arange(W0), indexing="ij")
        coords = np.stack([h_coords.ravel(), w_coords.ravel()], axis=1)  # (Ni, 2)

        tile_name = batch.get("hls_tile_name", "unknown")
        if isinstance(tile_name, (list, tuple)):
            tile_name = tile_name[0]
        tile_name = str(tile_name)

        meta_i = [(running_idx + j, int(h), int(w), tile_name) for j, (h, w) in enumerate(coords)]
        pixel_inputs.append(pix_flat)
        pixel_meta.extend(meta_i)
        running_idx += Ni
        items += 1


    # Final single-file save
    all_inputs = np.concatenate(pixel_inputs, axis=0) if pixel_inputs else np.zeros((0, T, E), dtype=SAVE_DTYPE)
    all_meta   = np.array(pixel_meta, dtype=object)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, inputs=all_inputs, meta=all_meta, T=T, E=E)
    print(f"[{split_name}] saved {all_inputs.shape[0]} pixels to {cache_path}")

    # Optional summary sidecar
    with open(os.path.join(out_dir, f"{split_name}_summary.pkl"), "wb") as f:
        pickle.dump(dict(N=all_inputs.shape[0], T=T, E=E, npz=cache_path), f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default=DEFAULT_MODEL_SIZE, help="Prithvi model size key")
    parser.add_argument("--out_dir",   type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--t",         type=int, default=12, help="Temporal length T")
    parser.add_argument("--e",         type=int, default=1024, help="Per-time embed size E (embed_dim * num_frames)")
    parser.add_argument("--save_every", type=int, default=0, help="Optional: checkpoint every N tiles")
    args = parser.parse_args()

    with open(f'configs/config_{args.model_size}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Match earlier setup (your dataloader uses 336 padded chips)
    config["pretrained_cfg"]["img_size"] = 224
    config["training"]["batch_size"] = 1
    config["validation"]["batch_size"] = 1
    config["test"]["batch_size"] = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(config)

    # Model (frozen encoder)
    weights_path = config["pretrained_cfg"]["prithvi_model_new_weight"]
    model = PrithviSeg(config["pretrained_cfg"], weights_path, True, n_classes=4, model_size=args.model_size).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # Sanity for E (optional)
    E_cfg = int(config["pretrained_cfg"]["embed_dim"]) * int(config["pretrained_cfg"]["num_frames"])
    if args.e != E_cfg:
        print(f"[warn] Provided E={args.e} differs from config-derived E={E_cfg}. Using E={args.e} as requested.")

    ensure_dir(args.out_dir)
    transform_and_save_split_pixels_tiled224(train_loader, model, device, "train", args.out_dir, T=args.t, E=args.e,
                                             cache_basename="train_224tiled", save_every=args.save_every)
    transform_and_save_split_pixels_tiled224(val_loader,   model, device, "val",   args.out_dir, T=args.t, E=args.e,
                                             cache_basename="val_224tiled",   save_every=args.save_every)
    transform_and_save_split_pixels_tiled224(test_loader,  model, device, "test",  args.out_dir, T=args.t, E=args.e,
                                             cache_basename="test_224tiled",  save_every=args.save_every)

    print("Done.")

if __name__ == "__main__":
    main()
