#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
import pickle

# project imports
sys.path.append("../")
from prithvi_hf.prithvi import PrithviSeg
from utils import data_path_paper_all_12month_match
from dataloader_fullsize_all import cycle_dataset

# ----------------------- Defaults / knobs -----------------------
DEFAULT_T          = 12           # ensure T*E matches channels (e.g., 12*1024=12288)
DEFAULT_E          = 1024
DEFAULT_K          = 256           # PCA target dims (E -> k)
DEFAULT_MODE       = "per_timestep"  # 'per_timestep' | 'meanstd' | 'concat'
DEFAULT_OUT_DIR    = "/projectnb/hlsfm/applications/lsp/outputs/HLS_composites_HP-LSP_PCA_Feats"
DEFAULT_SVD_SOLVER = "randomized" # 'randomized' or 'full'
SAVE_DTYPE         = np.float16
# ---------------------------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

@torch.no_grad()
def prithvi_forward_features(model: PrithviSeg, x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, T, H, W) on device
    returns: (B, T*E, H', W')  (e.g., (1, 12288, 21, 21))
    """
    model.eval()
    return model.forward_features(x)

def count_train_rows(train_loader, T: int) -> int:
    return len(train_loader) * T  # batch_size=1 assumed

@torch.no_grad()
def collect_train_rows_to_memmap(train_loader, model, device, T, E, memmap_path, dtype=np.float32):
    """
    First pass: (1, T*E, H, W) -> (T,E,H,W) -> GAP(H,W) -> (T,E), append to memmap.
    """
    n_rows = count_train_rows(train_loader, T)
    ensure_dir(os.path.dirname(memmap_path))
    X = np.memmap(memmap_path, mode="w+", dtype=dtype, shape=(n_rows, E))

    row_ptr = 0
    for batch in tqdm(train_loader, total=len(train_loader), desc="Collect (T,E) rows [train]"):
        x = batch["image"].to(device)[:, 0]               # (1, C, T, H, W)
        out = prithvi_forward_features(model, x)          # (1, T*E, H, W)
        _, TE, H, W = out.shape
        assert TE == T * E, f"Mismatch: expected channels {T*E}, got {TE}"

        feat = out.view(1, T, E, H, W).squeeze(0)         # (T, E, H, W)
        gap  = feat.mean(dim=(2, 3)).cpu().numpy()        # (T, E)

        X[row_ptr:row_ptr + T, :] = gap
        row_ptr += T

    X.flush()
    return memmap_path, n_rows

def fit_full_pca_memmap(memmap_path, n_rows, E, k, save_path, svd_solver="randomized", random_state=123):
    """
    Fit PCA on full memmap X (n_rows, E) and save mean/components.
    """
    X = np.memmap(memmap_path, mode="r", dtype=np.float32, shape=(n_rows, E))
    pca = PCA(n_components=k, svd_solver=svd_solver, random_state=random_state)
    tqdm(total=1, desc=f"Fitting PCA on {n_rows}x{E} (k={k})").update(1)  # progress placeholder
    pca.fit(X)

    ensure_dir(os.path.dirname(save_path))
    np.savez_compressed(
        save_path,
        components=pca.components_.astype(np.float32),  # (k, E)
        mean=pca.mean_.astype(np.float32),              # (E,)
        E=E, k=k
    )
    return save_path

def load_pca(pca_path, device):
    z = np.load(pca_path)
    comps = torch.from_numpy(z["components"]).float().to(device)  # (k, E)
    mean  = torch.from_numpy(z["mean"]).float().to(device)        # (E,)
    E     = int(z["E"]); k = int(z["k"])
    return comps, mean, E, k

@torch.no_grad()
def tile_out_to_array(out_TEHW: torch.Tensor,
                      comps: torch.Tensor,
                      mean: torch.Tensor,
                      T: int,
                      E: int,
                      k: int,
                      mode: str):
    """
    Convert single-tile Prithvi features (1,T*E,H,W) to:
      - 'per_timestep': Z ∈ (T, k)
      - 'meanstd'    : vec ∈ (2k,)
      - 'concat'     : vec ∈ (T*k,)
    """
    _, TE, H, W = out_TEHW.shape
    assert TE == T * E, f"Expected channels {T*E}, got {TE}"

    feat = out_TEHW.view(1, T, E, H, W).squeeze(0)  # (T,E,H,W)
    gap  = feat.mean(dim=(2, 3))                    # (T,E)

    X = gap - mean[None, :]                          # (T,E)
    Z = (X @ comps.t()).float().cpu().numpy()        # (T,k)

    if mode == "per_timestep":
        return Z  # (T,k)
    elif mode == "meanstd":
        return np.concatenate([Z.mean(0), Z.std(0)], axis=0)  # (2k,)
    elif mode == "concat":
        return Z.reshape(-1)                                  # (T*k,)
    else:
        raise ValueError(f"Unknown mode={mode}")

@torch.no_grad()
def transform_and_save_split(dataloader,
                             model,
                             device,
                             pca_path: str,
                             split_name: str,
                             out_dir_base: str,
                             T: int,
                             E: int,
                             k: int,
                             mode: str):
    """
    For each tile in split:
      - compute (1,T*E,H,W)
      - map to Z or vec (depending on mode)
      - save .npz and write split index CSV
    """
    comps, mean, E_check, k_check = load_pca(pca_path, device)
    assert E_check == E and k_check == k, "PCA meta mismatch."

    out_split_dir = ensure_dir(os.path.join(out_dir_base, split_name))
    index_csv = os.path.join(out_dir_base, f"{split_name}_index.csv")

    rows = []
    all_data = {} 
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Vectorize [{split_name}]"):
        x = batch["image"].to(device)[:, 0]  # (1,C,T,H,W)

        tile_name = batch["hls_tile_name"]
        if isinstance(tile_name, (list, tuple)):
            tile_name = tile_name[0]
        tile_name = str(tile_name)

        out = prithvi_forward_features(model, x)    # (1,T*E,H,W)
        arr = tile_out_to_array(out, comps, mean, T=T, E=E, k=k, mode=mode)
        
        save_path = os.path.join(out_split_dir, f"{tile_name}.npz")
        if mode == "per_timestep":
            np.savez_compressed(save_path,
                                Z=arr.astype(SAVE_DTYPE),  # (T,k)
                                tile=tile_name, T=T, E=E, k=k, mode=mode)
            dim = f"{arr.shape[0]}x{arr.shape[1]}"  # "T x k"
            all_data[tile_name] = arr
        else:
            np.savez_compressed(save_path,
                                vec=arr.astype(SAVE_DTYPE),
                                tile=tile_name, T=T, E=E, k=k, mode=mode)
            dim = f"{arr.shape[0]}"
            all_data[tile_name] = arr

        rows.append((tile_name, save_path, dim, mode, T, E, k))

    # write index CSV
    with open(index_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "path", "dim", "mode", "T", "E", "k"])
        for r in rows:
            w.writerow(list(r))
    print(f"[{split_name}] saved {len(rows)} items to {out_split_dir}; index: {index_csv}")

    with open(os.path.join(out_dir_base, f"{split_name}_all_data.pkl"), "wb") as f:
        pickle.dump(all_data, f)

def build_dataloaders(cfg):
    path_val   = data_path_paper_all_12month_match("validation")
    path_test  = data_path_paper_all_12month_match("testing")
    path_train = data_path_paper_all_12month_match("training")

    ds_train = cycle_dataset(path_train, split="training",  means = cfg["pretrained_cfg"]["mean"], stds = cfg["pretrained_cfg"]["std"])
    ds_val   = cycle_dataset(path_val,   split="validation", means = cfg["pretrained_cfg"]["mean"], stds = cfg["pretrained_cfg"]["std"])
    ds_test  = cycle_dataset(path_test,  split="testing", means = cfg["pretrained_cfg"]["mean"], stds = cfg["pretrained_cfg"]["std"])

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg["training"]["batch_size"],
        shuffle=cfg["training"]["shuffle"],
        num_workers=2
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg["validation"]["batch_size"],
        shuffle=cfg["validation"]["shuffle"],
        num_workers=2
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=cfg["test"]["batch_size"],
        shuffle=cfg["validation"]["shuffle"],
        num_workers=2
    )
    return train_loader, val_loader, test_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="300m", help="Prithvi model size key")
    parser.add_argument("--out_dir",   type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--k",         type=int, default=DEFAULT_K, help="PCA components (E->k)")
    parser.add_argument("--t",         type=int, default=DEFAULT_T, help="Temporal length T")
    parser.add_argument("--e",         type=int, default=DEFAULT_E, help="Per-time embed size E")
    parser.add_argument("--mode",      type=str, default=DEFAULT_MODE,
                        choices=["per_timestep", "meanstd", "concat"],
                        help="Save per-timestep (T,k) or aggregated vector")
    parser.add_argument("--svd_solver", type=str, default=DEFAULT_SVD_SOLVER, choices=["randomized", "full"],
                        help="PCA SVD solver")
    args = parser.parse_args()

    with open(f'configs/config_{args.model_size}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # match your earlier setup
    config["pretrained_cfg"]["img_size"] = 336
    config["training"]["batch_size"] = 1
    config["validation"]["batch_size"] = 1
    config["test"]["batch_size"] = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(config)

    # Model
    weights_path = config["pretrained_cfg"]["prithvi_model_new_weight"]
    model = PrithviSeg(config["pretrained_cfg"], weights_path, True, n_classes=4, model_size=args.model_size).to(device)

    # Pass 1: collect (T,E) rows into memmap (train only)
    ensure_dir(args.out_dir)
    memmap_path = os.path.join(args.out_dir, "pca_rows.dat")
    mm_path, n_rows = collect_train_rows_to_memmap(
        train_loader, model, device, T=args.t, E=args.e, memmap_path=memmap_path, dtype=np.float32
    )
    print(f"Collected {n_rows} rows into memmap: {mm_path}")

    # Fit PCA on full matrix
    pca_path = os.path.join(args.out_dir, f"pca_full_T{args.t}_E{args.e}_k{args.k}.npz")
    pca_path = fit_full_pca_memmap(mm_path, n_rows, args.e, args.k, pca_path,
                                   svd_solver=args.svd_solver, random_state=123)
    print(f"Saved PCA to: {pca_path}")

    # Pass 2: transform and save per-tile outputs (per-timestep by default)
    transform_and_save_split(train_loader, model, device, pca_path, "train",
                             out_dir_base=args.out_dir, T=args.t, E=args.e, k=args.k, mode=args.mode)
    transform_and_save_split(val_loader,   model, device, pca_path, "val",
                             out_dir_base=args.out_dir, T=args.t, E=args.e, k=args.k, mode=args.mode)
    transform_and_save_split(test_loader,  model, device, pca_path, "test",
                             out_dir_base=args.out_dir, T=args.t, E=args.e, k=args.k, mode=args.mode)

    print("Done.")

if __name__ == "__main__":
    main()
