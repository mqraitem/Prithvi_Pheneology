import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import h5py

# ===== helper functions (unchanged) =====

def day_of_year_to_decimal_month(day_of_year):
    decimal_month = np.zeros_like(day_of_year, dtype=float)

    # Days in each month for a non-leap year
    month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    cumulative_days = np.cumsum(month_days)
    start_days = np.insert(cumulative_days[:-1], 0, 0)

    invalid_mask = day_of_year == -1
    decimal_month[invalid_mask] = -1

    for i in range(12):
        mask = (day_of_year > start_days[i]) & (day_of_year <= cumulative_days[i])
        days_into_month = day_of_year[mask] - start_days[i]
        decimal_month[mask] = (i + 1) + (days_into_month - 1) / month_days[i]

    return decimal_month

def load_raster(path, crop=None):
    import rasterio
    if os.path.exists(path):
        with rasterio.open(path) as src:
            img = src.read()
            if crop:
                img = img[:, -crop[0]:, -crop[1]:]
    else:
        img = np.zeros((6, 330, 330))
    return img

def load_raster_input(path, target_size=330):
    import rasterio
    if os.path.exists(path):
        with rasterio.open(path) as src:
            img = src.read()  # (C, H, W)
        return img.astype(np.float32)
    else:
        return np.zeros((6, target_size, target_size), dtype=np.float32)

def load_raster_output(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read()

# ===== Patch Dataset =====

class cycle_dataset_patches(Dataset):
    """
    Build a dataset of spatio-temporal patches.

    Output shapes:
        inputs: (N, T, C, H, W)
        targets: (N, 4, H, W)
        meta rows: (image_idx, top_h, top_w, tile_name)

    Notes:
      - Non-overlapping patches by default (stride = patch_size).
      - Partial edge patches are dropped (use exact tiling or prepad the rasters if you need full coverage).
    """
    def __init__(
        self,
        data_dir,
        split,
        cache_path,
        data_percentage=1.0,
        target_size=330,
        regenerate=False,
        region_to_cut_name="EASTERN TEMPERATE FORESTS",
        h5_path=None,
        patch_size=(32, 32),   # <---- NEW
        stride=None,           # <---- optional; defaults to patch_size (non-overlap)
    ):
        """
        Args:
            data_dir: list of tuples [(image_paths, gt_path, hls_tile_name), ...]
            split: "train" / "val" / "test"
            cache_path: path to npz file where dataset is cached
            data_percentage: kept for filename compatibility
            target_size: unused here for read; kept for API compatibility
            regenerate: if True, rebuild dataset even if cache exists
            region_to_cut_name: used for means/stds cache path
            h5_path: optional features file; if provided, returns per-pixel feats averaged over patch
            patch_size: (H, W) of each patch
            stride: step for sliding window; if None, stride=patch_size (non-overlapping)
        """
        self.data_dir = data_dir
        self.split = split
        self.cache_path = cache_path
        self.data_percentage = data_percentage
        self.target_size = target_size
        self.region_to_cut_name = region_to_cut_name.replace(" ", "_").lower()

        self.h5_path = h5_path
        self.patch_h, self.patch_w = patch_size
        self.stride_h = self.patch_h if stride is None else stride[0]
        self.stride_w = self.patch_w if stride is None else stride[1]

        # correct gt indices
        self.correct_indices = [2, 5, 8, 11]
        self.correct_indices = [i - 1 for i in self.correct_indices]

        # load/compute means and stds (per-channel, same as pixel dataset)
        self.get_means_stds()

        if os.path.exists(self.cache_path) and not regenerate:
            print(f"[PatchDataset] Loading preprocessed dataset from {self.cache_path}")
            data = np.load(self.cache_path, allow_pickle=True)
            self.inputs = data['inputs']   # (N, T, C, H, W)
            self.targets = data['targets'] # (N, 4, H, W)
            self.meta = data['meta']       # (N, 4) objects
        else:
            print(f"[PatchDataset] Preprocessing {split} split into patches...")
            self._build_dataset()
            print(f"[PatchDataset] Saved to {self.cache_path}")

        self._h5 = None
        self._tile_info = {}  # tile -> (H0, W0, offset, N)

    # ---------- optional H5 opening ----------
    def _ensure_open(self):
        if self._h5 is None and self.h5_path is not None:
            self._h5 = h5py.File(self.h5_path, "r")
            idx = self._h5["index"][:]
            for rec in idx:
                tile = rec["tile"].decode("utf8")
                self._tile_info[tile] = (int(rec["H0"]), int(rec["W0"]), int(rec["offset"]), int(rec["N"]))

    # ---------- stats ----------
    def get_means_stds(self):
        """
        Compute or load the mean and standard deviation for image data.
        Uses correct batch-wise variance combination (Chan et al., 1979).
        """
        main_folder = f"paper_fullsize_all_12month_match_location_{self.region_to_cut_name}_{self.data_percentage}"
        means_stds_path = f"/usr4/cs505/mqraitem/ivc-ml/geo/data/{main_folder}/means_stds_distance.pkl"
        os.makedirs(os.path.dirname(means_stds_path), exist_ok=True)

        if os.path.exists(means_stds_path):
            with open(means_stds_path, 'rb') as f:
                self.means, self.stds = pickle.load(f)
            return
        elif self.split == "test":
            raise ValueError("Cannot compute mean and std for test split")

        global_mean = np.zeros(6, dtype=np.float64)
        global_var = np.zeros(6, dtype=np.float64)
        global_count = np.zeros(6, dtype=np.float64)

        for i in tqdm(range(len(self.data_dir))):
            image_path = self.data_dir[i][0]
            gt_path = self.data_dir[i][1]

            images = []
            for path in image_path:
                images.append(load_raster(path)[:, np.newaxis])

            img = np.concatenate(images, axis=1)  # (6, T, H, W)

            gt_mask = load_raster(gt_path)
            gt_mask = gt_mask[self.correct_indices, :, :]
            nan_mask = np.isnan(gt_mask)
            nan_mask = np.all(nan_mask, axis=0)  # (H, W)

            time_steps = img.shape[1]
            expanded_mask = np.repeat(nan_mask[np.newaxis, :, :], time_steps, axis=0)  # (T, H, W)
            expanded_mask = np.repeat(expanded_mask[np.newaxis, :, :, :], 6, axis=0)    # (6, T, H, W)

            img[expanded_mask] = 0
            img_flat = img.reshape(6, -1)
            mask_flat = ~expanded_mask.reshape(6, -1)

            for b in range(6):
                valid_values = img_flat[b][mask_flat[b]]
                n = len(valid_values)
                if n == 0:
                    continue
                batch_mean = valid_values.mean()
                batch_var = valid_values.var(ddof=1)
                m = global_count[b]
                mu1 = global_mean[b]
                mu2 = batch_mean
                v1 = global_var[b]
                v2 = batch_var

                combined_mean = (m / (m + n)) * mu1 + (n / (m + n)) * mu2 if (m + n) > 0 else mu2
                combined_var = (
                    (m / (m + n)) * v1
                    + (n / (m + n)) * v2
                    + (m * n / (m + n) ** 2) * (mu1 - mu2) ** 2
                    if (m + n) > 0 else v2
                )
                global_mean[b] = combined_mean
                global_var[b] = combined_var
                global_count[b] = m + n

        means = global_mean
        stds = np.sqrt(global_var)

        print(means)
        print(stds)

        with open(means_stds_path, 'wb') as f:
            pickle.dump([means, stds], f)

        self.means = means
        self.stds = stds

    # ---------- mask fixing for maps ----------
    def fix_final_mask_maps(self, final_mask):
        """
        final_mask: (..., 4, H, W) or (4, H, W)
        - Set NaNs -> -1 sentinel
        - Clamp [0, 365]
        - Convert to decimal month (elementwise)
        """
        fm = final_mask.copy()
        mask_nan = np.isnan(fm)
        fm[mask_nan] = -1

        valid_mask = ~mask_nan
        fm = np.where(valid_mask & (fm < 0), 0, fm)
        fm = np.where(valid_mask & (fm > 365), 365, fm)

        fm = day_of_year_to_decimal_month(fm)
        return fm.astype(np.float32)


    # ---------- dataset build (patch version) ----------


    def _edge_cover_starts(self, L: int, p: int, s: int):
        """
        Generate non-overlapping starts with stride s, except:
        - If (L - p) % s != 0, append a final start at L - p (edge-only overlap).
        - If L < p, return [0] and let caller pad.
        """
        if L <= p:
            return [0]
        starts = list(range(0, L - p + 1, s))
        last = L - p
        if starts[-1] != last:
            starts.append(last)  # minimal overlap for the edge
        return starts


    def _build_dataset(self):
        patch_inputs, patch_targets, patch_meta = [], [], []

        for img_idx in tqdm(range(len(self.data_dir))):
            image_paths, gt_path, hls_tile_name = self.data_dir[img_idx]

            # load and stack times: (C, T, H, W)
            imgs = [load_raster_input(p, target_size=self.target_size)[:, np.newaxis]
                    for p in image_paths]
            img = np.concatenate(imgs, axis=1)

            # normalize per channel
            means1 = self.means.reshape(-1, 1, 1, 1)
            stds1  = self.stds.reshape(-1, 1, 1, 1)
            img = (img - means1) / (stds1 + 1e-6)

            C, T, H, W = img.shape
            ph, pw = self.patch_h, self.patch_w
            sh, sw = self.stride_h, self.stride_w  # typically = (ph, pw) for non-overlap

            # load gt mask (4, H, W)
            gt_mask_full = load_raster_output(gt_path)[self.correct_indices, :, :]

            # If image is smaller than a patch, pad minimally so we can extract one patch
            pad_h = max(0, ph - H)
            pad_w = max(0, pw - W)
            if pad_h or pad_w:
                img = np.pad(img, ((0,0),(0,0),(0,pad_h),(0,pad_w)), mode="edge")
                gt_mask_full = np.pad(gt_mask_full, ((0,0),(0,pad_h),(0,pad_w)),
                                    mode="constant", constant_values=np.nan)
                H += pad_h; W += pad_w  # updated dims

            # Compute starts: non-overlap except last patch aligns to the edge if needed
            starts_h = self._edge_cover_starts(H, ph, sh)
            starts_w = self._edge_cover_starts(W, pw, sw)

            for top_h in starts_h:
                for top_w in starts_w:
                    img_patch = img[:, :, top_h:top_h+ph, top_w:top_w+pw]      # (C, T, ph, pw)
                    gt_patch  = gt_mask_full[:, top_h:top_h+ph, top_w:top_w+pw] # (4, ph, pw)

                    # skip patches that are entirely invalid
                    if np.isnan(gt_patch).all():
                        continue

                    gt_patch_fixed = self.fix_final_mask_maps(gt_patch)         # (4, ph, pw)

                    # (C, T, ph, pw) -> (T, C, ph, pw)
                    img_patch_tc = np.transpose(img_patch, (1, 0, 2, 3)).astype(np.float32)

                    patch_inputs.append(img_patch_tc)          # (T, C, ph, pw)
                    patch_targets.append(gt_patch_fixed)       # (4, ph, pw)
                    patch_meta.append((img_idx, int(top_h), int(top_w), hls_tile_name))

        if len(patch_inputs) == 0:
            raise RuntimeError("No valid patches were found. Check masks/paths/patch_size.")

        self.inputs  = np.stack(patch_inputs, axis=0)   # (N, T, C, H, W)
        self.targets = np.stack(patch_targets, axis=0)  # (N, 4, H, W)
        self.meta    = np.array(patch_meta, dtype=object)

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez_compressed(self.cache_path, inputs=self.inputs, targets=self.targets, meta=self.meta)
        # concat across images
        if len(patch_inputs) == 0:
            raise RuntimeError("No valid patches were found. Check your masks/paths/patch_size.")

        self.inputs  = np.stack(patch_inputs, axis=0)   # (N, T, C, H, W)
        self.targets = np.stack(patch_targets, axis=0)  # (N, 4, H, W)
        self.meta    = np.array(patch_meta, dtype=object)

        # cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            inputs=self.inputs,
            targets=self.targets,
            meta=self.meta
        )

    # ---------- torch dataset API ----------
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx])    # (T, C, H, W)
        y = torch.from_numpy(self.targets[idx])   # (4, H, W)
        _, top_h, top_w, tile = self.meta[idx]
        top_h, top_w, tile = int(top_h), int(top_w), str(tile)

        sample = {"image": x, "gt_mask": y, "patch_origin": (top_h, top_w), "tile": tile}

        return sample
