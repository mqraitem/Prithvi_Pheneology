import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import h5py
from tqdm import tqdm

# ===== helper functions =====

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
		_, h, w = img.shape
		return img.astype(np.float32)
	else:
		return np.zeros((6, target_size, target_size), dtype=np.float32)

def load_raster_output(path):
	import rasterio
	with rasterio.open(path) as src:
		return src.read()

# ===== Dataset =====

class cycle_dataset_pixels(Dataset):
	def __init__(self, data_dir, split, cache_path, data_percentage=1.0, target_size=330, regenerate=False, region_to_cut_name="EASTERN TEMPERATE FORESTS", h5_path=None):
		"""
		Args:
			data_dir: list of tuples [(image_paths, gt_path, hls_tile_name), ...]
			split: "train" / "val" / "test"
			cache_path: path to npz file where pixel dataset is cached
			data_percentage: used for naming stats file (like original code)
			target_size: padded size
			regenerate: if True, rebuild dataset even if cache exists
		"""
		self.data_dir = data_dir
		self.split = split
		self.cache_path = cache_path
		self.data_percentage = data_percentage
		self.target_size = target_size
		self.region_to_cut_name = region_to_cut_name
		self.region_to_cut_name = region_to_cut_name.replace(" ", "_").lower()

		self.h5_path = h5_path

		# correct gt indices
		self.correct_indices = [2, 5, 8, 11]
		self.correct_indices = [i - 1 for i in self.correct_indices]

		# load/compute means and stds
		self.get_means_stds()

		if os.path.exists(self.cache_path) and not regenerate:
			print(f"[PixelDataset] Loading preprocessed dataset from {self.cache_path}")
			data = np.load(self.cache_path, allow_pickle=True)
			self.inputs = data['inputs']   # (N, T, C)
			self.targets = data['targets'] # (N, 4)
			self.meta = data['meta']
		else:
			print(f"[PixelDataset] Preprocessing {split} split into pixels...")
			self._build_dataset()
			print(f"[PixelDataset] Saved to {self.cache_path}")


		self._h5 = None      # opened per worker
		self._tile_info = {} # tile -> (H0, W0, offset, N)

	def _ensure_open(self):
		if self._h5 is None:
			self._h5 = h5py.File(self.h5_path, "r")
			# load the index once (small)
			idx = self._h5["index"][:]  # structured array
			# build quick dict: tile -> (H0, W0, offset, N)
			for rec in idx:
				tile = rec["tile"].decode("utf8")
				self._tile_info[tile] = (int(rec["H0"]), int(rec["W0"]), int(rec["offset"]), int(rec["N"]))



	def get_means_stds(self):
		"""
		Compute or load the mean and standard deviation for image data.
		Uses correct batch-wise variance combination (Chan et al., 1979).
		"""
		main_folder = f"paper_fullsize_all_12month_match_location_{self.region_to_cut_name}_{self.data_percentage}"
		means_stds_path = f"/usr4/cs505/mqraitem/ivc-ml/geo/data/{main_folder}/means_stds_distance.pkl"
		os.makedirs(os.path.dirname(means_stds_path), exist_ok=True)

		# Load precomputed means and stds
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

			img = np.concatenate(images, axis=1)  # shape: (6, time_steps, 330, 330)

			gt_mask = load_raster(gt_path)
			gt_mask = gt_mask[self.correct_indices, :, :]
			nan_mask = np.isnan(gt_mask)
			
			#reduce the mask from 4, 330, 330 to 330,330 by using and operator
			nan_mask = np.all(nan_mask, axis=0)  # shape (330, 330)
			
			# Expand mask to match image shape: (6, time_steps, 330, 330)
			time_steps = img.shape[1]
			expanded_mask = np.repeat(nan_mask[np.newaxis, :, :], time_steps, axis=0)  # shape (time_steps, 330, 330)
			expanded_mask = np.repeat(expanded_mask[np.newaxis, :, :, :], 6, axis=0)  # shape (6, time_steps, 330, 330)

			img[expanded_mask] = 0
			img_flat = img.reshape(6, -1)
			mask_flat = ~expanded_mask.reshape(6, -1)

			for b in range(6):
				valid_values = img_flat[b][mask_flat[b]]
				n = len(valid_values)
				if n == 0:
					continue

				batch_mean = valid_values.mean()
				batch_var = valid_values.var(ddof=1)  # population variance (divided by n)

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
					if (m + n) > 0
					else v2
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

	# === mask fixing ===
	def fix_final_mask_batch(self, final_mask):
		# final_mask: (N, 4)
		mask = np.isnan(final_mask)
		final_mask = np.where(mask, -1, final_mask)

		valid_mask = ~mask
		final_mask = np.where(valid_mask & (final_mask < 0), 0, final_mask)
		final_mask = np.where(valid_mask & (final_mask > 365), 365, final_mask)

		# day_of_year_to_decimal_month works elementwise, so vectorize
		final_mask = day_of_year_to_decimal_month(final_mask)

		return final_mask.astype(np.float32)


	def _build_dataset(self):
		pixel_inputs, pixel_targets, pixel_meta = [], [], []

		for idx in tqdm(range(len(self.data_dir))):
			image_paths, gt_path, hls_tile_name = self.data_dir[idx]

			# load images
			imgs = [load_raster_input(p, target_size=self.target_size)[:, np.newaxis]
					for p in image_paths]
			img = np.concatenate(imgs, axis=1)  # (C, T, H, W)

			# normalize
			means1 = self.means.reshape(-1, 1, 1, 1)
			stds1 = self.stds.reshape(-1, 1, 1, 1)
			img = (img - means1) / (stds1 + 1e-6)

			# reshape pixels
			C, T, H, W = img.shape
			img_reshaped = img.reshape(C, T, H*W).transpose(2, 1, 0)  # (H*W, T, C)

			# load mask
			gt_mask = load_raster_output(gt_path)[self.correct_indices, :, :]  # (4, H, W)
			labels = gt_mask.reshape(4, H*W).transpose(1, 0)  # (H*W, 4)

			# fix masks in batch
			labels = self.fix_final_mask_batch(labels)  # vectorized version

			# drop completely invalid pixels (all -1)
			valid_idx = ~(labels == -1).all(axis=1)
			img_valid = img_reshaped[valid_idx]   # (N, T, C)
			labels_valid = labels[valid_idx]      # (N, 4)

			# build meta
			h_coords, w_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
			coords = np.stack([h_coords.ravel(), w_coords.ravel()], axis=1)  # (H*W, 2)
			coords = coords[valid_idx]
			meta = [(idx, h, w, hls_tile_name) for (h, w) in coords]

			pixel_inputs.append(img_valid.astype(np.float32))
			pixel_targets.append(labels_valid.astype(np.float32))
			pixel_meta.extend(meta)

		# concatenate across images
		self.inputs = np.concatenate(pixel_inputs, axis=0)   # (N, T, C)
		self.targets = np.concatenate(pixel_targets, axis=0) # (N, 4)
		self.meta = np.array(pixel_meta, dtype=object)

		os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
		np.savez_compressed(self.cache_path,
							inputs=self.inputs,
							targets=self.targets,
							meta=self.meta)


	# === torch dataset API ===
	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, idx):
		x = torch.from_numpy(self.inputs[idx])   # e.g., (T, C)
		y = torch.from_numpy(self.targets[idx])  # e.g., (4,)
		_, h, w, tile = self.meta[idx]
		h, w, tile = int(h), int(w), str(tile)

		sample = {"image": x, "gt_mask": y}


		if self.h5_path is not None:
			print("Loading feats from h5 file")
			self._ensure_open()
			H0, W0, offset, N = self._tile_info.get(tile, (0, 0, 0, 0))
			if N == 0 or not (0 <= h < H0 and 0 <= w < W0):
				# fallback if missing
				T = int(self._h5.attrs["T"]); E = int(self._h5.attrs["E"])
				sample["feats"] = torch.zeros(T, E, dtype=torch.float32)
				return sample

			row = offset + h * W0 + w
			feats_np = self._h5["inputs"][row, :, :]   # (T, E) — reads one chunk
			sample["feats"] = torch.from_numpy(np.asarray(feats_np)).float()
		
		return sample


# per_tile_fraction_sampler.py
import math
from typing import Iterator, List, Dict, Sequence, Optional
import numpy as np
import torch
from torch.utils.data import Sampler

class PerTileFractionSampler(Sampler[int]):
    """
    Each epoch: sample a fraction of indices from every tile.

    - fraction: 0.0–1.0 (e.g., 0.1 = 10% of pixels per tile per epoch)
    - cap_per_tile: optional hard cap per tile (e.g., 4096)
    - min_per_tile: ensure at least this many per tile (default 1)
    - replacement: sample with/without replacement
    - distributed: if True, evenly split sampled indices across replicas
    """
    def __init__(
        self,
        dataset,
        fraction: float = 0.1,
        cap_per_tile: Optional[int] = None,
        min_per_tile: int = 1,
        replacement: bool = False,
        shuffle_tiles: bool = True,
        seed: int = 0,
        distributed: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        super().__init__(dataset)
        assert 0.0 < fraction <= 1.0
        self.dataset = dataset
        self.fraction = fraction
        self.cap_per_tile = cap_per_tile
        self.min_per_tile = min_per_tile
        self.replacement = replacement
        self.shuffle_tiles = shuffle_tiles
        self.seed = seed

        # DDP settings
        if distributed:
            if num_replicas is None:
                num_replicas = torch.distributed.get_world_size()
            if rank is None:
                rank = torch.distributed.get_rank()
        self.distributed = distributed
        self.num_replicas = num_replicas or 1
        self.rank = rank or 0

        # Build tile -> indices map once
        # dataset.meta rows are (img_idx, h, w, tile)
        tile_to_idx: Dict[str, List[int]] = {}
        for i, m in enumerate(self.dataset.meta):
            tile = str(m[3])
            tile_to_idx.setdefault(tile, []).append(i)
        self.tile_to_idx = {k: np.asarray(v, dtype=np.int64) for k, v in tile_to_idx.items()}
        self.tiles: List[str] = list(self.tile_to_idx.keys())

        self.epoch = 0

    def __len__(self) -> int:
        # Expected number varies by epoch due to ceil; return an upper bound
        total = 0
        for tile in self.tiles:
            n = len(self.tile_to_idx[tile])
            m = max(self.min_per_tile, math.ceil(self.fraction * n))
            if self.cap_per_tile is not None:
                m = min(m, self.cap_per_tile)
            total += m
        # If distributed, each replica sees ~1/num_replicas samples
        return math.ceil(total / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)

        tiles = self.tiles.copy()
        if self.shuffle_tiles:
            rng.shuffle(tiles)

        sampled: List[int] = []
        for tile in tiles:
            idx = self.tile_to_idx[tile]
            n = len(idx)
            m = max(self.min_per_tile, math.ceil(self.fraction * n))
            if self.cap_per_tile is not None:
                m = min(m, self.cap_per_tile)

            if self.replacement:
                sel = rng.integers(0, n, size=m, endpoint=False)
                sampled_idx = idx[sel]
            else:
                if m >= n:
                    sampled_idx = idx.copy()
                    rng.shuffle(sampled_idx)
                else:
                    sampled_idx = idx[rng.choice(n, size=m, replace=False)]

            sampled.extend(sampled_idx.tolist())

        # shuffle all sampled indices for this epoch
        rng.shuffle(sampled)

        # DDP split
        if self.distributed and self.num_replicas > 1:
            # round-robin partition
            sampled = sampled[self.rank::self.num_replicas]

        return iter(sampled)

