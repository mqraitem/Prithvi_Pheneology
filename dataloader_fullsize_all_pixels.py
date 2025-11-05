import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

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
	def __init__(self, data_dir, split, cache_path, data_percentage=1.0, target_size=330, regenerate=False, region_to_cut_name="EASTERN TEMPERATE FORESTS", feats_path=None):
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

		self.feats_path = feats_path
		if self.feats_path is not None:
			self.feats_data = pickle.load(open(feats_path, "rb"))

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
		x = torch.from_numpy(self.inputs[idx])   # (T, C)
		y = torch.from_numpy(self.targets[idx])  # (4,)
		meta = self.meta[idx]

		to_return = {
			"image": x,
			"gt_mask": y
		}

		if self.feats_path is not None:
			feats = self.feats_data[meta[3]]
			feats = torch.from_numpy(feats)
			to_return["feats"] = feats

		return to_return
