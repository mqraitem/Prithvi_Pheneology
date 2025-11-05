import rasterio
import os
from tqdm import tqdm
import yaml
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
 
def load_raster(path,crop=None):

	if os.path.exists(path):
		with rasterio.open(path) as src:
			img = src.read()
			if crop:
				img = img[:, -crop[0]:, -crop[1]:]

	else:
		img = np.zeros((6, 330, 330))

	return img

def load_raster_input(path, target_size=330):

	if os.path.exists(path):

		with rasterio.open(path) as src:
			img = src.read()  # shape: (C, H, W)


		return img.astype(np.float32)

	else: 
		return np.zeros((6, target_size, target_size)).astype(np.float32)


def load_raster_output(path):
		with rasterio.open(path) as src:
			img = src.read()

		return img


def day_of_year_to_decimal_month(day_of_year):
	decimal_month = np.zeros_like(day_of_year, dtype=float)
	
	# Days in each month for a non-leap year
	month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
	cumulative_days = np.cumsum(month_days)
	start_days = np.insert(cumulative_days[:-1], 0, 0)  # start of each month
	
	# Handle invalid values (-1) first
	invalid_mask = day_of_year == -1		
	decimal_month[invalid_mask] = -1
	
	for i in range(12):
		# Get days that fall in the i-th month
		mask = (day_of_year > start_days[i]) & (day_of_year <= cumulative_days[i])
		days_into_month = day_of_year[mask] - start_days[i]
		decimal_month[mask] = (i + 1) + (days_into_month - 1) / month_days[i]  # -1 so day 1 is .0

	return decimal_month


def preprocess_image(image, mask, means, stds):
	"""
	Normalize a (bands, time, H, W) image using per-band means/stds (across all time steps).
	`image` is (bands, time, H, W)
	`mask` is (time, H, W) ground truth, possibly with NaNs
	Returns a torch tensor with shape (1, bands, time, H, W)
	"""


	number_of_channels = image.shape[0]  # bands
	number_of_time_steps = image.shape[1]
	
	bands, time, H, W = image.shape
	vh, vw = (330,330)  # e.g. 330, 330

	# Reshape for broadcasting
	means1 = means.reshape(bands, 1, 1, 1)
	stds1 = stds.reshape(bands, 1, 1, 1)

	# Initialize output with zeros (preserve padding)
	normalized = np.zeros_like(image, dtype=np.float32)

	# Normalize only valid region
	normalized[:, :, :vh, :vw] = (
		(image[:, :, :vh, :vw].astype(np.float32) - means1) / (stds1 + 1e-6)
	)

	# Identify invalid ground truth pixels (where mask is NaN)
	nan_mask = np.isnan(mask)  # shape: (time, H, W)
	nan_mask = np.all(nan_mask, axis=0)  # Reduce to (H, W) by ANDing across time
	nan_mask = np.repeat(nan_mask[np.newaxis, :, :], number_of_time_steps, axis=0)  # shape: (bands, H, W)

	# Expand nan_mask to match input image shape: (bands, time, H, W)
	expanded_mask = np.repeat(nan_mask[np.newaxis, :, :, :], number_of_channels, axis=0)
	# Mask input: zero out regions with invalid ground truth
	normalized[expanded_mask] = 0

	# Convert to torch tensor with batch dimension
	normalized_tensor = torch.from_numpy(
		normalized.reshape(1, number_of_channels, number_of_time_steps, *image.shape[-2:])
	).to(torch.float32)
	
	return normalized_tensor

class cycle_dataset_patches(Dataset):
	def __init__(self,path,split, data_percentage=1.0):
		
		self.data_dir=path
		self.split=split
		
		self.total_below_0 = 0 
		self.total_above_365 = 0
		self.total_nan = 0
		self.total = 0
		self.data_percentage = data_percentage

		self.correct_indices = [2, 5, 8, 11]
		self.correct_indices = [i - 1 for i in self.correct_indices]  # Convert to zero-based index

		self.get_means_stds()

	def get_means_stds(self):

		"""
		Compute or load the mean and standard deviation for image data.
		Uses correct batch-wise variance combination (Chan et al., 1979).
		"""

		main_folder = "paper_fullsize_all_12month_match_location" if self.data_percentage == 1.0 else f"paper_fullsize_all_12month_match_location_{self.data_percentage}"
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

	def __len__(self):
		return len(self.data_dir)
	
	
	def fix_final_mask(self,final_mask):	
		mask = np.isnan(final_mask)
		final_mask = np.where(mask, -1, final_mask) #so the nan values don't cause an error
		
		total_nan_before = np.sum(final_mask == -1)

		valid_mask = ~mask
		final_mask = np.where(valid_mask & (final_mask < 0), 0, final_mask)
		final_mask = np.where(valid_mask & (final_mask > 365), 365, final_mask)
		final_mask = day_of_year_to_decimal_month(final_mask)

		total_nan_after = np.sum(final_mask == -1)
		assert total_nan_before == total_nan_after, "Total nan before and after are not the same"

		return final_mask.astype(np.float32)
	
	def __getitem__(self,idx):
		
		image_path=self.data_dir[idx][0]
		output_path=self.data_dir[idx][1]
		hls_tile_name = self.data_dir[idx][2]

		images = []
		for path in image_path: 
			images.append(load_raster_input(path)[:, np.newaxis])

		gt_mask=load_raster_output(output_path)
		gt_mask = gt_mask[self.correct_indices, :, :]

		image = np.concatenate(images, axis=1)
		final_image=preprocess_image(image,gt_mask, self.means,self.stds)
		gt_mask = self.fix_final_mask(gt_mask)

		to_return = {
			"image": final_image,
			"image_unprocessed": image,
			"gt_mask": gt_mask,
			"hls_tile_name": hls_tile_name,
		}
	
		return to_return
	
