import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
import random
from collections import Counter

def print_trainable_parameters(model):
	"""
	Prints the number of trainable parameters in the model.
	"""
	trainable_params = 0
	all_param = 0
	for _, param in model.named_parameters():
		all_param += param.numel()
		if param.requires_grad:
			trainable_params += param.numel()
	print(
		f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
	)


def segmentation_loss_pixels(targets, preds, device, ignore_index=-1):
	"""
	Compute regression loss for pixel dataset.
	
	Args:
		targets: (B,) tensor of ground truth labels (float)
		preds:   (B,) or (B, num_outputs) tensor of predictions
		device:  torch device
		ignore_index: value in targets to ignore (default -1)
	"""
	criterion = nn.MSELoss(reduction="sum").to(device)

	# valid mask = targets not equal to ignore_index
	valid_mask = targets != ignore_index

	if valid_mask.sum() > 0:
		valid_pred = preds[valid_mask]
		valid_target = targets[valid_mask] / 12.0   # normalize like before
		loss = criterion(valid_pred, valid_target)
		return loss / valid_mask.sum().item()
	else:
		return torch.tensor(0.0, device=device)



def segmentation_loss(mask, pred, device, ignore_index=-1):
	mask = mask.float()  # Convert mask to float for regression loss
	
	criterion = nn.MSELoss(reduction="sum").to(device)

	loss = 0
	num_channels = pred.shape[1]  # Number of output channels
	total_valid_pixels = 0  # Counter for valid pixels

	for idx in range(num_channels):
		# Get valid mask (excluding ignore_index)
		valid_mask = mask[:, idx] != ignore_index

		if valid_mask.sum() > 0:  # Ensure there are valid pixels to compute loss

			valid_pred = pred[:, idx][valid_mask]  # Apply mask to predictions
			valid_target = mask[:, idx][valid_mask]  # Apply mask to ground truth

			valid_target = valid_target/12

			loss += criterion(valid_pred, valid_target)
			total_valid_pixels += valid_mask.sum().item()

	# Normalize by total valid pixels to avoid division by zero
	return loss / total_valid_pixels if total_valid_pixels > 0 else torch.tensor(0.0, device=device)

def get_masks_paper(data="train", device="cuda"):

	test_file = f"/usr4/cs505/mqraitem/ivc-ml/geo/data/LSP_{data}_samples.csv"
	
	data_paper_df = pd.read_csv(test_file)
	data_paper_df = data_paper_df[data_paper_df["version"] == "v1"]

	tiles_paper_masks = {}

	# Group by (year, tile)
	for (year, site_id, tile), group in data_paper_df.groupby(['years', "SiteID", 'tile']):
		# Initialize 320x320 mask with False
		mask = np.zeros((330, 330), dtype=bool)
		
		#subtract by 1 except 0 
		# Set True where (row, col) is mentioned in the group
		mask[group['row'].values, group['col'].values] = True
		
		# Store the mask with key (year, tile)

		mask = torch.Tensor(mask).bool().to(device)
		tiles_paper_masks[f"{year}_{site_id}_{tile}"] = mask
	
	return tiles_paper_masks

def compute_accuracy(gt_hls_tile, pred_hls_tile_avg, all_errors_hls_tile, hls_tile_n, paper_mask):

	for idx in range(4): 

		pred_idx = pred_hls_tile_avg[idx]
		gt_idx = gt_hls_tile[idx]

		pred_idx = pred_idx.flatten()
		gt_idx = gt_idx.flatten()

		mask = (gt_idx != -1) & paper_mask.flatten()
		pred_idx = pred_idx[mask]
		gt_idx = gt_idx[mask]

		errors = (pred_idx - gt_idx).detach().cpu().numpy()
		all_errors_hls_tile[hls_tile_n][idx] = np.mean(np.abs(errors))

	return all_errors_hls_tile


def eval_data_loader(data_loader,model, device, tiles_paper_masks, feats_path=None):

	model.eval()

	all_errors_hls_tile = {}

	eval_loss = 0.0
	with torch.no_grad():
		for _,data in tqdm(enumerate(data_loader), total=len(data_loader)):
			

			input = data["image"].to(device)[:, 0]
			ground_truth = data["gt_mask"].to(device)
			hls_tile_name = data["hls_tile_name"]
			if feats_path is not None:
				feats = [] 
				for hls_tile_name in hls_tile_name:
					feats.append(np.load(feats_path + hls_tile_name + ".npz")["Z"])

				feats = np.array(feats)
				feats = torch.from_numpy(feats).to(device).float()
				assert feats.shape[0] == input.shape[0], f"Feats shape {feats.shape} does not match input shape {input.shape}"
				predictions=model(input, z_ctx=feats)
			
			else:
				predictions=model(input)

			predictions = predictions[:, :, :330, :330]
			eval_loss += segmentation_loss(mask=data["gt_mask"].to(device),pred=predictions,device=device).item() * input.size(0)  # Multiply by batch size

			predictions = predictions * 12 
			pred_hls_tile_all = predictions  # Average over the last dimension	

			for gt_hls_tile, pred_hls_tile_avg, hls_tile_n in zip(ground_truth, pred_hls_tile_all, data["hls_tile_name"]): 

				assert hls_tile_n not in all_errors_hls_tile, f"Tile {hls_tile_n} already exists in all_errors_hls_tile"
				all_errors_hls_tile[hls_tile_n] = {i:0 for i in range(4)}  # Initialize errors for each of the 4 predicted dates
				all_errors_hls_tile = compute_accuracy(gt_hls_tile, pred_hls_tile_avg, all_errors_hls_tile, hls_tile_n, tiles_paper_masks[hls_tile_n])

	all_errors_time = {i:[] for i in range(4)}
	for tile in all_errors_hls_tile:
		for i in range(4):
			all_errors_time[i].append(all_errors_hls_tile[tile][i])

	acc_dataset_val = {i:np.mean(all_errors_time[i]) for i in range(4)}
	epoch_loss_val = eval_loss / len(data_loader.dataset)

	return acc_dataset_val, all_errors_hls_tile, epoch_loss_val


def eval_data_loader_patch(data_loader,model, device, tiles_paper_masks, patch_size):

	model.eval()

	all_errors_hls_tile = {}

	eval_loss = 0.0
	with torch.no_grad():
		for _,data in tqdm(enumerate(data_loader), total=len(data_loader)):
			

			input = data["image"].to(device)[:, 0]
			ground_truth = data["gt_mask"].to(device)

			# predictions=model(input)
			# predictions = predictions[:, :, :330, :330]
			predictions = model.forward_inference_tiled(input, patch_size=patch_size)


			eval_loss += segmentation_loss(mask=data["gt_mask"].to(device),pred=predictions,device=device).item() * input.size(0)  # Multiply by batch size

			predictions = predictions * 12 
			pred_hls_tile_all = predictions  # Average over the last dimension	

			for gt_hls_tile, pred_hls_tile_avg, hls_tile_n in zip(ground_truth, pred_hls_tile_all, data["hls_tile_name"]): 

				assert hls_tile_n not in all_errors_hls_tile, f"Tile {hls_tile_n} already exists in all_errors_hls_tile"
				all_errors_hls_tile[hls_tile_n] = {i:0 for i in range(4)}  # Initialize errors for each of the 4 predicted dates
				all_errors_hls_tile = compute_accuracy(gt_hls_tile, pred_hls_tile_avg, all_errors_hls_tile, hls_tile_n, tiles_paper_masks[hls_tile_n])

	all_errors_time = {i:[] for i in range(4)}
	for tile in all_errors_hls_tile:
		for i in range(4):
			all_errors_time[i].append(all_errors_hls_tile[tile][i])

	acc_dataset_val = {i:np.mean(all_errors_time[i]) for i in range(4)}
	epoch_loss_val = eval_loss / len(data_loader.dataset)

	return acc_dataset_val, all_errors_hls_tile, epoch_loss_val


def eval_data_loader_df(data_loader,model, device, tiles_paper_masks, feats_path=None):

	model.eval()

	#G_pred_DOY  M_pred_DOY  S_pred_DOY  D_pred_DOY
	data_df = { 
		"index":[],
		"years":[],
		"HLStile":[], 
		"SiteID": [],
		"row":[],
		"col":[],
		"version":[],
		"G_pred_DOY":[],
		"M_pred_DOY":[],
		"S_pred_DOY":[],
		"D_pred_DOY":[],
		"G_truth_DOY":[],
		"M_truth_DOY":[],
		"S_truth_DOY":[],
		"D_truth_DOY":[]
	}


	eval_loss = 0.0
	with torch.no_grad():
		for _,data in tqdm(enumerate(data_loader), total=len(data_loader)):
			
			input = data["image"].to(device)[:, 0]
			ground_truth = data["gt_mask"].to(device)
			hls_tile_name = data["hls_tile_name"]

			if feats_path is not None:
				feats = [] 
				for hls_tile_name in hls_tile_name:
					feats.append(np.load(feats_path + hls_tile_name + ".npz")["Z"])
				feats = np.array(feats)
				feats = torch.from_numpy(feats).to(device).float()
				assert feats.shape[0] == input.shape[0], f"Feats shape {feats.shape} does not match input shape {input.shape}"
				predictions=model(input, z_ctx=feats)
			else:
				predictions=model(input)

			predictions = predictions[:, :, :330, :330]
			predictions = predictions * 12 

			pred_hls_tile_all = predictions

			eval_loss += segmentation_loss(mask=data["gt_mask"].to(device),pred=predictions,device=device).item() * input.size(0)  # Multiply by batch size

			for gt_hls_tile, pred_hls_tile_avg, hls_tile_n in zip(ground_truth, pred_hls_tile_all, data["hls_tile_name"]): 
				
				mask_tilen = tiles_paper_masks[hls_tile_n] 
				year, siteid, hlstile = hls_tile_n.split("_")
				#get the row and col from the mask
				row, col = np.where(mask_tilen.cpu().numpy())
				#add all data for row/col, don't worry about -1
				#do it in parralel 

				for r, c in zip(row, col):
					data_df["index"].append(len(data_df["index"]))
					data_df["years"].append(year)
					data_df["HLStile"].append(hlstile)
					data_df["SiteID"].append(siteid)
					data_df["row"].append(r)
					data_df["col"].append(c)
					data_df["version"].append("v1")
					data_df["G_pred_DOY"].append(pred_hls_tile_avg[0, r, c].item()*30)
					data_df["M_pred_DOY"].append(pred_hls_tile_avg[1, r, c].item()*30)
					data_df["S_pred_DOY"].append(pred_hls_tile_avg[2, r, c].item()*30)
					data_df["D_pred_DOY"].append(pred_hls_tile_avg[3, r, c].item()*30)
					data_df["G_truth_DOY"].append(gt_hls_tile[0, r, c].item()*30)
					data_df["M_truth_DOY"].append(gt_hls_tile[1, r, c].item()*30)
					data_df["S_truth_DOY"].append(gt_hls_tile[2, r, c].item()*30)
					data_df["D_truth_DOY"].append(gt_hls_tile[3, r, c].item()*30)

	data_df = pd.DataFrame(data_df)
	return data_df	





def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filename):

	checkpoint = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'train_loss': train_loss,
		'val_loss':val_loss
	}
	torch.save(checkpoint, filename)
	print(f"Checkpoint saved at {filename}")
	
def data_path_paper_all_12month_match(mode, data_percentage=1.0, region_to_cut="EASTERN TEMPERATE FORESTS"): 

	region_to_cut_name = region_to_cut.replace(" ", "_").lower()
	data_dir_name = f"paper_fullsize_all_12month_match_location_{region_to_cut_name}_{data_percentage}"
	checkpoint_data = f"/usr4/cs505/mqraitem/ivc-ml/geo/data/{data_dir_name}/"

	if os.path.exists(f'{checkpoint_data}/data_pths_{mode}.pkl'):
		with open(f'{checkpoint_data}/data_pths_{mode}.pkl', 'rb') as f:
			data_dir = pickle.load(f)

		return data_dir

	hls_path = f"/projectnb/hlsfm/applications/lsp/outputs/HLS_composites_HP-LSP"
	lsp_path = f"/projectnb/hlsfm/applications/lsp/ancillary/HP_LSP"

	hls_tiles = [x for x in os.listdir(hls_path) if x.endswith('.tif')]
	lsp_tiles = []
	lsp_tiles.extend([x for x in os.listdir(f"{lsp_path}/A2019") if x.endswith('.tif')])
	lsp_tiles.extend([x for x in os.listdir(f"{lsp_path}/A2020") if x.endswith('.tif')])

	title_hls = ['_'.join(x.split('_')[3:5]).split(".")[0] for x in hls_tiles]
	title_hls = set(title_hls)

	title_hls_lsp = ["_".join(x.split('_')[3:5]) for x in lsp_tiles]
	title_hls_lsp = set(title_hls_lsp)

	hls_tiles_time = [] 
	lsp_tiles_time = []
	hls_tiles_name = []

	for year in ["2019", "2020"]:
		past_months = range(1, 13)
		timesteps = [f"{year}-{str(x).zfill(2)}" for x in past_months]

		for hls_tile in tqdm(title_hls):
			temp_ordered = [] 

			for timestep in timesteps:

				hls_tile_location = hls_tile.split("_")[0]
				hls_tile_name = hls_tile.split("_")[1]

				temp_ordered.append(f"{hls_path}/HLS_composite_{timestep}_{hls_tile_location}_{hls_tile_name}.tif")

			temp_lsp = f"{lsp_path}/A{year}/HLS_PhenoCam_A{year}_{hls_tile_location}_{hls_tile_name}_LSP_Date.tif" if f"HLS_PhenoCam_A{year}_{hls_tile_location}_{hls_tile_name}_LSP_Date.tif" in lsp_tiles else None

			hls_tiles_time.append(temp_ordered)
			lsp_tiles_time.append(temp_lsp)
			hls_tiles_name.append(f"{year}_{hls_tile}")

	#open training file 
	with open(f"{lsp_path}/HP-LSP_train_ids.csv", 'r') as f:
		train_ids = f.readlines()[0].replace("'", "").split(",")
		train_ids = [x.strip() for x in train_ids]

	with open(f"{lsp_path}/HP-LSP_test_ids.csv", 'r') as f:
		test_ids = f.readlines()[0].replace("'", "").split(",")
		test_ids = [x.strip() for x in test_ids]


	hls_tiles_val = [
		"2019_ME-1_T19TEL",
		"2019_FL-3_T17RML",
		"2020_WI-2_T15TYL",
		"2019_AZ-5_T12SVE",
		"2020_CO-2_T13TDE",
		"2020_OR-1_T10TEQ",
		"2019_MD-1_T18SUJ",
		"2020_ND-1_T14TLS"
	]

	hls_tiles_train = [x for x in hls_tiles_name if x.split("_")[1] in train_ids]
	hls_tiles_train = [x for x in hls_tiles_train if x not in hls_tiles_val]

	region_to_states = pickle.load(open("region_to_states.pkl", "rb"))

	if region_to_cut == "ALL":
		states_to_cut = []
		for regions in region_to_states.values():
			states_to_cut.extend(regions)
	else:
		states_to_cut = region_to_states[region_to_cut]
	
	states_to_keep = states_to_cut[:int(len(states_to_cut) * data_percentage)]
	
	if region_to_cut != "ALL":
		for region in region_to_states:
			if region != region_to_cut:
				states_to_keep.extend(region_to_states[region])

	hls_tiles_train = [x for x in hls_tiles_train if x.split("_")[1] in states_to_keep]
	hls_tiles_test = [x for x in hls_tiles_name if x.split("_")[1] in test_ids]
	data_dir_train = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_train]

	data_dir_val = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_val]
	data_dir_test = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_test]

	os.makedirs(checkpoint_data, exist_ok=True)
	with open(f'{checkpoint_data}/data_pths_training.pkl', 'wb') as f:
		pickle.dump(data_dir_train, f)
	
	with open(f'{checkpoint_data}/data_pths_validation.pkl', 'wb') as f:
		pickle.dump(data_dir_val, f)
	
	with open(f'{checkpoint_data}/data_pths_testing.pkl', 'wb') as f:
		pickle.dump(data_dir_test, f)

	if mode == 'training':	
		return data_dir_train
	elif mode == 'validation':
		return data_dir_val
	elif mode == "testing":
		return data_dir_test
	else: 
		raise ValueError(f"Unknown mode: {mode}. Expected 'training', 'validation', or 'testing'.")

