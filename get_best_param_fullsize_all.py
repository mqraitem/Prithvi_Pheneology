import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import argparse
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append("../")
from prithvi_hf.prithvi import PrithviSeg
from prithvi_hf.lsp_transformer_pixels import TemporalTransformer
from prithvi_hf.lsp_unet import UNet3DTimeAware
from prithvi_hf.lsp_transformer_spatial import TinyPrithviSeg
from prithvi_hf.lsp_mlp_pixels import PixelTemporalMLP
from prithvi_hf.lsp_transformer_pixels_feats import TemporalQFormer


from utils import data_path_paper_all_12month_match, eval_data_loader, get_masks_paper
from dataloader_fullsize_all import cycle_dataset

#######################################################################################

def main():

	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_size", type=str, default="300m", help="Size of the model to use")
	args = parser.parse_args()

	with open(f'configs/config_{args.model_size}.yaml', 'r') as f:
		config = yaml.safe_load(f)


	config["pretrained_cfg"]["img_size"] = 336

	device = "cuda"
	regions = ["paper_fullsize_12month_match_location"]

	for region in regions: 


		groups_dir = f"/usr4/cs505/mqraitem/ivc-ml/geo/checkpoints/distance/{region}"

		all_groups = os.listdir(groups_dir)

		if os.path.exists(f"results/best_param_{args.model_size}_{region}.csv"):
			param_df = pd.read_csv(f"results/best_param_{args.model_size}_{region}.csv")
			best_param_df_cached = param_df.to_dict(orient="list")

			already_done_groups = best_param_df_cached["Group"]
			all_groups = [group for group in all_groups if group not in already_done_groups]

			best_param_df = {}
			best_param_df["Group"] = best_param_df_cached["Group"]
			best_param_df["Best Param"] = best_param_df_cached["Best Param"]
		else: 
			best_param_df = {}
			best_param_df["Group"] = []
			best_param_df["Best Param"] = []
			all_groups = all_groups


		for group in tqdm(all_groups):

			data_percentage = group.split("_")[-1] 

			if "pixels" in group:
				region_to_cut = " ".join(group.split("_")[4:-1]).upper()
			else: 
				region_to_cut = " ".join(group.split("region-")[1].split("_")[:-1]).upper()

			path_val=data_path_paper_all_12month_match("validation", data_percentage, region_to_cut)
			cycle_dataset_val=cycle_dataset(path_val,split="val", data_percentage=data_percentage, region_to_cut_name=region_to_cut)
			val_dataloader=DataLoader(cycle_dataset_val,batch_size=config["validation"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=2)

			best_param = None 
			best_acc = 1000 

			for params in os.listdir(os.path.join(groups_dir, group)):

				if not params.endswith(".pth"):
					continue
				checkpoint = os.path.join(groups_dir, group, params)
				print(f"Loading checkpoint: {checkpoint}")

				# Load the model
				weights_path = None
				if "unet" in group:
					model = UNet3DTimeAware(
						in_ch=6,
						num_classes=4,
						base_ch=32,   # try 32 or 48; 32 is pretty lightweight
						depth=4,      # spatial pyramid /16; needs H,W divisible by 16
						k_t=3,
						norm='bn',
						dropout=0.0,
						time_pool='mean',  # 'conv' if you want a tiny learned temporal collapse
					)

				elif "spatial" in group:
					model = TinyPrithviSeg(
						in_ch=6,
						T=12,                 # match baseline seq_len
						img_size=336,
						patch=(1,16,16),      # keeps tokens small; 12×21×21 tokens
						d_model=132,           # modest width
						depth=3,              # 3 encoder layers
						nhead=4,              # 80 / 4 = 20 per head
						num_classes=4,
						up_depth=4,           # /16 -> /8 -> /4 -> /2 -> /1
					)

				elif "mlp" in group:
					model = PixelTemporalMLP(
						input_channels=6,
						seq_len=12,
						num_classes=4,
						hidden=128,
						layers=3,
						dropout=0.1,
					)

				elif "qtransformer" in group:
					model = TemporalQFormer(
						input_channels=6,
						ctx_channels=1024,
						seq_len=12,
						num_classes=4,
						d_model=128,
						nhead=4,
						num_layers=3,   # self-attn layers on pixel tokens
						dropout=0.1,
						fusion="qformer",   # try 'concat' for a very simple baseline
						num_xattn=2,        # number of cross-attn blocks
						mlp_ratio=4.0
					)

				elif "lsp" in group:
					model = TemporalTransformer(
						input_channels=6,
						seq_len=12,
						num_classes=4,
						d_model=128,
						nhead=4,
						num_layers=3,
						dropout=0.1
					)

				elif "sigmoidtemp" in group:
					from prithvi_hf.prithvi_temp import PrithviSeg
					model = PrithviSeg(config["pretrained_cfg"], weights_path, True, n_classes=4, model_size=args.model_size)

				else: 
					model=PrithviSeg(config["pretrained_cfg"], weights_path, True, n_classes=4, model_size=args.model_size)

				model=model.to(device)
				model.load_state_dict(torch.load(checkpoint)["model_state_dict"])

				pca_feats_path = config["data_dir"] + f"/HLS_composites_HP-LSP_PCA_Feats/"
				val_feats_path = pca_feats_path + "val/"

				if "qtransformer" in group:
					acc_dataset_val, _, _ = eval_data_loader(val_dataloader, model, device, get_masks_paper("train"), val_feats_path)
				else: 
					acc_dataset_val, _,  _ = eval_data_loader(val_dataloader, model, device, get_masks_paper("train"))

				print(f"Region: {region}")
				print(f"Parameters: {params}")
				print(f"Test avg acc: {np.mean(list(acc_dataset_val.values()))}")	

				if np.mean(list(acc_dataset_val.values())) < best_acc:
					best_acc = np.mean(list(acc_dataset_val.values()))
					best_param = params

			print(f"Best parameters: {best_param}")
			best_param_df["Group"].append(group)
			best_param_df["Best Param"].append(best_param)


	os.makedirs("results", exist_ok=True)
	best_param_df = pd.DataFrame(best_param_df)
	best_param_df.to_csv(f"results/best_param_{args.model_size}_{region}.csv", index=False)

if __name__ == "__main__":
	main()
