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

from utils import get_masks_paper, eval_data_loader_df

from utils import data_path_paper_all_12month_match
from dataloader_fullsize_all import cycle_dataset


ALL_DAYS = list(range(1, 31))

#######################################################################################

def main():
	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_size", type=str, default="300m", help="Size of the model to use")
	args = parser.parse_args()

	with open(f'configs/config_{args.model_size}.yaml', 'r') as f:
		config = yaml.safe_load(f)

	device = "cuda"
	config["pretrained_cfg"]["img_size"] = 336
	
	results = {
		"Model": [],
		"Region": [],
		"Accuracy": [],
		"Date": [], 
		"Tile": [], 
	}

	results_ci = {
		"Model": [],
		"Region": [],
		"Accuracy": [],
		"Date": [], 
	}

	cis_names = [str(x) + " Days" for x in ALL_DAYS]
	for ci_name in cis_names:
		results_ci[f"CI {ci_name}"] = []

	region = "paper_fullsize_12month_match_location"
	best_param_df = pd.read_csv(f"results/best_param_{args.model_size}_{region}.csv")
	model_names = best_param_df["Group"].values

	for model_name in model_names:

		# if "north" in model_name:
		# 	continue
		# if "eastern" in model_name:
		# 	continue

		data_percentage = model_name.split("_")[-1] 
		if "pixels" in model_name:
			region_to_cut = " ".join(model_name.split("_")[4:-1]).upper()
		else: 
			region_to_cut = " ".join(model_name.split("region-")[1].split("_")[:-1]).upper()

		path_test=data_path_paper_all_12month_match("testing", data_percentage, region_to_cut) 
		cycle_dataset_test=cycle_dataset(path_test,split="test", data_percentage=data_percentage, region_to_cut_name=region_to_cut)

		# path_test=data_path_paper_all_12month_match("testing") 
		# cycle_dataset_test=cycle_dataset(path_test,split="test")
		
		test_dataloader=DataLoader(cycle_dataset_test,batch_size=config["test"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=2)


		data_loader = test_dataloader
		data_loader_name = "test"

		if os.path.exists(f"results/{model_name}_{data_loader_name}.csv"):
			print(f"Results for {model_name} on {data_loader_name} already exist, skipping...")
			continue

		config_dir = f"/usr4/cs505/mqraitem/ivc-ml/geo/checkpoints/distance/{region}/{model_name}/"
		best_param = best_param_df[best_param_df["Group"] == model_name]["Best Param"].values[0] 

		print(f"Best parameters: {best_param}")

		if "unet" in model_name:
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

		elif "spatial" in model_name:
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

		elif "mlp" in model_name:
			model = PixelTemporalMLP(
				input_channels=6,
				seq_len=12,
				num_classes=4,
				hidden=128,
				layers=3,
				dropout=0.1,
			)

		elif "qtransformer" in model_name:
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
							

		elif "lsp" in model_name:
			model = TemporalTransformer(
				input_channels=6,
				seq_len=12,
				num_classes=4,
				d_model=128,
				nhead=4,
				num_layers=3,
				dropout=0.1
			)

		elif "sigmoidtemp" in model_name:
			from prithvi_hf.prithvi_temp import PrithviSeg
			model = PrithviSeg(config["pretrained_cfg"], None, True, n_classes=4, model_size=args.model_size)


		else: 
			model=PrithviSeg(config["pretrained_cfg"], None, True, n_classes=4, model_size=args.model_size)

		model=model.to(device)
		model.load_state_dict(torch.load(os.path.join(config_dir, best_param))["model_state_dict"])

		if "qtransformer" in model_name:
			pca_feats_path = config["data_dir"] + f"/HLS_composites_HP-LSP_PCA_Feats/"
			test_feats_path = pca_feats_path + "test/"
			out_df = eval_data_loader_df(data_loader, model, device, get_masks_paper(data_loader_name), test_feats_path)
		else: 
			out_df = eval_data_loader_df(data_loader, model, device, get_masks_paper(data_loader_name))
		
		out_df.to_csv(f"results/{model_name}_{data_loader_name}.csv", index=False)


if __name__ == "__main__":
	main()


