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
from prithvi_hf.unet import UNet3D
from prithvi_hf.prithvi_mini import TinyPrithviSeg
from prithvi_hf.lsp_mlp_pixels import PixelTemporalMLP
from prithvi_hf.lsp_transformer_patches import TemporalTransformerPerPatch
from prithvi_hf.prithvi_lora import PrithviSegLora

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

		if model_name not in ["freeze-False_loadcheckpoint-True_fullsize_sigmoid_region-all_1.0", "transformer_lsp_fullsize_pixels_all_1.0", "loadcheckpoint-True_fullsize_sigmoid_lora_region-all_1.0", "freeze-False_loadcheckpoint-True_fullsize_sigmoid_sampler_region-all_1.0"]:
			continue

		data_percentage = model_name.split("_")[-1] 
		if "pixels" in model_name or "patch" in model_name:
			region_to_cut = " ".join(model_name.split("_")[4:-1]).upper()
		else: 
			region_to_cut = " ".join(model_name.split("region-")[1].split("_")[:-1]).upper()

		print(data_percentage, region_to_cut)

		split = "train"
		if split == "train":
			path_test=data_path_paper_all_12month_match("training", data_percentage, region_to_cut) 
			cycle_dataset_test=cycle_dataset(path_test,split="training", data_percentage=data_percentage, region_to_cut_name=region_to_cut)
			
			train_dataloader=DataLoader(cycle_dataset_test,batch_size=1,shuffle=config["training"]["shuffle"],num_workers=2)

			data_loader = train_dataloader
			data_loader_name = "train"

		elif split == "val":
			path_test=data_path_paper_all_12month_match("validation", data_percentage, region_to_cut) 
			cycle_dataset_test=cycle_dataset(path_test,split="validation", data_percentage=data_percentage, region_to_cut_name=region_to_cut)
			
			val_dataloader=DataLoader(cycle_dataset_test,batch_size=1,shuffle=config["validation"]["shuffle"],num_workers=2)

			data_loader = val_dataloader
			data_loader_name = "val"
		
		else: 
			raise ValueError(f"Invalid split: {split}")


		if os.path.exists(f"results_{split}/{model_name}_{data_loader_name}.csv"):
			print(f"Results for {model_name} on {data_loader_name} already exist, skipping...")
			continue

		config_dir = f"/usr4/cs505/mqraitem/ivc-ml/geo/checkpoints/distance/{region}/{model_name}/"
		best_param = best_param_df[best_param_df["Group"] == model_name]["Best Param"].values[0] 

		print(f"Best parameters: {best_param}")


		if "unet" in model_name:
			model = UNet3D(
				in_channel=6,
				n_classes=4,
				timesteps=12,
				dropout=0.1
			)

		elif "lora" in model_name:
			r_param = int(best_param.split("_")[3].replace("r-", ""))
			alpha_param = int(best_param.split("_")[4].replace("alpha-", "").replace(".pth", ""))
			lora_dict = {
				"Lora_peft_layer_name_pre": config["Lora_peft_layer_name"][0],
				"Lora_peft_layer_name_suffix": config["Lora_peft_layer_name"][1],
				"LP_layer_no_start": config["Lora_peft_layer_no"][0],
				"LP_layer_no_end": config["Lora_peft_layer_no"][1]
			}
			model = PrithviSegLora(config["pretrained_cfg"], lora_dict, None, True, n_classes=4, model_size=args.model_size, r=r_param, alpha=alpha_param)


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

		elif "lsp" in model_name and "patch" in model_name:
			patch_size = int(model_name.split("_")[3].replace("patch", ""))
			model = TemporalTransformerPerPatch(
				input_channels=6,
				seq_len=12,
				num_classes=4,
				d_model=128,
				nhead=4,
				num_layers=3,
				dropout=0.1,
				patch_size=(patch_size, patch_size),
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


		else: 
			model=PrithviSeg(config["pretrained_cfg"], None, True, n_classes=4, model_size=args.model_size)

		model=model.to(device)
		model.load_state_dict(torch.load(os.path.join(config_dir, best_param))["model_state_dict"])
		out_df = eval_data_loader_df(data_loader, model, device, get_masks_paper(split))
		
		out_df.to_csv(f"results_{split}/{model_name}_{data_loader_name}.csv", index=False)


if __name__ == "__main__":
	main()


