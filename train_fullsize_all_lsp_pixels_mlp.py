import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import wandb
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from prithvi_hf.lsp_mlp_pixels import PixelTemporalMLP
from utils import segmentation_loss, segmentation_loss_pixels, eval_data_loader, get_masks_paper, print_trainable_parameters, save_checkpoint,str2bool

from utils import data_path_paper_all_12month_match
from dataloader_fullsize_all_pixels import cycle_dataset_pixels
from dataloader_fullsize_all import cycle_dataset
from utils import print_trainable_parameters

#######################################################################################

def main():

	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the model")
	parser.add_argument("--logging", type=str2bool, default=False, help="Whether to log the results or not")
	parser.add_argument("--group_name", type=str, default="default", help="Group name for wandb")
	parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
	parser.add_argument("--data_percentage", type=float, default=1.0, help="Data percentage to use")
	parser.add_argument("--region_to_cut", type=str, default="EASTERN_TEMPERATE_FORESTS", help="Region to cut")

	args = parser.parse_args()
	args.region_to_cut_name = args.region_to_cut.lower()
	args.region_to_cut = args.region_to_cut.replace("_", " ")

	wandb_config = {
		"learningrate": args.learning_rate,
		"batch_size": args.batch_size,
		"data_percentage": args.data_percentage,
		"region_to_cut": args.region_to_cut,
	}

	args.model_size = "300m"
	wandb_name = str(wandb_config["learningrate"]) + "_batch_size-" + str(args.batch_size)

	with open(f'configs/config_{args.model_size}.yaml', 'r') as f:
		config = yaml.safe_load(f)

	config["training"]["n_iteration"] = 200
	config["pretrained_cfg"]["img_size"] = 336

	group_name = args.group_name 

	if args.logging: 
		wandb.init(
				project=f"location_33_{args.data_percentage}_{args.region_to_cut_name}",
				group=group_name,
				config = wandb_config, 
				name=wandb_name,
				)
		wandb.run.log_code(".")

	path_train=data_path_paper_all_12month_match("training", args.data_percentage, args.region_to_cut)
	path_val=data_path_paper_all_12month_match("validation", args.data_percentage, args.region_to_cut)
	path_test=data_path_paper_all_12month_match("testing", args.data_percentage, args.region_to_cut)

	cache_path_train=f"{config['data_dir']}/HLS_composites_HP-LSP_Pixels/cycle_dataset_pixels_train_{args.data_percentage}_{args.region_to_cut_name}.npz"

	cycle_dataset_train=cycle_dataset_pixels(path_train,split="training",cache_path=cache_path_train, data_percentage=args.data_percentage, region_to_cut_name=args.region_to_cut)
	cycle_dataset_val=cycle_dataset(path_val,split="validation", data_percentage=args.data_percentage, region_to_cut_name=args.region_to_cut)
	cycle_dataset_test=cycle_dataset(path_test,split="testing", data_percentage=args.data_percentage, region_to_cut_name=args.region_to_cut)


	config["training"]["batch_size"] = args.batch_size
	config["validation"]["batch_size"] = 1
	config["test"]["batch_size"] = 1

	train_dataloader=DataLoader(cycle_dataset_train,batch_size=config["training"]["batch_size"],shuffle=config["training"]["shuffle"],num_workers=2)
	val_dataloader=DataLoader(cycle_dataset_val,batch_size=config["validation"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=2)
	test_dataloader=DataLoader(cycle_dataset_test,batch_size=config["test"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=2)

	device = "cuda"

	model = PixelTemporalMLP(
		input_channels=6,
		seq_len=12,
		num_classes=4,
		hidden=128,
		layers=3,
		dropout=0.1,
	)

	print_trainable_parameters(model)
	model=model.to(device)

	group_name_checkpoint = f"{group_name}_{args.data_percentage}"
	checkpoint_dir = config["training"]["checkpoint_dir"] + f"/paper_fullsize_12month_match_location/{group_name_checkpoint}"
	os.makedirs(checkpoint_dir, exist_ok=True)
	checkpoint = f"{checkpoint_dir}/{wandb_name}.pth"
	
	optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)

	best_acc_val=100
	for epoch in range(config["training"]["n_iteration"]):

		loss_i=0.0

		print("iteration started")
		model.train()

		for j,batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

			input = batch_data["image"]
			mask = batch_data["gt_mask"]	

			input=input.to(device)
			mask=mask.to(device)

			optimizer.zero_grad()
			out=model(input, processing_images=False)

			loss=segmentation_loss_pixels(mask,out,device=device)
			loss_i += loss.item() * input.size(0)  # Multiply by batch size

			loss.backward()
			optimizer.step()

			if j%500==0:
				to_print = f"Epoch: {epoch}, iteration: {j}, loss: {loss.item()} \n "
				print(to_print)


		epoch_loss_train = loss_i / len(train_dataloader.dataset)

		# Validation Phase
		acc_dataset_val, _, epoch_loss_val = eval_data_loader(val_dataloader, model, device, get_masks_paper("train"))
		acc_dataset_test, _, epoch_loss_test = eval_data_loader(test_dataloader, model, device, get_masks_paper("test"))
		
		if args.logging: 
			to_log = {} 
			to_log["epoch"] = epoch + 1 
			to_log["val_loss"] = epoch_loss_val
			to_log["test_loss"] = epoch_loss_test
			to_log["train_loss"] = epoch_loss_train
			to_log["learning_rate"] = optimizer.param_groups[0]['lr']
			for idx in range(4):
				to_log[f"acc_val_{idx}"] = acc_dataset_val[idx]
				to_log[f"acc_test_{idx}"] = acc_dataset_test[idx]
			wandb.log(to_log)


		print("="*100)
		to_print = f"Epoch: {epoch}, val_loss: {epoch_loss_val} \n "
		for idx in range(4):
			to_print += f"acc_val_{idx}: {acc_dataset_val[idx]} \n "

		print(to_print)
		print("="*100)

		scheduler.step(epoch_loss_val)
		acc_dataset_val_mean = np.mean(list(acc_dataset_val.values()))

		if acc_dataset_val_mean<best_acc_val:
			save_checkpoint(model, optimizer, epoch, epoch_loss_train, epoch_loss_val, checkpoint)
			best_acc_val=acc_dataset_val_mean

	model.load_state_dict(torch.load(checkpoint)["model_state_dict"])

	acc_dataset_val, _, epoch_loss_val = eval_data_loader(val_dataloader, model, device, get_masks_paper("train"))
	acc_dataset_test, _, _ = eval_data_loader(test_dataloader, model, device, get_masks_paper("test"))

	if args.logging:
		for idx in range(4): 
			wandb.run.summary[f"best_acc_val_{idx}"] = acc_dataset_val[idx]
			wandb.run.summary[f"best_acc_test_{idx}"] = acc_dataset_test[idx]
		wandb.run.summary[f"best_avg_acc_val"] = np.mean(list(acc_dataset_val.values()))
		wandb.run.summary[f"best_avg_acc_test"] = np.mean(list(acc_dataset_test.values()))

	if args.logging: 
		wandb.finish()
	
	
if __name__ == "__main__":
	main()
