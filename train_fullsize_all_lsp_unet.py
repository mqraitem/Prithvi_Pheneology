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

from prithvi_hf.lsp_unet import UNet3DTimeAware

from utils import segmentation_loss, eval_data_loader, get_masks_paper, save_checkpoint,str2bool
from utils import data_path_paper_all_12month_match
from utils import print_trainable_parameters

from dataloader_fullsize_all import cycle_dataset

#######################################################################################

def main():

	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the model")
	parser.add_argument("--freeze", type=str2bool, default=False, help="Whether to unfreeze the model or not")
	parser.add_argument("--logging", type=str2bool, default=False, help="Whether to log the results or not")
	parser.add_argument("--model_size", type=str, default="300m", help="Size of the model to use")
	parser.add_argument("--load_checkpoint", type=str2bool, default=False, help="Whether to load a checkpoint or not")
	parser.add_argument("--group_name", type=str, default="default", help="Group name for wandb")
	parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
	parser.add_argument("--data_percentage", type=float, default=1.0, help="Data percentage to use")
	parser.add_argument("--using_sampler", type=str2bool, default=False, help="Whether to use sampler or not")
	parser.add_argument("--region_to_cut", type=str, default="EASTERN_TEMPERATE_FORESTS", help="Region to cut")

	args = parser.parse_args()
	args.region_to_cut_name = args.region_to_cut.lower()
	args.region_to_cut = args.region_to_cut.replace("_", " ")

	wandb_config = {
		"learningrate": args.learning_rate,
		"freeze": args.freeze,
		"model_size": args.model_size,
		"load_checkpoint": args.load_checkpoint, 
		"batch_size": args.batch_size,
		"data_percentage": args.data_percentage,
	}

	wandb_name = str(wandb_config["learningrate"]) + "_batch_size-" + str(args.batch_size)

	with open(f'configs/config_{args.model_size}.yaml', 'r') as f:
		config = yaml.safe_load(f)

	config["training"]["n_iteration"] = 200
	config["pretrained_cfg"]["img_size"] = 336

	config["training"]["batch_size"] = args.batch_size
	config["validation"]["batch_size"] = args.batch_size
	config["test"]["batch_size"] = args.batch_size

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

	cycle_dataset_train=cycle_dataset(path_train,split="training", data_percentage=args.data_percentage, region_to_cut_name=args.region_to_cut)
	cycle_dataset_val=cycle_dataset(path_val,split="validation", data_percentage=args.data_percentage, region_to_cut_name=args.region_to_cut)
	cycle_dataset_test=cycle_dataset(path_test,split="testing", data_percentage=args.data_percentage, region_to_cut_name=args.region_to_cut)

	from torch.utils.data import WeightedRandomSampler

	sampler = WeightedRandomSampler(
		weights=cycle_dataset_train.sample_weights,
		num_samples=len(cycle_dataset_train),
		replacement=True
	)


	if args.using_sampler:
		train_dataloader=DataLoader(cycle_dataset_train,batch_size=config["training"]["batch_size"],num_workers=1, sampler=sampler)
	else:
		train_dataloader=DataLoader(cycle_dataset_train,batch_size=config["training"]["batch_size"],shuffle=config["training"]["shuffle"],num_workers=2)

	val_dataloader=DataLoader(cycle_dataset_val,batch_size=config["validation"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=1)
	test_dataloader=DataLoader(cycle_dataset_test,batch_size=config["test"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=1)



	device = "cuda"
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
	
	model=model.to(device)
	print_trainable_parameters(model)

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

			input=input.to(device)[:, 0]
			mask=mask.to(device)

			optimizer.zero_grad()
			out=model(input)

			out = out[:, :, :330, :330]

			loss=segmentation_loss(mask=mask,pred=out,device=device)
			loss_i += loss.item() * input.size(0)  # Multiply by batch size

			loss.backward()
			optimizer.step()

			if j%10==0:
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
