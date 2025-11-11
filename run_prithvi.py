import os 

records_dir = "records"

ALL_REGIONS = [
    "ALL",
]


# # ###############################################################
# # # * Prithvi Pretrained LORA
# # ###############################################################

# # 8, 8
# # 16, 32
# # 64, 32
# # 32, 64

# load_checkpoint = True
# for batch_size in [2]:
#     for data_percentage in [1.0]:
#         for region_to_cut in ALL_REGIONS:
#             region_to_cut_name = region_to_cut.replace(" ", "_").lower()
#             group_name = f"loadcheckpoint-{load_checkpoint}_fullsize_sigmoid_lora_region-{region_to_cut_name}"
            
#             for learning_rate in [0.001, 0.0001, 0.00001]:
#                 for r, alpha in [(8, 8), (16, 32), (64, 32), (32, 64)]:
#                     name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}_r-{r}_alpha-{alpha}"
#                     if os.path.exists(f"{records_dir}/{name}"):
#                         file_content = open(f"{records_dir}/{name}", "r").readlines()
#                         last_line = file_content[-1]
#                         if "wandb: Find logs" in last_line: 
#                             continue

#                     command = f"qsub -v args=' --r {r} --alpha {alpha} --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_prithvi_lora.sh"
#                     os.system(command)
# # ###############################################################

# ###############################################################
# # * Prithvi Pretrained Temporal Only
# ###############################################################
load_checkpoint = False
for freeze in [False]:
    for batch_size in [2]:
        for data_percentage in [1.0]:
            for region_to_cut in ALL_REGIONS:
                region_to_cut_name = region_to_cut.replace(" ", "_").lower()
                group_name = f"freeze-{freeze}_loadcheckpoint-{load_checkpoint}_fullsize_sigmoid_temporalonly_region-{region_to_cut_name}"
                
                for learning_rate in [0.001, 0.0001, 0.00001, 0.000001]:
                    
                    name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
                    if os.path.exists(f"{records_dir}/{name}"):
                        file_content = open(f"{records_dir}/{name}", "r").readlines()
                        last_line = file_content[-1]
                        if "wandb: Find logs" in last_line: 
                            continue

                    command = f"qsub -v args=' --temporal_only True --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate} --freeze {freeze}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_prithvi.sh"
                    os.system(command)
# ###############################################################

# # ###############################################################
# # # * Shallow Transformer Patch
# # ###############################################################
# for learning_rate in [0.001, 0.0001, 0.00001, 0.000001]:
#     for patch_size in [2]:
#         for batch_size in [int(264/patch_size), int(512/patch_size)]:
#             for data_percentage in [1.0]:
#                 for region_to_cut in ALL_REGIONS:

                    
#                     region_to_cut_name = region_to_cut.replace(" ", "_").lower()
#                     group_name = f"transformer_lsp_fullsize_patch{patch_size}_{region_to_cut_name}"

#                     name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
                    
#                     if os.path.exists(f"{records_dir}/{name}"):
#                         file_content = open(f"{records_dir}/{name}", "r").readlines()
#                         last_line = file_content[-1]
#                         if "wandb: Find logs" in last_line: 
#                             continue

#                         print("haha: ", name)

#                     command = f"qsub -v args=' --patch_size {patch_size} --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_lsp_patch_transformer.sh"
#                     os.system(command)
# # ###############################################################



# # ###############################################################
# # # * Mini Prithvi
# # ###############################################################
# for batch_size in [2]:
#     for data_percentage in [1.0]:
#         for region_to_cut in ALL_REGIONS:
#             region_to_cut_name = region_to_cut.replace(" ", "_").lower()
#             group_name = f"transformer_lsp_fullsize_spatial_region-{region_to_cut_name}"
            
#             for learning_rate in [0.001, 0.0001, 0.00001, 0.000001]:
                
#                 name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
#                 if os.path.exists(f"{records_dir}/{name}"):
#                     file_content = open(f"{records_dir}/{name}", "r").readlines()
#                     last_line = file_content[-1]
#                     if "wandb: Find logs" in last_line: 
#                         continue

#                 command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_mini_prithvi.sh"
#                 os.system(command)
# # ###############################################################


# # ###############################################################
# # # * MLP
# # ###############################################################
# for learning_rate in [0.001, 0.0001, 0.00001, 0.000001]:
#     for batch_size in [264, 512]:
#         for data_percentage in [1.0]:
#             for region_to_cut in ALL_REGIONS:

                
#                 region_to_cut_name = region_to_cut.replace(" ", "_").lower()
#                 group_name = f"mlp_lsp_fullsize_pixels_{region_to_cut_name}"

#                 name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
                
#                 if os.path.exists(f"{records_dir}/{name}"):
#                     file_content = open(f"{records_dir}/{name}", "r").readlines()
#                     last_line = file_content[-1]
#                     if "wandb: Find logs" in last_line: 
#                         continue

#                     # if "Disk quota exceeded" not in last_line:
#                     #     continue
#                     print("haha: ", name)

#                 command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_lsp_pixels_mlp.sh"
#                 os.system(command)
# # ###############################################################




# # ###############################################################
# # # * Prithvi Random
# # ###############################################################
# load_checkpoint = False
# for freeze in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for region_to_cut in ALL_REGIONS:
#                 region_to_cut_name = region_to_cut.replace(" ", "_").lower()
#                 group_name = f"freeze-{freeze}_loadcheckpoint-{load_checkpoint}_fullsize_sigmoid_region-{region_to_cut_name}"
                
#                 for learning_rate in [0.001, 0.0001, 0.00001, 0.000001]:
                    
#                     name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
#                     if os.path.exists(f"{records_dir}/{name}"):
#                         file_content = open(f"{records_dir}/{name}", "r").readlines()
#                         last_line = file_content[-1]
#                         if "wandb: Find logs" in last_line: 
#                             continue

#                     command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate} --freeze {freeze}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_prithvi.sh"
#                     os.system(command)
# # ###############################################################


# # ###############################################################
# # # * Prithvi Pretrained
# # ###############################################################
# load_checkpoint = True
# for freeze in [False]:
#     for batch_size in [2]:
#         for data_percentage in [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]:
#             for region_to_cut in ALL_REGIONS:
#                 region_to_cut_name = region_to_cut.replace(" ", "_").lower()
#                 group_name = f"freeze-{freeze}_loadcheckpoint-{load_checkpoint}_fullsize_sigmoid_region-{region_to_cut_name}"
                
#                 for learning_rate in [0.001, 0.0001, 0.00001, 0.000001]:
                    
#                     name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
#                     if os.path.exists(f"{records_dir}/{name}"):
#                         file_content = open(f"{records_dir}/{name}", "r").readlines()
#                         last_line = file_content[-1]
#                         if "wandb: Find logs" in last_line: 
#                             continue

#                     command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate} --freeze {freeze}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_prithvi.sh"
#                     os.system(command)
# # ###############################################################




# # ###############################################################
# # # * Shallow Transformer 
# # ###############################################################
# for learning_rate in [0.001, 0.0001, 0.00001, 0.000001]:
#     for batch_size in [264, 512]:
#         for data_percentage in [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]:
#             for region_to_cut in ALL_REGIONS:

                
#                 region_to_cut_name = region_to_cut.replace(" ", "_").lower()
#                 group_name = f"transformer_lsp_fullsize_pixels_{region_to_cut_name}"

#                 name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
                
#                 if os.path.exists(f"{records_dir}/{name}"):
#                     file_content = open(f"{records_dir}/{name}", "r").readlines()
#                     last_line = file_content[-1]
#                     if "wandb: Find logs" in last_line: 
#                         continue

#                     # if "Disk quota exceeded" not in last_line:
#                     #     continue
#                     print("haha: ", name)

#                 command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_lsp_pixels_transformer.sh"
#                 os.system(command)
# # ###############################################################


# ###############################################################
# # * Shallow Transformer Full Image
# ##############################################################
# for learning_rate in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
#     for batch_size in [1]:
#         for data_percentage in [1.0]:
#             for region_to_cut in ALL_REGIONS:

#                 region_to_cut_name = region_to_cut.replace(" ", "_").lower()
#                 group_name = f"transformer_lsp_fullsize_pixelsimage_{region_to_cut_name}"

#                 name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
                
#                 if os.path.exists(f"{records_dir}/{name}"):
#                     file_content = open(f"{records_dir}/{name}", "r").readlines()
#                     last_line = file_content[-1]
#                     if "wandb: Find logs" in last_line: 
#                         continue

#                     # if "Disk quota exceeded" not in last_line:
#                     #     continue
#                     print("haha: ", name)

#                 command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_lsp_pixels_transformer_image.sh"
#                 os.system(command)
# ##############################################################


# # ###############################################################
# # # * UNet
# # ###############################################################
# for batch_size in [2]:
#     for data_percentage in [1.0]:
#         for region_to_cut in ALL_REGIONS:
#             region_to_cut_name = region_to_cut.replace(" ", "_").lower()
#             group_name = f"unet_lsp_fullsize_region-{region_to_cut_name}"
            
#             for learning_rate in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
                
#                 name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
#                 if os.path.exists(f"{records_dir}/{name}"):
#                     file_content = open(f"{records_dir}/{name}", "r").readlines()
#                     last_line = file_content[-1]
#                     if "wandb: Find logs" in last_line: 
#                         continue

#                 command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_unet.sh"
#                 os.system(command)
# ###############################################################

# # ###############################################################
# # # * Prithvi Pretrained Sampler
# # ###############################################################
# load_checkpoint = True
# for freeze in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for region_to_cut in ALL_REGIONS:
#                 region_to_cut_name = region_to_cut.replace(" ", "_").lower()
#                 group_name = f"freeze-{freeze}_loadcheckpoint-{load_checkpoint}_fullsize_sigmoid_sampler_region-{region_to_cut_name}"
                
#                 for learning_rate in [0.001, 0.0001, 0.00001, 0.000001]:
                    
#                     name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
#                     if os.path.exists(f"{records_dir}/{name}"):
#                         file_content = open(f"{records_dir}/{name}", "r").readlines()
#                         last_line = file_content[-1]
#                         if "wandb: Find logs" in last_line: 
#                             continue

#                     command = f"qsub -v args=' --using_sampler True --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate} --freeze {freeze}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_prithvi.sh"
#                     os.system(command)
# # ###############################################################

# ###############################################################
# # * Conv
# ###############################################################
# for batch_size in [1,2]:
#     for data_percentage in [1.0]:
#         for region_to_cut in ALL_REGIONS:
#             region_to_cut_name = region_to_cut.replace(" ", "_").lower()
#             group_name = f"conv_lsp_fullsize_region-{region_to_cut_name}"
            
#             for learning_rate in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
                
#                 name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
#                 if os.path.exists(f"{records_dir}/{name}"):
#                     file_content = open(f"{records_dir}/{name}", "r").readlines()
#                     last_line = file_content[-1]
#                     if "wandb: Find logs" in last_line: 
#                         continue

#                 command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_lsp_conv.sh"
#                 os.system(command)

# ###############################################################

