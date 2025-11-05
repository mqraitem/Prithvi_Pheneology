import os 

records_dir = "records"

ALL_REGIONS = [
    "ALL",
]


# ###############################################################
# # * Prithvi Pretrained Temp
# ###############################################################
load_checkpoint = True
for freeze in [False]:
    for batch_size in [2]:
        for data_percentage in [1.0]:
            for region_to_cut in ALL_REGIONS:
                region_to_cut_name = region_to_cut.replace(" ", "_").lower()
                group_name = f"freeze-{freeze}_loadcheckpoint-{load_checkpoint}_fullsize_sigmoidtemp_region-{region_to_cut_name}"
                
                for learning_rate in [0.001, 0.0001, 0.00001, 0.000001]:
                    
                    name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
                    if os.path.exists(f"{records_dir}/{name}"):
                        file_content = open(f"{records_dir}/{name}", "r").readlines()
                        last_line = file_content[-1]
                        if "wandb: Find logs" in last_line: 
                            continue

                    command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate} --freeze {freeze}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_temp.sh"
                    os.system(command)
# ###############################################################


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

#                 command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_lsp_spatial.sh"
#                 os.system(command)
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

#                 command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_lsp_unet.sh"
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

#                     command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate} --freeze {freeze}'  -o {records_dir}/{name} run_scripts/train_fullsize_all.sh"
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

#                     command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate} --freeze {freeze}'  -o {records_dir}/{name} run_scripts/train_fullsize_all.sh"
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

#                 command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_lsp_pixels.sh"
#                 os.system(command)
# # ###############################################################



# # ##############################################################
# # * QTransformer
# # ##############################################################
# for learning_rate in [0.001, 0.0001, 0.00001, 0.000001]:
#     for batch_size in [264, 512]:
#         for data_percentage in [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]:
#             for region_to_cut in ALL_REGIONS:

                
#                 region_to_cut_name = region_to_cut.replace(" ", "_").lower()
#                 group_name = f"qtransformer_lsp_fullsize_pixels_{region_to_cut_name}"

#                 name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
                
#                 if os.path.exists(f"{records_dir}/{name}"):
#                     file_content = open(f"{records_dir}/{name}", "r").readlines()
#                     last_line = file_content[-1]
#                     if "wandb: Find logs" in last_line: 
#                         continue

#                     # if "Disk quota exceeded" not in last_line:
#                     #     continue
#                     print("haha: ", name)

#                 command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_lsp_pixels_feats.sh"
#                 os.system(command)
# # ##############################################################



# # ###############################################################
# # # * Shallow Transformer Full Image
# # ##############################################################
# # for learning_rate in [0.01, 0.001, 0.0001, 0.00001, 0.000001]:
# #     for batch_size in [1]:
# #         for data_percentage in [1.0]:
# #             for region_to_cut in ALL_REGIONS:

# #                 region_to_cut_name = region_to_cut.replace(" ", "_").lower()
# #                 group_name = f"transformer_lsp_fullsize_pixelsimage_{region_to_cut_name}"

# #                 name = f"{group_name}_{learning_rate}_batch_size-{batch_size}_data_percentage-{data_percentage}_region-{region_to_cut_name}"
                
# #                 if os.path.exists(f"{records_dir}/{name}"):
# #                     file_content = open(f"{records_dir}/{name}", "r").readlines()
# #                     last_line = file_content[-1]
# #                     if "wandb: Find logs" in last_line: 
# #                         continue

# #                     # if "Disk quota exceeded" not in last_line:
# #                     #     continue
# #                     print("haha: ", name)

# #                 command = f"qsub -v args=' --region_to_cut {region_to_cut} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_fullsize_all_lsp_pixels_image.sh"
# #                 os.system(command)
# # ##############################################################


