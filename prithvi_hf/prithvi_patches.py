import torch
import torch.nn.functional as F
from torch import nn
import random
    

class PrithviReshape(nn.Module):
    def __init__(self,
                patch_size,
                input_size):
            super().__init__()
            self.patch_size = patch_size
            self.input_size = input_size
            self.view_size = int(self.input_size / self.patch_size[-1])
    
    def forward(self, latent):
        latent = latent[:, 1:, :]
        latent = latent.transpose(1,2)
        latent = latent.reshape(
            latent.shape[0],
            -1,
            self.view_size,
            self.view_size)

        return latent


class PrithviBackbone(nn.Module):
    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None,
                 reshape: bool = True):
        super().__init__()
        self.prithvi_ckpt_path = prithvi_ckpt_path
        self.prithvi_params = prithvi_params

        from prithvi_hf.prithvi_mae import PrithviMAE
        self.model = PrithviMAE(**self.prithvi_params)
        if self.prithvi_ckpt_path is not None:
            checkpoint = torch.load(self.prithvi_ckpt_path, weights_only=False)


            if "encoder.pos_embed" not in checkpoint.keys():
                key = "model" if "model" in checkpoint.keys() else "state_dict"
                keys = list(checkpoint[key].keys())
                checkpoint = checkpoint[key]
            else:
                keys = list(checkpoint.keys())
            
            for k in keys:
                if ((prithvi_params["encoder_only"]) and ("decoder" in k)) or "pos_embed" in k:
                    del checkpoint[k]
                elif "prithvi" in k:
                    print(f"Warning: renaming prithvi layer {k}")
                    new_k = k.replace("prithvi.", "")
                    checkpoint[new_k] = checkpoint[k]
                elif k in self.model.state_dict() and checkpoint[k].shape != self.model.state_dict()[k].shape:
                    print(f"Warning: size mismatch for layer {k}, deleting: {checkpoint[k].shape} != {self.model.state_dict()[k].shape}")
                    del checkpoint[k]

            _ = self.model.load_state_dict(checkpoint, strict=False)
            
        self.reshaper = PrithviReshape(prithvi_params["patch_size"], prithvi_params["img_size"]) if reshape else nn.Identity()


    def forward(self, data):
        if isinstance(data, dict):
            chip = data.get("chip")
            temporal = data.get("temporal_coords")
            location = data.get("location_coords")
        else:
            chip = data
            temporal = None
            location = None
        
        if self.prithvi_params["encoder_only"]:
            latent = self.model.forward_features(chip,
                                    temporal,
                                    location)
        else:
            latent, mask, ids_restore = self.model.encoder(chip, temporal, location, 0.0)
            latent = self.model.decoder(latent,
                                ids_restore,
                                temporal,
                                location,
                                input_size=(self.prithvi_params["num_frames"], self.prithvi_params["img_size"], self.prithvi_params["img_size"]))
        return self.reshaper(latent)


class Upscaler(nn.Module):
    def __init__(self, embed_dim: int, depth: int, dropout: bool = True):
        super().__init__()

        def build_block(in_ch, out_ch): return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=2,
                stride=2),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout(0.1) if dropout else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

        self.upscale_blocks = nn.Sequential(
            *[build_block(int(embed_dim // 2**i), int(embed_dim // 2**(i+1))) for i in range(depth)]
        )

    def forward(self, x):
        return self.upscale_blocks(x)


class PrithviSegPatches(nn.Module): 
    def __init__(self,
                 prithvi_params: dict,
                 prithvi_ckpt_path: str = None,
                 reshape: bool = True, 
                 n_classes: int = 1,
                 model_size: str="300m"):
        super().__init__()

        self.backbone = PrithviBackbone(prithvi_params, prithvi_ckpt_path, reshape)

        if model_size == "300m":
            print("Dim: ", prithvi_params["embed_dim"]*prithvi_params["num_frames"])
            self.head = nn.Sequential(
                Upscaler(prithvi_params["embed_dim"]*prithvi_params["num_frames"] , 4),
                nn.Conv2d(in_channels=768, out_channels=n_classes, kernel_size=1),
            )

        else: 
            raise ValueError(f"model_size {model_size} not supported")

        self.n_frames = prithvi_params["num_frames"]
        self.n_classes = n_classes  

    def forward_patches(self, x, mask=None, patch_size=64, patch_fraction=0.25):
        """
        Patch-based forward for training.
        Args:
            x: (B, C, T, H, W)
            mask: (B, T, H, W) ground truth mask
            patch_size: spatial patch size (multiple of 16)
            patch_fraction: fraction of patches to sample per image
        Returns:
            pred_patches: model predictions for selected patches
            mask_patches: corresponding ground truth patches
        """
        B, C, T, H, W = x.shape
        device = x.device

        # Compute patch grid
        n_h = H // patch_size
        n_w = W // patch_size
        total_patches = n_h * n_w
        n_sample = max(1, int(total_patches * patch_fraction))

        all_coords = [(i * patch_size, j * patch_size)
                      for i in range(n_h) for j in range(n_w)]
        selected = random.sample(all_coords, n_sample)

        pred_patches = []
        mask_patches = []

        for (y, x0) in selected:
            img_patch = x[:, :, :, y:y + patch_size, x0:x0 + patch_size]
            gt_patch = mask[:, :, y:y + patch_size, x0:x0 + patch_size] if mask is not None else None

            # Forward pass through backbone + head
            out_patch = self.backbone(img_patch)
            out_patch = self.head(out_patch)
            out_patch = torch.sigmoid(out_patch)

            pred_patches.append(out_patch)
            if gt_patch is not None:
                mask_patches.append(gt_patch)

        pred_patches = torch.cat(pred_patches, dim=0)
        mask_patches = torch.cat(mask_patches, dim=0) if mask_patches else None
        return pred_patches, mask_patches

    @torch.no_grad()
    def forward_inference_tiled(self, x, patch_size=64):
        """
        Full-image inference using tiling.
        Args:
            x: (B, C, T, H, W)
            patch_size: spatial tile size (multiple of 16)
            overlap: overlap (in pixels) between adjacent patches
        Returns:
            output stitched to (B, C, H, W)
        """
        overlap = patch_size // 4
        B, C, T, H, W = x.shape
        stride = patch_size - overlap
        full_pred = torch.zeros(B, self.n_classes, H, W, device=x.device)
        count = torch.zeros_like(full_pred)

        for y in range(0, H - patch_size + 1, stride):
            for x0 in range(0, W - patch_size + 1, stride):
                patch = x[:, :, :, y:y + patch_size, x0:x0 + patch_size]
                out = self.backbone(patch)
                out = self.head(out)
                out = torch.sigmoid(out)
                full_pred[:, :, y:y + patch_size, x0:x0 + patch_size] += out
                count[:, :, y:y + patch_size, x0:x0 + patch_size] += 1

        return full_pred / count.clamp(min=1)
