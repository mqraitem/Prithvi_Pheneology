import torch
import torch.nn.functional as F
from torch import nn
    

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


# class Upscaler(nn.Module):
#     def __init__(self, embed_dim: int, depth: int, dropout: bool = True):
#         super().__init__()

#         def build_block(in_ch, out_ch): return nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=in_ch,
#                 out_channels=out_ch,
#                 kernel_size=2,
#                 stride=2),
#             nn.BatchNorm2d(out_ch),
#             nn.GELU(),
#             nn.Dropout(0.1) if dropout else nn.Identity(),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True))

#         self.upscale_blocks = nn.Sequential(
#             *[build_block(int(embed_dim // 2**i), int(embed_dim // 2**(i+1))) for i in range(depth)]
#         )

#     def forward(self, x):
#         return self.upscale_blocks(x)

import torch
from torch import nn
import torch.nn.functional as F


class ResidualUnit(nn.Module):
    """2×(Conv3x3 -> GN -> GELU) with residual & optional channel projection."""
    def __init__(self, channels: int, dropout: bool = True, num_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=channels),
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(0.1) if dropout else nn.Identity()

    def forward(self, x):
        y = self.block(x)
        y = self.drop(y)
        return self.act(x + y)


class ResidualUpsampleBlock(nn.Module):
    """
    Upsample by 2× (bilinear) -> Conv3x3(in->out) -> GN -> GELU -> ResidualUnit(out)
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = True, num_groups: int = 8):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.GELU(),
        )
        self.residual = ResidualUnit(out_ch, dropout=dropout, num_groups=num_groups)

    def forward(self, x):
        x = self.upsample(x)
        x = self.reduce(x)
        x = self.residual(x)
        return x


class Upscaler(nn.Module):
    """
    Residual decoder:
    For depth d, channels go: embed_dim -> embed_dim/2 -> ... -> embed_dim/2^d
    """
    def __init__(self, embed_dim: int, depth: int, dropout: bool = True, num_groups: int = 8):
        super().__init__()

        def chans(i):  # i = 0..depth
            return int(embed_dim // (2 ** i))

        blocks = []
        for i in range(depth):
            in_ch, out_ch = chans(i), chans(i + 1)
            blocks.append(ResidualUpsampleBlock(in_ch, out_ch, dropout=dropout, num_groups=num_groups))
        self.upscale_blocks = nn.Sequential(*blocks)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, "bias", None) is not None and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.upscale_blocks(x)


class PrithviSeg(nn.Module): 
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

    def forward(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        x = self.head(x)
        x = x.view(batch_size, self.n_classes, x.size(2), x.size(3))   
        x = torch.sigmoid(x)
        return x
