import torch
import torch.nn as nn

class PixelTemporalMLP(nn.Module):
    """
    Super-simple per-pixel MLP for phenology.

    - Input (processing_images=True):  x ∈ R[B, C, T, H, W]
      Each (H,W) pixel gets its own feature vector of length (C*T).
    - Output: logits ∈ R[B, num_classes, H, W]
    - No spatial mixing. Pure per-pixel temporal classification.

    Args:
        input_channels:  C
        seq_len:         T
        num_classes:     number of phenology classes
        hidden:          hidden width for the MLP
        layers:          number of Linear blocks (>=2 recommended)
        dropout:         dropout probability between layers
    """
    def __init__(
        self,
        input_channels: int = 6,
        seq_len: int = 12,
        num_classes: int = 4,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = input_channels * seq_len
        assert layers >= 2, "Use at least 2 layers (input->hidden, hidden->out)."

        blocks = []
        # input -> hidden
        blocks += [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        # hidden -> hidden (layers-2 times)
        for _ in range(layers - 2):
            blocks += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*blocks)
        # hidden -> classes
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x, processing_images: bool = True):
        """
        If processing_images:
            x: (B, C, T, H, W) -> logits: (B, num_classes, H, W)
        Else:
            x: (N, C*T)        -> logits: (N, num_classes)
        """
        if processing_images:
            B, C, T, H, W = x.shape
            # per-pixel vectors of length C*T
            x = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, T * C)  # (B*H*W, C*T)
        else: 
            x = x.reshape(x.size(0), -1)


        feats = self.mlp(x)                  # (N, hidden)
        logits = self.head(feats)            # (N, num_classes)

        if processing_images:
            logits = logits.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, classes, H, W)

        return logits
