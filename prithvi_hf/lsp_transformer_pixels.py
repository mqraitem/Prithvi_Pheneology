import torch
import torch.nn as nn
import math
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TemporalTransformer(nn.Module):
    def __init__(self, input_channels=6, seq_len=4, num_classes=4, d_model=128,
                 nhead=4, num_layers=3, dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_channels, d_model, bias=False)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, num_classes)

    def forward(self, x, processing_images= True, chunk_size=1000):
        # Input shape: (B, C, T, H, W)

        if processing_images:
            x = x.permute(0, 3, 4, 2, 1)
            B, H, W, T, C = x.shape

            x = x.reshape(B * H * W, T, C)      # (B*H*W, T, C)

        # Project input and apply positional encoding
        x = self.input_proj(x)             # (N, T, d_model)
        x = self.pos_encoder(x)

        N = x.size(0)
        outputs = []

        for i in range(0, N, chunk_size):
            x_chunk = x[i:i + chunk_size]                          # (chunk_size, T, d_model)
            x_chunk = self.transformer_encoder(x_chunk)           # (chunk_size, T, d_model)
            x_chunk = x_chunk.mean(dim=1)                         # (chunk_size, d_model)
            x_chunk = self.dropout(x_chunk)
            x_chunk = self.output_proj(x_chunk)                   # (chunk_size, num_classes)
            outputs.append(x_chunk)

        x = torch.cat(outputs, dim=0)  # (B*H*W, num_classes)
       
        if processing_images:
            x = x.view(B, H, W, -1)        # (B, H, W, num_classes)
            #permute to (B, last, H, W)
            x = x.permute(0, 3, 1, 2)      # (B, num_classes, H, W)

        return x