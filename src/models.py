import torch
import torch.nn.functional as F
import torch.nn as nn
from src.modules import *
from src.train_utils import quantize_values

class CNN_MLP(nn.Module):
    """
    CNN feature extraction with pooling and fully connected classification head.
    Optionally supports supervised contrastive learning.
    """
    def __init__(self,
                 feat_dim,
                 kernel_size, # CNN
                 hidden_size, # Classification head
                 num_classes,
                 cnn_out_channels=None,
                 num_cnn_layers=3,
                 use_attention_pooling=False,
                 attn_hidden_dim=None,                 
                 pooling_mode='both', # Only used if attPool is False
                 dropout=0.1,
                 # Contrastive learning parameters
                 use_contrastive=False,
                 proj_hidden_dim=64,
                 proj_out_dim=32,
                 proj_num_layers=2,
                 # Duration embedding parameters
                 use_duration=False,
                 dur_mean=None,
                 dur_std=None,
                 dur_num_bins=32,
                 dur_min_val=-3.0,
                 dur_max_val=3.0,
                 main_item=None):
        super().__init__()

        self.main_item = main_item
        self.cnn_out_channels = cnn_out_channels if cnn_out_channels else feat_dim * 2
        self.use_attention_pooling = use_attention_pooling
        self.use_contrastive = use_contrastive
        self.use_duration = use_duration

        # CNN Encoder
        self.cnn_encoder = Conv1dStack(
            in_channels=feat_dim,
            hidden_channels=self.cnn_out_channels,
            out_channels=self.cnn_out_channels,
            kernel_size=kernel_size,
            num_layers=num_cnn_layers,
            activation='relu',
            dropout=dropout,
            norm='layer'
        )

        # Pooling
        if use_attention_pooling:
            self.attn_pool = AttentionPooling(
                embed_dim=self.cnn_out_channels,
                hidden_dim=attn_hidden_dim,
                dropout=dropout)
            self.embed_dim = self.cnn_out_channels
        else:
            self.pooling = GlobalPooling(mode=pooling_mode)
            self.embed_dim = self.cnn_out_channels * 2 if pooling_mode == 'both' else self.cnn_out_channels

        # Duration embedding
        if use_duration:
            assert dur_mean is not None and dur_std is not None, \
                "dur_mean and dur_std must be provided when use_duration=True"
            self.register_buffer('dur_mean', torch.tensor(dur_mean, dtype=torch.float32))
            self.register_buffer('dur_std', torch.tensor(dur_std, dtype=torch.float32))
            self.dur_num_bins = dur_num_bins
            self.dur_min_val = dur_min_val
            self.dur_max_val = dur_max_val
            
            # Duration embedding: maps quantized duration to embed_dim
            self.dur_embedding = nn.Embedding(dur_num_bins, self.embed_dim)
            self.dur_layer_norm = nn.LayerNorm(self.embed_dim)

        # Classification head
        self.fc = MLP(
            in_features=self.embed_dim,
            hidden_features=hidden_size,
            out_features=num_classes,
            num_layers=1,
            activation='relu',
            dropout=dropout,
            norm='layer'
        )
        
        # Projection head for contrastive learning
        if use_contrastive:
            self.projection_head = ProjectionHead(
                in_dim=self.embed_dim,
                hidden_dim=proj_hidden_dim,
                out_dim=proj_out_dim,
                num_layers=proj_num_layers,
                dropout=dropout
            )

    def forward(self, batch, return_attention=False, return_embeddings=False):
        x = batch[self.main_item]       # (B, T)
        x = x.unsqueeze(1)                   # (B, 1, T)

        cnn_out = self.cnn_encoder(x)        # (B, C, T)
        cnn_out = cnn_out.permute(0, 2, 1)   # (B, T, C)

        attn_weights = None
        if self.use_attention_pooling:
            pooled, attn_weights = self.attn_pool(cnn_out)  # (B, C), (B, T)
            embeddings = pooled
        else:
            embeddings = self.pooling(cnn_out)  # (B, C) or (B, 2C)

        # Add duration embedding if enabled
        if self.use_duration:
            dur = batch['dur']  # (B,)
            dur_bins = quantize_values(
                dur, 
                mean=self.dur_mean, 
                std=self.dur_std,
                num_bins=self.dur_num_bins,
                min_val=self.dur_min_val,
                max_val=self.dur_max_val
            )  # (B,)
            dur_emb = self.dur_embedding(dur_bins)  # (B, embed_dim)
            dur_emb = self.dur_layer_norm(dur_emb)
            embeddings = embeddings + dur_emb  # (B, embed_dim)

        logits = self.fc(embeddings)
        
        # Build return values
        outputs = [logits]
        
        if return_embeddings and self.use_contrastive:
            projected = self.projection_head(embeddings)
            outputs.append(projected)
        
        if return_attention and attn_weights is not None:
            outputs.append(attn_weights)
        
        # Return single value if only logits, otherwise tuple
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
