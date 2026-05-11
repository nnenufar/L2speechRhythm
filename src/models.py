import torch
from transformers import WavLMModel
import torch.nn.functional as F
import torch.nn as nn
from src.modules import *
from src.train_utils import quantize_values

class CNN_MLP(nn.Module):
    """
    CNN feature extraction with pooling and fully connected classification/regression head.
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
                 attn_pool_mode='sigmoid',  # 'sigmoid' or 'softmax'
                 attn_tau=2.0,  # Temperature for attention
                 pooling_mode='both', # Only used if attPool is False
                 dropout=0.1,
                 # CNN stride for downsampling
                 cnn_stride=1,
                 # LSTM parameters
                 use_lstm=False,
                 lstm_out_mode = "all",
                 lstm_hidden_size=64,
                 lstm_num_layers=1,
                 lstm_bidirectional=True,
                 main_item=None,
                 # F0 stats 
                 f0_mean=None,
                 f0_std=None,
                 # Task type: 'classification' or 'regression'
                 task='classification',
                 # Regression target normalization stats
                 target_mean=None,
                 target_std=None,
                 # Utterance ID embedding
                 num_utterances=0,
                 utterance_embed_dim=0):
        super().__init__()

        self.main_item = main_item
        self.cnn_out_channels = cnn_out_channels if cnn_out_channels else feat_dim * 2
        self.use_attention_pooling = use_attention_pooling
        self.use_lstm = use_lstm
        self.lstm_out_mode = lstm_out_mode
        self.f0_mean = f0_mean
        self.f0_std = f0_std
        self.attn_pool_mode = attn_pool_mode
        self.attn_tau = attn_tau
        self.task = task
        self.target_mean = target_mean
        self.target_std = target_std

        # Utterance ID embedding
        self.use_utterance_embedding = (utterance_embed_dim > 0 and num_utterances > 0)
        if self.use_utterance_embedding:
            self.utterance_embedding = nn.Embedding(num_utterances, utterance_embed_dim)
        self._utterance_embed_dim = utterance_embed_dim

        # CNN Encoder (with optional stride for downsampling)
        self.cnn_encoder = Conv1dStack(
            in_channels=feat_dim,
            hidden_channels=self.cnn_out_channels,
            out_channels=self.cnn_out_channels,
            kernel_size=kernel_size,
            num_layers=num_cnn_layers,
            activation='relu',
            dropout=dropout,
            norm='layer',
            stride=cnn_stride
        )

        # Optional LSTM layer after CNN
        if use_lstm:
            self.lstm_encoder = LSTMEncoder(
                input_size=self.cnn_out_channels,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                dropout=dropout,
                bidirectional=lstm_bidirectional,
                output_mode=lstm_out_mode 
            )
            
            if lstm_out_mode == 'last':
                # LSTM with 'last' mode outputs (B, H*D) directly, no pooling needed
                self.embed_dim = self.lstm_encoder.output_size
                self.skip_pooling = True
            elif lstm_out_mode == 'all':
                # LSTM with 'all' mode outputs (B, T, H*D), use attention pooling
                self.lstm_attn_pool = AttentionPooling(
                    embed_dim=self.lstm_encoder.output_size,
                    hidden_dim=attn_hidden_dim,
                    dropout=dropout,
                    mode=attn_pool_mode
                )
                self.embed_dim = self.lstm_encoder.output_size
                self.skip_pooling = False  # Will use lstm_attn_pool
            else:  # 'pool' mode
                self.embed_dim = self.lstm_encoder.output_size
                self.skip_pooling = True
        else:
            self.pre_pool_dim = self.cnn_out_channels
            self.skip_pooling = False

            # Attention Pooling)
            if use_attention_pooling:
                self.attn_pool = AttentionPooling(
                    embed_dim=self.pre_pool_dim,
                    hidden_dim=attn_hidden_dim,
                    dropout=dropout,
                    mode=attn_pool_mode)
                self.embed_dim = self.pre_pool_dim
            else:
                # Use Global Pooling
                self.pooling = GlobalPooling(mode=pooling_mode)
                self.embed_dim = self.pre_pool_dim * 2 if pooling_mode == 'both' else self.pre_pool_dim

        fc_in_features = self.embed_dim + utterance_embed_dim if self.use_utterance_embedding else self.embed_dim

        # Output head: classification or regression
        if task == 'regression':
            # Regression head: outputs single value
            self.fc = MLP(
                in_features=fc_in_features,
                hidden_features=hidden_size,
                out_features=1,  # Single output for regression
                num_layers=1,
                activation='relu',
                dropout=dropout,
                norm='layer'
            )
        else:
            # Classification head
            self.fc = MLP(
                in_features=fc_in_features,
                hidden_features=hidden_size,
                out_features=num_classes,
                num_layers=1,
                activation='relu',
                dropout=dropout,
                norm='layer'
            )

    def forward(self, batch, return_attention=False):
        x = batch[self.main_item]            # (B, T) for 1D features or (B, T, F) or (B, F, T) for 2D features
        
        if self.main_item == 'f0':
            mask = batch['voiced_mask'].bool()
            f0_norm = torch.zeros_like(x)
            f0_norm[mask] = (x[mask] - self.f0_mean) / (self.f0_std + 1e-8) # Normalize using train set statistics
            x = f0_norm

        # Handle different input dimensions
        if x.dim() == 2:
            # 1D features like envelope, f0: (B, T) -> (B, 1, T)
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            # 2D features: (B, T, F) -> (B, F, T) for Conv1D
            # This applies to egemaps, f0_wavelet, etc.
            x = x.permute(0, 2, 1)

        cnn_out = self.cnn_encoder(x)        # (B, C, T)
        cnn_out = cnn_out.permute(0, 2, 1)   # (B, T, C)

        # Optional LSTM processing
        if self.use_lstm:
            if self.lstm_out_mode == "last":
                embeddings = self.lstm_encoder(cnn_out)  # (B, H*D) with 'last' mode
                attn_weights = None  # No attention when using LSTM 'last' mode
            elif self.lstm_out_mode == "all":
                lstm_out = self.lstm_encoder(cnn_out)  # (B, T, H*D)
                embeddings, attn_weights = self.lstm_attn_pool(lstm_out, tau=self.attn_tau)  # (B, H*D), (B, T)
            else:  # 'pool' mode
                embeddings = self.lstm_encoder(cnn_out)  # (B, H*D)
                attn_weights = None
        else:
            features = cnn_out
            attn_weights = None
            if self.use_attention_pooling:
                pooled, attn_weights = self.attn_pool(features, tau=self.attn_tau)  # (B, C), (B, T)
                embeddings = pooled
            else:
                embeddings = self.pooling(features)  # (B, C) or (B, 2C)

        # Concatenate utterance ID embedding if enabled
        if self.use_utterance_embedding:
            utterance_id = batch['utterance_id']  # (B,)
            utt_embed = self.utterance_embedding(utterance_id)  # (B, utterance_embed_dim)
            embeddings = torch.cat([embeddings, utt_embed], dim=-1)  # (B, embed_dim + utt_embed_dim)

        logits = self.fc(embeddings)
        
        # For regression, squeeze the output and optionally denormalize
        if self.task == 'regression':
            logits = logits.squeeze(-1)  # (B,) instead of (B, 1)
        
        if return_attention and attn_weights is not None:
            return logits, attn_weights
        
        return logits
    

class WAV_LM(nn.Module):
    def __init__(self, model_name, num_classes, hidden_size=128, attn_hidden_dim=None, 
                 dropout=0.1, freeze_encoder=True, cache_dir=None, local_files_only=False,
                 attn_pool_mode='sigmoid', attn_tau=2.0, use_attention_pooling=True):
        super().__init__()
        self.attn_tau = attn_tau
        self.use_attn_pooling = use_attention_pooling
        self.model = WavLMModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        
        # Optionally freeze the encoder
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get hidden size from the model config
        self.encoder_dim = self.model.config.hidden_size  # 1024 for large
        
        # Attention pooling
        self.attn_pool = AttentionPooling(
            embed_dim=self.encoder_dim,
            hidden_dim=attn_hidden_dim,
            dropout=dropout,
            mode=attn_pool_mode
        )
        
        # Classification head
        self.fc = MLP(
            in_features=self.encoder_dim,
            hidden_features=hidden_size,
            out_features=num_classes,
            num_layers=1,
            activation='relu',
            dropout=dropout,
            norm='layer'
        )
    
    def forward(self, batch, return_attention=False):
        waveform = batch['waveform']  # (B, T_samples)
        
        # Get WavLM hidden states
        outputs = self.model(waveform)
        hidden_states = outputs.last_hidden_state  # (B, T_frames, D)

        if self.use_attn_pooling:
            # Attention pooling
            embeddings, attn_weights = self.attn_pool(hidden_states, tau=self.attn_tau)  # (B, D), (B, T_frames)
        else:
            embeddings = hidden_states.mean(dim=1)  # (B, D)
            attn_weights = None

        # Classification
        logits = self.fc(embeddings)
        
        if return_attention:
            return logits, attn_weights
        
        return logits