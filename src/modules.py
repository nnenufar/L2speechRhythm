import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# Common Building Blocks
# ==============================================================================

class LinearBlock(nn.Module):
    """
    A linear layer followed by optional normalization, activation, and dropout.
    """
    def __init__(self, in_features, out_features, activation='relu', dropout=0.0, norm=None):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(in_features, out_features)]
        
        if norm == 'layer':
            layers.append(nn.LayerNorm(out_features))
        elif norm == 'batch':
            layers.append(nn.BatchNorm1d(out_features))
        
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable hidden layers.
    """
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, 
                 activation='relu', dropout=0.0, norm=None):
        super(MLP, self).__init__()
        layers = []
        
        # Input layer
        layers.append(LinearBlock(in_features, hidden_features, activation, dropout, norm))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(LinearBlock(hidden_features, hidden_features, activation, dropout, norm))
        
        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class Conv1dBlock(nn.Module):
    """
    A 1D convolutional layer followed by optional normalization, activation, and dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same',
                 activation='relu', dropout=0.0, norm=None):
        super(Conv1dBlock, self).__init__()
        
        # Handle padding for strided convolutions
        if stride > 1:
            # For strided conv, 'same' padding isn't straightforward
            # Use manual padding to maintain expected downsampling
            padding_val = kernel_size // 2
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                  stride=stride, padding=padding_val)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                  stride=stride, padding=padding)
        
        self.norm = None
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == 'layer':
            self.norm = None  # LayerNorm applied in forward
            self.use_layer_norm = True
            self.layer_norm = nn.LayerNorm(out_channels)
        else:
            self.use_layer_norm = False
        
        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x):
        # x shape: (B, C, T)
        x = self.conv(x)
        
        if self.norm is not None:
            x = self.norm(x)
        elif hasattr(self, 'use_layer_norm') and self.use_layer_norm:
            x = x.permute(0, 2, 1)  # (B, T, C)
            x = self.layer_norm(x)
            x = x.permute(0, 2, 1)  # (B, C, T)
        
        if self.activation is not None:
            x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class Conv1dStack(nn.Module):
    """
    A stack of 1D convolutional blocks.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, 
                 num_layers=2, activation='relu', dropout=0.0, norm='batch', stride=1):
        super(Conv1dStack, self).__init__()
        layers = []
        
        # First layer (with stride for downsampling if specified)
        layers.append(Conv1dBlock(in_channels, hidden_channels, kernel_size, 
                                   stride=stride, activation=activation, dropout=dropout, norm=norm))
        
        # Hidden layers (can also use stride for progressive downsampling)
        for _ in range(num_layers - 2):
            layers.append(Conv1dBlock(hidden_channels, hidden_channels, kernel_size,
                                       stride=stride, activation=activation, dropout=dropout, norm=norm))
        
        # Output layer
        if num_layers > 1:
            layers.append(Conv1dBlock(hidden_channels, out_channels, kernel_size,
                                       stride=stride, activation=activation, dropout=dropout, norm=norm))
        
        self.stack = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.stack(x)


class GlobalPooling(nn.Module):
    """
    Global pooling layer supporting mean, max, or both (concatenated).
    """
    def __init__(self, mode='mean'):
        super(GlobalPooling, self).__init__()
        assert mode in ['mean', 'max', 'both'], "mode must be 'mean', 'max', or 'both'"
        self.mode = mode
    
    def forward(self, x, mask=None):
        # x shape: (B, T, C)
        if mask is not None:
            # mask shape: (B, T) or (B, T, 1)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            x = x * mask
        
        if self.mode == 'mean':
            if mask is not None:
                return x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return x.mean(dim=1)
        elif self.mode == 'max':
            if mask is not None:
                x = x.masked_fill(~mask.bool(), float('-inf'))
            return x.max(dim=1)[0]
        else:  # both
            if mask is not None:
                mean_pool = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                x_masked = x.masked_fill(~mask.bool(), float('-inf'))
                max_pool = x_masked.max(dim=1)[0]
            else:
                mean_pool = x.mean(dim=1)
                max_pool = x.max(dim=1)[0]
            return torch.cat([mean_pool, max_pool], dim=-1)


class AttentionPooling(nn.Module):
    """
    Attention pooling over sequence length (bins).
    Input:  x of shape (B, T, C)
    Output: pooled of shape (B, C) and attn weights of shape (B, T)
    
    Args:
        embed_dim: Input embedding dimension
        hidden_dim: Hidden dimension for attention scorer (None for simple linear)
        dropout: Dropout rate
        mode: Attention normalization mode ('sigmoid' or 'softmax')
    """
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.0, mode='sigmoid', use_norm=True):
        super().__init__()
        assert mode in ['sigmoid', 'softmax'], "mode must be 'sigmoid' or 'softmax'"
        self.mode = mode
        self.use_norm = use_norm
        
        if hidden_dim is None:
            self.score = nn.Linear(embed_dim, 1)
        else:
            self.score = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        self.norm = nn.LayerNorm(embed_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, mask=None, tau=2.0):
        # x: (B, T, C)
        logits = self.score(x).squeeze(-1)  # (B, T)

        if mask is not None:
            # mask: (B, T) with 1 for valid, 0 for invalid
            logits = logits.masked_fill(mask == 0, float("-inf"))

        if self.mode == 'softmax':
            attn = F.softmax(logits / tau, dim=-1)  # (B, T)
        else:  # sigmoid
            g = torch.sigmoid(logits / tau)
            attn = g / (g.sum(dim=1, keepdim=True) + 1e-8)
        
        if self.dropout is not None:
            attn = self.dropout(attn)

        pooled = torch.einsum("btc,bt->bc", x, attn)  # (B, C)
        pooled = self.norm(pooled)

        return pooled, attn

class LSTMEncoder(nn.Module):
    """
    LSTM encoder for sequential processing.
    
    Args:
        input_size: Input feature dimension
        hidden_size: LSTM hidden state dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate (applied between LSTM layers if num_layers > 1)
        bidirectional: Whether to use bidirectional LSTM
        output_mode: How to produce output:
            - 'last': Return last hidden state (B, H*D)
            - 'all': Return all timesteps (B, T, H*D)
            - 'pool': Return mean-pooled hidden states (B, H*D)
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, 
                 bidirectional=True, output_mode='all'):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_size = hidden_size * self.num_directions
        self.output_mode = output_mode
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.layer_norm = nn.LayerNorm(self.output_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: Input tensor of shape (B, T, C)
            lengths: Optional sequence lengths for packed sequence (B,)
        
        Returns:
            output: Depending on output_mode:
                - 'last': (B, H*D)
                - 'all': (B, T, H*D)
                - 'pool': (B, H*D)
        """
        if lengths is not None:
            # Pack sequence for variable length inputs
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
        
        # lstm_out: (B, T, H*D)
        # h_n: (num_layers * num_directions, B, H)
        
        if self.output_mode == 'last':
            # Concatenate last hidden states from both directions
            if self.bidirectional:
                # h_n[-2] is forward, h_n[-1] is backward
                output = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, H*2)
            else:
                output = h_n[-1]  # (B, H)
        elif self.output_mode == 'pool':
            # Mean pool over time dimension
            if lengths is not None:
                # Masked mean pooling
                mask = torch.arange(lstm_out.size(1), device=lstm_out.device).unsqueeze(0) < lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).float()
                output = (lstm_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                output = lstm_out.mean(dim=1)  # (B, H*D)
        else:  # 'all'
            output = lstm_out  # (B, T, H*D)
        
        output = self.layer_norm(output)
        
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output


# ==============================================================================
# Projection Head (for embedding tasks)
# ==============================================================================

class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps representations to a lower-dimensional space where contrastive loss is applied.
    
    Args:
        in_dim: Input dimension
        hidden_dim: Hidden layer dimension
        out_dim: Output (projection) dimension
        num_layers: Number of layers (1 = linear, 2+ = MLP)
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, num_layers=2, dropout=0.0):
        super(ProjectionHead, self).__init__()
        
        if num_layers == 1:
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            layers = []
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.proj = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.proj(x)


# ==============================================================================
# Contrastive Loss
# ==============================================================================

def per_utterance_contrastive_loss(embeddings, utterance_ids, is_l1, temperature=0.07):
    """
    Per-utterance supervised contrastive loss.
    Anchors are L1 samples only. Positives are other L1 samples of the same utterance.
    Negatives are L2 samples of the same utterance only (no cross-utterance negatives).

    Args:
        embeddings: (B, D) L2-normalized encoder outputs
        utterance_ids: (B,) integer tensor, same value for samples of same utterance
        is_l1: (B,) boolean tensor, True for L1 (anchor) samples
        temperature: scalar temperature

    Returns:
        scalar loss
    """
    device = embeddings.device
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    unique_utterances = torch.unique(utterance_ids)

    total_loss = torch.tensor(0.0, device=device)
    num_anchors = 0

    for utt in unique_utterances:
        utt_mask = utterance_ids == utt
        l1_mask = utt_mask & is_l1
        l2_mask = utt_mask & ~is_l1

        l1_indices = torch.where(l1_mask)[0]
        l2_indices = torch.where(l2_mask)[0]

        if len(l1_indices) < 2 or len(l2_indices) == 0:
            continue

        for anchor_idx in l1_indices:
            pos_mask = l1_mask.clone()
            pos_mask[anchor_idx] = False

            pos_sim = sim[anchor_idx][pos_mask]
            neg_sim = sim[anchor_idx][l2_mask]

            pos_exp = torch.exp(pos_sim)
            neg_exp = torch.exp(neg_sim)

            numerator = pos_exp.sum()
            denominator = numerator + neg_exp.sum()

            loss_i = -torch.log(numerator / (denominator + 1e-8))
            total_loss = total_loss + loss_i
            num_anchors += 1

    if num_anchors == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / num_anchors


# ==============================================================================
# Text Encoder
# ==============================================================================

class TextEncoder(nn.Module):
    """
    Phoneme sequence encoder: embedding → conv stack → bidirectional LSTM.
    Standard architecture for TTS text encoding.
    """
    def __init__(self, vocab_size, embed_dim, conv_channels, kernel_size,
                 lstm_hidden, lstm_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = Conv1dStack(
            in_channels=embed_dim,
            hidden_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            num_layers=3,
            activation='relu',
            dropout=dropout,
            norm='layer'
        )
        self.lstm = LSTMEncoder(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=True,
            output_mode='all'
        )
        self.output_dim = self.lstm.output_size

    def forward(self, phoneme_ids, lengths):
        x = self.embedding(phoneme_ids)
        x = x.permute(0, 2, 1)
        x = self.convs(x)
        x = x.permute(0, 2, 1)
        x = self.lstm(x, lengths)
        return x


# ==============================================================================
# Cross-Attention
# ==============================================================================

class AdditiveCrossAttention(nn.Module):
    """
    Bahdanau-style additive cross-attention.
    Queries attend to keys, producing a context vector at each query position.
    """
    def __init__(self, query_dim, key_dim, attn_dim, dropout=0.0):
        super().__init__()
        self.W_q = nn.Linear(query_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(key_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, queries, keys, key_lengths=None):
        q_proj = self.W_q(queries)
        k_proj = self.W_k(keys)

        q_proj = q_proj.unsqueeze(2)
        k_proj = k_proj.unsqueeze(1)
        energy = torch.tanh(q_proj + k_proj)
        energy = self.v(energy).squeeze(-1)

        if key_lengths is not None:
            mask = torch.arange(keys.size(1), device=keys.device).unsqueeze(0) < key_lengths.unsqueeze(1)
            energy = energy.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(energy, dim=-1)

        if self.dropout is not None:
            attn = self.dropout(attn)

        context = torch.bmm(attn, keys)
        return context, attn


# ==============================================================================
# Monotonicity Regularization
# ==============================================================================

def monotonicity_loss(attn_weights, margin=1.0):
    """
    Penalize backward movement in cross-attention alignment.
    Computes expected alignment position per audio timestep and penalizes
    decreases relative to the previous timestep.

    Args:
        attn_weights: (B, T_audio, T_text) attention matrix
        margin: Allowed backward drift in positions before penalty applies

    Returns:
        scalar loss
    """
    T_text = attn_weights.size(-1)
    positions = torch.arange(T_text, device=attn_weights.device, dtype=attn_weights.dtype)
    expected_pos = (attn_weights * positions).sum(dim=-1)

    prev_pos = expected_pos[:, :-1]
    curr_pos = expected_pos[:, 1:]
    penalty = F.relu(prev_pos - curr_pos + margin)

    return penalty.mean()