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
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
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
                 num_layers=2, activation='relu', dropout=0.0, norm='batch'):
        super(Conv1dStack, self).__init__()
        layers = []
        
        # First layer
        layers.append(Conv1dBlock(in_channels, hidden_channels, kernel_size, 
                                   activation=activation, dropout=dropout, norm=norm))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(Conv1dBlock(hidden_channels, hidden_channels, kernel_size,
                                       activation=activation, dropout=dropout, norm=norm))
        
        # Output layer
        if num_layers > 1:
            layers.append(Conv1dBlock(hidden_channels, out_channels, kernel_size,
                                       activation=activation, dropout=dropout, norm=norm))
        
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
    """
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            # simple and often sufficient
            self.score = nn.Linear(embed_dim, 1)
        else:
            # slightly richer scorer can help if attention looks too uniform
            self.score = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, mask=None, tau=2.0):
        # x: (B, T, C)
        logits = self.score(x).squeeze(-1)  # (B, T)

        if mask is not None:
            # mask: (B, T) with 1 for valid, 0 for invalid
            logits = logits.masked_fill(mask == 0, float("-inf"))

        g = torch.sigmoid(logits / tau)
        attn = g / (g.sum(dim=1, keepdim=True) + 1e-8)
        if self.dropout is not None:
            attn = self.dropout(attn)

        pooled = torch.einsum("btc,bt->bc", x, attn)  # (B, C)
        pooled = self.norm(pooled)

        return pooled, attn


# ==============================================================================
# Contrastive Learning Modules
# ==============================================================================

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon) from https://arxiv.org/abs/2004.11362
    
    Uses labels to define positive pairs (same class) and negative pairs (different class).
    
    Args:
        temperature: Temperature scaling parameter (default: 0.07)
        base_temperature: Base temperature for normalization (default: 0.07)
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features, labels):
        """
        Args:
            features: Hidden vectors of shape (B, D) - should be L2 normalized
            labels: Ground truth labels of shape (B,)
        
        Returns:
            loss: Supervised contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Need at least 2 samples to compute contrastive loss
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # L2 normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)  # (B, B)
        
        # Clamp similarity to prevent numerical issues
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)
        
        # Create mask for positive pairs (same label, excluding self)
        labels = labels.contiguous().view(-1, 1)
        mask_positives = torch.eq(labels, labels.T).float().to(device)  # (B, B)
        
        # Remove self-contrast (diagonal)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        mask_positives = mask_positives.masked_fill(self_mask, 0)
        
        # Count positives per anchor
        num_positives = mask_positives.sum(dim=1)  # (B,)
        
        # Check if there are any valid positive pairs
        if num_positives.sum() == 0:
            # No positive pairs in batch - return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Apply temperature scaling
        logits = similarity_matrix / self.temperature
        
        # For numerical stability, subtract max
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Mask out self-contrast for denominator
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits.masked_fill(self_mask, 0)
        
        # Compute log-softmax denominator (sum over all except self)
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute log-prob for all pairs
        log_prob = logits - log_sum_exp
        
        # Mask out self for log_prob as well
        log_prob = log_prob.masked_fill(self_mask, 0)
        
        # Compute mean log-likelihood over positive pairs
        # Only consider anchors that have at least one positive
        mask_valid = num_positives > 0
        
        # Mean log-prob of positive pairs for each anchor
        mean_log_prob_pos = (mask_positives * log_prob).sum(dim=1) / (num_positives + 1e-8)
        
        # Loss (only for valid anchors)
        loss = -(self.base_temperature / self.temperature) * mean_log_prob_pos
        loss = loss[mask_valid].mean()
        
        # Final safety check
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss


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