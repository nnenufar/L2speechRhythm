from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def setup_experiment_dir(exp_name, timestamp=None):
    """
    Create experiment directory structure and return paths.
    
    Args:
        exp_name: Name of the experiment
        timestamp: Optional timestamp string for subdirectories
    
    Returns:
        dict with keys: exp_dir, logs_dir, plots_dir, checkpoints_dir, 
                        att_plots_dir, test_results_dir
    """
    exp_dir = Path("exp") / exp_name
    logs_dir = exp_dir / "logs"
    plots_dir = exp_dir / "plots"
    
    # Create timestamped subdirectories
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    checkpoints_dir = exp_dir / "checkpoints" / timestamp
    att_plots_dir = exp_dir / "att_plots" / timestamp
    test_results_dir = exp_dir / "test_results" / timestamp
    
    # Create directories
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    att_plots_dir.mkdir(parents=True, exist_ok=True)
    test_results_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'exp_dir': exp_dir,
        'logs_dir': logs_dir,
        'plots_dir': plots_dir,
        'checkpoints_dir': checkpoints_dir,
        'att_plots_dir': att_plots_dir,
        'test_results_dir': test_results_dir
    }

def setup_logger(logs_dir, exp_name, log_to_file=True):
    """
    Setup logging configuration with optional file and console handlers.
    """
    # Create logger
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    if log_to_file:
        # File handler - logs everything to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        log_file = logs_dir / f"training_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
    
    # Console handler - logs to terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    if log_to_file:
        logger.addHandler(file_handler)
        logger.info(f"Logging to: {log_file}")
    logger.addHandler(console_handler)
    
    return logger

def plot_training_curves(train_losses, train_accs, train_f1s,
                         val_losses, val_accs, val_f1s,
                         plots_dir, timestamp):
    """
    Plot and save training curves (loss, accuracy, and F1 score).
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot loss
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    if val_losses:
        ax1.plot(epochs_range, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    if train_accs:
        ax2.plot(epochs_range, train_accs, 'b-o', label='Training Accuracy', linewidth=2, markersize=4)
    if val_accs:
        ax2.plot(epochs_range, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot F1 score
    if train_f1s:
        ax3.plot(epochs_range, train_f1s, 'b-o', label='Training F1', linewidth=2, markersize=4)
    if val_f1s:
        ax3.plot(epochs_range, val_f1s, 'r-s', label='Validation F1', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('F1 Score', fontsize=12)
    ax3.set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = plots_dir / f'training_curves_{timestamp}.png'
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    return plot_file


def plot_regression_curves(train_losses, train_rmses, train_pearson_rs,
                           val_losses, val_rmses, val_pearson_rs,
                           plots_dir, timestamp):
    """
    Plot and save regression training curves (loss, RMSE, and Pearson correlation).
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot loss
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    if val_losses:
        ax1.plot(epochs_range, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot RMSE
    if train_rmses:
        ax2.plot(epochs_range, train_rmses, 'b-o', label='Training RMSE', linewidth=2, markersize=4)
    if val_rmses:
        ax2.plot(epochs_range, val_rmses, 'r-s', label='Validation RMSE', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('Training and Validation RMSE', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot Pearson correlation
    if train_pearson_rs:
        ax3.plot(epochs_range, train_pearson_rs, 'b-o', label='Training Pearson r', linewidth=2, markersize=4)
    if val_pearson_rs:
        ax3.plot(epochs_range, val_pearson_rs, 'r-s', label='Validation Pearson r', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Pearson r', fontsize=12)
    ax3.set_title('Training and Validation Pearson Correlation', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = plots_dir / f'regression_curves_{timestamp}.png'
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    return plot_file

def save_checkpoint(model, optimizer, epoch, train_loss, train_acc, val_loss, val_acc, 
                   checkpoints_dir, logger, is_best=False, utterance_str2int=None):
    """
    Save model checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
    }
    if utterance_str2int is not None:
        checkpoint['utterance_str2int'] = utterance_str2int
    
    if is_best:
        # Delete previous best model
        for f in checkpoints_dir.glob("best_model_epoch*.pth"):
            try:
                os.remove(f)
            except Exception as e:
                logger.warning(f"Could not delete {f}: {e}")
        # Save new best model
        checkpoint_path = checkpoints_dir / f'best_model_epoch{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Best model saved to {checkpoint_path}")
    else:
        checkpoint_path = checkpoints_dir / f'checkpoint_epoch{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

def create_weighted_sampler(train_dataset, train_label_counts):
    """
    Create a WeightedRandomSampler based on class frequencies in the training dataset.
    Only applicable for classification tasks.

    Args:
        train_dataset: The training dataset object.
        train_label_counts: A dictionary with class labels as keys and their counts as values.

    Returns:
        sampler: A WeightedRandomSampler object.
    """
    cls2int = train_dataset.labels_str2int
    
    # Check if this is a regression task (labels map to floats)
    first_mapped_value = next(iter(cls2int.values()))
    if isinstance(first_mapped_value, float):
        raise ValueError("Weighted sampler is not supported for regression tasks. "
                        "Set 'use_weighted_sampler': false in your config.")
    
    counts_int = {cls2int[k]: v for k, v in train_label_counts.items()}
    num_classes = len(cls2int)
    class_weights = torch.zeros(num_classes, dtype=torch.float)
    for k, v in counts_int.items():
        class_weights[k] = 1.0 / max(v, 1)

    sample_weights = []
    for key_bytes in train_dataset.lmdb_keys:
        key = key_bytes.decode('utf-8')
        lbl_str = train_dataset.labels[key]
        lbl_int = cls2int[lbl_str]
        sample_weights.append(class_weights[lbl_int].item())

    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def plot_attention_weights(attn_weights, labels, predictions, att_plots_dir, epoch, 
                           num_samples=4, timestamp=None):
    """
    Plot and save attention weight visualizations for a batch of samples.
    
    Args:
        attn_weights: Attention weights tensor of shape:
                      - (B, T) for attention pooling
                      - (B, T, T) for self-attention
                      - (B, num_heads, T, T) for multi-head self-attention
        labels: Ground truth labels
        predictions: Model predictions
        att_plots_dir: Directory to save attention plots
        epoch: Current epoch number
        num_samples: Number of samples to plot
        timestamp: Timestamp string for file naming
    
    Returns:
        plot_file: Path to the saved plot
    """
    import numpy as np
    
    # Handle different attention weight shapes
    attn_dim = attn_weights.dim()
    
    if attn_dim == 4:
        # (B, num_heads, T, T) - average over heads
        attn_weights = attn_weights.mean(dim=1)
        is_pooling_attention = False
    elif attn_dim == 3:
        # (B, T, T) - self-attention
        is_pooling_attention = False
    elif attn_dim == 2:
        # (B, T) - attention pooling weights
        is_pooling_attention = True
    else:
        raise ValueError(f"Unexpected attention weight shape: {attn_weights.shape}")
    
    # Convert to numpy
    attn_weights = attn_weights.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    
    
    # Limit number of samples to plot
    num_samples = min(num_samples, attn_weights.shape[0])
    
    if is_pooling_attention:
        # Plot bar charts for attention pooling weights
        fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 4))
        if num_samples == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            attn = attn_weights[i]
            x = np.arange(len(attn))
            
            ax.bar(x, attn, alpha=0.7, color='steelblue')
            ax.set_xlabel('Sequence Position (Frequency Bin)', fontsize=10)
            ax.set_ylabel('Attention Weight', fontsize=10)
            
            label = labels[i]
            pred = predictions[i]
            correct = "✓" if label == pred else "✗"
            ax.set_title(f'Sample {i+1}\nTrue: {label}, Pred: {pred} {correct}', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Attention Pooling Weights - Epoch {epoch}', fontsize=14, fontweight='bold')
    else:
        # Plot heatmaps for self-attention
        fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 4))
        if num_samples == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            attn = attn_weights[i]
            
            im = ax.imshow(attn, cmap='viridis', aspect='auto')
            ax.set_xlabel('Key Position', fontsize=10)
            ax.set_ylabel('Query Position', fontsize=10)
            
            label = labels[i]
            pred = predictions[i]
            correct = "✓" if label == pred else "✗"
            ax.set_title(f'Sample {i+1}\nTrue: {label}, Pred: {pred} {correct}', fontsize=11)
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Self-Attention Weights - Epoch {epoch}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    ts = timestamp if timestamp else datetime.now().strftime('%Y%m%d_%H%M')
    plot_file = att_plots_dir / f'attention_epoch{epoch}_{ts}.png'
    plt.savefig(plot_file, bbox_inches='tight', dpi=150)
    plt.close()
    
    return plot_file


def plot_attention_summary(attn_weights, labels, att_plots_dir, epoch, timestamp=None):
    """
    Plot averaged attention patterns grouped by class.
    
    Args:
        attn_weights: Attention weights tensor of shape:
                      - (B, T) for attention pooling
                      - (B, T, T) for self-attention
                      - (B, num_heads, T, T) for multi-head self-attention
        labels: Ground truth labels
        att_plots_dir: Directory to save attention plots
        epoch: Current epoch number
        timestamp: Timestamp string for file naming
    
    Returns:
        plot_file: Path to the saved plot
    """
    import numpy as np
    
    # Handle different attention weight shapes
    attn_dim = attn_weights.dim()
    
    if attn_dim == 4:
        # (B, num_heads, T, T) - average over heads
        attn_weights = attn_weights.mean(dim=1)
        is_pooling_attention = False
    elif attn_dim == 3:
        # (B, T, T) - self-attention
        is_pooling_attention = False
    elif attn_dim == 2:
        # (B, T) - attention pooling weights
        is_pooling_attention = True
    else:
        raise ValueError(f"Unexpected attention weight shape: {attn_weights.shape}")
    
    # Convert to numpy
    attn_weights = attn_weights.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Get unique classes
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    if is_pooling_attention:
        # Plot line charts for attention pooling weights by class
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes + 1))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_attn_mean = attn_weights[mask].mean(axis=0)
            class_attn_std = attn_weights[mask].std(axis=0)
            x = np.arange(len(class_attn_mean))
            
            ax.plot(x, class_attn_mean, label=f'Class {label} (n={mask.sum()})', 
                   color=colors[i], linewidth=2)
            ax.fill_between(x, class_attn_mean - class_attn_std, class_attn_mean + class_attn_std,
                           alpha=0.2, color=colors[i])
        
        # Overall average
        overall_mean = attn_weights.mean(axis=0)
        ax.plot(np.arange(len(overall_mean)), overall_mean, label=f'Overall (n={len(labels)})',
               color='black', linewidth=2, linestyle='--')
        
        ax.set_xlabel('Sequence Position (Frequency Bin)', fontsize=12)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_title(f'Attention Pooling Weights by Class - Epoch {epoch}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
    else:
        # Plot heatmaps for self-attention
        fig, axes = plt.subplots(1, num_classes + 1, figsize=(5 * (num_classes + 1), 4))
        
        # Plot per-class averaged attention
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_attn = attn_weights[mask].mean(axis=0)
            
            im = axes[i].imshow(class_attn, cmap='viridis', aspect='auto')
            axes[i].set_xlabel('Key Position', fontsize=10)
            axes[i].set_ylabel('Query Position', fontsize=10)
            axes[i].set_title(f'Class {label}\n(n={mask.sum()})', fontsize=11)
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Plot overall averaged attention
        overall_attn = attn_weights.mean(axis=0)
        im = axes[-1].imshow(overall_attn, cmap='viridis', aspect='auto')
        axes[-1].set_xlabel('Key Position', fontsize=10)
        axes[-1].set_ylabel('Query Position', fontsize=10)
        axes[-1].set_title(f'Overall Average\n(n={len(labels)})', fontsize=11)
        plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Attention Patterns by Class - Epoch {epoch}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    ts = timestamp if timestamp else datetime.now().strftime('%Y%m%d_%H%M')
    plot_file = att_plots_dir / f'attention_summary_epoch{epoch}_{ts}.png'
    plt.savefig(plot_file, bbox_inches='tight', dpi=150)
    plt.close()
    
    return plot_file


def compute_dataset_statistics(dataset, item_key, max_samples=None):
    """
    Compute descriptive statistics for a given item across the entire dataset.
    
    Args:
        dataset: Dataset object (e.g., DatasetLMDB)
        item_key: Key of the item to compute statistics for (e.g., 'envelope_spectrum')
        max_samples: Optional limit on number of samples to use (for speed)
    
    Returns:
        Dictionary with statistics:
            - mean: Global mean across all values
            - std: Global standard deviation
            - min: Minimum value
            - max: Maximum value
            - median: Median value
    """
    import numpy as np
    
    all_values = []
    
    num_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    item_key = item_key[0]
    
    for i in range(num_samples):
        sample = dataset[i]
        
        item = sample[item_key]
        
        # Convert to numpy if tensor
        if isinstance(item, torch.Tensor):
            item = item.numpy()

        if item_key == 'f0':
            log_f0 = item
            voiced_mask = sample['voiced_mask'].numpy().astype(bool)
            values = log_f0[voiced_mask]
        else:        
            values = item.flatten()
        
        all_values.extend(values.tolist())
    
    all_values = np.array(all_values)
    
    stats = {
        'mean': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'min': float(np.min(all_values)),
        'max': float(np.max(all_values)),
        'median': float(np.median(all_values)),
        'num_samples': num_samples,
        'total_values': len(all_values)
    }
    
    return stats

def quantize_values(values, mean, std, num_bins=32, min_val=-3.0, max_val=3.0):
    """
    Normalize and quantize continuous values to integer bin indices.
    
    Args:
        values: Tensor of shape (B,) or numpy array
        mean: Mean value for normalization (from training set)
        std: Standard deviation for normalization (from training set)
        num_bins: Number of quantization bins
        min_val: Minimum normalized value (in std units, e.g., -3.0)
        max_val: Maximum normalized value (in std units, e.g., 3.0)
    
    Returns:
        Quantized indices of shape (B,) with values in [0, num_bins-1]
    
    Example:
        >>> dur_bins = quantize_values(batch['dur'], mean=5.0, std=2.0, num_bins=32)
    """
    # Convert to tensor if numpy
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float32)
    
    # Normalize using statistics
    values_normalized = (values - mean) / (std + 1e-8)
    
    # Clamp to valid range
    values_clamped = torch.clamp(values_normalized, min_val, max_val)
    
    # Scale to [0, 1]
    values_scaled = (values_clamped - min_val) / (max_val - min_val)
    
    # Quantize to bins [0, num_bins-1]
    bins = (values_scaled * (num_bins - 1)).long()
    
    # Ensure valid range
    bins = torch.clamp(bins, 0, num_bins - 1)
    
    return bins


def plot_latent_space(encoder, dataloader, output_path, device, max_samples=2000):
    """
    Visualize the learned latent space using UMAP.
    Runs the encoder on samples and produces a 2D scatter plot colored by L1 vs L2.

    Args:
        encoder: RhythmEncoder (or RhythmContrastiveModel) that returns embeddings
        dataloader: DataLoader yielding batches with 'envelope' and 'label'
        output_path: Path to save the plot
        device: torch device
        max_samples: Maximum number of samples to use for visualization
    """
    import umap

    encoder.eval()
    all_embeddings = []
    all_labels = []
    is_duration_model = hasattr(encoder, 'lstm_v') and not hasattr(encoder, 'encoder')

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['label'].cpu().numpy()

            if is_duration_model:
                emb = encoder(batch['v_dur'].to(device), batch['c_dur'].to(device))
            elif hasattr(encoder, 'encoder'):
                emb = encoder(batch['envelope'].to(device))
            else:
                emb = encoder(batch['envelope'].to(device))

            all_embeddings.append(emb.cpu().numpy() if isinstance(emb, torch.Tensor) else emb[0].cpu().numpy())
            all_labels.append(labels)

            if sum(len(e) for e in all_embeddings) >= max_samples:
                break

    all_embeddings = np.concatenate(all_embeddings, axis=0)[:max_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:max_samples]

    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(all_embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))

    l1_mask = all_labels == 0
    l2_mask = all_labels == 1

    ax.scatter(reduced[l1_mask, 0], reduced[l1_mask, 1], c='steelblue', label='L1 (native)',
               alpha=0.6, s=10, edgecolors='none')
    ax.scatter(reduced[l2_mask, 0], reduced[l2_mask, 1], c='coral', label='L2 (non-native)',
               alpha=0.6, s=10, edgecolors='none')

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Rhythm Encoder Latent Space (UMAP)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def compute_val_centroid_distance(model, val_loader, device):
    """
    Per-utterance L1-L2 centroid cosine distance, averaged over validation utterances.
    Higher = better separation of native vs non-native speakers on unseen utterances.
    Works with both RhythmContrastiveModel and DurationContrastiveModel.
    """
    from collections import defaultdict
    from src.dataloaders import parse_identifier

    model.eval()
    utt_embs = defaultdict(lambda: {'L1': [], 'L2': []})
    is_duration_model = hasattr(model, 'lstm_v') and not hasattr(model, 'encoder')

    with torch.no_grad():
        for batch in val_loader:
            labels = batch['label']
            identifiers = batch['identifier']

            if is_duration_model:
                emb = model(batch['v_dur'].to(device), batch['c_dur'].to(device))
            elif hasattr(model, 'encoder'):
                emb = model(batch['envelope'].to(device))
            else:
                emb = model(batch['envelope'].to(device))

            emb_np = emb.cpu().numpy() if isinstance(emb, torch.Tensor) else emb[0].cpu().numpy()

            for i, identifier in enumerate(identifiers):
                _, utt_id = parse_identifier(identifier)
                cat = 'L1' if labels[i].item() == 0 else 'L2'
                utt_embs[utt_id][cat].append(emb_np[i])

    distances = []
    for utt_id, groups in utt_embs.items():
        if len(groups['L1']) == 0 or len(groups['L2']) == 0:
            continue
        c_l1 = np.mean(groups['L1'], axis=0)
        c_l2 = np.mean(groups['L2'], axis=0)
        cos_sim = np.dot(c_l1, c_l2) / (np.linalg.norm(c_l1) * np.linalg.norm(c_l2) + 1e-8)
        distances.append(1.0 - cos_sim)

    if not distances:
        return 0.0

    return float(np.mean(distances))