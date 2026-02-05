from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import os

def setup_experiment_dir(exp_name, timestamp=None):
    """
    Create experiment directory structure and return paths.
    
    Args:
        exp_name: Name of the experiment
        timestamp: Optional timestamp string for attention plots subdirectory
    
    Returns:
        exp_dir, logs_dir, plots_dir, checkpoints_dir, att_plots_dir, contrastive_plots_dir
    """
    exp_dir = Path("exp") / exp_name
    logs_dir = exp_dir / "logs"
    plots_dir = exp_dir / "plots"
    checkpoints_dir = exp_dir / "checkpoints"
    
    # Create timestamped subdirectory for attention plots
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    att_plots_dir = exp_dir / "att_plots" / timestamp
    contrastive_plots_dir = exp_dir / "contrastive_plots" / timestamp
    
    # Create directories
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    att_plots_dir.mkdir(parents=True, exist_ok=True)
    contrastive_plots_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir, logs_dir, plots_dir, checkpoints_dir, att_plots_dir, contrastive_plots_dir

def setup_logger(logs_dir, exp_name):
    """
    Setup logging configuration with both file and console handlers.
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
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_file}")
    
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

def save_checkpoint(model, optimizer, epoch, train_loss, train_acc, val_loss, val_acc, 
                   checkpoints_dir, logger, is_best=False):
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

    Args:
        train_dataset: The training dataset object.
        train_label_counts: A dictionary with class labels as keys and their counts as values.

    Returns:
        sampler: A WeightedRandomSampler object.
    """
    cls2int = train_dataset.labels_str2int
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


def plot_contrastive_embeddings(model, dataloader, device, contrastive_plots_dir, epoch, 
                                 labels_int2str=None, timestamp=None, max_samples=1000):
    """
    Visualize contrastive embeddings in 2D space using t-SNE.
    
    Args:
        model: The model (must support return_embeddings=True)
        dataloader: DataLoader to get samples from
        device: Device to run inference on
        contrastive_plots_dir: Directory to save plots
        epoch: Current epoch number
        labels_int2str: Optional mapping from label indices to strings
        timestamp: Timestamp string for file naming
        max_samples: Maximum number of samples to plot (for speed)
    
    Returns:
        plot_file: Path to the saved plot
        metrics: Dictionary with embedding quality metrics
    """
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import pdist, squareform
    
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
            
            # Get projected embeddings
            _, embeddings = model(batch_device, return_embeddings=True)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(batch['label'])
            
            # Limit samples for speed
            total_samples = sum(e.shape[0] for e in all_embeddings)
            if total_samples >= max_samples:
                break
    
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    # Limit to max_samples
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    # Compute embedding quality metrics
    metrics = compute_embedding_metrics(embeddings, labels)
    
    # t-SNE projection
    perplexity = min(30, len(embeddings) - 1)  # Adjust perplexity for small datasets
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Get unique labels and colors
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: t-SNE visualization
    ax1 = axes[0]
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = labels_int2str.get(label, f'Class {label}') if labels_int2str else f'Class {label}'
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], label=f'{label_name} (n={mask.sum()})',
                   alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax1.set_title(f'Contrastive Embeddings (t-SNE) - Epoch {epoch}', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Metrics summary
    ax2 = axes[1]
    ax2.axis('off')
    
    metrics_text = f"""
    Embedding Quality Metrics (Epoch {epoch})
    {'='*40}
    
    Silhouette Score: {metrics['silhouette_score']:.4f}
    (Range: -1 to 1, higher is better)
    
    Intra-class Distance: {metrics['intra_class_distance']:.4f}
    (Average distance within same class, lower is better)
    
    Inter-class Distance: {metrics['inter_class_distance']:.4f}
    (Average distance between classes, higher is better)
    
    Separation Ratio: {metrics['separation_ratio']:.4f}
    (Inter/Intra ratio, higher is better)
    
    {'='*40}
    Total samples: {len(embeddings)}
    """
    
    ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    
    # Save plot
    ts = timestamp if timestamp else datetime.now().strftime('%Y%m%d_%H%M')
    plot_file = contrastive_plots_dir / f'embeddings_epoch{epoch}_{ts}.png'
    plt.savefig(plot_file, bbox_inches='tight', dpi=150)
    plt.close()
    
    return plot_file, metrics


def compute_embedding_metrics(embeddings, labels):
    """
    Compute metrics to quantify embedding quality.
    
    Args:
        embeddings: Numpy array of shape (N, D)
        labels: Numpy array of shape (N,)
    
    Returns:
        Dictionary with embedding quality metrics
    """
    import numpy as np
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import pdist, squareform, cosine
    
    # Need at least 2 samples per class for meaningful metrics
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2 or min(counts) < 2:
        return {
            'silhouette_score': 0.0,
            'intra_class_distance': 0.0,
            'inter_class_distance': 0.0,
            'separation_ratio': 0.0
        }
    
    # Silhouette score: -1 (bad) to 1 (good)
    try:
        silhouette = silhouette_score(embeddings, labels, metric='cosine')
    except:
        silhouette = 0.0
    
    # Compute pairwise cosine distances
    # Cosine distance = 1 - cosine_similarity
    distances = squareform(pdist(embeddings, metric='cosine'))
    
    # Intra-class distances (same class)
    intra_distances = []
    # Inter-class distances (different class)
    inter_distances = []
    
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra_distances.append(distances[i, j])
            else:
                inter_distances.append(distances[i, j])
    
    intra_mean = np.mean(intra_distances) if intra_distances else 0.0
    inter_mean = np.mean(inter_distances) if inter_distances else 0.0
    separation_ratio = inter_mean / (intra_mean + 1e-8)
    
    return {
        'silhouette_score': float(silhouette),
        'intra_class_distance': float(intra_mean),
        'inter_class_distance': float(inter_mean),
        'separation_ratio': float(separation_ratio)
    }


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
        
        # Flatten and collect all values
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