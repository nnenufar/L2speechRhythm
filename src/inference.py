import torch
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import models
from src.dataloaders import DatasetLMDB, collate_fn, processor_ssl
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_MAPPING = {
    "cnn": models.CNN_MLP,
    "ssl": models.WAV_LM 
}

COLLATE_FUNC_MAPPING = {
    "ssl": processor_ssl,
    "pad": collate_fn
}


def _build_dataset_kwargs(config, utterance_str2int):
    dataset_kwargs = {**config['dataset_params']}
    if utterance_str2int is not None:
        dataset_kwargs['external_utterance_str2int'] = utterance_str2int
        dataset_kwargs['external_utterance_int2str'] = {v: k for k, v in utterance_str2int.items()}
    return dataset_kwargs


def load_model(checkpoint_path, config):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loading model from {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')}) on device {device}.")

    model_kwargs = {**config['model_params']}
    utterance_str2int = checkpoint.get('utterance_str2int')
    if utterance_str2int is not None:
        model_kwargs['num_utterances'] = len(utterance_str2int)

    model = MODEL_MAPPING.get(config['model_type'])(
        **model_kwargs,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def load_speaker_metadata(csv_path):
    """Load speaker metadata from CSV file."""
    if csv_path is None or not Path(csv_path).exists():
        return None
    
    df = pd.read_csv(csv_path)
    metadata_dict = {}
    for _, row in df.iterrows():
        metadata_dict[row['identifier']] = {
            'ibt': row['ibt'],
            'l1': row['l1'],
            'spkID': row['spkID']
        }
    return metadata_dict


def run_inference(model, dataloader, labels_int2str, speaker_int2str, utterance_int2str,
                  speaker_metadata=None, return_attention=False):
    """Run inference on entire dataset."""
    results = []
    all_attention_weights = []
    all_identifiers = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
            labels = batch['label']
            identifiers = batch.get('identifier', None)
            speaker_ids = batch['speaker_id']
            utterance_ids = batch['utterance_id']

            if return_attention:
                try:
                    outputs, attn_weights = model(batch_device, return_attention=True)
                    all_attention_weights.append(attn_weights.cpu())
                except TypeError:
                    outputs = model(batch_device)
                    attn_weights = None
            else:
                outputs = model(batch_device)
            
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                speaker_id_str = speaker_int2str.get(speaker_ids[i].item(), 'unknown')
                utterance_id_str = utterance_int2str.get(utterance_ids[i].item(), 'unknown')
                identifier_str = identifiers[i] if identifiers else f"{speaker_id_str}_{utterance_id_str}"
                
                result = {
                    'identifier': identifier_str,
                    'true_label_int': labels[i].item(),
                    'true_label_str': labels_int2str.get(labels[i].item(), 'unknown'),
                    'pred_label_int': predicted[i].item(),
                    'pred_label_str': labels_int2str.get(predicted[i].item(), 'unknown'),
                    'confidence': probabilities[i, predicted[i]].item(),
                    'prob_class_0': probabilities[i, 0].item(),
                    'prob_class_1': probabilities[i, 1].item() if probabilities.shape[1] > 1 else 0.0,
                    'correct': labels[i].item() == predicted[i].item(),
                    'speaker_id': speaker_id_str,
                    'utterance_id': utterance_id_str,
                }
                
                if speaker_metadata and identifier_str in speaker_metadata:
                    meta = speaker_metadata[identifier_str]
                    result['ibt'] = meta['ibt']
                    result['l1'] = meta['l1']
                
                results.append(result)
                all_identifiers.append(identifier_str)
    
    # Store attention weights as list (variable lengths) with identifiers
    if return_attention and all_attention_weights:
        # Flatten the list of batch attention weights to individual samples
        flattened_attn = []
        for batch_attn in all_attention_weights:
            for i in range(batch_attn.shape[0]):
                flattened_attn.append(batch_attn[i])
        all_attention_weights = flattened_attn
    else:
        all_attention_weights = None
    
    return results, all_attention_weights, all_identifiers


def compute_and_save_metrics(results, output_dir, config, checkpoint_info):
    """Compute and save inference metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions CSV
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "predictions.csv", index=False)
    
    # Compute statistics
    correct = sum(r['correct'] for r in results)
    total = len(results)
    accuracy = 100 * correct / total if total > 0 else 0
    
    class_stats = {}
    for label_str in set(r['true_label_str'] for r in results):
        class_results = [r for r in results if r['true_label_str'] == label_str]
        class_correct = sum(r['correct'] for r in class_results)
        class_total = len(class_results)
        class_stats[label_str] = {
            'correct': class_correct,
            'total': class_total,
            'accuracy': 100 * class_correct / class_total if class_total > 0 else 0
        }
    
    summary = {
        'checkpoint_epoch': checkpoint_info.get('epoch', 'unknown'),
        'total_samples': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'per_class_stats': class_stats,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    return summary


def filter_by_condition(results, identifiers, attn_weights, condition):
    """
    Filter results by a condition string.
    
    Args:
        results: List of result dictionaries
        identifiers: List of identifiers
        attn_weights: List of attention weight tensors (variable lengths)
        condition: Condition string, e.g., "l1=='US'" or "ibt>0.5"
    
    Returns:
        Filtered results, identifiers, and attention weights
    """
    df = pd.DataFrame(results)
    
    try:
        mask = df.eval(condition)
    except Exception as e:
        print(f"Error evaluating condition '{condition}': {e}")
        print("Available columns:", df.columns.tolist())
        return [], [], None
    
    filtered_indices = mask[mask].index.tolist()
    filtered_results = [results[i] for i in filtered_indices]
    filtered_identifiers = [identifiers[i] for i in filtered_indices]
    
    if attn_weights is not None:
        filtered_attn = [attn_weights[i] for i in filtered_indices]
    else:
        filtered_attn = None
    
    return filtered_results, filtered_identifiers, filtered_attn


def plot_attention_samples(attn_weights, results, identifiers, output_dir, group_name, 
                           num_samples=8, labels_int2str=None):
    """Plot attention weights for individual samples."""
    if attn_weights is None or len(results) == 0:
        return
    
    output_dir = Path(output_dir)
    num_samples = min(num_samples, len(results))
    
    # Select samples (mix of correct and incorrect if possible)
    correct_indices = [i for i, r in enumerate(results) if r['correct']]
    incorrect_indices = [i for i, r in enumerate(results) if not r['correct']]
    
    selected = []
    for i in range(num_samples):
        if i % 2 == 0 and incorrect_indices:
            selected.append(incorrect_indices.pop(0))
        elif correct_indices:
            selected.append(correct_indices.pop(0))
        elif incorrect_indices:
            selected.append(incorrect_indices.pop(0))
    
    if not selected:
        return
    
    ncols = min(4, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten() if len(selected) > 1 else [axes]
    
    for ax_idx, sample_idx in enumerate(selected):
        ax = axes[ax_idx]
        attn = attn_weights[sample_idx]
        if isinstance(attn, torch.Tensor):
            attn = attn.cpu().numpy()
        attn = np.squeeze(attn)
        result = results[sample_idx]
        
        x = np.arange(len(attn))
        ax.bar(x, attn, alpha=0.7, color='steelblue')
        
        # Highlight top positions
        top_k = min(5, len(attn))
        top_indices = np.argsort(attn)[-top_k:]
        for idx in top_indices:
            ax.bar(idx, attn[idx], color='coral', alpha=0.8)
        
        correct_str = "✓" if result['correct'] else "✗"
        ax.set_title(f"{identifiers[sample_idx][:20]}\n"
                    f"T:{result['true_label_str']} P:{result['pred_label_str']} "
                    f"({result['confidence']:.2f}) {correct_str}", fontsize=9)
        ax.set_xlabel('Position', fontsize=8)
        ax.set_ylabel('Attention', fontsize=8)
        ax.tick_params(labelsize=7)
    
    # Hide unused axes
    for ax_idx in range(len(selected), len(axes)):
        axes[ax_idx].set_visible(False)
    
    plt.suptitle(f"Attention Weights - {group_name} (n={len(results)})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"attention_samples_{group_name}.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_summary(attn_weights, results, output_dir, group_name):
    """Plot attention summary statistics by class."""
    if attn_weights is None or len(results) == 0:
        return
    
    output_dir = Path(output_dir)
    
    # Convert list of variable-length tensors to numpy arrays
    # Normalize each to same length by interpolation
    attn_list = []
    for attn in attn_weights:
        if isinstance(attn, torch.Tensor):
            attn = attn.cpu().numpy()
        attn_list.append(np.squeeze(attn))
    
    # Find median length and interpolate all to that length
    lengths = [len(a) for a in attn_list]
    target_len = int(np.median(lengths))
    
    attn_normalized = []
    for attn in attn_list:
        if len(attn) == target_len:
            attn_normalized.append(attn)
        else:
            # Interpolate to target length
            x_old = np.linspace(0, 1, len(attn))
            x_new = np.linspace(0, 1, target_len)
            attn_interp = np.interp(x_new, x_old, attn)
            attn_normalized.append(attn_interp)
    
    attn_np = np.array(attn_normalized)
    
    labels = np.array([r['true_label_int'] for r in results])
    unique_labels = np.unique(labels)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean attention by class
    ax1 = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels) + 1))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_attn_mean = attn_np[mask].mean(axis=0)
        class_attn_std = attn_np[mask].std(axis=0)
        x = np.arange(len(class_attn_mean))
        
        label_str = results[np.where(mask)[0][0]]['true_label_str']
        ax1.plot(x, class_attn_mean, label=f'{label_str} (n={mask.sum()})', 
                color=colors[i], linewidth=2)
        ax1.fill_between(x, class_attn_mean - class_attn_std, class_attn_mean + class_attn_std,
                        alpha=0.2, color=colors[i])
    
    ax1.set_xlabel('Normalized Position', fontsize=11)
    ax1.set_ylabel('Attention Weight', fontsize=11)
    ax1.set_title(f'Mean Attention by Class - {group_name}', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Attention difference (if binary classification)
    ax2 = axes[1]
    if len(unique_labels) == 2:
        mask_0 = labels == unique_labels[0]
        mask_1 = labels == unique_labels[1]
        diff = attn_np[mask_1].mean(axis=0) - attn_np[mask_0].mean(axis=0)
        x = np.arange(len(diff))
        
        colors_diff = ['coral' if d > 0 else 'steelblue' for d in diff]
        ax2.bar(x, diff, color=colors_diff, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Normalized Position', fontsize=11)
        ax2.set_ylabel('Attention Difference', fontsize=11)
        
        label_0 = results[np.where(mask_0)[0][0]]['true_label_str']
        label_1 = results[np.where(mask_1)[0][0]]['true_label_str']
        ax2.set_title(f'Attention Diff ({label_1} - {label_0})', fontsize=12, fontweight='bold')
    else:
        # Overall distribution
        overall_mean = attn_np.mean(axis=0)
        overall_std = attn_np.std(axis=0)
        x = np.arange(len(overall_mean))
        ax2.bar(x, overall_mean, yerr=overall_std, alpha=0.7, color='steelblue', capsize=2)
        ax2.set_xlabel('Normalized Position', fontsize=11)
        ax2.set_ylabel('Attention Weight', fontsize=11)
        ax2.set_title(f'Overall Attention Distribution - {group_name}', fontsize=12, fontweight='bold')
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"attention_summary_{group_name}.png", dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    """Main inference function."""
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    exp_name = config.get('exp_name', 'unnamed')
    output_dir = Path("inference") / args.exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    collateFunc = COLLATE_FUNC_MAPPING.get(config['collate_fn'])
    
    print("="*70)
    print(f"Inference: {exp_name}")

    print("="*70)
    
    # Check if results already exist
    predictions_path = output_dir / "predictions.csv"
    summary_path = output_dir / "summary.json"
    
    if predictions_path.exists() and summary_path.exists() and not args.force:
        print(f"\nResults already exist at {output_dir}")
        print("Loading existing results... (use --force to recompute)")
        results = pd.read_csv(predictions_path).to_dict('records')
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        checkpoint_info = {'epoch': summary.get('checkpoint_epoch', 'unknown')}
        
        # Still need to load model and dataset for attention if groups specified
        if args.group:
            print("\nLoading model for attention computation...")
            model, checkpoint_info = load_model(args.checkpoint, config)
            utt_str2int = checkpoint_info.get('utterance_str2int')
            dataset_kwargs = _build_dataset_kwargs(config, utt_str2int)
            dataset = DatasetLMDB(**dataset_kwargs, split=args.split)
    else:
        # Load model
        print(f"\nLoading model from: {args.checkpoint}")
        model, checkpoint_info = load_model(args.checkpoint, config)
        print(f"  Checkpoint epoch: {checkpoint_info.get('epoch', 'unknown')}")
        
        # Load dataset
        print(f"\nLoading dataset (split: {args.split})...")
        utt_str2int = checkpoint_info.get('utterance_str2int')
        dataset_kwargs = _build_dataset_kwargs(config, utt_str2int)
        dataset = DatasetLMDB(**dataset_kwargs, split=args.split)
        print(f"  Dataset size: {len(dataset)}")
        
        # Load metadata
        speaker_metadata = load_speaker_metadata(args.metadata_csv) if args.metadata_csv else None
        
        # Determine if attention should be computed
        use_attention = args.group is not None
        
        # Run inference
        print("\nRunning inference...")
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                collate_fn=collateFunc, shuffle=False)

        results, attn_weights, identifiers = run_inference(
            model, dataloader, dataset.labels_int2str,
            dataset.speaker_int2str, dataset.utterance_int2str,
            speaker_metadata, return_attention=use_attention
        )
        
        # Save metrics
        print(f"\nSaving results to: {output_dir}")
        summary = compute_and_save_metrics(results, output_dir, config, checkpoint_info)
    
    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Accuracy: {summary['accuracy']:.2f}%")
    for label, stats in summary['per_class_stats'].items():
        print(f"    {label}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2f}%)")
    
    # Process groups if specified
    if args.group:
        print("\n" + "="*70)
        print("Processing attention for groups")
        print("="*70)
        
        # Need attention weights - recompute if we loaded from cache
        if 'attn_weights' not in dir() or attn_weights is None:
            print("\nComputing attention weights...")
            dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                    collate_fn=collateFunc, shuffle=False)
            speaker_metadata = load_speaker_metadata(args.metadata_csv) if args.metadata_csv else None
            results, attn_weights, identifiers = run_inference(
                model, dataloader, dataset.labels_int2str,
                dataset.speaker_int2str, dataset.utterance_int2str,
                speaker_metadata, return_attention=True
            )
        
        # Create attention output directory
        attn_dir = output_dir / "attention"
        attn_dir.mkdir(exist_ok=True)
        
        for group_spec in args.group:
            # Parse group: "name:condition" or just "condition"
            if ':' in group_spec:
                group_name, condition = group_spec.split(':', 1)
            else:
                group_name = group_spec.replace(' ', '_').replace('==', '_eq_').replace('>', '_gt_').replace('<', '_lt_')
                condition = group_spec
            
            print(f"\n  Group '{group_name}': {condition}")
            
            # Filter by condition
            filtered_results, filtered_ids, filtered_attn = filter_by_condition(
                results, identifiers, attn_weights, condition
            )
            
            if len(filtered_results) == 0:
                print(f"    No samples match condition")
                continue
            
            print(f"    Found {len(filtered_results)} samples")
            
            # Compute group accuracy
            correct = sum(r['correct'] for r in filtered_results)
            acc = 100 * correct / len(filtered_results)
            print(f"    Accuracy: {acc:.2f}%")
            
            # Plot attention
            plot_attention_samples(filtered_attn, filtered_results, filtered_ids, 
                                  attn_dir, group_name, num_samples=8)
            plot_attention_summary(filtered_attn, filtered_results, attn_dir, group_name)
            print(f"    Plots saved to {attn_dir}")
    
    print("\n" + "="*70)
    print(f"All results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the JSON configuration file.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint (.pth file).')
    parser.add_argument('--exp_id', type=str, required=True,
                        help='Experiment ID for output directory.')
    parser.add_argument('--split', type=str, default='Test',
                        choices=['Train', 'Development', 'Test'],
                        help='Dataset split to run inference on.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference.')
    parser.add_argument('--metadata_csv', type=str, default=None,
                        help='Path to CSV with speaker metadata.')
    parser.add_argument('--group', type=str, nargs='+', default=None,
                        help='Group conditions for attention analysis. '
                             'Format: "name:condition" or just "condition". '
                             'Examples: --group "US:l1==\'US\'" "correct==True" "ibt>0.5"')
    parser.add_argument('--force', action='store_true',
                        help='Force recomputation even if results exist.')
    
    args = parser.parse_args()
    main(args)
