# filepath: /home/joao.lima/experiments/rhythm_classifier/src/inference.py
import torch
import json
import argparse
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from src import models
from src.dataloaders import DatasetLMDB, collate_fn
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_MAPPING = {
    "cnn": models.CNN_MLP,
    "lstm": models.LSTM_with_MultiHeadAttention,
    "deepSet": models.DeepSet,
    "setTransformer": models.SetTransformer
}


def load_speaker_metadata(csv_path):
    """
    Load speaker metadata from CSV file.
    
    Args:
        csv_path: Path to the CSV file with speaker metadata
    
    Returns:
        metadata_dict: Dictionary mapping identifier to {ibt, l1, spkID}
    """
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


def load_model(checkpoint_path, config):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        config: Model configuration dictionary
    
    Returns:
        model: Loaded model in eval mode
        checkpoint: Full checkpoint dictionary
    """
    model = MODEL_MAPPING.get(config['model_type'])(
        **config['model_params'],
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def run_inference_batch(model, dataloader, labels_int2str, speaker_int2str, utterance_int2str,
                        speaker_metadata=None, return_attention=False):
    """
    Run inference on a dataset using batched processing.
    
    Args:
        model: Trained model
        dataloader: DataLoader for inference
        labels_int2str: Mapping from label indices to strings
        speaker_int2str: Mapping from speaker indices to strings
        utterance_int2str: Mapping from utterance indices to strings
        speaker_metadata: Dictionary mapping identifier to {ibt, l1, spkID}
        return_attention: Whether to return attention weights (if model supports it)
    
    Returns:
        results: List of dictionaries with predictions and metadata
        all_attention_weights: List of attention weight tensors (if return_attention=True)
    """
    results = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
            labels = batch['label']
            identifiers = batch.get('identifier', None)  # This is now a list of strings
            speaker_ids = batch['speaker_id']
            utterance_ids = batch['utterance_id']

            # Try to get attention weights if requested
            if return_attention:
                try:
                    outputs, attn_weights = model(batch_device, return_attention=True)
                    all_attention_weights.append(attn_weights.cpu())
                except TypeError:
                    # Model doesn't support return_attention
                    outputs = model(batch_device)
            else:
                outputs = model(batch_device)
            
            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results for each sample in batch
            for i in range(len(labels)):
                # Convert speaker and utterance IDs to strings
                speaker_id_int = speaker_ids[i].item()
                speaker_id_str = speaker_int2str.get(speaker_id_int, 'unknown')
                utterance_id_int = utterance_ids[i].item()
                utterance_id_str = utterance_int2str.get(utterance_id_int, 'unknown')
                
                result = {
                    'true_label_int': labels[i].item(),
                    'true_label_str': labels_int2str.get(labels[i].item(), 'unknown'),
                    'pred_label_int': predicted[i].item(),
                    'pred_label_str': labels_int2str.get(predicted[i].item(), 'unknown'),
                    'confidence': probabilities[i, predicted[i]].item(),
                    'probabilities': probabilities[i].cpu().numpy().tolist(),
                    'correct': labels[i].item() == predicted[i].item(),
                    'speaker_id': speaker_id_str,
                    'utterance_id': utterance_id_str,
                }
                
                # Add identifier and speaker metadata if available
                if identifiers is not None:
                    identifier_str = identifiers[i]  # Already a string
                    result['identifier'] = identifier_str
                    
                    if speaker_metadata and identifier_str in speaker_metadata:
                        meta = speaker_metadata[identifier_str]
                        result['ibt'] = meta['ibt']
                        result['l1'] = meta['l1']
                
                results.append(result)
    
    return results, all_attention_weights if return_attention else None


def save_results(results, output_dir, exp_id, config, checkpoint_info, dataset_stats=None):
    """
    Save inference results to files.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save results
        exp_id: Experiment identifier
        config: Model configuration
        checkpoint_info: Information about the loaded checkpoint
        dataset_stats: Dictionary with dataset statistics (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV for easy analysis
    df = pd.DataFrame(results)
    csv_path = output_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)
    
    # Compute summary statistics
    correct = sum(r['correct'] for r in results)
    total = len(results)
    accuracy = 100 * correct / total if total > 0 else 0
    
    # Per-class accuracy
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
        'exp_id': exp_id,
        'model_type': config['model_type'],
        'checkpoint_epoch': checkpoint_info.get('epoch', 'unknown'),
        'total_samples': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'per_class_stats': class_stats,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add dataset statistics if provided
    if dataset_stats:
        summary['dataset_stats'] = dataset_stats
    
    summary = {
        'exp_id': exp_id,
        'model_type': config['model_type'],
        'checkpoint_epoch': checkpoint_info.get('epoch', 'unknown'),
        'total_samples': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'per_class_stats': class_stats,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save summary as JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save config for reference
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return summary


def inference(args):
    """
    Main inference function.
    """
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create experiment ID and output directory
    model_name = config['model_type']
    exp_id = args.exp_id if args.exp_id else datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path("inference") / f"{model_name}_{exp_id}"
    
    print("="*70)
    print(f"Inference - Model: {model_name} | Exp ID: {exp_id}")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint, config)
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Checkpoint val_acc: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    
    # Load dataset
    print(f"\nLoading dataset (split: {args.split})...")
    dataset = DatasetLMDB(**config['dataset_params'], split=args.split)
    labels_int2str = dataset.labels_int2str
    
    # Get dataset statistics
    num_unique_speakers = len(dataset.speaker_str2int)
    num_unique_utterances = len(dataset.utterance_str2int)
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Unique speakers: {num_unique_speakers}")
    print(f"  Unique utterances: {num_unique_utterances}")
    print(f"  Label mapping: {dataset.labels_str2int}")
    
    # Load speaker metadata if provided
    speaker_metadata = None
    if args.metadata_csv:
        print(f"\nLoading speaker metadata from: {args.metadata_csv}")
        speaker_metadata = load_speaker_metadata(args.metadata_csv)
        if speaker_metadata:
            print(f"  Loaded metadata for {len(speaker_metadata)} samples")
        else:
            print("  Warning: Could not load metadata")
    
    # Check if model uses attention
    use_attention = config['model_params'].get('use_attention', False) and args.save_attention
    
    # Run inference
    print("\nRunning inference...")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            collate_fn=collate_fn, shuffle=False)
    print(f"  Running on all {len(dataset)} samples")
        
    results, attention_weights = run_inference_batch(
        model, dataloader, labels_int2str,
        dataset.speaker_int2str, dataset.utterance_int2str,
        speaker_metadata, return_attention=use_attention
    )    # Save results
    print(f"\nSaving results to: {output_dir}")
    dataset_stats = {
        'split': args.split,
        'num_samples': len(dataset),
        'num_unique_speakers': num_unique_speakers,
        'num_unique_utterances': num_unique_utterances
    }
    summary = save_results(results, output_dir, exp_id, config, checkpoint, dataset_stats)
    
    # Print summary
    print("\n" + "="*70)
    print("Inference Summary")
    print("="*70)
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Correct predictions: {summary['correct_predictions']}")
    print(f"  Accuracy: {summary['accuracy']:.2f}%")
    print("\n  Per-class statistics:")
    for label, stats in summary['per_class_stats'].items():
        print(f"    {label}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2f}%)")
    print("="*70)
    
    # Save attention weights if requested
    if attention_weights and args.save_attention:
        attn_path = output_dir / "attention_weights.pt"
        torch.save(attention_weights, attn_path)
        print(f"\nAttention weights saved to: {attn_path}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the JSON configuration file (same as used for training).')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint (.pth file).')
    parser.add_argument('--split', type=str, default='Test',
                        choices=['Train', 'Development', 'Test'],
                        help='Dataset split to run inference on.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference.')
    parser.add_argument('--exp_id', type=str, default=None,
                        help='Experiment ID for output directory naming.')
    parser.add_argument('--metadata_csv', type=str, default=None,
                        help='Path to CSV file with speaker metadata (columns: path, spkID, identifier, ibt, l1).')
    parser.add_argument('--save_attention', action='store_true',
                        help='Save attention weights if model supports it.')
    args = parser.parse_args()
    
    inference(args)
