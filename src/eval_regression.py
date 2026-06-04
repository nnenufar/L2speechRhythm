"""
Evaluate a trained RhythmRegressor on a dataset split.
Saves per-sample predictions CSV and aggregate metrics JSON.

Usage:
    python -m src.eval_regression \
        --config config/rhythm_regression_speechocean.json \
        --checkpoint exp/rhythm_regression_speechocean/checkpoints/<ts>/best_model.pth \
        --split Test
"""

import torch
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from src.models import RhythmRegressor
from src.dataloaders import DatasetLMDB, collate_fn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='Test',
                        choices=['Train', 'Development', 'Test'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    exp_name = config.get('exp_name', 'unnamed')
    output_dir = Path('exp') / exp_name / 'eval'
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    target_mean = config['model_params'].get('target_mean')
    target_std = config['model_params'].get('target_std')

    task = config['model_params'].get('task', 'classification')
    is_regression = task == 'regression'

    dataset = DatasetLMDB(
        config['dataset_params']['lmdb_path'],
        data_source=config['dataset_params']['data_source'],
        split=args.split,
        items=config['dataset_params']['items'],
        test_size=config['dataset_params'].get('test_size', 0.20),
        target_mean=target_mean,
        target_std=target_std,
    )

    if is_regression and target_mean is None:
        raw_labels = np.array([float(v) for v in dataset.labels.values()])
        target_mean = float(np.mean(raw_labels))
        target_std = float(np.std(raw_labels))
        dataset.target_mean = target_mean
        dataset.target_std = target_std
        print(f"Auto-computed target normalization: mean={target_mean:.4f}, std={target_std:.4f}")

    print(f"Split: {args.split}, samples: {len(dataset)}")

    model = RhythmRegressor(**config['model_params']).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if target_mean is None:
        target_mean = checkpoint.get('target_mean')
        target_std = checkpoint.get('target_std')

    print(f"Loaded checkpoint epoch {checkpoint.get('epoch', '?')}")

    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=False)

    all_identifiers = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            targets = batch_dev['label'].cpu().numpy()
            outputs = model(batch_dev)
            preds = outputs.cpu().numpy()

            all_identifiers.extend(batch['identifier'])
            all_targets.extend(targets.tolist())
            all_preds.extend(preds.tolist())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    if is_regression and target_mean is not None and target_std is not None:
        all_targets_denorm = all_targets * target_std + target_mean
        all_preds_denorm = all_preds * target_std + target_mean
    else:
        all_targets_denorm = all_targets
        all_preds_denorm = all_preds

    rmse = float(np.sqrt(mean_squared_error(all_targets_denorm, all_preds_denorm)))
    mae = float(mean_absolute_error(all_targets_denorm, all_preds_denorm))
    pearson_r, _ = pearsonr(all_targets_denorm, all_preds_denorm)
    spearman_r, _ = spearmanr(all_targets_denorm, all_preds_denorm)

    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, Pearson r: {pearson_r:.4f}, Spearman r: {spearman_r:.4f}")

    df = pd.DataFrame({
        'identifier': all_identifiers,
        'ground_truth': all_targets_denorm,
        'prediction': all_preds_denorm,
    })
    csv_path = output_dir / f'predictions_{args.split}_{ts}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")

    summary = {
        'exp_name': exp_name,
        'split': args.split,
        'num_samples': len(all_targets),
        'rmse': rmse,
        'mae': mae,
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'target_mean': target_mean,
        'target_std': target_std,
    }
    summary_path = output_dir / f'summary_{args.split}_{ts}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
