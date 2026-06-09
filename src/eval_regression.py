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

from src.models import RhythmRegressor
from src.dataloaders import DatasetLMDB, collate_fn
from src.evaluation import collect_regression_predictions
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

    dataset = DatasetLMDB(
        config['dataset_params']['lmdb_path'],
        data_source=config['dataset_params']['data_source'],
        split=args.split,
        items=config['dataset_params']['items'],
        test_size=config['dataset_params'].get('test_size', 0.20),
        label_column=config['dataset_params'].get('label_column', 'fluency'),
    )
    print(f"Split: {args.split}, samples: {len(dataset)}")

    model = RhythmRegressor(**config['model_params']).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint epoch {checkpoint.get('epoch', '?')}")

    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=False)

    results = collect_regression_predictions(model, dataloader, device)
    m = results['metrics']

    print(f"RMSE: {m['rmse']:.4f}, MAE: {m['mae']:.4f}, "
          f"Pearson r: {m['pearson_r']:.4f}, Spearman r: {m['spearman_r']:.4f}")

    df = pd.DataFrame({
        'identifier': results['identifiers'],
        'ground_truth': results['targets'],
        'prediction': results['preds'],
    })
    csv_path = output_dir / f'predictions_{args.split}_{ts}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")

    summary = {
        'exp_name': exp_name,
        'split': args.split,
        'num_samples': m['num_samples'],
        'rmse': m['rmse'],
        'mae': m['mae'],
        'pearson_r': m['pearson_r'],
        'spearman_r': m['spearman_r'],
        'checkpoint_epoch': checkpoint.get('epoch'),
        'target_mean': m['target_mean'],
        'target_std': m['target_std'],
    }
    summary_path = output_dir / f'summary_{args.split}_{ts}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
