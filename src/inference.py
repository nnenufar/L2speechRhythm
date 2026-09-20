import json
from datetime import datetime
from pathlib import Path

import lmdb
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataloaders import DatasetLMDB, collate_fn
from src.evaluation import collect_regression_predictions
from src.train import COLLATE_FUNC_MAPPING, MODEL_MAPPING

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_test_inference(config, checkpoint_path, split='Test', output_dir=None):
    exp_name = config.get('exp_name', 'unnamed')
    model_type = config.get('model_type')
    if model_type not in MODEL_MAPPING:
        raise ValueError(f"Unknown model_type: {model_type}")

    dataset_params = config['dataset_params'].copy()
    dataset_params.pop('vc_features_path', None)
    lmdb_path = config['dataset_params']['lmdb_path']

    vc_features = None
    num_tokens = None
    max_phones = None
    if model_type == 'duration_regressor':
        vc_path = config['dataset_params'].get(
            'vc_features_path', 'data/speechocean/vc_features.json'
        )
        with open(vc_path) as f:
            vc_data = json.load(f)
        vc_features = vc_data['samples']
        num_tokens = len(vc_data['vocab'])
        max_v = max(max(len(p) for p in s.get('v_phones', [])) for s in vc_features.values())
        max_c = max(max(len(p) for p in s.get('c_phones', [])) for s in vc_features.values())
        max_phones = max(max_v, max_c)

    lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    train_dataset = DatasetLMDB(
        lmdb_path, **dataset_params, split='Train', vc_features=vc_features, env=lmdb_env
    )
    split_dataset = DatasetLMDB(
        lmdb_path, **dataset_params, split=split, vc_features=vc_features, env=lmdb_env
    )

    collate_func = COLLATE_FUNC_MAPPING.get(config.get('collate_fn', 'pad'), collate_fn)
    loader = DataLoader(
        split_dataset, batch_size=config['batch_size'], collate_fn=collate_func, shuffle=False
    )

    model_kwargs = {**config['model_params']}
    if model_type == 'duration_regressor':
        model_kwargs['num_tokens'] = num_tokens
        model_kwargs['max_phones'] = max_phones
    model = MODEL_MAPPING[model_type](**model_kwargs).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    results = collect_regression_predictions(model, loader, device)
    m = results['metrics']
    if output_dir is None:
        output_dir = Path('exp') / exp_name / 'eval'
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    df = pd.DataFrame({
        'identifier': results['identifiers'],
        'ground_truth': results['targets'],
        'prediction': results['preds'],
    })
    csv_path = output_dir / f'predictions_{split}_{ts}.csv'
    summary_path = output_dir / f'summary_{split}_{ts}.json'
    df.to_csv(csv_path, index=False)

    summary = {
        'exp_name': exp_name,
        'split': split,
        'checkpoint': str(checkpoint_path),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'num_samples': m['num_samples'],
        'rmse': m['rmse'],
        'mae': m['mae'],
        'pearson_r': m['pearson_r'],
        'spearman_r': m['spearman_r'],
        'target_mean': m['target_mean'],
        'target_std': m['target_std'],
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Test inference complete: {len(df)} samples")
    print(f'  predictions -> {csv_path}')
    print(f'  summary     -> {summary_path}')
    print(f"  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
          f"Pearson r={m['pearson_r']:.4f}  Spearman r={m['spearman_r']:.4f}")
    return summary
