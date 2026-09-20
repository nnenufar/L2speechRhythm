import argparse
import json
from datetime import datetime

import lmdb
import numpy as np
import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader

from src import models
from src.dataloaders import DatasetLMDB, collate_fn
from src.evaluation import collect_regression_predictions
from src.task_spec import get_task_spec
from src.train_utils import (
    plot_regression_curves,
    save_checkpoint,
    setup_experiment_dir,
    setup_logger,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = datetime.now().strftime('%Y%m%d_%H%M')

MODEL_MAPPING = {
    'rhythm_regressor': models.RhythmRegressor,
    'duration_regressor': models.DurationRegressor,
}

COLLATE_FUNC_MAPPING = {
    'pad': collate_fn,
}


def _build_epoch_metrics(train_loss, train_metrics, val_metrics, lr):
    metrics = {'train_loss': train_loss, 'val_loss': val_metrics['loss'], 'lr': lr}
    for key, value in train_metrics.items():
        metrics[f'train_{key}'] = value
    for key, value in val_metrics.items():
        if key != 'loss':
            metrics[f'val_{key}'] = value
    return metrics


def _to_wandb_metrics(metrics):
    out = {}
    for key, value in metrics.items():
        if key == 'lr':
            out['lr'] = value
        elif key.startswith('train_'):
            out[f"train/{key[6:]}"] = value
        elif key.startswith('val_'):
            out[f"val/{key[4:]}"] = value
    return out


def train(config, trial=None, save_ckpt=False):
    exp_name = config.get('exp_name')
    dirs = setup_experiment_dir(exp_name, timestamp)
    exp_dir = dirs['exp_dir']
    plots_dir = dirs['plots_dir']
    checkpoints_dir = dirs['checkpoints_dir']
    dev_results_dir = dirs['dev_results_dir']

    logger = setup_logger(dirs['logs_dir'], exp_name, log_to_file=False)

    wandb_config = config.get('wandb', {})
    wandb_enabled = wandb_config.get('enabled', True)
    if wandb_enabled:
        wandb.init(
            project=wandb_config.get('project', exp_name),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('run_name', f'{exp_name}_{timestamp}'),
            dir=str(exp_dir),
            config=config,
            tags=wandb_config.get('tags'),
            mode=wandb_config.get('mode', 'offline'),
        )

    task = config['model_params'].get('task', 'regression')
    if task != 'regression':
        raise ValueError('Only regression tasks are supported.')
    task_spec = get_task_spec(task)

    target_mean = config['model_params'].get('target_mean')
    target_std = config['model_params'].get('target_std')

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
    dev_dataset = DatasetLMDB(
        lmdb_path, **dataset_params, split='Development', vc_features=vc_features, env=lmdb_env
    )
    test_dataset = DatasetLMDB(
        lmdb_path, **dataset_params, split='Test', vc_features=vc_features, env=lmdb_env
    )

    collate_func = COLLATE_FUNC_MAPPING.get(config.get('collate_fn', 'pad'))
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], collate_fn=collate_func, shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config['batch_size'], collate_fn=collate_func, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], collate_fn=collate_func, shuffle=False
    )

    model_kwargs = {**config['model_params']}
    if model_type == 'duration_regressor':
        model_kwargs['num_tokens'] = num_tokens
        model_kwargs['max_phones'] = max_phones
    model = MODEL_MAPPING[model_type](**model_kwargs).to(device)

    criterion = task_spec.criterion
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training_params']['learning_rate'],
        weight_decay=config['training_params']['weight_decay'],
    )

    train_losses = []
    val_losses = []
    train_rmses, train_maes, train_pearson_rs = [], [], []
    val_rmses, val_maes, val_pearson_rs = [], [], []

    best_val_loss = float('inf')
    best_val_metric = -float('inf')
    best_model_path = None
    epochs_without_improvement = 0

    plot_every = config['training_params'].get('plot_every', 5)
    save_every = config['training_params'].get('save_every', 10)
    use_early_stopping = config['training_params'].get('use_early_stopping', True)
    early_stop_patience = config['training_params'].get('early_stop_patience', 15)

    for epoch in range(config['training_params']['num_epochs']):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = task_spec.prepare_labels(batch)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(task_spec.prediction_from_outputs(outputs))
            train_labels.extend(labels.detach().cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        train_metrics = task_spec.compute_train_metrics(
            train_preds, train_labels, target_mean, target_std
        )
        train_rmses.append(train_metrics['rmse'])
        train_maes.append(train_metrics['mae'])
        train_pearson_rs.append(train_metrics['pearson_r'])

        val_metrics = task_spec.eval_metrics(model, dev_loader, criterion, device, target_mean, target_std)
        val_losses.append(val_metrics['loss'])
        val_rmses.append(val_metrics['rmse'])
        val_maes.append(val_metrics['mae'])
        val_pearson_rs.append(val_metrics['pearson_r'])

        lr_value = config['training_params']['learning_rate']
        epoch_metrics = _build_epoch_metrics(avg_train_loss, train_metrics, val_metrics, lr_value)
        for line in task_spec.summary_lines(epoch_metrics):
            logger.info(line)
        if wandb_enabled:
            wandb.log(_to_wandb_metrics(epoch_metrics), step=epoch + 1)

        current_val_metric = epoch_metrics[task_spec.best_metric_key]
        if trial is not None:
            value = float(current_val_metric)
            if np.isfinite(value):
                trial.report(value, step=epoch)
                if trial.should_prune():
                    if lmdb_env is not None:
                        lmdb_env.close()
                    import optuna
                    raise optuna.exceptions.TrialPruned()

        is_improved = current_val_metric > best_val_metric
        if is_improved:
            best_val_metric = current_val_metric
            epochs_without_improvement = 0
            train_key, val_key = task_spec.checkpoint_metric_keys
            save_checkpoint(
                model, optimizer, epoch + 1, avg_train_loss,
                epoch_metrics[train_key], val_losses[-1], epoch_metrics[val_key],
                checkpoints_dir, logger, is_best=True,
            )
            best_model_path = checkpoints_dir / f'best_model_epoch{epoch + 1}.pth'
        else:
            epochs_without_improvement += 1

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]

        if use_early_stopping and epochs_without_improvement >= early_stop_patience:
            logger.info(f'Early stopping triggered after {early_stop_patience} epochs.')
            break

        if (epoch + 1) % plot_every == 0:
            plot_file = plot_regression_curves(
                train_losses, train_rmses, train_pearson_rs,
                val_losses, val_rmses, val_pearson_rs,
                plots_dir, timestamp,
            )
            logger.info(f'Training curves saved to {plot_file}')

        if save_ckpt and (epoch + 1) % save_every == 0:
            train_key, val_key = task_spec.checkpoint_metric_keys
            save_checkpoint(
                model, optimizer, epoch + 1, avg_train_loss,
                epoch_metrics[train_key], val_losses[-1], epoch_metrics[val_key],
                checkpoints_dir, logger, is_best=False,
            )

    if best_model_path is not None and best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        checkpoint_epoch = checkpoint.get('epoch')
    else:
        checkpoint_epoch = len(train_losses)

    results = collect_regression_predictions(model, dev_loader, device)
    m = results['metrics']
    dev_metrics = {
        'loss': float('nan'),
        'rmse': m['rmse'],
        'mae': m['mae'],
        'pearson_r': m['pearson_r'],
        'spearman_r': m['spearman_r'],
    }
    for line in task_spec.dev_lines(dev_metrics):
        logger.info(line)

    eval_dir = exp_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    dev_results = {
        'exp_name': exp_name,
        'timestamp': timestamp,
        'task': 'regression',
        'dev_rmse': dev_metrics['rmse'],
        'dev_mae': dev_metrics['mae'],
        'dev_pearson_r': dev_metrics['pearson_r'],
        'dev_spearman_r': dev_metrics['spearman_r'],
        'best_val_loss': best_val_loss,
        'best_val_spearman_r': best_val_metric,
        'total_epochs': len(train_losses),
        'config': config,
    }
    out_path = dev_results_dir / f'dev_results_{timestamp}.json'
    with open(out_path, 'w') as f:
        json.dump(dev_results, f, indent=2)

    if wandb_enabled:
        wandb.summary['best_val_loss'] = best_val_loss
        wandb.summary['best_val_spearman_r'] = best_val_metric
        wandb.summary['dev_rmse'] = dev_metrics['rmse']
        wandb.finish()

    if lmdb_env is not None:
        lmdb_env.close()
    return best_val_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--save_ckpt', action='store_true')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    train(config, save_ckpt=args.save_ckpt)
