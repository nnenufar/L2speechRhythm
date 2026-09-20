from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.evaluation import evaluate_regression


@dataclass(frozen=True)
class TaskSpec:
    name: str
    is_regression: bool
    loss_name: str
    criterion: nn.Module
    prepare_labels: Callable
    prediction_from_outputs: Callable
    compute_train_metrics: Callable
    eval_metrics: Callable
    summary_lines: Callable
    best_metric_key: str
    best_metric_label: str
    higher_is_better: bool
    checkpoint_metric_keys: Tuple[str, str]
    dev_lines: Callable


def _prepare_labels(batch):
    return batch['label'].float()


def _prediction_from_outputs(outputs):
    return outputs.detach().cpu().numpy()


def _compute_metrics(preds, labels, target_mean, target_std):
    if target_mean is not None and target_std is not None:
        preds = preds * target_std + target_mean
        labels = labels * target_std + target_mean

    rmse = float(np.sqrt(mean_squared_error(labels, preds)))
    mae = float(mean_absolute_error(labels, preds))
    if np.std(preds) > 1e-6:
        pearson_r = float(pearsonr(labels, preds)[0])
    else:
        pearson_r = 0.0
    return {'rmse': rmse, 'mae': mae, 'pearson_r': pearson_r}


def _eval_metrics(model, dataloader, criterion, device, target_mean, target_std):
    loss, rmse, mae, pearson_r, spearman_r = evaluate_regression(
        model, dataloader, criterion, device, target_mean, target_std
    )
    return {
        'loss': float(loss),
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
    }


def _summary_lines(metrics):
    return (
        f"  Train Loss: {metrics['train_loss']:.4f}",
        f"  Train RMSE: {metrics['train_rmse']:.4f} | Train MAE: {metrics['train_mae']:.4f} | Train Pearson r: {metrics['train_pearson_r']:.4f}",
        f"  Val Loss: {metrics['val_loss']:.4f} | Val RMSE: {metrics['val_rmse']:.4f} | Val MAE: {metrics['val_mae']:.4f}",
        f"  Val Pearson r: {metrics['val_pearson_r']:.4f} | Val Spearman r: {metrics['val_spearman_r']:.4f}",
    )


def _dev_lines(metrics):
    return (
        f"Dev Loss: {metrics['loss']:.4f}",
        f"Dev RMSE: {metrics['rmse']:.4f}",
        f"Dev MAE: {metrics['mae']:.4f}",
        f"Dev Pearson r: {metrics['pearson_r']:.4f}",
        f"Dev Spearman r: {metrics['spearman_r']:.4f}",
    )


def get_task_spec(task='regression'):
    if task != 'regression':
        raise ValueError('Only regression tasks are supported.')
    return TaskSpec(
        name='regression',
        is_regression=True,
        loss_name='MSELoss for regression',
        criterion=nn.MSELoss(),
        prepare_labels=_prepare_labels,
        prediction_from_outputs=_prediction_from_outputs,
        compute_train_metrics=_compute_metrics,
        eval_metrics=_eval_metrics,
        summary_lines=_summary_lines,
        best_metric_key='val_spearman_r',
        best_metric_label='Spearman r',
        higher_is_better=True,
        checkpoint_metric_keys=('train_rmse', 'val_rmse'),
        dev_lines=_dev_lines,
    )
