from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

from src.evaluation import evaluate_classification, evaluate_regression


@dataclass(frozen=True)
class TaskSpec:
    name: str
    is_regression: bool
    loss_name: str
    criterion: nn.Module
    prepare_labels: Callable[[Dict[str, torch.Tensor]], torch.Tensor]
    prediction_from_outputs: Callable[[torch.Tensor], np.ndarray]
    compute_train_metrics: Callable[[np.ndarray, np.ndarray, Optional[float], Optional[float]], Dict[str, float]]
    eval_metrics: Callable[[nn.Module, object, nn.Module, torch.device, Optional[float], Optional[float]], Dict[str, float]]
    summary_lines: Callable[[Dict[str, float]], Tuple[str, ...]]
    best_metric_key: str
    best_metric_label: str
    higher_is_better: bool
    checkpoint_metric_keys: Tuple[str, str]
    test_lines: Callable[[Dict[str, float]], Tuple[str, ...]]


def _prepare_labels_classification(batch):
    return batch['label']


def _prepare_labels_regression(batch):
    return batch['label'].float()


def _prediction_from_logits(outputs):
    _, predicted = torch.max(outputs, 1)
    return predicted.detach().cpu().numpy()


def _prediction_from_regression(outputs):
    return outputs.detach().cpu().numpy()


def _compute_classification_metrics(preds, labels, _target_mean, _target_std):
    accuracy = 100 * (preds == labels).sum() / len(labels)
    f1 = f1_score(labels, preds, average='macro')
    return {
        'accuracy': float(accuracy),
        'f1': float(f1),
    }


def _compute_regression_metrics(preds, labels, target_mean, target_std):
    if target_mean is not None and target_std is not None:
        preds_denorm = preds * target_std + target_mean
        labels_denorm = labels * target_std + target_mean
    else:
        preds_denorm = preds
        labels_denorm = labels

    rmse = np.sqrt(mean_squared_error(labels_denorm, preds_denorm))
    mae = mean_absolute_error(labels_denorm, preds_denorm)

    if np.std(preds) > 1e-6:
        pearson_r, _ = pearsonr(labels_denorm, preds_denorm)
    else:
        pearson_r = 0.0

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'pearson_r': float(pearson_r),
    }


def _eval_classification(model, dataloader, criterion, device, _target_mean, _target_std):
    loss, accuracy, f1 = evaluate_classification(model, dataloader, criterion, device, "Evaluation")
    return {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'f1': float(f1),
    }


def _eval_regression(model, dataloader, criterion, device, target_mean, target_std):
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


def _classification_summary(metrics):
    return (
        f"  Train Loss: {metrics['train_loss']:.4f}",
        f"  Train Acc: {metrics['train_accuracy']:.2f}% | Train F1: {metrics['train_f1']:.4f}",
        f"  Val Loss: {metrics['val_loss']:.4f} | Val Acc: {metrics['val_accuracy']:.2f}% | Val F1: {metrics['val_f1']:.4f}",
    )


def _regression_summary(metrics):
    return (
        f"  Train Loss: {metrics['train_loss']:.4f}",
        f"  Train RMSE: {metrics['train_rmse']:.4f} | Train MAE: {metrics['train_mae']:.4f} | Train Pearson r: {metrics['train_pearson_r']:.4f}",
        f"  Val Loss: {metrics['val_loss']:.4f} | Val RMSE: {metrics['val_rmse']:.4f} | Val MAE: {metrics['val_mae']:.4f}",
        f"  Val Pearson r: {metrics['val_pearson_r']:.4f} | Val Spearman r: {metrics['val_spearman_r']:.4f}",
    )


def _classification_test_lines(metrics):
    return (
        f"Test Loss: {metrics['loss']:.4f}",
        f"Test Accuracy: {metrics['accuracy']:.2f}%",
        f"Test F1: {metrics['f1']:.4f}",
    )


def _regression_test_lines(metrics):
    return (
        f"Test Loss: {metrics['loss']:.4f}",
        f"Test RMSE: {metrics['rmse']:.4f}",
        f"Test MAE: {metrics['mae']:.4f}",
        f"Test Pearson r: {metrics['pearson_r']:.4f}",
        f"Test Spearman r: {metrics['spearman_r']:.4f}",
    )


def get_task_spec(task):
    if task == 'regression':
        return TaskSpec(
            name='regression',
            is_regression=True,
            loss_name='MSELoss for regression',
            criterion=nn.MSELoss(),
            prepare_labels=_prepare_labels_regression,
            prediction_from_outputs=_prediction_from_regression,
            compute_train_metrics=_compute_regression_metrics,
            eval_metrics=_eval_regression,
            summary_lines=_regression_summary,
            best_metric_key='val_pearson_r',
            best_metric_label='Pearson r',
            higher_is_better=True,
            checkpoint_metric_keys=('train_rmse', 'val_rmse'),
            test_lines=_regression_test_lines,
        )

    return TaskSpec(
        name='classification',
        is_regression=False,
        loss_name='CrossEntropyLoss for classification',
        criterion=nn.CrossEntropyLoss(),
        prepare_labels=_prepare_labels_classification,
        prediction_from_outputs=_prediction_from_logits,
        compute_train_metrics=_compute_classification_metrics,
        eval_metrics=_eval_classification,
        summary_lines=_classification_summary,
        best_metric_key='val_f1',
        best_metric_label='F1',
        higher_is_better=True,
        checkpoint_metric_keys=('train_accuracy', 'val_accuracy'),
        test_lines=_classification_test_lines,
    )
