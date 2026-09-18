import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_regression(model, dataloader, criterion, device, target_mean=None, target_std=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch['label'].float()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    if target_mean is not None and target_std is not None:
        all_preds = all_preds * target_std + target_mean
        all_labels = all_labels * target_std + target_mean

    rmse = float(np.sqrt(mean_squared_error(all_labels, all_preds)))
    mae = float(mean_absolute_error(all_labels, all_preds))
    if np.std(all_preds) > 1e-6 and np.std(all_labels) > 1e-6:
        pearson_r = float(pearsonr(all_labels, all_preds)[0])
        spearman_r = float(spearmanr(all_labels, all_preds)[0])
    else:
        pearson_r = 0.0
        spearman_r = 0.0

    return total_loss / len(dataloader), rmse, mae, pearson_r, spearman_r


def collect_regression_predictions(model, dataloader, device):
    model.eval()
    all_identifiers = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            targets = batch_dev['label'].cpu().numpy()
            preds = model(batch_dev).cpu().numpy()
            all_identifiers.extend(batch['identifier'])
            all_targets.extend(targets.tolist())
            all_preds.extend(preds.tolist())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    rmse = float(np.sqrt(mean_squared_error(all_targets, all_preds)))
    mae = float(mean_absolute_error(all_targets, all_preds))
    pearson_r, _ = pearsonr(all_targets, all_preds)
    spearman_r, _ = spearmanr(all_targets, all_preds)

    return {
        'identifiers': all_identifiers,
        'targets': all_targets,
        'preds': all_preds,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'pearson_r': pearson_r,
            'spearman_r': spearman_r,
            'target_mean': float(np.mean(all_targets)),
            'target_std': float(np.std(all_targets)),
            'num_samples': len(all_targets),
        },
    }
