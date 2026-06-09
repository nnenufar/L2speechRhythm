import numpy as np
import torch
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr


def evaluate_classification(model, dataloader, criterion, device, split_name="Development"):
    """
    Evaluate model on a given dataset (classification).

    Args:
        model: The model to evaluate
        dataloader: DataLoader for the evaluation dataset
        criterion: Loss function
        device: Device to run evaluation on
        split_name: Name of the split (for logging)

    Returns:
        avg_loss: Average loss over the dataset
        accuracy: Accuracy percentage
        f1: F1 score
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            labels = batch['label']

            outputs = model(batch)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, f1


def evaluate_regression(model, dataloader, criterion, device, target_mean=None, target_std=None):
    """
    Evaluate regression model on a given dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for the evaluation dataset
        criterion: Loss function (MSELoss)
        device: Device to run evaluation on
        target_mean: Mean of target values (for denormalization)
        target_std: Std of target values (for denormalization)

    Returns:
        avg_loss: Average MSE loss over the dataset (on normalized scale)
        rmse: Root Mean Squared Error (in original scale)
        mae: Mean Absolute Error (in original scale)
        pearson_r: Pearson correlation coefficient
        spearman_r: Spearman correlation coefficient
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            labels = batch['label'].float()

            outputs = model(batch)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if target_mean is not None and target_std is not None:
        all_preds_denorm = all_preds * target_std + target_mean
        all_labels_denorm = all_labels * target_std + target_mean
    else:
        all_preds_denorm = all_preds
        all_labels_denorm = all_labels

    rmse = np.sqrt(mean_squared_error(all_labels_denorm, all_preds_denorm))
    mae = mean_absolute_error(all_labels_denorm, all_preds_denorm)

    if np.std(all_preds) > 1e-6 and np.std(all_labels) > 1e-6:
        pearson_r, _ = pearsonr(all_labels_denorm, all_preds_denorm)
        spearman_r, _ = spearmanr(all_labels_denorm, all_preds_denorm)
    else:
        pearson_r = 0.0
        spearman_r = 0.0

    return avg_loss, rmse, mae, pearson_r, spearman_r


def collect_regression_predictions(model, dataloader, device):
    """
    Run inference on a regression dataset and collect per-sample predictions.

    Args:
        model: Regression model (outputs a scalar per sample).
        dataloader: DataLoader iterating over samples with 'identifier' and 'label' keys.
        device: torch device.

    Returns:
        results: dict with keys:
            'identifiers' (list[str]), 'targets' (np.ndarray), 'preds' (np.ndarray),
            'metrics' (dict with 'rmse', 'mae', 'pearson_r', 'spearman_r',
                      'target_mean', 'target_std', 'num_samples')
    """
    model.eval()
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
