import torch
import torch.optim as optim
import json
import argparse
import numpy as np
import pandas as pd
import lmdb
from datetime import datetime
import wandb
from src import models
from src.task_spec import get_task_spec
from src.train_utils import (
    setup_experiment_dir, setup_logger, plot_training_curves, 
    save_checkpoint,
    plot_attention_weights, plot_attention_summary,
    plot_regression_curves
)
from src.dataloaders import DatasetLMDB, DatasetRhythmFeatures, collate_fn, processor_ssl
from src.evaluation import collect_regression_predictions
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime('%Y%m%d_%H%M')

MODEL_MAPPING = {
    "cnn": models.CNN_MLP,
    "ssl": models.WAV_LM,
    "rhythm_regressor": models.RhythmRegressor,
    "duration_regressor": models.DurationRegressor,
    "rhythm_feature_mlp": models.RhythmFeatureMLP,
}

COLLATE_FUNC_MAPPING = {
    "ssl": processor_ssl,
    "pad": collate_fn
}


def build_epoch_metrics(train_loss, train_metrics, val_metrics, lr):
    metrics = {
        'train_loss': train_loss,
        'val_loss': val_metrics['loss'],
        'lr': lr,
    }
    for key, value in train_metrics.items():
        metrics[f"train_{key}"] = value
    for key, value in val_metrics.items():
        if key != 'loss':
            metrics[f"val_{key}"] = value
    return metrics


def to_wandb_metrics(metrics):
    wandb_metrics = {}
    for key, value in metrics.items():
        if key == 'lr':
            wandb_metrics['lr'] = value
        elif key.startswith('train_'):
            wandb_metrics[f"train/{key[6:]}"] = value
        elif key.startswith('val_'):
            wandb_metrics[f"val/{key[4:]}"] = value
    return wandb_metrics

def train(config, trial=None, save_ckpt=False):
    # Setup experiment directories
    exp_name = config.get('exp_name')
    dirs = setup_experiment_dir(exp_name, timestamp)
    exp_dir = dirs['exp_dir']
    logs_dir = dirs['logs_dir']
    plots_dir = dirs['plots_dir']
    checkpoints_dir = dirs['checkpoints_dir']
    att_plots_dir = dirs['att_plots_dir']
    dev_results_dir = dirs['dev_results_dir']
    
    # Setup logger (console only; metrics go to wandb)
    logger = setup_logger(logs_dir, exp_name, log_to_file=False)

    # Setup wandb
    wandb_config = config.get('wandb', {})
    wandb_enabled = wandb_config.get('enabled', True)
    if wandb_enabled:
        wandb.init(
            project=wandb_config.get('project', exp_name),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('run_name', f"{exp_name}_{timestamp}"),
            dir=str(exp_dir),
            config=config,
            tags=wandb_config.get('tags'),
            mode=wandb_config.get('mode', 'offline'),
        )
    
    # Determine task type
    task = config['model_params'].get('task', 'classification')
    task_spec = get_task_spec(task)
    is_regression = task_spec.is_regression
    
    # Get target normalization stats for regression (needed before loading datasets)
    target_mean = config['model_params'].get('target_mean')
    target_std = config['model_params'].get('target_std')
    
    logger.info("="*70)
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Task type: {task}")
    logger.info("="*70)
    
    # Log configuration
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Load VC features for duration_regressor
    vc_features = None
    num_tokens = None
    max_phones = None
    if config.get('model_type') == 'duration_regressor':
        vc_path = config['dataset_params'].get('vc_features_path', 'data/speechocean/vc_features.json')
        with open(vc_path, 'r') as f:
            vc_data = json.load(f)
        vc_features = vc_data['samples']
        num_tokens = len(vc_data['vocab'])
        max_v = max(max(len(p) for p in s.get('v_phones', [])) for s in vc_features.values())
        max_c = max(max(len(p) for p in s.get('c_phones', [])) for s in vc_features.values())
        max_phones = max(max_v, max_c)
        logger.info(f"Loaded VC features: {len(vc_features)} samples, {num_tokens} tokens, "
                    f"max_phones V={max_v} C={max_c}")

    # Load datasets
    logger.info("\n" + "-"*70)
    logger.info("Loading datasets...")
    
    # Pass normalization stats to dataset for regression tasks
    dataset_params = config['dataset_params'].copy()
    dataset_params.pop('vc_features_path', None)
    if is_regression and target_mean is not None and target_std is not None:
        dataset_params['target_mean'] = target_mean
        dataset_params['target_std'] = target_std
        logger.info(f"Target normalization: mean={target_mean}, std={target_std}")
    
    is_rhythm_feature_mlp = config.get('model_type') == 'rhythm_feature_mlp'

    lmdb_env = None
    if is_rhythm_feature_mlp:
        train_dataset = DatasetRhythmFeatures(**dataset_params, split='Train')
    else:
        lmdb_env = lmdb.open(dataset_params['lmdb_path'], readonly=True, lock=False, readahead=False, meminit=False)
        train_dataset = DatasetLMDB(**dataset_params, split='Train', vc_features=vc_features, env=lmdb_env)

    utterance_embed_dim = config['model_params'].get('utterance_embed_dim', 0)
    train_utterance_str2int = None
    train_utterance_int2str = None
    if utterance_embed_dim > 0:
        train_utterance_str2int = train_dataset.utterance_str2int
        train_utterance_int2str = train_dataset.utterance_int2str
        logger.info(f"Using utterance ID embeddings: dim={utterance_embed_dim}, "
                    f"num_utterances={len(train_utterance_str2int)}")

    if is_rhythm_feature_mlp:
        dev_dataset = DatasetRhythmFeatures(**dataset_params, split='Development',
                                            external_utterance_str2int=train_utterance_str2int,
                                            external_utterance_int2str=train_utterance_int2str,
                                            external_feature_mean=train_dataset.feature_mean,
                                            external_feature_std=train_dataset.feature_std)
        test_dataset = DatasetRhythmFeatures(**dataset_params, split='Test',
                                             external_utterance_str2int=train_utterance_str2int,
                                             external_utterance_int2str=train_utterance_int2str,
                                             external_feature_mean=train_dataset.feature_mean,
                                             external_feature_std=train_dataset.feature_std)
    else:
        dev_dataset = DatasetLMDB(**dataset_params, split='Development',
                                  external_utterance_str2int=train_utterance_str2int,
                                  external_utterance_int2str=train_utterance_int2str,
                                  vc_features=vc_features, env=lmdb_env)
        test_dataset = DatasetLMDB(**dataset_params, split='Test',
                                   external_utterance_str2int=train_utterance_str2int,
                                   external_utterance_int2str=train_utterance_int2str,
                                   vc_features=vc_features, env=lmdb_env)

    # Compute class/label distribution on train
    from collections import Counter
    train_label_counts = Counter(train_dataset.labels.values())
    dev_label_counts = Counter(dev_dataset.labels.values())
    test_label_counts = Counter(test_dataset.labels.values())
    
    if is_regression:
        # For regression, compute label statistics
        train_labels = list(train_dataset.labels.values())
        logger.info(f"Train target stats: mean={np.mean(train_labels):.2f}, std={np.std(train_labels):.2f}, "
                   f"min={np.min(train_labels):.2f}, max={np.max(train_labels):.2f}")
    else:
        logger.info(f"Train class distribution: {dict(train_label_counts)}")

    collateFunc = COLLATE_FUNC_MAPPING.get(config['collate_fn'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              collate_fn=collateFunc, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'], 
                            collate_fn=collateFunc, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            collate_fn=collateFunc, shuffle=False)
    
    num_classes = len(train_dataset.labels_str2int)
    logger.info(f"Sucessfully loaded dataset\nNumber of classes/unique values: {num_classes}\nLabel mapping: {train_dataset.labels_str2int}")
    logger.info(f"Number of datapoints: {len(train_dataset)} (Train), {len(dev_dataset)} (Dev), {len(test_dataset)} (Test)")
    
    # Initialize model
    logger.info("\n" + "-"*70)
    logger.info("Initializing model...")
    model_kwargs = {**config['model_params']}
    if num_tokens is not None:
        model_kwargs['num_tokens'] = num_tokens
    if max_phones is not None:
        model_kwargs['max_phones'] = max_phones
    if utterance_embed_dim > 0:
        model_kwargs['num_utterances'] = len(train_utterance_str2int)
    model = MODEL_MAPPING.get(config['model_type'])(
        **model_kwargs,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {config['model_type']} | Total parameters: {total_params:,} | "
                f"Trainable parameters: {trainable_params:,} | Device: {device}")
    
    # Loss function based on task
    criterion = task_spec.criterion
    logger.info(f"Using {task_spec.loss_name}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training_params']['learning_rate'],
        weight_decay=config['training_params']['weight_decay']
    )
    
    # Training tracking
    train_losses = []
    val_losses = []
    
    if is_regression:
        train_rmses = []
        train_maes = []
        train_pearson_rs = []
        val_rmses = []
        val_maes = []
        val_pearson_rs = []
    else:
        train_accs = []
        train_f1s = []
        val_accs = []
        val_f1s = []
    
    best_val_loss = float('inf')
    best_val_metric = -float('inf') if task_spec.higher_is_better else float('inf')
    best_model_path = None
    epochs_without_improvement = 0
    
    plot_every = config['training_params'].get('plot_every', 5)
    save_every = config['training_params'].get('save_every', 10)
    use_early_stopping = config['training_params'].get('use_early_stopping', True)
    early_stop_patience = config['training_params'].get('early_stop_patience', 15)
    
    # Check if model uses attention (self-attention or attention pooling)
    use_attention = config['model_params'].get('use_attention', False) or \
                    config['model_params'].get('use_attention_pooling', False) or \
                    config['model_params'].get('lstm_out_mode') == "all"
    
    logger.info("\n" + "="*70 + "Starting training" + "\n" + "="*70)
    
    # Training loop
    for epoch in range(config['training_params']['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        logger.info(f"Epoch [{epoch+1}/{config['training_params']['num_epochs']}] - Training")
        
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            labels = task_spec.prepare_labels(batch)

            optimizer.zero_grad()
            
            outputs = model(batch)
            loss = criterion(outputs, labels)

            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item()
            
            train_preds.extend(task_spec.prediction_from_outputs(outputs))
            train_labels.extend(labels.detach().cpu().numpy())
        
        # Training statistics
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        train_metrics = task_spec.compute_train_metrics(
            train_preds, train_labels, target_mean, target_std
        )

        if is_regression:
            train_rmses.append(train_metrics['rmse'])
            train_maes.append(train_metrics['mae'])
            train_pearson_rs.append(train_metrics['pearson_r'])
        else:
            train_accs.append(train_metrics['accuracy'])
            train_f1s.append(train_metrics['f1'])
        
        # Validation phase
        logger.info(f"Epoch [{epoch+1}/{config['training_params']['num_epochs']}] - Validation")
        
        val_metrics = task_spec.eval_metrics(
            model, dev_loader, criterion, device, target_mean, target_std
        )
        val_loss = val_metrics['loss']
        val_losses.append(val_loss)
        if is_regression:
            val_rmses.append(val_metrics['rmse'])
            val_maes.append(val_metrics['mae'])
            val_pearson_rs.append(val_metrics['pearson_r'])
        else:
            val_accs.append(val_metrics['accuracy'])
            val_f1s.append(val_metrics['f1'])
        
        # Log epoch summary
        lr_value = config['training_params']['learning_rate']
        epoch_metrics = build_epoch_metrics(avg_train_loss, train_metrics, val_metrics, lr_value)

        logger.info("-"*70)
        logger.info(f"Epoch [{epoch+1}/{config['training_params']['num_epochs']}] Summary:")
        for line in task_spec.summary_lines(epoch_metrics):
            logger.info(line)
        
        # Log metrics to wandb
        if wandb_enabled:
            wandb.log(to_wandb_metrics(epoch_metrics), step=epoch + 1)

        # Check if best model
        current_val_metric = epoch_metrics[task_spec.best_metric_key]

        # Report to Optuna and prune underperforming trials early
        if trial is not None:
            value = float(current_val_metric)
            if np.isfinite(value):
                trial.report(value, step=epoch)
                if trial.should_prune():
                    logger.info(f"Trial pruned at epoch {epoch+1}")
                    if lmdb_env is not None:
                        lmdb_env.close()
                    import optuna
                    raise optuna.exceptions.TrialPruned()

        if task_spec.higher_is_better:
            is_improved = current_val_metric > best_val_metric
        else:
            is_improved = current_val_metric < best_val_metric

        if is_improved:
            best_val_metric = current_val_metric
            epochs_without_improvement = 0
            logger.info(f"  New best validation {task_spec.best_metric_label}: {best_val_metric:.4f}")
            train_metric_key, val_metric_key = task_spec.checkpoint_metric_keys
            save_checkpoint(
                model, optimizer, epoch+1, avg_train_loss, epoch_metrics[train_metric_key],
                val_loss, epoch_metrics[val_metric_key], checkpoints_dir, logger, is_best=True,
                utterance_str2int=train_utterance_str2int
            )
            best_model_path = checkpoints_dir / f'best_model_epoch{epoch+1}.pth'
        else:
            epochs_without_improvement += 1
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"  New best validation loss: {best_val_loss:.4f}")
        
        logger.info("-"*70 + "\n")
        
        # Early stopping check
        if use_early_stopping and epochs_without_improvement >= early_stop_patience:
            logger.info(f"Early stopping triggered after {early_stop_patience} epochs without improvement")
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
            logger.info(f"Best validation {task_spec.best_metric_label}: {best_val_metric:.4f}")
            break
        
        # Plot training curves periodically
        if (epoch + 1) % plot_every == 0:
            if is_regression:
                plot_file = plot_regression_curves(
                    train_losses, train_rmses, train_pearson_rs,
                    val_losses, val_rmses, val_pearson_rs,
                    plots_dir, timestamp
                )
            else:
                plot_file = plot_training_curves(
                    train_losses, train_accs, train_f1s, 
                    val_losses, val_accs, val_f1s,
                    plots_dir, timestamp
                )
            logger.info(f"Training curves saved to {plot_file}\n")
            
            # Plot attention weights if model uses attention (only for classification)
            if use_attention and not is_regression:
                model.eval()
                with torch.no_grad():
                    # Get a batch from validation set for attention visualization
                    val_batch = next(iter(dev_loader))
                    val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                for k, v in val_batch.items()}
                    val_labels = val_batch['label']
                    
                    outputs, attn_weights = model(val_batch, return_attention=True)
                    _, val_preds = torch.max(outputs, 1)
                    
                    # Plot individual attention maps
                    att_plot_file = plot_attention_weights(
                        attn_weights, val_labels, val_preds, 
                        att_plots_dir, epoch+1, num_samples=4, timestamp=timestamp
                    )
                    logger.info(f"Attention weights saved to {att_plot_file}")
                    
                    # Plot attention summary by class
                    att_summary_file = plot_attention_summary(
                        attn_weights, val_labels,
                        att_plots_dir, epoch+1, timestamp=timestamp
                    )
                    logger.info(f"Attention summary saved to {att_summary_file}\n")
                model.train()
        
        # Save periodic checkpoint
        if save_ckpt and (epoch + 1) % save_every == 0:
            train_metric_key, val_metric_key = task_spec.checkpoint_metric_keys
            save_checkpoint(
                model, optimizer, epoch+1, avg_train_loss, epoch_metrics[train_metric_key],
                val_loss, epoch_metrics[val_metric_key], checkpoints_dir, logger, is_best=False,
                utterance_str2int=train_utterance_str2int
            )
    
    # Final plot
    if is_regression:
        plot_file = plot_regression_curves(
            train_losses, train_rmses, train_pearson_rs,
            val_losses, val_rmses, val_pearson_rs,
            plots_dir, timestamp
        )
    else:
        plot_file = plot_training_curves(
            train_losses, train_accs, train_f1s,
            val_losses, val_accs, val_f1s,
            plots_dir, timestamp
        )
    logger.info(f"\nFinal training curves saved to {plot_file}")
    
    # Save final model
    train_metric_key, val_metric_key = task_spec.checkpoint_metric_keys
    save_checkpoint(
        model, optimizer, len(train_losses),
        epoch_metrics['train_loss'], epoch_metrics[train_metric_key],
        epoch_metrics['val_loss'], epoch_metrics[val_metric_key],
        checkpoints_dir, logger, is_best=False,
        utterance_str2int=train_utterance_str2int
    )
    
    # Final evaluation on dev set
    logger.info("\n" + "="*70)
    logger.info("Evaluating on dev set...")

    checkpoint_epoch = None
    if best_model_path is not None and best_model_path.exists():
        logger.info(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        checkpoint_epoch = checkpoint.get('epoch')
        logger.info(f"Best model at epoch {checkpoint_epoch} loaded")
    else:
        logger.info("No best model checkpoint found; evaluating with current model")

    if is_regression:
        results = collect_regression_predictions(model, dev_loader, device)
        all_targets = results['targets']
        all_preds = results['preds']
        m = results['metrics']

        dev_metrics = {
            'loss': float('nan'),
            'rmse': m['rmse'],
            'mae': m['mae'],
            'pearson_r': m['pearson_r'],
            'spearman_r': m['spearman_r'],
        }

        eval_dir = exp_dir / 'eval'
        eval_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            'identifier': results['identifiers'],
            'ground_truth': all_targets,
            'prediction': all_preds,
        })
        csv_path = eval_dir / f'predictions_Dev_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Predictions saved to {csv_path}")

        summary = {
            'exp_name': exp_name,
            'split': 'Dev',
            'num_samples': m['num_samples'],
            'rmse': m['rmse'],
            'mae': m['mae'],
            'pearson_r': m['pearson_r'],
            'spearman_r': m['spearman_r'],
            'checkpoint_epoch': checkpoint_epoch if checkpoint_epoch is not None else len(train_losses),
            'target_mean': m['target_mean'],
            'target_std': m['target_std'],
        }
        summary_path = eval_dir / f'summary_Dev_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")

    else:
        dev_metrics = task_spec.eval_metrics(
            model, dev_loader, criterion, device, target_mean, target_std
        )

    for line in task_spec.dev_lines(dev_metrics):
        logger.info(line)

    if is_regression:
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
            'config': config
        }
    else:
        dev_results = {
            'exp_name': exp_name,
            'timestamp': timestamp,
            'task': 'classification',
            'dev_loss': dev_metrics['loss'],
            'dev_accuracy': dev_metrics['accuracy'],
            'dev_f1': dev_metrics['f1'],
            'best_val_loss': best_val_loss,
            'best_val_f1': best_val_metric,
            'total_epochs': len(train_losses),
            'config': config
        }

    dev_results_file = dev_results_dir / f'dev_results_{timestamp}.json'
    with open(dev_results_file, 'w') as f:
        json.dump(dev_results, f, indent=2)
    logger.info(f"Dev results saved to {dev_results_file}")
    
    logger.info("\n" + "="*70)
    logger.info("Training completed successfully!")
    logger.info(f"  Best validation loss: {best_val_loss:.4f}")
    if is_regression:
        logger.info(f"  Best validation {task_spec.best_metric_label}: {best_val_metric:.4f}")
        logger.info(f"  Final dev RMSE: {dev_metrics['rmse']:.4f}")
        logger.info(f"  Final dev Pearson r: {dev_metrics['pearson_r']:.4f}")
    else:
        logger.info(f"  Best validation {task_spec.best_metric_label}: {best_val_metric:.4f}")
        logger.info(f"  Final dev loss: {dev_metrics['loss']:.4f}")
        logger.info(f"  Final dev accuracy: {dev_metrics['accuracy']:.2f}%")
        logger.info(f"  Final dev F1: {dev_metrics['f1']:.4f}")
    logger.info(f"  Experiment directory: {exp_dir}")
    logger.info("="*70)

    if wandb_enabled:
        if is_regression:
            wandb.summary['best_val_loss'] = best_val_loss
            wandb.summary['best_val_spearman_r'] = best_val_metric
            wandb.summary['dev_loss'] = dev_metrics['loss']
            wandb.summary['dev_rmse'] = dev_metrics['rmse']
            wandb.summary['dev_mae'] = dev_metrics['mae']
            wandb.summary['dev_pearson_r'] = dev_metrics['pearson_r']
            wandb.summary['dev_spearman_r'] = dev_metrics['spearman_r']
        else:
            wandb.summary['best_val_loss'] = best_val_loss
            wandb.summary['best_val_f1'] = best_val_metric
            wandb.summary['dev_loss'] = dev_metrics['loss']
            wandb.summary['dev_accuracy'] = dev_metrics['accuracy']
            wandb.summary['dev_f1'] = dev_metrics['f1']
        wandb.finish()

    if lmdb_env is not None:
        lmdb_env.close()

    return best_val_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence classification model.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the JSON configuration file.')
    parser.add_argument('--save_ckpt', action='store_true',
                        help='Save checkpoint after the specified number of epochs. If not used, will only save the best model.')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    train(config, save_ckpt=args.save_ckpt)