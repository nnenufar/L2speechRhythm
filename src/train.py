import torch
import torch.optim as optim
import json
import argparse
import numpy as np
from datetime import datetime
import wandb
from src import models
from src.task_spec import get_task_spec
from src.train_utils import (
    setup_experiment_dir, setup_logger, plot_training_curves, 
    save_checkpoint, create_weighted_sampler,
    plot_attention_weights, plot_attention_summary,
    plot_regression_curves
)
from src.dataloaders import DatasetLMDB, collate_fn, processor_ssl
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime('%Y%m%d_%H%M')

MODEL_MAPPING = {
    "cnn": models.CNN_MLP,
    "ssl": models.WAV_LM,
    "rhythm_regressor": models.RhythmRegressor,
    "duration_regressor": models.DurationRegressor,
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

def train(config):
    # Setup experiment directories
    exp_name = config.get('exp_name')
    dirs = setup_experiment_dir(exp_name, timestamp)
    exp_dir = dirs['exp_dir']
    logs_dir = dirs['logs_dir']
    plots_dir = dirs['plots_dir']
    checkpoints_dir = dirs['checkpoints_dir']
    att_plots_dir = dirs['att_plots_dir']
    test_results_dir = dirs['test_results_dir']
    
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
    
    # Get target normalization stats for regression
    # If not provided, compute from training data
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
    if config.get('model_type') == 'duration_regressor':
        vc_path = config['dataset_params'].get('vc_features_path', 'data/speechocean/vc_features.json')
        with open(vc_path, 'r') as f:
            vc_data = json.load(f)
        vc_features = vc_data['samples']
        num_tokens = len(vc_data['vocab'])
        logger.info(f"Loaded VC features: {len(vc_features)} samples, {num_tokens} tokens")

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
    
    train_dataset = DatasetLMDB(**dataset_params, split='Train', vc_features=vc_features)

    if is_regression and target_mean is None:
        raw_labels = np.array([float(v) for v in train_dataset.labels.values()])
        target_mean = float(np.mean(raw_labels))
        target_std = float(np.std(raw_labels))
        train_dataset.target_mean = target_mean
        train_dataset.target_std = target_std
        logger.info(f"Auto-computed target normalization: mean={target_mean:.4f}, std={target_std:.4f}")

    utterance_embed_dim = config['model_params'].get('utterance_embed_dim', 0)
    train_utterance_str2int = None
    train_utterance_int2str = None
    if utterance_embed_dim > 0:
        train_utterance_str2int = train_dataset.utterance_str2int
        train_utterance_int2str = train_dataset.utterance_int2str
        logger.info(f"Using utterance ID embeddings: dim={utterance_embed_dim}, "
                    f"num_utterances={len(train_utterance_str2int)}")

    dev_dataset = DatasetLMDB(**dataset_params, split='Development',
                              external_utterance_str2int=train_utterance_str2int,
                              external_utterance_int2str=train_utterance_int2str,
                              vc_features=vc_features,
                              target_mean=target_mean, target_std=target_std)
    test_dataset = DatasetLMDB(**dataset_params, split='Test',
                               external_utterance_str2int=train_utterance_str2int,
                               external_utterance_int2str=train_utterance_int2str,
                               vc_features=vc_features,
                               target_mean=target_mean, target_std=target_std)

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

    # Optional WeightedRandomSampler (only for classification)
    use_weighted_sampler = config['training_params'].get('use_weighted_sampler', False) and not is_regression

    collateFunc = COLLATE_FUNC_MAPPING.get(config['collate_fn'])

    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(train_dataset, train_label_counts)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=collateFunc, sampler=train_sampler)
        dev_sampler = create_weighted_sampler(dev_dataset, dev_label_counts)
        dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'], collate_fn=collateFunc, sampler=dev_sampler)
        test_sampler = create_weighted_sampler(test_dataset, test_label_counts)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=collateFunc, sampler=test_sampler)
    else:
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
    use_scheduler = config['training_params'].get('use_scheduler')

    if use_scheduler:
        scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=config['training_params'].get('max_lr', 0.01),
                    epochs=config['training_params']['num_epochs'],
                    steps_per_epoch=len(train_loader),
                    pct_start=config['training_params'].get('warmup_pct', 0.1),
                    anneal_strategy='cos'
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            
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
        lr_value = scheduler.get_last_lr()[0] if use_scheduler else config['training_params']['learning_rate']
        epoch_metrics = build_epoch_metrics(avg_train_loss, train_metrics, val_metrics, lr_value)

        logger.info("-"*70)
        logger.info(f"Epoch [{epoch+1}/{config['training_params']['num_epochs']}] Summary:")
        for line in task_spec.summary_lines(epoch_metrics):
            logger.info(line)
        
        if use_scheduler:
            logger.info(f"  Learning Rate: {lr_value:.6f} (Using Scheduler)")
        else:
            logger.info(f"  Learning Rate: {lr_value:.6f} (Not using Scheduler)")
        
        # Log metrics to wandb
        if wandb_enabled:
            wandb.log(to_wandb_metrics(epoch_metrics), step=epoch + 1)

        # Check if best model
        current_val_metric = epoch_metrics[task_spec.best_metric_key]
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
                utterance_str2int=train_utterance_str2int,
                target_mean=target_mean, target_std=target_std
            )
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
        if args.save_ckpt and (epoch + 1) % save_every == 0:
            train_metric_key, val_metric_key = task_spec.checkpoint_metric_keys
            save_checkpoint(
                model, optimizer, epoch+1, avg_train_loss, epoch_metrics[train_metric_key],
                val_loss, epoch_metrics[val_metric_key], checkpoints_dir, logger, is_best=False,
                utterance_str2int=train_utterance_str2int,
                target_mean=target_mean, target_std=target_std
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
        utterance_str2int=train_utterance_str2int,
        target_mean=target_mean, target_std=target_std
    )
    
    # Final evaluation on test set
    logger.info("\n" + "="*70)
    logger.info("Evaluating on test set...")
    
    test_metrics = task_spec.eval_metrics(
        model, test_loader, criterion, device, target_mean, target_std
    )
    for line in task_spec.test_lines(test_metrics):
        logger.info(line)

    if is_regression:
        test_results = {
            'exp_name': exp_name,
            'timestamp': timestamp,
            'task': 'regression',
            'test_loss': test_metrics['loss'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_pearson_r': test_metrics['pearson_r'],
            'test_spearman_r': test_metrics['spearman_r'],
            'best_val_loss': best_val_loss,
            'best_val_pearson_r': best_val_metric,
            'total_epochs': len(train_losses),
            'config': config
        }
    else:
        test_results = {
            'exp_name': exp_name,
            'timestamp': timestamp,
            'task': 'classification',
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1'],
            'best_val_loss': best_val_loss,
            'best_val_f1': best_val_metric,
            'total_epochs': len(train_losses),
            'config': config
        }
    
    test_results_file = test_results_dir / f'test_results_{timestamp}.json'
    with open(test_results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Test results saved to {test_results_file}")
    
    logger.info("\n" + "="*70)
    logger.info("Training completed successfully!")
    logger.info(f"  Best validation loss: {best_val_loss:.4f}")
    if is_regression:
        logger.info(f"  Best validation {task_spec.best_metric_label}: {best_val_metric:.4f}")
        logger.info(f"  Final test RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"  Final test Pearson r: {test_metrics['pearson_r']:.4f}")
    else:
        logger.info(f"  Best validation {task_spec.best_metric_label}: {best_val_metric:.4f}")
        logger.info(f"  Final test loss: {test_metrics['loss']:.4f}")
        logger.info(f"  Final test accuracy: {test_metrics['accuracy']:.2f}%")
        logger.info(f"  Final test F1: {test_metrics['f1']:.4f}")
    logger.info(f"  Experiment directory: {exp_dir}")
    logger.info("="*70)

    if wandb_enabled:
        if is_regression:
            wandb.summary['best_val_loss'] = best_val_loss
            wandb.summary['best_val_pearson_r'] = best_val_metric
            wandb.summary['test_loss'] = test_metrics['loss']
            wandb.summary['test_rmse'] = test_metrics['rmse']
            wandb.summary['test_mae'] = test_metrics['mae']
            wandb.summary['test_pearson_r'] = test_metrics['pearson_r']
            wandb.summary['test_spearman_r'] = test_metrics['spearman_r']
        else:
            wandb.summary['best_val_loss'] = best_val_loss
            wandb.summary['best_val_f1'] = best_val_metric
            wandb.summary['test_loss'] = test_metrics['loss']
            wandb.summary['test_accuracy'] = test_metrics['accuracy']
            wandb.summary['test_f1'] = test_metrics['f1']
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence classification model.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the JSON configuration file.')
    parser.add_argument('--save_ckpt', action='store_true',
                        help='Save checkpoint after the specified number of epochs. If not used, will only save the best model.')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    train(config)