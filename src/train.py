import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import f1_score
from src import models
from src.train_utils import (
    setup_experiment_dir, setup_logger, plot_training_curves, 
    save_checkpoint, create_weighted_sampler,
    plot_attention_weights, plot_attention_summary,
    plot_contrastive_embeddings
)
from src.modules import SupervisedContrastiveLoss
from src.dataloaders import DatasetLMDB, collate_fn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime('%Y%m%d_%H%M')

MODEL_MAPPING = {
    "cnn": models.CNN_MLP,
}

def evaluate(model, dataloader, criterion, device, logger, split_name="Development"):
    """
    Evaluate model on a given dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the evaluation dataset
        criterion: Loss function
        device: Device to run evaluation on
        logger: Logger instance
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
            labels = batch['label']  # Keep as long for CrossEntropyLoss
            
            outputs = model(batch)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get class with highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return avg_loss, accuracy, f1

def train(config):
    # Setup experiment directories
    exp_name = config.get('exp_name')
    exp_dir, logs_dir, plots_dir, checkpoints_dir, att_plots_dir, contrastive_plots_dir = setup_experiment_dir(exp_name, timestamp)
    
    # Setup logger
    logger = setup_logger(logs_dir, exp_name)
    
    logger.info("="*70)
    logger.info(f"Starting experiment: {exp_name}")
    logger.info("="*70)
    
    # Log configuration
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Load datasets
    logger.info("\n" + "-"*70)
    logger.info("Loading datasets...")
    train_dataset = DatasetLMDB(**config['dataset_params'], split='Train')
    dev_dataset = DatasetLMDB(**config['dataset_params'], split='Development')
    test_dataset = DatasetLMDB(**config['dataset_params'], split='Test')

    # Compute class distribution on train
    from collections import Counter
    train_label_counts = Counter(train_dataset.labels.values())
    dev_label_counts = Counter(dev_dataset.labels.values())
    test_label_counts = Counter(test_dataset.labels.values())
    logger.info(f"Train class distribution: {dict(train_label_counts)}")

    # Optional WeightedRandomSampler
    use_weighted_sampler = config['training_params'].get('use_weighted_sampler', False)

    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(train_dataset, train_label_counts)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=collate_fn, sampler=train_sampler)
        dev_sampler = create_weighted_sampler(dev_dataset, dev_label_counts)
        dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'], collate_fn=collate_fn, sampler=dev_sampler)
        test_sampler = create_weighted_sampler(test_dataset, test_label_counts)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=collate_fn, sampler=test_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                  collate_fn=collate_fn, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'], 
                                collate_fn=collate_fn, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                                collate_fn=collate_fn, shuffle=False)
    
    num_classes = len(train_dataset.labels_str2int)
    logger.info(f"Sucessfully loaded dataset\nNumber of classes: {num_classes}\nLabel mapping: {train_dataset.labels_str2int}")
    logger.info(f"Number of datapoints: {len(train_dataset)} (Train), {len(dev_dataset)} (Dev), {len(test_dataset)} (Test)")
    
    # Initialize model
    logger.info("\n" + "-"*70)
    logger.info("Initializing model...")
    model = MODEL_MAPPING.get(config['model_type'])(
        **config['model_params'],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {config['model_type']} | Total parameters: {total_params:,} | "
                f"Trainable parameters: {trainable_params:,} | Device: {device}")
    
    # Use CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Check if using contrastive learning
    use_contrastive = config['model_params'].get('use_contrastive', False)
    if use_contrastive:
        contrastive_weight = config['training_params'].get('contrastive_weight', 0.5)
        contrastive_temp = config['training_params'].get('contrastive_temperature', 0.07)
        contrastive_criterion = SupervisedContrastiveLoss(temperature=contrastive_temp)
        logger.info(f"Using Supervised Contrastive Loss: weight={contrastive_weight}, temp={contrastive_temp}")

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
    train_ce_losses = []  # Track CE loss separately when using contrastive
    train_contrastive_losses = []  # Track contrastive loss separately
    train_accs = []
    train_f1s = []
    val_losses = []
    val_accs = []
    val_f1s = []
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    
    plot_every = config['training_params'].get('plot_every', 5)
    save_every = config['training_params'].get('save_every', 10)
    use_early_stopping = config['training_params'].get('use_early_stopping', True)
    early_stop_patience = config['training_params'].get('early_stop_patience', 15)
    
    # Check if model uses attention (self-attention or attention pooling)
    use_attention = config['model_params'].get('use_attention', False) or \
                    config['model_params'].get('use_attention_pooling', False)
    
    logger.info("\n" + "="*70 + "Starting training" + "\n" + "="*70)
    
    # Training loop
    for epoch in range(config['training_params']['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_ce_loss = 0
        train_con_loss = 0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []
        
        logger.info(f"Epoch [{epoch+1}/{config['training_params']['num_epochs']}] - Training")
        
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            labels = batch['label']  # Keep as long for CrossEntropyLoss

            optimizer.zero_grad()
            
            # Forward pass with optional contrastive embeddings
            if use_contrastive:
                outputs, embeddings = model(batch, return_embeddings=True)
                ce_loss = criterion(outputs, labels)
                contrastive_loss = contrastive_criterion(embeddings, labels)
                
                # Handle potential NaN in contrastive loss (can happen with small batches)
                if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                    loss = ce_loss
                    contrastive_loss_val = 0.0
                else:
                    loss = ce_loss + contrastive_weight * contrastive_loss
                    contrastive_loss_val = contrastive_loss.item()
                
                train_ce_loss += ce_loss.item()
                train_con_loss += contrastive_loss_val
            else:
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
            _, predicted = torch.max(outputs, 1)  # Get class with highest score
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        
        # Training statistics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_f1 = f1_score(all_train_labels, all_train_preds, average='binary')
        
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)
        train_f1s.append(train_f1)
        
        # Track separate losses for contrastive learning
        if use_contrastive:
            avg_ce_loss = train_ce_loss / len(train_loader)
            avg_con_loss = train_con_loss / len(train_loader)
            train_ce_losses.append(avg_ce_loss)
            train_contrastive_losses.append(avg_con_loss)
        
        # Validation phase
        logger.info(f"Epoch [{epoch+1}/{config['training_params']['num_epochs']}] - Validation")
        val_loss, val_accuracy, val_f1 = evaluate(model, dev_loader, criterion, device, logger, "Development")
        
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        val_f1s.append(val_f1)
        
        # Log epoch summary
        logger.info("-"*70)
        logger.info(f"Epoch [{epoch+1}/{config['training_params']['num_epochs']}] Summary:")
        if use_contrastive:
            logger.info(f"  Train Loss: {avg_train_loss:.4f} (CE: {avg_ce_loss:.4f}, Contrastive: {avg_con_loss:.4f})")
        else:
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Train Acc: {train_accuracy:.2f}% | Train F1: {train_f1:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Val F1: {val_f1:.4f}")
        if use_scheduler:
            logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f} (Using Scheduler)")
        else:
            logger.info(f"  Learning Rate: {config['training_params']['learning_rate']:.6f} (Not using Scheduler)")
        
        # Check if best model (based on validation F1 score)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            logger.info(f"  New best validation F1: {best_val_f1:.4f}")
            save_checkpoint(model, optimizer, epoch+1, avg_train_loss, train_accuracy, 
                          val_loss, val_accuracy, checkpoints_dir, logger, is_best=True)
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
            logger.info(f"Best validation F1: {best_val_f1:.4f}")
            break
        
        # Plot training curves periodically
        if (epoch + 1) % plot_every == 0:
            plot_file = plot_training_curves(train_losses, train_accs, train_f1s, 
                                            val_losses, val_accs, val_f1s,
                                            plots_dir, timestamp)
            logger.info(f"Training curves saved to {plot_file}\n")
            
            # Plot attention weights if model uses attention
            if use_attention:
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
            
            # Plot contrastive embeddings if using contrastive learning
            if use_contrastive:
                emb_plot_file, emb_metrics = plot_contrastive_embeddings(
                    model, dev_loader, device, contrastive_plots_dir, epoch+1,
                    labels_int2str=train_dataset.labels_int2str,
                    timestamp=timestamp, max_samples=500
                )
                logger.info(f"Contrastive embeddings saved to {emb_plot_file}")
                logger.info(f"  Silhouette: {emb_metrics['silhouette_score']:.4f} | "
                           f"Sep Ratio: {emb_metrics['separation_ratio']:.4f}\n")
                model.train()
        
        # Save periodic checkpoint
        if args.save_ckpt and (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch+1, avg_train_loss, train_accuracy, 
                          val_loss, val_accuracy, checkpoints_dir, logger, is_best=False)
    
    # Final plot
    plot_file = plot_training_curves(train_losses, train_accs, train_f1s,
                                    val_losses, val_accs, val_f1s,
                                    plots_dir, timestamp)
    logger.info(f"\nFinal training curves saved to {plot_file}")
    
    # Save final model
    save_checkpoint(model, optimizer, len(train_losses), 
                   train_losses[-1], train_accs[-1], val_losses[-1], val_accs[-1], 
                   checkpoints_dir, logger, is_best=False)
    
    # Final evaluation on test set
    logger.info("\n" + "="*70)
    logger.info("Evaluating on test set...")
    test_loss, test_accuracy, test_f1 = evaluate(model, test_loader, criterion, device, logger, "Test")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
    logger.info(f"Test F1: {test_f1:.4f}")
    
    logger.info("\n" + "="*70)
    logger.info("Training completed successfully!")
    logger.info(f"  Best validation loss: {best_val_loss:.4f}")
    logger.info(f"  Best validation F1: {best_val_f1:.4f}")
    logger.info(f"  Final test loss: {test_loss:.4f}")
    logger.info(f"  Final test accuracy: {test_accuracy:.2f}%")
    logger.info(f"  Final test F1: {test_f1:.4f}")
    logger.info(f"  Experiment directory: {exp_dir}")
    logger.info("="*70)

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