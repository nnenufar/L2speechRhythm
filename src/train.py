import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import argparse
import logging
from pathlib import Path
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from src.models import CNN_RNN_Classifier, LSTM_with_MultiHeadAttention, rhythm_spectrum_encoder
from src.dataloaders import DatasetLMDB, collate_fn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime('%Y%m%d_%H%M')

MODEL_MAPPING = {
    "cnn_rnn": CNN_RNN_Classifier,
    "lstm": LSTM_with_MultiHeadAttention,
    "spec_encoder": rhythm_spectrum_encoder,
}

def setup_experiment_dir(exp_name):
    """
    Create experiment directory structure and return paths.
    """
    exp_dir = Path("exp") / exp_name
    logs_dir = exp_dir / "logs"
    plots_dir = exp_dir / "plots"
    checkpoints_dir = exp_dir / "checkpoints"
    
    # Create directories
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    return exp_dir, logs_dir, plots_dir, checkpoints_dir

def setup_logger(logs_dir, exp_name):
    """
    Setup logging configuration with both file and console handlers.
    """
    # Create logger
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler - logs everything to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_file = logs_dir / f"training_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler - logs to terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_file}")
    
    return logger

def plot_training_curves(train_losses, train_accs, val_losses, val_accs, plots_dir, timestamp):
    """
    Plot and save training curves (loss and accuracy).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    if val_losses:
        ax1.plot(epochs_range, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    if train_accs:
        ax2.plot(epochs_range, train_accs, 'b-o', label='Training Accuracy', linewidth=2, markersize=4)
    if val_accs:
        ax2.plot(epochs_range, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = plots_dir / f'training_curves_{timestamp}.png'
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    return plot_file

def save_checkpoint(model, optimizer, epoch, train_loss, train_acc, val_loss, val_acc, 
                   checkpoints_dir, logger, is_best=False):
    """
    Save model checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
    }
    
    if is_best:
        # Delete previous best model
        for f in checkpoints_dir.glob("best_model_epoch*.pth"):
            try:
                os.remove(f)
            except Exception as e:
                logger.warning(f"Could not delete {f}: {e}")
        # Save new best model
        checkpoint_path = checkpoints_dir / f'best_model_epoch{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Best model saved to {checkpoint_path}")
    else:
        checkpoint_path = checkpoints_dir / f'checkpoint_epoch{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

def evaluate(model, dataloader, criterion, device, logger, split_name="Validation"):
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
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['label']
            
            outputs = model(batch)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def train(config):
    # Setup experiment directories
    exp_name = config.get('exp_name')
    exp_dir, logs_dir, plots_dir, checkpoints_dir = setup_experiment_dir(exp_name)
    
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
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
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
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    plot_every = config['training_params'].get('plot_every', 5)
    save_every = config['training_params'].get('save_every', 10)
    early_stop_patience = config['training_params'].get('early_stop_patience', 15)
    
    logger.info("\n" + "="*70 + "Starting training" + "\n" + "="*70)
    
    # Training loop
    for epoch in range(config['training_params']['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        logger.info(f"Epoch [{epoch+1}/{config['training_params']['num_epochs']}] - Training")
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['label']

            optimizer.zero_grad()
            outputs = model(batch)
            break
        break
    #         loss = criterion(outputs, labels)

    #         loss.backward()
    #         optimizer.step()
    #         if use_scheduler:
    #             scheduler.step()
            
    #         # Calculate metrics
    #         train_loss += loss.item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         train_total += labels.size(0)
    #         train_correct += (predicted == labels).sum().item()
        
    #     # Training statistics
    #     avg_train_loss = train_loss / len(train_loader)
    #     train_accuracy = 100 * train_correct / train_total
        
    #     train_losses.append(avg_train_loss)
    #     train_accs.append(train_accuracy)
        
    #     # Validation phase
    #     logger.info(f"Epoch [{epoch+1}/{config['training_params']['num_epochs']}] - Validation")
    #     val_loss, val_accuracy = evaluate(model, dev_loader, criterion, device, logger, "Development")
        
    #     val_losses.append(val_loss)
    #     val_accs.append(val_accuracy)
        
    #     # Log epoch summary
    #     logger.info("-"*70)
    #     logger.info(f"Epoch [{epoch+1}/{config['training_params']['num_epochs']}] Summary:")
    #     logger.info(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
    #     logger.info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
    #     if use_scheduler:
    #         logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f} (Using Scheduler)")
    #     else:
    #         logger.info(f"  Learning Rate: {config['training_params']['learning_rate']:.6f} (Not using Scheduler)")
        
    #     # Check if best model (based on validation loss)
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         epochs_without_improvement = 0
    #         logger.info(f"  New best validation loss: {best_val_loss:.4f}")
    #         save_checkpoint(model, optimizer, epoch+1, avg_train_loss, train_accuracy, 
    #                       val_loss, val_accuracy, checkpoints_dir, logger, is_best=True)
    #     else:
    #         epochs_without_improvement += 1
        
    #     if val_accuracy > best_val_acc:
    #         best_val_acc = val_accuracy
    #         logger.info(f"  New best validation accuracy: {best_val_acc:.2f}%")
        
    #     logger.info("-"*70 + "\n")
        
    #     # Early stopping check
    #     if epochs_without_improvement >= early_stop_patience:
    #         logger.info(f"Early stopping triggered after {early_stop_patience} epochs without improvement")
    #         logger.info(f"Best validation loss: {best_val_loss:.4f}")
    #         logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    #         break
        
    #     # Plot training curves periodically
    #     if (epoch + 1) % plot_every == 0:
    #         plot_file = plot_training_curves(train_losses, train_accs, val_losses, val_accs, 
    #                                         plots_dir, timestamp)
    #         logger.info(f"Training curves saved to {plot_file}\n")
        
    #     # Save periodic checkpoint
    #     if args.save_ckpt and (epoch + 1) % save_every == 0:
    #         save_checkpoint(model, optimizer, epoch+1, avg_train_loss, train_accuracy, 
    #                       val_loss, val_accuracy, checkpoints_dir, logger, is_best=False)
    
    # # Final plot
    # plot_file = plot_training_curves(train_losses, train_accs, val_losses, val_accs, 
    #                                 plots_dir, timestamp)
    # logger.info(f"\nFinal training curves saved to {plot_file}")
    
    # # Save final model
    # save_checkpoint(model, optimizer, len(train_losses), 
    #                train_losses[-1], train_accs[-1], val_losses[-1], val_accs[-1], 
    #                checkpoints_dir, logger, is_best=False)
    
    # # Final evaluation on test set
    # logger.info("\n" + "="*70)
    # logger.info("Evaluating on test set...")
    # test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, logger, "Test")
    # logger.info(f"Test Loss: {test_loss:.4f}")
    # logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # logger.info("\n" + "="*70)
    # logger.info("Training completed successfully!")
    # logger.info(f"  Best validation loss: {best_val_loss:.4f}")
    # logger.info(f"  Best validation accuracy: {best_val_acc:.2f}%")
    # logger.info(f"  Final test loss: {test_loss:.4f}")
    # logger.info(f"  Final test accuracy: {test_accuracy:.2f}%")
    # logger.info(f"  Experiment directory: {exp_dir}")
    # logger.info("="*70)

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