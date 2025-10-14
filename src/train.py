import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from src.models import CNN_RNN_Classifier
from src.dataloaders import DatasetLMDB, collate_fn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_MAPPING = {
    "cnn_rnn": CNN_RNN_Classifier
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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

def plot_training_curves(train_losses, train_accs, val_losses, val_accs, plots_dir, epoch):
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
    plot_file = plots_dir / f'training_curves_epoch_{epoch}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
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
        checkpoint_path = checkpoints_dir / 'best_model.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Best model saved to {checkpoint_path}")
    else:
        checkpoint_path = checkpoints_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

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
    
    # Load dataset
    logger.info("\n" + "-"*70)
    dataset = DatasetLMDB(config['lmdb_path'], config['audio_source'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_fn, shuffle=True)
    num_classes = len(dataset.labels_str2int)
    logger.info(f"Sucessfully loaded dataset\nNumber of classes: {num_classes}\nNumber of datapoints: {len(dataset)}")
    
    # Initialize model
    logger.info("\n" + "-"*70)
    logger.info("Initializing model...")
    model = MODEL_MAPPING.get(config['model_type'])(
        **config['model_params'],
        num_classes=num_classes
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {config['model_type']}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Device: {device}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training_params']['learning_rate']
    )
    
    # Training tracking
    train_losses = []
    train_accs = []
    val_losses = []  # TODO: add validation
    val_accs = []
    best_train_loss = float('inf')
    
    plot_every = config['training_params'].get('plot_every', 5)  # Plot every N epochs
    save_every = config['training_params'].get('save_every', 10)  # Save checkpoint every N epochs
    
    logger.info("\n" + "="*70)
    logger.info("Starting training" + "\n")
    
    # Training loop
    for epoch in range(config['training_params']['num_epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        logger.info(f"Epoch [{epoch+1}/{config['training_params']['num_epochs']}]")
        
        for (feats, intervals, labels) in dataloader:
            feats = feats.to(device)
            intervals = intervals.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(feats, intervals)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Epoch statistics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        train_losses.append(avg_loss)
        train_accs.append(accuracy)
        
        logger.info("-"*70)
        logger.info(f"Epoch [{epoch+1}/{config['training_params']['num_epochs']}] Summary:")
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  Accuracy: {accuracy:.2f}%")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check if best model
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            logger.info(f" New best training loss: {best_train_loss:.4f}")
            save_checkpoint(model, optimizer, epoch+1, avg_loss, accuracy, 
                          None, None, checkpoints_dir, logger, is_best=True)
        
        logger.info("-"*70 + "\n")
        
        # Plot training curves periodically
        if (epoch + 1) % plot_every == 0:
            plot_file = plot_training_curves(train_losses, train_accs, val_losses, val_accs, 
                                            plots_dir, epoch+1)
            logger.info(f" Training curves saved to {plot_file}\n")
        
        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, epoch+1, avg_loss, accuracy, 
                          None, None, checkpoints_dir, logger, is_best=False)
    
    # Final plot
    plot_file = plot_training_curves(train_losses, train_accs, val_losses, val_accs, 
                                    plots_dir, config['training_params']['num_epochs'])
    logger.info(f"\n Final training curves saved to {plot_file}")
    
    # Save final model
    save_checkpoint(model, optimizer, config['training_params']['num_epochs'], 
                   train_losses[-1], train_accs[-1], None, None, 
                   checkpoints_dir, logger, is_best=False)
    
    logger.info("\n" + "="*70)
    logger.info("Training completed successfully!")
    logger.info(f"  Best training loss: {best_train_loss:.4f}")
    logger.info(f"  Final training accuracy: {train_accs[-1]:.2f}%")
    logger.info(f"  Experiment directory: {exp_dir}")
    logger.info("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence classification model.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the JSON configuration file.')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    train(config)