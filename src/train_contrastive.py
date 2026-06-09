import torch
import torch.optim as optim
import json
import argparse
import lmdb
from datetime import datetime
import wandb
from src.models import RhythmContrastiveModel
from src.modules import per_utterance_contrastive_loss
from src.train_utils import (
    setup_experiment_dir, setup_logger, plot_latent_space, compute_val_centroid_distance
)
from src.dataloaders import DatasetLMDB, collate_fn, ContrastiveBatchSampler
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime('%Y%m%d_%H%M')


def train(config, test_mode=False):
    exp_name = config.get('exp_name', 'rhythm_encoder_contrastive')
    dirs = setup_experiment_dir(exp_name, timestamp)
    exp_dir = dirs['exp_dir']
    logs_dir = dirs['logs_dir']
    plots_dir = dirs['plots_dir']
    checkpoints_dir = dirs['checkpoints_dir']

    logger = setup_logger(logs_dir, exp_name, log_to_file=False)

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

    logger.info("=" * 70)
    logger.info(f"Starting contrastive pretraining: {exp_name}")
    logger.info("=" * 70)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n" + "-" * 70)
    logger.info("Loading datasets...")

    dataset_kwargs = config['dataset_params'].copy()

    lmdb_env = lmdb.open(dataset_kwargs['lmdb_path'], readonly=True, lock=False, readahead=False, meminit=False)

    train_dataset = DatasetLMDB(**dataset_kwargs, split='Train', env=lmdb_env)
    val_dataset = DatasetLMDB(**dataset_kwargs, split='Development', env=lmdb_env)

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    utterances_per_batch = config['training_params']['utterances_per_batch']
    l1_per_utterance = config['training_params']['l1_per_utterance']
    l2_per_utterance = config['training_params']['l2_per_utterance']

    batch_sampler = ContrastiveBatchSampler(
        dataset=train_dataset,
        csv_path='data/arctic/arctic_metadata.csv',
        utterances_per_batch=utterances_per_batch,
        l1_per_utterance=l1_per_utterance,
        l2_per_utterance=l2_per_utterance,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False
    )

    effective_batch = utterances_per_batch * (l1_per_utterance + l2_per_utterance)
    logger.info(f"Train batches: {len(batch_sampler)}, Val samples: {len(val_dataset)}")
    logger.info(f"Effective batch: {effective_batch}")

    logger.info("\n" + "-" * 70)
    logger.info("Initializing model...")

    model = RhythmContrastiveModel(**config['model_params']).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,} | Device: {device}")

    temperature = config['training_params'].get('temperature', 0.07)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training_params']['learning_rate'],
        weight_decay=config['training_params'].get('weight_decay', 0.0)
    )

    num_epochs = config['training_params']['num_epochs']

    if test_mode:
        logger.info("\n" + "=" * 70)
        logger.info("TEST MODE: single batch forward/backward check")
        logger.info("=" * 70)

        batch = next(iter(train_loader))
        x = batch['envelope'].to(device)
        utterance_ids = batch['utterance_id'].to(device)
        is_l1 = (batch['label'].to(device) == 0)

        optimizer.zero_grad()
        embeddings = model(x)
        loss = per_utterance_contrastive_loss(embeddings, utterance_ids, is_l1, temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        logger.info(f"  Envelope shape: {x.shape}, Embeddings: {embeddings.shape}")
        logger.info(f"  Loss: {loss.item():.6f}, Grad norm: {grad_norm:.4f}")
        logger.info(f"  Utterance IDs: {utterance_ids.unique().tolist()}")
        logger.info(f"  L1/L2: {is_l1.sum().item()}/{(~is_l1).sum().item()}")
        logger.info("Test mode PASSED")
        return

    logger.info("\n" + "=" * 70 + "Starting training" + "\n" + "=" * 70)

    best_val_metric = -1.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            x = batch['envelope'].to(device)
            utterance_ids = batch['utterance_id'].to(device)
            is_l1 = (batch['label'].to(device) == 0)

            optimizer.zero_grad()
            embeddings = model(x)
            loss = per_utterance_contrastive_loss(embeddings, utterance_ids, is_l1, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_dist = compute_val_centroid_distance(model, val_loader, device)

        lr_value = config['training_params']['learning_rate']

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.6f} "
            f"| Val centroid dist: {val_dist:.4f} | LR: {lr_value:.6f}"
        )

        if wandb_enabled:
            wandb.log({
                'train_loss': avg_loss,
                'val_centroid_distance': val_dist,
                'lr': lr_value,
            }, step=epoch + 1)

        if val_dist > best_val_metric:
            best_val_metric = val_dist
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_centroid_distance': val_dist,
            }, checkpoints_dir / 'best_model.pth')
            logger.info(f"  New best model (val centroid dist: {best_val_metric:.4f})")

    logger.info("\n" + "=" * 70)
    logger.info(f"Training completed. Best val centroid distance: {best_val_metric:.4f}")
    logger.info("Generating latent space plot...")

    latent_plot_path = plots_dir / f'latent_space_{timestamp}.png'
    plot_latent_space(model, train_loader, latent_plot_path, device)
    logger.info(f"Latent space plot saved to {latent_plot_path}")

    if wandb_enabled:
        wandb.summary['best_val_centroid_distance'] = best_val_metric
        wandb.finish()

    lmdb_env.close()

    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrastive pretraining of rhythm encoder.")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    train(config, test_mode=args.test)
