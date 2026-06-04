"""
Contrastive pretraining of the duration-based rhythm model (V/C intervals).
Same-sentence L1-L1 positive, L1-L2 negative pairs from ARCTIC.

Usage:
    python -m src.train_contrastive_duration --config config/duration_contrastive_arctic.json
"""

import torch
import torch.optim as optim
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import wandb
from src.models import DurationContrastiveModel
from src.modules import per_utterance_contrastive_loss
from src.train_utils import (
    setup_experiment_dir, setup_logger, plot_latent_space, compute_val_centroid_distance
)
from src.dataloaders import DatasetLMDB, collate_fn, ContrastiveBatchSampler, parse_identifier
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime('%Y%m%d_%H%M')


def train(config, test_mode=False):
    exp_name = config.get('exp_name', 'duration_contrastive')
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
    logger.info(f"Starting duration contrastive pretraining: {exp_name}")
    logger.info("=" * 70)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n" + "-" * 70)
    logger.info("Loading duration features...")

    dur_path = config['dataset_params'].get('vc_features_path', 'data/arctic/vc_durations.json')
    with open(dur_path, 'r') as f:
        vc_durations = json.load(f)
    logger.info(f"Duration features: {len(vc_durations)} samples")

    logger.info("\n" + "-" * 70)
    logger.info("Loading datasets...")

    dataset_kwargs = config['dataset_params'].copy()
    dataset_kwargs.pop('vc_features_path', None)

    train_dataset = DatasetLMDB(**dataset_kwargs, split='Train', vc_features=vc_durations)
    val_dataset = DatasetLMDB(**dataset_kwargs, split='Development', vc_features=vc_durations)

    for ds, name in [(train_dataset, 'Train'), (val_dataset, 'Development')]:
        before = len(ds)
        valid = []
        for k in ds.lmdb_keys:
            spk, utt = parse_identifier(k.decode('utf-8'))
            if f'{spk}_{utt}' in vc_durations:
                valid.append(k)
        ds.lmdb_keys = valid
        if before - len(ds) > 0:
            logger.info(f"  {name}: dropped {before - len(ds)} samples missing VC data "
                        f"({len(ds)} remaining)")

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

    model = DurationContrastiveModel(**config['model_params']).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,} | Device: {device}")

    temperature = config['training_params'].get('temperature', 0.07)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training_params']['learning_rate'],
        weight_decay=config['training_params'].get('weight_decay', 0.0)
    )

    use_scheduler = config['training_params'].get('use_scheduler', False)
    if use_scheduler:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['training_params'].get('max_lr', 0.01),
            epochs=config['training_params']['num_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=config['training_params'].get('warmup_pct', 0.1),
            anneal_strategy='cos'
        )

    num_epochs = config['training_params']['num_epochs']

    if test_mode:
        logger.info("\n" + "=" * 70)
        logger.info("TEST MODE: single batch forward/backward check")
        logger.info("=" * 70)

        batch = next(iter(train_loader))
        v_dur = batch['v_dur'].to(device)
        c_dur = batch['c_dur'].to(device)
        utterance_ids = batch['utterance_id'].to(device)
        is_l1 = (batch['label'].to(device) == 0)

        optimizer.zero_grad()
        embeddings = model(v_dur, c_dur)
        loss = per_utterance_contrastive_loss(embeddings, utterance_ids, is_l1, temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        logger.info(f"  v_dur shape: {v_dur.shape}, c_dur shape: {c_dur.shape}")
        logger.info(f"  Embeddings: {embeddings.shape}, norm: {embeddings.norm(dim=-1)[:3].tolist()}")
        logger.info(f"  Loss: {loss.item():.6f}, Grad norm: {grad_norm:.4f}")
        logger.info(f"  Utterances: {utterance_ids.unique().tolist()}")
        logger.info(f"  L1/L2: {is_l1.sum().item()}/{(~is_l1).sum().item()}")
        logger.info("Test mode PASSED")
        return

    logger.info("\n" + "=" * 70 + "Starting training" + "\n" + "=" * 70)

    best_val_metric = -1.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            v_dur = batch['v_dur'].to(device)
            c_dur = batch['c_dur'].to(device)
            utterance_ids = batch['utterance_id'].to(device)
            is_l1 = (batch['label'].to(device) == 0)

            optimizer.zero_grad()
            embeddings = model(v_dur, c_dur)
            loss = per_utterance_contrastive_loss(embeddings, utterance_ids, is_l1, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if use_scheduler:
                scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_dist = compute_val_centroid_distance(model, val_loader, device, use_text=False)

        if epoch == 0:
            model.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader))
                vd = val_batch['v_dur']
                cd = val_batch['c_dur']
                emb = model(vd.to(device), cd.to(device))
                emb_np = emb.cpu().numpy()
                norms = np.linalg.norm(emb_np, axis=1)
                sims = emb_np @ emb_np.T
                np.fill_diagonal(sims, 0)
                logger.info(f"  [DEBUG val] emb shape={emb.shape}, norms min/mean/max: "
                            f"{norms.min():.4f}/{norms.mean():.4f}/{norms.max():.4f}")
                logger.info(f"  [DEBUG val] pairwise cos sim (excl diag): "
                            f"min={sims.min():.4f}, max={sims.max():.4f}, mean={sims.mean():.4f}")
                logger.info(f"  [DEBUG val] v_dur range: {vd[vd!=0].min():.6f}-{vd[vd!=0].max():.6f}, "
                            f"c_dur range: {cd[cd!=0].min():.6f}-{cd[cd!=0].max():.6f}")
                logger.info(f"  [DEBUG val] v_interval counts: {(vd!=0).sum(dim=1).tolist()}")
                logger.info(f"  [DEBUG val] c_interval counts: {(cd!=0).sum(dim=1).tolist()}")

        lr_value = scheduler.get_last_lr()[0] if use_scheduler else config['training_params']['learning_rate']

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

    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duration contrastive pretraining.")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test', action='store_true',
                        help='Run single batch forward/backward check and exit.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    train(config, test_mode=args.test)
