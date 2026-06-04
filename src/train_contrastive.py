import torch
import torch.optim as optim
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
import wandb
from src.models import RhythmContrastiveModel
from src.modules import per_utterance_contrastive_loss, monotonicity_loss
from src.train_utils import (
    setup_experiment_dir, setup_logger, plot_latent_space, compute_val_centroid_distance
)
from src.dataloaders import DatasetLMDB, collate_fn, ContrastiveBatchSampler
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime('%Y%m%d_%H%M')


def _parse_prompts(filepath):
    mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'\(\s*(\S+)\s+"(.*)"\s*\)', line)
            if match:
                utt_id = match.group(1).replace('arctic_', '')
                mapping[utt_id] = match.group(2)
    return mapping


def _build_phoneme_mapping(output_path, prompts_path):
    from g2p_en import G2p
    g2p = G2p()

    transcripts = _parse_prompts(prompts_path)

    phoneme_seqs = {}
    for utt_id, text in sorted(transcripts.items()):
        phonemes = g2p(text)
        phonemes = [p for p in phonemes if p not in (' ', '')]
        phoneme_seqs[utt_id] = phonemes

    all_phonemes = set()
    for seq in phoneme_seqs.values():
        all_phonemes.update(seq)
    vocab = {'<pad>': 0}
    for i, ph in enumerate(sorted(all_phonemes)):
        vocab[ph] = i + 1

    mapping = {utt_id: [vocab[p] for p in seq] for utt_id, seq in phoneme_seqs.items()}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'vocab': vocab, 'mapping': mapping}, f)

    return vocab, mapping


def load_phoneme_mapping(mapping_path, prompts_path):
    mapping_path = Path(mapping_path)
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            data = json.load(f)
        return data['vocab'], data['mapping']

    print(f"Phoneme mapping not found at {mapping_path}. Building it now...")
    return _build_phoneme_mapping(mapping_path, prompts_path)


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

    use_text = config['model_params'].get('use_text', False)

    phoneme_vocab = None
    phoneme_mapping = None
    if use_text:
        logger.info("\n" + "-" * 70)
        logger.info("Loading phoneme mapping...")
        mapping_path = config['dataset_params']['phoneme_mapping_path']
        prompts_path = config['dataset_params'].get('prompts_path',
                                                     '/hadatasets/joao.lima/data/arctic_cmu/PROMPTS')
        phoneme_vocab, phoneme_mapping = load_phoneme_mapping(mapping_path, prompts_path)
        logger.info(f"Phoneme vocabulary size: {len(phoneme_vocab)}")

    logger.info("\n" + "-" * 70)
    logger.info("Loading datasets...")

    dataset_kwargs = config['dataset_params'].copy()
    dataset_kwargs.pop('phoneme_mapping_path', None)
    dataset_kwargs.pop('prompts_path', None)

    train_dataset = DatasetLMDB(**dataset_kwargs, split='Train', phoneme_mapping=phoneme_mapping)
    val_dataset = DatasetLMDB(**dataset_kwargs, split='Development', phoneme_mapping=phoneme_mapping)

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

    effective_batch_size = utterances_per_batch * (l1_per_utterance + l2_per_utterance)
    logger.info(f"Train batches: {len(batch_sampler)}, Val samples: {len(val_dataset)}")
    logger.info(f"Utterances per batch: {utterances_per_batch}, "
                f"L1/utt: {l1_per_utterance}, L2/utt: {l2_per_utterance}, "
                f"Effective batch: {effective_batch_size}")

    logger.info("\n" + "-" * 70)
    logger.info("Initializing model...")

    model_kwargs = {**config['model_params']}
    if use_text:
        model_kwargs['num_phonemes'] = len(phoneme_vocab)
    model = RhythmContrastiveModel(**model_kwargs).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,} | Device: {device}")

    temperature = config['training_params'].get('temperature', 0.07)
    monotonicity_coeff = config['training_params'].get('monotonicity_coeff', 0.1)

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
        logger.info("TEST MODE: Running single batch forward/backward check")
        logger.info("=" * 70)

        batch = next(iter(train_loader))
        x = batch['envelope'].to(device)
        utterance_ids = batch['utterance_id'].to(device)
        is_l1 = (batch['label'].to(device) == 0)

        if use_text:
            embeddings, attn_weights = model(x, batch['phoneme_ids'].to(device),
                                              batch['phoneme_lengths'].to(device))
        else:
            embeddings, attn_weights = model(x)

        optimizer.zero_grad()
        contrastive_loss = per_utterance_contrastive_loss(embeddings, utterance_ids, is_l1, temperature)
        if attn_weights is not None:
            mono_loss = monotonicity_loss(attn_weights)
            loss = contrastive_loss + monotonicity_coeff * mono_loss
        else:
            mono_loss = torch.tensor(0.0)
            loss = contrastive_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

        logger.info(f"  Envelope shape: {batch['envelope'].shape}")
        logger.info(f"  Embeddings shape: {embeddings.shape}")
        logger.info(f"  Embeddings norm: {embeddings.norm(dim=-1)[:3].tolist()}")
        logger.info(f"  Contrastive loss: {contrastive_loss.item():.6f}")
        if attn_weights is not None:
            logger.info(f"  Attention shape: {attn_weights.shape}")
            logger.info(f"  Monotonicity loss: {mono_loss.item():.6f}")
        logger.info(f"  Total loss: {loss.item():.6f} | Grad norm: {total_grad:.4f}")
        logger.info(f"  Utterance IDs: {utterance_ids.unique().tolist()}")
        logger.info(f"  L1/L2: {is_l1.sum().item()}/{(~is_l1).sum().item()}")
        logger.info("Test mode PASSED")
        logger.info("=" * 70)
        return

    logger.info("\n" + "=" * 70 + "Starting training" + "\n" + "=" * 70)

    best_val_metric = -1.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_mono_loss = 0.0

        for batch in train_loader:
            x = batch['envelope'].to(device)
            utterance_ids = batch['utterance_id'].to(device)
            is_l1 = (batch['label'].to(device) == 0)

            if use_text:
                embeddings, attn_weights = model(x, batch['phoneme_ids'].to(device),
                                                  batch['phoneme_lengths'].to(device))
            else:
                embeddings, attn_weights = model(x)

            optimizer.zero_grad()

            contrastive_loss = per_utterance_contrastive_loss(
                embeddings, utterance_ids, is_l1, temperature
            )
            if attn_weights is not None:
                mono_loss = monotonicity_loss(attn_weights)
                loss = contrastive_loss + monotonicity_coeff * mono_loss
            else:
                mono_loss = torch.tensor(0.0)
                loss = contrastive_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if use_scheduler:
                scheduler.step()

            epoch_loss += loss.item()
            epoch_contrastive_loss += contrastive_loss.item()
            epoch_mono_loss += mono_loss.item()

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_contrastive = epoch_contrastive_loss / n_batches
        avg_mono = epoch_mono_loss / n_batches

        val_dist = compute_val_centroid_distance(model, val_loader, device, use_text)

        lr_value = scheduler.get_last_lr()[0] if use_scheduler else config['training_params']['learning_rate']

        log_msg = f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.6f} (contrastive: {avg_contrastive:.6f}"
        if use_text:
            log_msg += f", mono: {avg_mono:.6f}"
        log_msg += f") | Val centroid dist: {val_dist:.4f} | LR: {lr_value:.6f}"
        logger.info(log_msg)

        wandb_log = {
            'train_loss': avg_loss,
            'train_contrastive_loss': avg_contrastive,
            'val_centroid_distance': val_dist,
            'lr': lr_value,
        }
        if use_text:
            wandb_log['train_mono_loss'] = avg_mono
        if wandb_enabled:
            wandb.log(wandb_log, step=epoch + 1)

        if val_dist > best_val_metric:
            best_val_metric = val_dist
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_centroid_distance': val_dist,
            }
            checkpoint_path = checkpoints_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
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
    parser = argparse.ArgumentParser(description="Contrastive pretraining of rhythm encoder.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the JSON configuration file.')
    parser.add_argument('--test', action='store_true',
                        help='Run single batch forward/backward check and exit.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    train(config, test_mode=args.test)
