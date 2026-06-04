"""
Analyze the trained rhythm encoder: cross-attention weights and per-utterance latent space.

Usage:
    python -m src.analyze_encoder \
        --config config/rhythm_encoder_contrastive.json \
        --checkpoint exp/rhythm_encoder_contrastive/checkpoints/best_model.pth \
        --output_dir analysis/rhythm_encoder

Analysis A: Overlay cross-attention weights on the envelope for random samples.
Analysis B: UMAP latent space for individual utterances, split by L1/L2.
"""

import torch
import json
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA

from src.models import RhythmContrastiveModel
from src.train_contrastive import load_phoneme_mapping
from src.dataloaders import DatasetLMDB, collate_fn, parse_identifier
from torch.utils.data import DataLoader
import umap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4


def plot_attention_overlay(identifier, is_l1, envelope, attn_weights, phoneme_ids,
                           vocab_inv, output_path):
    """
    envelope: (T,) raw envelope values
    attn_weights: (T_audio, T_text) cross-attention, T_audio <= T
    phoneme_ids: (T_text,) int phoneme ids
    """
    T_env = len(envelope)
    T_attn = attn_weights.shape[0]

    x_attn = np.linspace(0, T_env - 1, T_attn)
    attn_intensity = attn_weights.sum(axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(16, 7),
                              gridspec_kw={'height_ratios': [2, 3, 1]})

    speaker_type = 'L1 (native)' if is_l1 else 'L2 (non-native)'
    axes[0].set_title(f'{identifier}  [{speaker_type}]')
    axes[0].plot(envelope, 'steelblue', linewidth=0.4)
    axes[0].set_ylabel('Envelope')

    im = axes[1].imshow(attn_weights.T, aspect='auto', origin='lower', cmap='inferno',
                         extent=[0, T_env - 1, 0, attn_weights.shape[1] - 1],
                         interpolation='bilinear')
    axes[1].set_ylabel('Phoneme index')
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.02)
    cbar.set_label('Attention')

    axes[2].plot(x_attn, attn_intensity, 'coral', linewidth=0.8)
    axes[2].fill_between(x_attn, attn_intensity, alpha=0.3, color='coral')
    axes[2].set_ylabel('Attn intensity')
    axes[2].set_xlabel('Envelope frame')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _centroid_distance(l1_embs, l2_embs):
    c_l1 = np.mean(l1_embs, axis=0)
    c_l2 = np.mean(l2_embs, axis=0)
    cos_sim = np.dot(c_l1, c_l2) / (np.linalg.norm(c_l1) * np.linalg.norm(c_l2) + 1e-8)
    return 1.0 - cos_sim


def plot_per_utterance_latent(utterance_embeddings, output_path):
    """
    utterance_embeddings: dict {utt_id: {'L1': [...emb...], 'L2': [...emb...]}}
    Each value is a list of numpy embedding vectors.
    """
    flat_embs = []
    plot_groups = []

    for utt_id, groups in utterance_embeddings.items():
        for emb in groups['L1']:
            flat_embs.append(emb)
            plot_groups.append((utt_id, 'L1'))
        for emb in groups['L2']:
            flat_embs.append(emb)
            plot_groups.append((utt_id, 'L2'))

    if len(flat_embs) < 2:
        return

    flat_embs = np.array(flat_embs)
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(flat_embs)

    utt_ids = list(utterance_embeddings.keys())
    n_utts = len(utt_ids)
    ncols = min(3, n_utts)
    nrows = (n_utts + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows * ncols == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for idx, utt_id in enumerate(utt_ids):
        ax = axes[idx]
        l1_mask = np.array([g[0] == utt_id and g[1] == 'L1' for g in plot_groups])
        l2_mask = np.array([g[0] == utt_id and g[1] == 'L2' for g in plot_groups])

        ax.scatter(reduced[:, 0], reduced[:, 1], c='lightgray', alpha=0.15, s=5,
                   edgecolors='none')

        ax.scatter(reduced[l1_mask, 0], reduced[l1_mask, 1], c='steelblue',
                   label=f'L1 ({l1_mask.sum()})', alpha=0.8, s=15, edgecolors='none')
        ax.scatter(reduced[l2_mask, 0], reduced[l2_mask, 1], c='coral',
                   label=f'L2 ({l2_mask.sum()})', alpha=0.8, s=15, edgecolors='none')

        dist = _centroid_distance(
            np.array(utterance_embeddings[utt_id]['L1']),
            np.array(utterance_embeddings[utt_id]['L2'])
        )
        ax.set_title(f'{utt_id}  (dist={dist:.3f})')
        ax.legend(fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_utts, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Rhythm Encoder — Per-Utterance Latent Space (UMAP)', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analysis_a(model, dataset, vocab_inv, output_dir, num_samples):
    """Cross-attention overlay on envelope. Requires use_text=True."""
    if not model.encoder.use_text:
        print("  Skipped: model was trained without text (use_text=False).")
        return

    output_dir = Path(output_dir) / 'attention_overlay'
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn,
                        num_workers=NUM_WORKERS)
    model.eval()

    done = 0
    with torch.no_grad():
        for batch in loader:
            if done >= num_samples:
                break

            x = batch['envelope'].to(device)
            pid = batch['phoneme_ids'].to(device)
            plen = batch['phoneme_lengths'].to(device)
            identifier = batch['identifier'][0]
            is_l1 = batch['label'].item() == 0

            _, attn_weights = model.encoder(x, pid, plen)

            env_np = x[0].cpu().numpy()
            attn_np = attn_weights[0].cpu().numpy()
            pid_np = pid[0, :plen[0].item()].cpu().numpy()

            if attn_np.shape[0] < 2 or attn_np.shape[1] < 2:
                continue

            out_path = output_dir / f'attn_{identifier}.png'
            plot_attention_overlay(
                identifier, is_l1, env_np, attn_np, pid_np, vocab_inv, out_path
            )
            print(f"  [{done+1}/{num_samples}] {identifier}")
            done += 1


def plot_overall_dev_latent(utt_to_samples, output_path):
    """Plot all dev samples in 2D UMAP, colored by L1 vs L2."""
    flat_embs = []
    flat_labels = []
    for groups in utt_to_samples.values():
        for emb in groups['L1']:
            flat_embs.append(emb)
            flat_labels.append(0)
        for emb in groups['L2']:
            flat_embs.append(emb)
            flat_labels.append(1)

    if len(flat_embs) < 2:
        return

    flat_embs = np.array(flat_embs)
    flat_labels = np.array(flat_labels)

    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(flat_embs)

    fig, ax = plt.subplots(figsize=(8, 6))
    l1_mask = flat_labels == 0
    l2_mask = flat_labels == 1

    ax.scatter(reduced[l1_mask, 0], reduced[l1_mask, 1], c='steelblue', label='L1 (native)',
               alpha=0.6, s=15, edgecolors='none')
    ax.scatter(reduced[l2_mask, 0], reduced[l2_mask, 1], c='coral', label='L2 (non-native)',
               alpha=0.6, s=15, edgecolors='none')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Dev Set — All Utterances (UMAP)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analysis_b(model, dataset, output_dir, num_utterances):
    """Per-utterance latent space using all available L1 and L2 samples."""
    output_dir = Path(output_dir) / 'per_utterance'
    output_dir.mkdir(parents=True, exist_ok=True)

    use_text = model.encoder.use_text

    utt_to_samples = defaultdict(lambda: {'L1': [], 'L2': []})

    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,
                        num_workers=NUM_WORKERS)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            x = batch['envelope'].to(device)
            labels = batch['label']
            identifiers = batch['identifier']

            if use_text:
                pid = batch['phoneme_ids'].to(device)
                plen = batch['phoneme_lengths'].to(device)
                emb, _ = model.encoder(x, pid, plen)
            else:
                emb, _ = model.encoder(x)

            emb_np = emb.cpu().numpy()

            for i, identifier in enumerate(identifiers):
                _, utt_id = parse_identifier(identifier)
                category = 'L1' if labels[i].item() == 0 else 'L2'
                utt_to_samples[utt_id][category].append(emb_np[i])

    valid = {
        k: v for k, v in utt_to_samples.items()
        if len(v['L1']) >= 2 and len(v['L2']) >= 2
    }

    if not valid:
        print("  No utterances with at least 2 L1 and 2 L2 samples found.")
        return

    valid_keys = list(valid.keys())
    random.shuffle(valid_keys)
    selected = {k: valid[k] for k in valid_keys[:num_utterances]}

    suffix = random.randint(100000, 999999)
    out_path = output_dir / f'per_utterance_latent_{suffix}.png'
    plot_per_utterance_latent(selected, out_path)
    print(f"  Plotted {len(selected)} utterances -> {out_path}")

    overall_path = output_dir / f'dev_overall_latent_{suffix}.png'
    plot_overall_dev_latent(utt_to_samples, overall_path)
    print(f"  Overall dev latent space -> {overall_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='analysis/rhythm_encoder')
    parser.add_argument('--num_samples_a', type=int, default=8,
                        help='Number of random samples for attention overlay (Analysis A)')
    parser.add_argument('--num_utterances_b', type=int, default=6,
                        help='Number of utterances for per-utterance latent space (Analysis B)')
    parser.add_argument('--skip_a', action='store_true')
    parser.add_argument('--skip_b', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    use_text = config['model_params'].get('use_text', False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phoneme_vocab = None
    phoneme_mapping = None
    vocab_inv = None
    if use_text:
        mapping_path = config['dataset_params']['phoneme_mapping_path']
        prompts_path = config['dataset_params'].get('prompts_path',
                                                     '/hadatasets/joao.lima/data/arctic_cmu/PROMPTS')
        phoneme_vocab, phoneme_mapping = load_phoneme_mapping(mapping_path, prompts_path)
        vocab_inv = {v: k for k, v in phoneme_vocab.items()}
        print(f"Phoneme vocabulary: {len(phoneme_vocab)} entries")

    dataset_params = {**config['dataset_params']}
    dataset_params.pop('phoneme_mapping_path', None)
    dataset_params.pop('prompts_path', None)
    train_dataset = DatasetLMDB(**dataset_params, split='Train', phoneme_mapping=phoneme_mapping)
    dev_dataset = DatasetLMDB(**dataset_params, split='Development', phoneme_mapping=phoneme_mapping)
    print(f"Train: {len(train_dataset)} samples, Dev: {len(dev_dataset)} samples")

    model_kwargs = {**config['model_params']}
    if use_text:
        model_kwargs['num_phonemes'] = len(phoneme_vocab)
    model = RhythmContrastiveModel(**model_kwargs).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint epoch {checkpoint.get('epoch', '?')}")

    if not args.skip_a:
        print(f"\n--- Analysis A: cross-attention overlay ({args.num_samples_a} samples) ---")
        analysis_a(model, train_dataset, vocab_inv, output_dir, args.num_samples_a)

    if not args.skip_b:
        print(f"\n--- Analysis B: per-utterance latent space "
              f"({args.num_utterances_b} dev utterances, all samples) ---")
        analysis_b(model, dev_dataset, output_dir, args.num_utterances_b)

    print(f"\nDone. Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
