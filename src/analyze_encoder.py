"""
Analyze the trained rhythm encoder: per-utterance and overall latent space (UMAP).

Usage:
    python -m src.analyze_encoder \
        --config config/exp_A1_env_fluency.json \
        --checkpoint exp/exp_A1_env_fluency/checkpoints/<ts>/best_model_epoch38.pth \
        --output_dir analysis/rhythm_encoder
"""

import torch
import json
import argparse
import random
import numpy as np
import lmdb
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import umap

from src.models import RhythmRegressor, RhythmContrastiveModel
from src.dataloaders import DatasetLMDB, collate_fn, parse_identifier
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_overall_latent(embs, labels, is_regression, output_path):
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embs)

    fig, ax = plt.subplots(figsize=(8, 6))
    if is_regression:
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis',
                             alpha=0.6, s=15, edgecolors='none')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Score')
    else:
        for label_val, label_name, color in [(0, 'L1 (native)', 'steelblue'),
                                              (1, 'L2 (non-native)', 'coral')]:
            mask = labels == label_val
            ax.scatter(reduced[mask, 0], reduced[mask, 1], c=color, label=label_name,
                       alpha=0.6, s=15, edgecolors='none')
        ax.legend()
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Latent Space (UMAP)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Overall latent space -> {output_path}")


def _centroid_distance(l1_embs, l2_embs):
    c_l1 = np.mean(l1_embs, axis=0)
    c_l2 = np.mean(l2_embs, axis=0)
    cos_sim = np.dot(c_l1, c_l2) / (np.linalg.norm(c_l1) * np.linalg.norm(c_l2) + 1e-8)
    return float(1.0 - cos_sim)


def plot_per_utterance_latent(utt_to_embs, utt_to_labels, is_regression, output_path):
    utt_ids = sorted(utt_to_embs.keys())

    flat_embs = []
    plot_groups = []
    for utt_id in utt_ids:
        for i, emb in enumerate(utt_to_embs[utt_id]):
            flat_embs.append(emb)
            label = utt_to_labels[utt_id][i]
            plot_groups.append((utt_id, label))

    if len(flat_embs) < 2:
        print("  Not enough samples for per-utterance plot")
        return

    flat_embs = np.array(flat_embs)
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(flat_embs)

    n_utts = len(utt_ids)
    ncols = min(3, n_utts)
    nrows = (n_utts + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows * ncols == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for idx, utt_id in enumerate(utt_ids):
        ax = axes[idx]
        mask = np.array([g[0] == utt_id for g in plot_groups])
        utt_labels = np.array([g[1] for g in plot_groups])

        ax.scatter(reduced[:, 0], reduced[:, 1], c='lightgray', alpha=0.15, s=5,
                   edgecolors='none')

        if is_regression:
            ax.scatter(reduced[mask, 0], reduced[mask, 1], c=utt_labels[mask],
                       cmap='viridis', alpha=0.8, s=15, edgecolors='none')
            ax.set_title(f'{utt_id}  (n={mask.sum()})')
        else:
            for label_val, label_name, color in [(0, 'L1', 'steelblue'),
                                                  (1, 'L2', 'coral')]:
                sub_mask = mask & (utt_labels == label_val)
                ax.scatter(reduced[sub_mask, 0], reduced[sub_mask, 1], c=color,
                           label=f'{label_name} ({sub_mask.sum()})', alpha=0.8, s=15,
                           edgecolors='none')
            if any(utt_labels == 0) and any(utt_labels == 1):
                dist = _centroid_distance(
                    np.array([e for j, e in enumerate(utt_to_embs[utt_id]) if utt_to_labels[utt_id][j] == 0]),
                    np.array([e for j, e in enumerate(utt_to_embs[utt_id]) if utt_to_labels[utt_id][j] == 1]),
                )
                ax.set_title(f'{utt_id}  (dist={dist:.3f})')
            else:
                ax.set_title(f'{utt_id}  (n={mask.sum()})')
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
    print(f"  Per-utterance latent space -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='analysis/rhythm_encoder')
    parser.add_argument('--num_utterances', type=int, default=6,
                        help='Number of utterances for per-utterance latent space')
    parser.add_argument('--split', type=str, default='Development',
                        choices=['Train', 'Development', 'Test'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_params = {**config['dataset_params']}

    lmdb_env = lmdb.open(dataset_params['lmdb_path'], readonly=True, lock=False,
                         readahead=False, meminit=False)

    dataset = DatasetLMDB(**dataset_params, split=args.split, env=lmdb_env)
    print(f"Split {args.split}: {len(dataset)} samples")

    model_type = config.get('model_type', 'rhythm_regressor')
    model_kwargs = {**config['model_params']}
    if model_type == 'rhythm_regressor':
        model = RhythmRegressor(**model_kwargs).to(device)
    elif model_type == 'rhythm_encoder':
        model = RhythmContrastiveModel(**model_kwargs).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder_state = {k.removeprefix('encoder.'): v
                     for k, v in checkpoint['model_state_dict'].items()
                     if k.startswith('encoder.')}
    model.encoder.load_state_dict(encoder_state, strict=False)
    model.eval()
    print(f"Loaded checkpoint epoch {checkpoint.get('epoch', '?')}")

    is_regression = config['model_params'].get('task') == 'regression'

    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    utt_to_embs = defaultdict(list)
    utt_to_labels = defaultdict(list)
    all_embs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            emb = model.encoder(batch_dev['envelope'])
            emb_np = emb.cpu().numpy()
            labels_np = batch_dev['label'].cpu().numpy()

            for i, identifier in enumerate(batch['identifier']):
                _, utt_id = parse_identifier(identifier)
                utt_to_embs[utt_id].append(emb_np[i])
                utt_to_labels[utt_id].append(labels_np[i])
                all_embs.append(emb_np[i])
                all_labels.append(labels_np[i])

    all_embs = np.array(all_embs)
    all_labels = np.array(all_labels)

    suffix = random.randint(100000, 999999)

    overall_path = output_dir / f'overall_latent_{suffix}.png'
    plot_overall_latent(all_embs, all_labels, is_regression, overall_path)

    valid = {
        k: v for k, v in utt_to_embs.items() if len(v) >= 1
    }
    valid_labels = {k: utt_to_labels[k] for k in valid}

    if not valid:
        print("  No utterances with samples found.")
    else:
        valid_keys = sorted(valid.keys())
        random.shuffle(valid_keys)
        selected = {k: valid[k] for k in valid_keys[:args.num_utterances]}
        selected_labels = {k: valid_labels[k] for k in selected}

        per_utt_path = output_dir / f'per_utterance_latent_{suffix}.png'
        plot_per_utterance_latent(selected, selected_labels, is_regression, per_utt_path)

    lmdb_env.close()
    print(f"\nDone. Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
