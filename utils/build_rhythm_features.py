"""
Build per-utterance rhythm-metric features from pre-extracted V/C duration sequences.

Computes the six classic speech-rhythm metrics for each utterance:
    npvi_v, npvi_c  (normalized pairwise variability index, Grabe & Low 2002)
    rpvi_v, rpvi_c  (raw pairwise variability index)
    delta_v, delta_c (standard deviation of the duration sequence)

Durations are converted to milliseconds before computing rPVI and delta (nPVI is
unit-invariant). Utterances with fewer than 2 vocalic or 2 intervocalic segments are
dropped (nPVI/rPVI are undefined for N < 2).

Usage:
    python utils/build_rhythm_features.py \
        --alignments data/speechocean/vc_alignments.json \
        --metadata data/speechocean/speechocean_metadata.csv \
        --output data/speechocean/rhythm_features.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _npvi(dat):
    """Normalized PVI (Grabe & Low 2002), unit-invariant."""
    return sum([abs((k - k1) / ((k + k1) / 2)) for (k, k1) in zip(dat, dat[1:])]) / (sum(1 for _ in dat) - 1)


def _rpvi(dat):
    """Raw PVI (same as nPVI but without the ratio normalization)."""
    return sum([abs(k - k1) for (k, k1) in zip(dat, dat[1:])]) / (sum(1 for _ in dat) - 1)


def _delta(dat):
    """Population standard deviation of the duration sequence."""
    return float(np.std(dat))


def compute_features(v_durs_s, c_durs_s):
    """Convert seconds -> ms and return the 6 features as a list."""
    v = np.asarray(v_durs_s, dtype=float) * 1000.0
    c = np.asarray(c_durs_s, dtype=float) * 1000.0
    return [
        _npvi(v), _npvi(c),
        _rpvi(v), _rpvi(c),
        _delta(v), _delta(c),
    ]


FEATURE_COLS = ['npvi_v', 'npvi_c', 'rpvi_v', 'rpvi_c', 'delta_v', 'delta_c']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alignments', type=str, default='data/speechocean/vc_alignments.json')
    parser.add_argument('--metadata', type=str, default='data/speechocean/speechocean_metadata.csv')
    parser.add_argument('--output', type=str, default='data/speechocean/rhythm_features.csv')
    args = parser.parse_args()

    with open(args.alignments, 'r') as f:
        alignments = json.load(f)

    metadata = pd.read_csv(args.metadata)
    metadata_ids = set(metadata['identifier'].astype(str))

    rows = []
    dropped_few_segments = 0
    dropped_no_metadata = 0

    for key, data in alignments.items():
        identifier = f"{key}.WAV"
        if identifier not in metadata_ids:
            dropped_no_metadata += 1
            continue

        v_durs = data['vocalic']['durations']
        c_durs = data['intervocalic']['durations']

        if len(v_durs) < 2 or len(c_durs) < 2:
            dropped_few_segments += 1
            continue

        features = compute_features(v_durs, c_durs)
        rows.append([identifier] + features)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=['identifier'] + FEATURE_COLS)
    df.to_csv(output_path, index=False)

    print(f"Wrote {len(df)} utterances to {output_path}")
    print(f"Dropped for <2 V/C segments: {dropped_few_segments}")
    print(f"Dropped for missing metadata identifier: {dropped_no_metadata}")


if __name__ == '__main__':
    main()
