"""
Build Speechocean metadata CSV from LMDB keys, fluency scores JSON, and utt2spk files.

Usage:
    python utils/create_speechocean_table.py \
        --lmdb data/speechocean/rtm_feats.lmdb \
        --scores /path/to/scores.json \
        --train_utt2spk /path/to/train/utt2spk \
        --test_utt2spk /path/to/test/utt2spk \
        --output data/speechocean/speechocean_metadata.csv
"""

import argparse
import json
import lmdb
import pandas as pd
from pathlib import Path


def _load_utt2spk(utt2spk_path):
    """Load utt2spk file and return a set of utterance IDs."""
    utt_ids = set()
    with open(utt2spk_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            uttID, _ = line.split(maxsplit=1)
            utt_ids.add(uttID)
    return utt_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb', type=str, default='data/speechocean/rtm_feats.lmdb')
    parser.add_argument('--scores', type=str, required=True,
                        help='Path to scores JSON file or directory of JSON files.')
    parser.add_argument('--train_utt2spk', type=str, required=True,
                        help='Path to train/utt2spk file.')
    parser.add_argument('--test_utt2spk', type=str, required=True,
                        help='Path to test/utt2spk file.')
    parser.add_argument('--output', type=str, default='data/speechocean/speechocean_metadata.csv')
    args = parser.parse_args()

    train_utt_ids = _load_utt2spk(args.train_utt2spk)
    test_utt_ids = _load_utt2spk(args.test_utt2spk)
    overlap = train_utt_ids & test_utt_ids
    if overlap:
        raise ValueError(f"Overlap between train and test utt2spk: {len(overlap)} IDs")
    print(f"Loaded {len(train_utt_ids)} train and {len(test_utt_ids)} test utterance IDs from utt2spk")

    scores_path = Path(args.scores)
    if scores_path.is_dir():
        all_scores = {}
        for jf in sorted(scores_path.glob('*.json')):
            with open(jf, 'r') as f:
                all_scores.update(json.load(f))
    else:
        with open(args.scores, 'r') as f:
            all_scores = json.load(f)

    print(f"Loaded {len(all_scores)} score entries")

    scores_df = pd.DataFrame([
        {'uttID': k, 'fluency': v.get('fluency'), 'prosodic': v.get('prosodic')}
        for k, v in all_scores.items()
    ])

    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        lmdb_keys = [key.decode('utf-8') for key in txn.cursor().iternext(values=False)]

    rows = []
    for key in lmdb_keys:
        spkID, uttID_wav = key.split('_', 1)
        uttID = uttID_wav.replace('.WAV', '').replace('.wav', '')
        identifier = key
        rows.append({'spkID': spkID, 'uttID': uttID, 'identifier': identifier})

    df_wavs = pd.DataFrame(rows)
    print(f"Found {len(df_wavs)} LMDB entries")

    df_merged = df_wavs.merge(scores_df, on='uttID', how='left')

    missing = df_merged['fluency'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} entries have no fluency score")

    def _assign_split(uttID):
        if uttID in train_utt_ids:
            return 'train'
        elif uttID in test_utt_ids:
            return 'test'
        return None

    df_merged['split'] = df_merged['uttID'].apply(_assign_split)
    unassigned = df_merged['split'].isna().sum()
    if unassigned > 0:
        print(f"Dropping {unassigned} entries not found in any utt2spk")
        df_merged = df_merged[df_merged['split'].notna()].copy()

    print(f"Split distribution: {df_merged['split'].value_counts().to_dict()}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    print(f"Saved {len(df_merged)} entries to {output_path}")


if __name__ == '__main__':
    main()
