"""
Prepare MFA input for Speechocean: create .lab transcript files and .wav symlinks.

Reads transcripts from Speechocean text files, reads LMDB keys for sample IDs,
writes one .lab and one .wav symlink per sample to mfa_input/.

Usage:
    python utils/prepare_mfa_input.py
"""

import os
import pickle
import lmdb
from pathlib import Path
from collections import defaultdict


TEXT_PATHS = [
    '/hadatasets/aims/speechocean762/test/text',
    '/hadatasets/aims/speechocean762/train/text',
]
LMDB_PATH = 'data/speechocean/rtm_feats.lmdb'
WAVS_DIR = Path('data/speechocean/wavs')
OUTPUT_DIR = Path('data/speechocean/mfa_input')


def load_transcripts():
    utt_to_text = {}
    seen = set()
    for path in TEXT_PATHS:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                utt_id, text = line.split('\t', 1)
                if utt_id in seen:
                    continue
                seen.add(utt_id)
                utt_to_text[utt_id] = text
    print(f"Loaded {len(utt_to_text)} unique transcripts")
    return utt_to_text


def main():
    utt_to_text = load_transcripts()

    env = lmdb.open(LMDB_PATH, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        lmdb_keys = [key.decode('utf-8') for key in txn.cursor().iternext(values=False)]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing_text = 0
    missing_wav = 0
    created = 0

    for key in lmdb_keys:
        spk_id, utt_id_wav = key.split('_', 1)
        utt_id = utt_id_wav.replace('.WAV', '').replace('.wav', '')

        text = utt_to_text.get(utt_id)
        if text is None:
            missing_text += 1
            continue

        wav_src = WAVS_DIR / spk_id / f'{utt_id}.WAV'
        if not wav_src.exists():
            missing_wav += 1
            continue

        spk_dir = OUTPUT_DIR / spk_id
        spk_dir.mkdir(exist_ok=True)

        lab_path = spk_dir / f'{utt_id}.lab'
        with open(lab_path, 'w') as f:
            f.write(text + '\n')

        wav_dst = spk_dir / f'{utt_id}.wav'
        if not wav_dst.exists():
            os.symlink(wav_src.resolve(), wav_dst)

        created += 1

    print(f"Created {created} .lab + .wav pairs in {OUTPUT_DIR}")
    if missing_text:
        print(f"  Missing transcripts: {missing_text}")
    if missing_wav:
        print(f"  Missing wav files: {missing_wav}")


if __name__ == '__main__':
    main()
