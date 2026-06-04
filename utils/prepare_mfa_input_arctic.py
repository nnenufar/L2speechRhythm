"""
Prepare MFA input for ARCTIC: create .lab transcript files and .wav symlinks.

Reads transcripts from ARCTIC PROMPTS, reads LMDB keys for sample IDs,
writes per-speaker subdirectories with .lab and .wav per sample.

Usage:
    python utils/prepare_mfa_input_arctic.py
"""

import os
import sys
import pickle
import lmdb
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataloaders import parse_identifier


PROMPTS_PATH = '/hadatasets/joao.lima/data/arctic_cmu/PROMPTS'
LMDB_PATH = 'data/arctic/rtm_feats_bark_f0_egemaps.lmdb'
WAVS_CMU = Path('data/arctic/wavs_cmu')
WAVS_L2 = Path('data/arctic/wavs_l2')
OUTPUT_DIR = Path('data/arctic/mfa_input')


def load_transcripts():
    utt_to_text = {}
    with open(PROMPTS_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'\(\s*(\S+)\s+"(.*)"\s*\)', line)
            if match:
                utt_id = match.group(1).replace('arctic_', '')
                utt_to_text[utt_id] = match.group(2)
    print(f"Loaded {len(utt_to_text)} transcripts")
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
        spk_id, utt_id = parse_identifier(key)

        text = utt_to_text.get(utt_id)
        if text is None:
            missing_text += 1
            continue

        wav_filename = f'arctic_{utt_id}.wav'
        wav_src = WAVS_CMU / spk_id / wav_filename
        if not wav_src.exists():
            wav_src = WAVS_L2 / spk_id / wav_filename
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
