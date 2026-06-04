"""
Build phoneme mapping from ARCTIC PROMPTS file.
Reads transcripts, converts to ARPABET phonemes via g2p-en, builds vocabulary,
and saves a JSON mapping file.

Usage:
    python utils/build_phoneme_mapping.py \
        --prompts /hadatasets/joao.lima/data/arctic_cmu/PROMPTS \
        --output data/arctic/phoneme_mapping.json
"""

import argparse
import json
import re
from pathlib import Path


def parse_prompts(filepath):
    """Parse PROMPTS file, return dict {utt_id: transcript}."""
    mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'\(\s*(\S+)\s+"(.*)"\s*\)', line)
            if match:
                utt_id = match.group(1).replace('arctic_', '')
                transcript = match.group(2)
                mapping[utt_id] = transcript
    return mapping


def build_vocab(phoneme_sequences):
    """Build phoneme vocabulary from list of phoneme sequences."""
    all_phonemes = set()
    for seq in phoneme_sequences:
        all_phonemes.update(seq)
    vocab = {'<pad>': 0}
    for i, ph in enumerate(sorted(all_phonemes)):
        vocab[ph] = i + 1
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=str,
                        default='/hadatasets/joao.lima/data/arctic_cmu/PROMPTS')
    parser.add_argument('--output', type=str,
                        default='data/arctic/phoneme_mapping.json')
    args = parser.parse_args()

    from g2p_en import G2p
    g2p = G2p()

    transcripts = parse_prompts(args.prompts)
    print(f"Parsed {len(transcripts)} utterances from {args.prompts}")

    phoneme_sequences = {}
    for utt_id, text in sorted(transcripts.items()):
        phonemes = g2p(text)
        phonemes = [p for p in phonemes if p not in (' ', '')]
        phoneme_sequences[utt_id] = phonemes

    vocab = build_vocab(phoneme_sequences.values())
    print(f"Phoneme vocabulary size: {len(vocab)}")

    mapping = {utt_id: [vocab[p] for p in seq] for utt_id, seq in phoneme_sequences.items()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'vocab': vocab, 'mapping': mapping}, f)

    print(f"Saved phoneme mapping to {output_path}")


if __name__ == "__main__":
    main()
