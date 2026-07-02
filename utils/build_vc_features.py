"""
Build phone vocabulary and tokenized V/C features from MFA alignments.

Vocab contains individual phones only. Each interval stores a list of phone IDs.

Input:  data/speechocean/vc_alignments.json
Output: data/speechocean/vc_features.json

Usage:
    python utils/build_vc_features.py
"""

import json
from pathlib import Path


INPUT_PATH = Path('data/speechocean/vc_alignments.json')
OUTPUT_PATH = Path('data/speechocean/vc_features.json')


def main():
    with open(INPUT_PATH, 'r') as f:
        alignments = json.load(f)

    all_phones = set()
    for data in alignments.values():
        for category in ('vocalic', 'intervocalic'):
            for phones in data[category]['phones']:
                all_phones.update(phones)

    vocab = {'<pad>': 0}
    for i, ph in enumerate(sorted(all_phones), start=1):
        vocab[ph] = i

    result = {}
    for identifier, data in alignments.items():
        v_phone_ids = [[vocab[p] for p in phones] for phones in data['vocalic']['phones']]
        c_phone_ids = [[vocab[p] for p in phones] for phones in data['intervocalic']['phones']]
        result[identifier] = {
            'v_phones': v_phone_ids,
            'v_phone_durs': data['vocalic']['phone_durs'],
            'v_phone_durs_z': data['vocalic'].get('phone_durs_z', data['vocalic']['phone_durs']),
            'v_dur': data['vocalic']['durations'],
            'c_phones': c_phone_ids,
            'c_phone_durs': data['intervocalic']['phone_durs'],
            'c_phone_durs_z': data['intervocalic'].get('phone_durs_z', data['intervocalic']['phone_durs']),
            'c_dur': data['intervocalic']['durations'],
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump({'vocab': vocab, 'samples': result}, f)

    print(f"Vocab size: {len(vocab)}, saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
