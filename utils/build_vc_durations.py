"""
Extract duration-only V/C features from phone alignments.

Usage:
    python utils/build_vc_durations.py --input data/arctic/vc_alignments.json \
                                        --output data/arctic/vc_durations.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/arctic/vc_alignments.json')
    parser.add_argument('--output', type=str, default='data/arctic/vc_durations.json')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, 'r') as f:
        alignments = json.load(f)

    result = {}
    for identifier, data in alignments.items():
        result[identifier] = {
            'v_dur': data['vocalic']['durations'],
            'c_dur': data['intervocalic']['durations'],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved {len(result)} utterances -> {output_path}")


if __name__ == '__main__':
    main()
