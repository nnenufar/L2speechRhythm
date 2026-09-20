"""
Add z-scored per-phone durations to VC alignments.

Computes per-phone mean/std from phone_alignments.json, then adds
phone_durs_z fields to vc_alignments.json (parallel to phone_durs).

Usage:
    # Pipeline-integrated:
    python utils/add_phone_durs_z.py \
        --phone_alignments data/speechocean/phone_alignments.json \
        --vc_alignments data/speechocean/vc_alignments.json \
        --output data/speechocean/vc_alignments.json \
        --stats_output data/speechocean/phone_duration_stats.json

    # Standalone (after alignments already exist):
    python utils/add_phone_durs_z.py \
        --phone_alignments data/speechocean/phone_alignments.json \
        --vc_alignments data/speechocean/vc_alignments.json \
        --output data/speechocean/vc_alignments.json \
        --stats_output data/speechocean/phone_duration_stats.json
"""

import argparse
import json
from pathlib import Path


def compute_phone_stats(phone_alignments, std_floor=1e-6):
    durs_by_phone = {}

    for entry in phone_alignments.values():
        for ph, dur in zip(entry['phones'], entry['durations']):
            durs_by_phone.setdefault(ph, []).append(dur)

    stats = {}
    for ph, durs in durs_by_phone.items():
        n = len(durs)
        mean = sum(durs) / n
        if n == 1:
            std = std_floor
        else:
            var = sum((d - mean) ** 2 for d in durs) / n
            std = max(var ** 0.5, std_floor)
        stats[ph] = {'mean': round(mean, 6), 'std': round(std, 6), 'n': n}

    return stats


def add_z_scores(vc_alignments, stats):
    for entry in vc_alignments.values():
        for category in ('vocalic', 'intervocalic'):
            cat = entry[category]
            cat['phone_durs_z'] = [
                [round((d - stats[p]['mean']) / stats[p]['std'], 6)
                 for p, d in zip(phones, durs)]
                for phones, durs in zip(cat['phones'], cat['phone_durs'])
            ]


def main():
    parser = argparse.ArgumentParser(description='Add z-scored per-phone durations to VC alignments.')
    parser.add_argument('--phone_alignments', type=str, required=True,
                        help='Path to phone_alignments.json (flat per-phone data)')
    parser.add_argument('--vc_alignments', type=str, required=True,
                        help='Path to vc_alignments.json (input; will be augmented)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to write augmented vc_alignments.json')
    parser.add_argument('--stats_output', type=str, required=True,
                        help='Path to write phone_duration_stats.json')
    args = parser.parse_args()

    with open(args.phone_alignments, 'r') as f:
        phone_alignments = json.load(f)

    with open(args.vc_alignments, 'r') as f:
        vc_alignments = json.load(f)

    stats = compute_phone_stats(phone_alignments)
    print(f"Computed stats for {len(stats)} unique phones")

    add_z_scores(vc_alignments, stats)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(vc_alignments, f, indent=2)
    print(f"Saved augmented VC alignments to {output_path}")

    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved phone duration stats to {stats_path}")

    total_v = sum(len(v['vocalic']['phone_durs_z']) for v in vc_alignments.values())
    total_c = sum(len(v['intervocalic']['phone_durs_z']) for v in vc_alignments.values())
    print(f"Total V segments with z-scores: {total_v}, C segments: {total_c}")


if __name__ == '__main__':
    main()
