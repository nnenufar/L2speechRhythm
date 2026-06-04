"""
Parse MFA TextGrid alignments into phone intervals JSON.

Reads all .TextGrid files from the MFA output, extracts phone-tier intervals,
and saves a JSON mapping identifier → {phones, intervals, durations}.

Usage:
    python utils/parse_alignments.py --input_dir data/speechocean/alignments \
                                      --output data/speechocean/phone_alignments.json
"""

import argparse
import json
from pathlib import Path
from praatio import textgrid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/speechocean/alignments')
    parser.add_argument('--output', type=str, default='data/speechocean/phone_alignments.json')
    args = parser.parse_args()

    alignments_dir = Path(args.input_dir)
    output_path = Path(args.output)

    tg_files = sorted(alignments_dir.glob('*/*.TextGrid'))
    if not tg_files:
        print(f"No .TextGrid files found in {alignments_dir}")
        return

    result = {}
    skipped = 0

    for tg_path in tg_files:
        spk_id = tg_path.parent.name
        identifier = f'{spk_id}_{tg_path.stem}'  # e.g. SPEAKER0001_000010011
        tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)

        try:
            phone_tier = tg.getTier('phones')
        except KeyError:
            skipped += 1
            continue

        phones = []
        intervals = []
        durations = []

        for entry in phone_tier.entries:
            label = entry.label.strip()
            if not label:
                continue
            phones.append(label)
            start = round(entry.start, 6)
            end = round(entry.end, 6)
            intervals.append([start, end])
            durations.append(round(end - start, 6))

        result[identifier] = {
            'phones': phones,
            'intervals': intervals,
            'durations': durations,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved {len(result)} alignments to {output_path}")
    if skipped:
        print(f"  Skipped {skipped} files missing 'phones' tier")


if __name__ == '__main__':
    main()
