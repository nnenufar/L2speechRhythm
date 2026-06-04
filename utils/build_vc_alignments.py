"""
Build vocalic (V) and intervocalic (C) segments from phone alignments.

V segments: consecutive vowels (ARPABET with optional stress markers).
C segments: everything between V segments (consonants + silence).

Usage:
    python utils/build_vc_alignments.py --input data/speechocean/phone_alignments.json \
                                         --output data/speechocean/vc_alignments.json
"""

import argparse
import json
import re
from pathlib import Path


VOWELS_REGEX = re.compile(
    r'(?:AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UW|UH)[012]?'
)

SILENCE_TOKENS = {'SIL', 'SP', 'SPN', '', '<SIL>', 'spn'}


def _is_vowel(phone):
    return bool(VOWELS_REGEX.fullmatch(phone))


def _is_silence(phone):
    return phone in SILENCE_TOKENS


def build_vc_segments(phones, intervals, durations):
    """Process a single utterance's phone sequence into V and C segments."""
    v_intervals = []
    v_durations = []
    v_phones = []

    c_intervals = []
    c_durations = []
    c_phones = []

    # Which segment type we're currently accumulating
    # True = V, False = C, None = haven't started yet
    in_v = None

    cur_start = None
    cur_end = None
    cur_dur = 0.0
    cur_phones = []

    for phone, (start, end), dur in zip(phones, intervals, durations):
        is_v = _is_vowel(phone) and not _is_silence(phone)

        if in_v is None:
            # First phone
            in_v = is_v
            cur_start = start
            cur_end = end
            cur_dur = dur
            cur_phones = [phone]
        elif in_v == is_v:
            # Continuing same segment type
            cur_end = end
            cur_dur += dur
            cur_phones.append(phone)
        else:
            # Transition: close current segment, start new one
            if in_v:
                v_intervals.append([cur_start, cur_end])
                v_durations.append(round(cur_dur, 6))
                v_phones.append(cur_phones)
            else:
                c_intervals.append([cur_start, cur_end])
                c_durations.append(round(cur_dur, 6))
                c_phones.append(cur_phones)

            in_v = is_v
            cur_start = start
            cur_end = end
            cur_dur = dur
            cur_phones = [phone]

    # Close final segment
    if cur_phones:
        if in_v:
            v_intervals.append([cur_start, cur_end])
            v_durations.append(round(cur_dur, 6))
            v_phones.append(cur_phones)
        else:
            c_intervals.append([cur_start, cur_end])
            c_durations.append(round(cur_dur, 6))
            c_phones.append(cur_phones)

    return {
        'vocalic': {
            'intervals': v_intervals,
            'durations': v_durations,
            'phones': v_phones,
        },
        'intervocalic': {
            'intervals': c_intervals,
            'durations': c_durations,
            'phones': c_phones,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/speechocean/phone_alignments.json')
    parser.add_argument('--output', type=str, default='data/speechocean/vc_alignments.json')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, 'r') as f:
        alignments = json.load(f)

    result = {}
    for identifier, data in alignments.items():
        result[identifier] = build_vc_segments(
            data['phones'], data['intervals'], data['durations']
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    total_v = sum(len(v['vocalic']['intervals']) for v in result.values())
    total_c = sum(len(v['intervocalic']['intervals']) for v in result.values())
    print(f"Saved {len(result)} utterances -> {output_path}")
    print(f"  Total V segments: {total_v}, C segments: {total_c}")


if __name__ == '__main__':
    main()
