import argparse
import json
import pickle
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 14,         # default text
    'axes.titlesize': 18,    # title
    'axes.labelsize': 16,    # axis labels
    'xtick.labelsize': 14,   # x tick labels
    'ytick.labelsize': 14,   # y tick labels
    'legend.fontsize': 14    # legend
})


def parse_identifier(identifier: str):
    """
    Parse an identifier string into speaker_id and utterance_id.
    Supports two formats:
      <speaker_id>_<dataset_name>_<utterance_id>   (e.g. ljm_arctic_a0572.wav)
      <speaker_id>_<utterance_id>                  (e.g. SPEAKER0001_000010011.WAV)
    """
    parts = identifier.replace('.WAV', '').split('_')
    speaker_id = parts[0]
    if len(parts) >= 3:
        utterance_id = '_'.join(parts[2:])
    else:
        utterance_id = '_'.join(parts[1:])
    return speaker_id, utterance_id


def load_sample(lmdb_path, key):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        entry = pickle.loads(txn.get(key.encode('utf-8')))
    env.close()

    result = {}
    for item in ['waveform', 'envelope', 'envelope_derivative']:
        if item in entry:
            result[item] = np.array(entry[item], dtype=np.float32)
    result['key'] = entry.get('key', key)
    return result


def load_vc_intervals(vc_alignments_path, lookup_key):
    if not vc_alignments_path or not Path(vc_alignments_path).exists():
        return None

    with open(vc_alignments_path, 'r') as f:
        vc_align = json.load(f)
    entry = vc_align.get(lookup_key, {})
    vocalic = entry.get('vocalic', {})
    intervocalic = entry.get('intervocalic', {})
    return {
        'v_intervals': vocalic.get('intervals', []),
        'v_phones': vocalic.get('phones', []),
        'v_phone_durs': vocalic.get('phone_durs', []),
        'c_intervals': intervocalic.get('intervals', []),
        'c_phones': intervocalic.get('phones', []),
        'c_phone_durs': intervocalic.get('phone_durs', []),
    }


def shade_intervals(ax, v_intervals, c_intervals, v_color='salmon', c_color='lightblue', alpha=0.35):
    for t_start, t_end in v_intervals:
        ax.axvspan(t_start, t_end, color=v_color, alpha=alpha, zorder=0)
    for t_start, t_end in c_intervals:
        ax.axvspan(t_start, t_end, color=c_color, alpha=alpha, zorder=0)


def label_phones(ax, intervals, phones_seqs, phone_durs_seqs, y=0.98, color='black', fontsize=6):
    for (t_start, t_end), phones, durs in zip(intervals, phones_seqs, phone_durs_seqs):
        offset = t_start
        for phone, dur in zip(phones, durs):
            mid = offset + dur / 2
            ax.text(mid, y, phone, transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=10, color=color, zorder=5)
            offset += dur


def plot_sample(sample, vc_intervals, output_path=None, sample_rate=16000):
    waveform = sample.get('waveform')
    envelope = sample.get('envelope')
    envelope_derivative = sample.get('envelope_derivative')

    # Infer utterance duration from whichever signal is available
    if waveform is not None:
        duration = len(waveform) / sample_rate
    elif envelope is not None:
        duration = len(envelope) / sample_rate
    else:
        raise ValueError("No waveform or envelope found in sample")

    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=False, gridspec_kw={'hspace': 0.003, 'height_ratios': [2, 1.5, 1.5]})

    plt.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.95,
        bottom=0.09,
        hspace=0
    )

    # Row 1: Waveform
    ax = axes[0]
    if waveform is not None:
        t_wav = np.linspace(0, duration, len(waveform))
        ax.plot(t_wav, waveform, color='black', linewidth=0.5)
    if vc_intervals:
        shade_intervals(ax, vc_intervals['v_intervals'], vc_intervals['c_intervals'])
        label_phones(ax, vc_intervals['v_intervals'], vc_intervals['v_phones'],
                     vc_intervals['v_phone_durs'], y=0.98, color='darkred')
        label_phones(ax, vc_intervals['c_intervals'], vc_intervals['c_phones'],
                     vc_intervals['c_phone_durs'], y=0.86, color='darkblue')
    ax.set_ylabel('Amplitude')
    ax.set_title('Input Sample Visualization')

    # Row 2: Envelope
    ax = axes[1]
    if envelope is not None:
        t_env = np.linspace(0, duration, len(envelope))
        ax.plot(t_env, envelope, color='darkgreen', linewidth=1.5)
    if vc_intervals:
        shade_intervals(ax, vc_intervals['v_intervals'], vc_intervals['c_intervals'])
    ax.set_ylabel('Envelope')

    # Row 3: Envelope derivative
    ax = axes[2]
    if envelope_derivative is not None:
        t_der = np.linspace(0, duration, len(envelope_derivative))
        ax.plot(t_der, envelope_derivative, color='#6a0dad', linewidth=1.5)
    if vc_intervals:
        shade_intervals(ax, vc_intervals['v_intervals'], vc_intervals['c_intervals'])
    ax.set_ylabel(f'Envelope\nderivative')
    ax.set_xlabel('Time (s)')

    # Legend for V/C shading
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor='salmon', alpha=0.35, label='V (vocalic)'),
        Patch(facecolor='lightblue', alpha=0.35, label='C (intervocalic)'),
    ]
    axes[0].legend(handles=legend_patches, loc='upper right', fontsize=12)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
            ax.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False
            )
            ax.set_yticks([])

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize a speech sample with waveform, envelope, and envelope derivative.')
    parser.add_argument('--lmdb_path', type=str, required=True,
                        help='Path to the LMDB database (e.g. data/speechocean/rtm_feats.lmdb)')
    parser.add_argument('--key', type=str, required=True,
                        help='Sample identifier as stored in the LMDB (e.g. SPEAKER0001_000010011.WAV)')
    parser.add_argument('--vc_alignments', type=str, default=None,
                        help='Path to vc_alignments.json for V/C interval shading')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the figure (e.g. plots/sample.png). If not provided, displays interactively.')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate (default: 16000)')

    args = parser.parse_args()

    sample = load_sample(args.lmdb_path, args.key)

    vc_intervals = None
    if args.vc_alignments:
        spk_id, utt_id = parse_identifier(args.key)
        lookup_key = f'{spk_id}_{utt_id}'
        vc_intervals = load_vc_intervals(args.vc_alignments, lookup_key)
        if vc_intervals:
            n_v = len(vc_intervals['v_intervals'])
            n_c = len(vc_intervals['c_intervals'])
            print(f'Loaded {n_v} V intervals, {n_c} C intervals')
        else:
            print(f'No VC intervals found for key "{lookup_key}" in {args.vc_alignments}')

    plot_sample(sample, vc_intervals, args.output, args.sample_rate)


if __name__ == '__main__':
    main()
