import json
import pickle
import lmdb
import torch
import numpy as np
from pathlib import Path


MODEL_MAPPING = {}
try:
    from src.models import RhythmRegressor, DurationRegressor
    MODEL_MAPPING = {
        "rhythm_regressor": RhythmRegressor,
        "duration_regressor": DurationRegressor,
    }
except ImportError:
    pass


TEXT_PATHS = [
    '/hadatasets/aims/speechocean762/train/text',
    '/hadatasets/aims/speechocean762/test/text',
]


def load_model(config_path, checkpoint_path, device='cpu'):
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_type = config['model_type']
    model_params = dict(config['model_params'])

    if model_type == 'duration_regressor':
        vc_path = config['dataset_params'].get(
            'vc_features_path', 'data/speechocean/vc_features.json')
        with open(vc_path, 'r') as f:
            vc_data = json.load(f)
        num_tokens = len(vc_data['vocab'])
        samples = vc_data['samples']
        max_v = max(max(len(p) for p in s.get('v_phones', [])) for s in samples.values())
        max_c = max(max(len(p) for p in s.get('c_phones', [])) for s in samples.values())
        model_params['num_tokens'] = num_tokens
        model_params['max_phones'] = max(max_v, max_c)

    model_cls = MODEL_MAPPING[model_type]
    model = model_cls(**model_params).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def load_utterance_from_lmdb(lmdb_path, key, items=None):
    if items is None:
        items = ['envelope', 'waveform', 'intervals']
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        entry = pickle.loads(txn.get(key.encode('utf-8')))
    env.close()

    result = {}
    for item in items:
        if item in entry:
            if item == 'waveform':
                result[item] = np.array(entry[item], dtype=np.float32)
            else:
                result[item] = np.array(entry[item], dtype=np.float32)
    result['key'] = entry.get('key', key)
    return result


def load_transcriptions():
    utt_to_text = {}
    for path in TEXT_PATHS:
        if not Path(path).exists():
            continue
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                utt_id, text = line.split('\t', 1)
                utt_to_text[utt_id] = text
    return utt_to_text


def load_word_boundaries(spk_id, utt_id):
    tg_path = Path(f'data/speechocean/alignments/{spk_id}_{utt_id}.TextGrid')
    if not tg_path.exists():
        return []

    boundaries = []
    with open(tg_path, 'r') as f:
        lines = f.readlines()

    in_words_tier = False
    in_interval = False
    xmin = xmax = None
    text = None

    for line in lines:
        stripped = line.strip()
        if 'name = "words"' in stripped:
            in_words_tier = True
            continue
        if in_words_tier and 'name = "phones"' in stripped:
            break
        if not in_words_tier:
            continue

        if stripped.startswith('intervals ['):
            in_interval = True
            continue
        if in_interval and stripped.startswith('xmin = '):
            xmin = float(stripped.split('=')[1].strip())
            continue
        if in_interval and stripped.startswith('xmax = '):
            xmax = float(stripped.split('=')[1].strip())
            continue
        if in_interval and stripped.startswith('text = '):
            raw = stripped.split('=', 1)[1].strip().strip('"')
            text = raw if raw else None
            if text:
                boundaries.append((text, xmin, xmax))
            in_interval = False
            continue

    return boundaries


def load_vc_data(vc_features_path, lookup_key):
    with open(vc_features_path, 'r') as f:
        vc_data = json.load(f)
    samples = vc_data.get('samples', vc_data)
    return samples.get(lookup_key)


def load_vc_intervals(vc_alignments_path, lookup_key):
    with open(vc_alignments_path, 'r') as f:
        vc_align = json.load(f)
    entry = vc_align.get(lookup_key, {})
    vocalic = entry.get('vocalic', {})
    intervocalic = entry.get('intervocalic', {})
    return {
        'v_intervals': vocalic.get('intervals', []),
        'v_durations': vocalic.get('durations', []),
        'c_intervals': intervocalic.get('intervals', []),
        'c_durations': intervocalic.get('durations', []),
    }


def load_utterance_labels(metadata_path, identifier, label_column='fluency'):
    import pandas as pd
    df = pd.read_csv(metadata_path)
    match = df[df['identifier'] == identifier]
    if len(match) == 0:
        return None
    return match[label_column].values[0]


def build_vc_batch(vc_entry, device='cpu'):
    batch = {}
    for prefix in ('v', 'c'):
        phone_seqs = vc_entry[f'{prefix}_phones']
        n_intervals = len(phone_seqs)

        if n_intervals == 0:
            batch[f'{prefix}_phones'] = torch.zeros(1, 0, 0, dtype=torch.long, device=device)
            batch[f'{prefix}_phone_durs'] = torch.zeros(1, 0, 0, dtype=torch.float32, device=device)
            batch[f'{prefix}_dur'] = torch.zeros(1, 0, dtype=torch.float32, device=device)
            continue

        L_max = max(len(seq) for seq in phone_seqs)

        phones = torch.zeros(1, n_intervals, L_max, dtype=torch.long)
        for i, seq in enumerate(phone_seqs):
            phones[0, i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        batch[f'{prefix}_phones'] = phones.to(device)

        dur_seqs = vc_entry.get(f'{prefix}_phone_durs', [[0.0]*len(s) for s in phone_seqs])
        durs_padded = torch.zeros(1, n_intervals, L_max, dtype=torch.float32)
        for i, seq in enumerate(dur_seqs):
            durs_padded[0, i, :len(seq)] = torch.tensor(seq, dtype=torch.float32)
        batch[f'{prefix}_phone_durs'] = durs_padded.to(device)

        total_durs = vc_entry.get(f'{prefix}_dur', [0.0]*n_intervals)
        batch[f'{prefix}_dur'] = torch.tensor(total_durs, dtype=torch.float32).unsqueeze(0).to(device)

    return batch


def compute_envelope_time(envelope_length, waveform_length, sample_rate=16000):
    duration = waveform_length / sample_rate
    return np.linspace(0, duration, envelope_length)


def resolve_key(utt_id, spk_id='SPEAKER0001'):
    return f'{spk_id}_{utt_id}.WAV'


def get_envelope_downsample_factor(model_params):
    stride = model_params.get('cnn_stride', 1)
    num_layers = model_params.get('num_cnn_layers', 3)
    return stride ** num_layers
