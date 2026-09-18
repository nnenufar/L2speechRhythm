import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_sources import DATA_SOURCE_HANDLERS, get_labels_for_source


def process_labels(labels):
    unique_labels = sorted(set(labels))
    try:
        numeric_labels = [float(label) for label in unique_labels]
        is_regression = True
    except (TypeError, ValueError):
        is_regression = False

    if is_regression:
        str2int = {label: float(label) for label in unique_labels}
        int2str = {float(label): label for label in unique_labels}
    else:
        str2int = {label: i for i, label in enumerate(unique_labels)}
        int2str = {i: label for label, i in str2int.items()}
    return str2int, int2str


def parse_identifier(identifier):
    parts = identifier.split('_')
    speaker_id = parts[0]
    if len(parts) >= 3:
        utterance_id = Path('_'.join(parts[2:])).stem
    else:
        utterance_id = Path('_'.join(parts[1:])).stem
    return speaker_id, utterance_id


def process_identifiers(identifiers):
    speaker_ids = []
    utterance_ids = []
    for identifier in identifiers:
        speaker_id, utterance_id = parse_identifier(identifier)
        speaker_ids.append(speaker_id)
        utterance_ids.append(utterance_id)

    unique_speakers = sorted(set(speaker_ids))
    speaker_str2int = {spk: i for i, spk in enumerate(unique_speakers)}
    speaker_int2str = {i: spk for spk, i in speaker_str2int.items()}

    unique_utterances = sorted(set(utterance_ids))
    utterance_str2int = {utt: i for i, utt in enumerate(unique_utterances)}
    utterance_int2str = {i: utt for utt, i in utterance_str2int.items()}
    return speaker_str2int, speaker_int2str, utterance_str2int, utterance_int2str


def _pad_2d(items, pad_value=0):
    N_max = max(t.shape[0] for t in items)
    L_max = max((t.shape[1] for t in items if t.numel() > 0), default=1)
    out = torch.full((len(items), N_max, L_max), pad_value, dtype=items[0].dtype)
    for i, t in enumerate(items):
        if t.numel() == 0:
            continue
        N, L = t.shape
        out[i, :N, :L] = t
    return out


def collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        items = [sample[key] for sample in batch]
        if isinstance(items[0], str):
            collated[key] = items
        elif items[0].dim() == 2:
            collated[key] = _pad_2d(
                [it.long() if it.dtype != torch.float32 else it for it in items],
                pad_value=0,
            )
        elif items[0].dim() == 1:
            collated[key] = torch.nn.utils.rnn.pad_sequence(items, batch_first=True)
        else:
            collated[key] = torch.stack(items)
    return collated


def subset_lmdb(env, keys_to_use):
    with env.begin() as txn:
        return [
            key for key in txn.cursor().iternext(values=False)
            if key.decode('utf-8') in keys_to_use
        ]


class DatasetLMDB(Dataset):
    def __init__(self, lmdb_path, data_source, split, items, test_size=0.2, random_seed=42,
                 target_mean=None, target_std=None,
                 external_utterance_str2int=None, external_utterance_int2str=None,
                 phoneme_mapping=None, vc_features=None, label_column='fluency',
                 env=None):
        possible_splits = ['Train', 'Development', 'Test']
        possible_items = [
            'key', 'beats', 'envelope_spectrum', 'envelope_derivative', 'feats',
            'intervals', 'envelope', 'spectrum_freq_bins', 'dur', 'waveform', 'f0',
            'voiced_mask', 'egemaps', 'f0_wavelet',
            'v_phones', 'v_plen', 'v_phone_durs', 'v_phone_durs_z', 'v_dur',
            'c_phones', 'c_plen', 'c_phone_durs', 'c_phone_durs_z', 'c_dur',
        ]
        assert data_source in set(DATA_SOURCE_HANDLERS)
        assert split in possible_splits
        assert set(items).issubset(possible_items)

        self.lmdb_path = lmdb_path
        self.split = split
        self.random_seed = random_seed
        self.test_size = test_size
        self.items = items
        self.phoneme_mapping = phoneme_mapping
        self.vc_features = vc_features
        self.target_mean = target_mean
        self.target_std = target_std

        self.env = env if env is not None else lmdb.open(
            lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        )
        self.labels = get_labels_for_source(
            data_source, split, test_size, random_seed, label_column=label_column
        )
        self.lmdb_keys = subset_lmdb(self.env, self.labels.keys())

        self.labels_str2int, self.labels_int2str = process_labels(list(self.labels.values()))

        identifiers = [key.decode('utf-8') for key in self.lmdb_keys]
        if external_utterance_str2int is not None and external_utterance_int2str is not None:
            self.utterance_str2int = external_utterance_str2int
            self.utterance_int2str = external_utterance_int2str
        else:
            _, _, self.utterance_str2int, self.utterance_int2str = process_identifiers(identifiers)

        speaker_ids = [parse_identifier(identifier)[0] for identifier in identifiers]
        unique_speakers = sorted(set(speaker_ids))
        self.speaker_str2int = {spk: i for i, spk in enumerate(unique_speakers)}
        self.speaker_int2str = {i: spk for spk, i in self.speaker_str2int.items()}

    def __len__(self):
        return len(self.lmdb_keys)

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            entry = pickle.loads(txn.get(self.lmdb_keys[idx]))

        result = {}
        key = entry['key']
        for item in self.items:
            if item in (
                'v_phones', 'v_plen', 'v_phone_durs', 'v_phone_durs_z', 'v_dur',
                'c_phones', 'c_plen', 'c_phone_durs', 'c_phone_durs_z', 'c_dur',
            ):
                continue
            if item == 'waveform':
                result[item] = np.array(entry[item], dtype=np.float32)
            else:
                result[item] = torch.tensor(np.array(entry[item], dtype=np.float32))

        label = self.labels[key]
        label = self.labels_str2int.get(label)
        if self.target_mean is not None and self.target_std is not None:
            label = (float(label) - self.target_mean) / self.target_std
            result['label'] = torch.tensor(label, dtype=torch.float32)
        elif isinstance(label, float):
            result['label'] = torch.tensor(label, dtype=torch.float32)
        else:
            result['label'] = torch.tensor(label, dtype=torch.long)

        speaker_id, utterance_id = parse_identifier(key)
        result['identifier'] = key
        result['speaker_id'] = torch.tensor(self.speaker_str2int[speaker_id])
        result['utterance_id'] = torch.tensor(self.utterance_str2int.get(utterance_id, 0))

        vc_items_needed = any(item in self.items for item in (
            'v_phones', 'v_plen', 'v_phone_durs', 'v_phone_durs_z', 'v_dur',
            'c_phones', 'c_plen', 'c_phone_durs', 'c_phone_durs_z', 'c_dur',
        ))
        if vc_items_needed:
            spk_id, utt_id = parse_identifier(key)
            lookup_key = f'{spk_id}_{utt_id}'
            vc = self.vc_features.get(lookup_key) if self.vc_features is not None else None
            if vc is not None:
                has_phones = 'v_phones' in vc
                for prefix in ('v', 'c'):
                    cat = f'{prefix}_phones'
                    if has_phones:
                        intervals = vc[cat]
                        if intervals:
                            L_max = max(len(seq) for seq in intervals)
                            padded = torch.zeros(len(intervals), L_max, dtype=torch.long)
                            plen = torch.zeros(len(intervals), dtype=torch.long)
                            for i, seq in enumerate(intervals):
                                padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                                plen[i] = len(seq)
                            result[cat] = padded
                            result[f'{prefix}_plen'] = plen
                        else:
                            result[cat] = torch.zeros(0, 0, dtype=torch.long)
                            result[f'{prefix}_plen'] = torch.zeros(0, dtype=torch.long)

                        for dur_key in (f'{prefix}_phone_durs', f'{prefix}_phone_durs_z'):
                            seqs = vc.get(dur_key, [])
                            pdur = torch.zeros(len(intervals), L_max, dtype=torch.float32)
                            for i, seq in enumerate(seqs):
                                pdur[i, :len(seq)] = torch.tensor(seq, dtype=torch.float32)
                            result[dur_key] = pdur
                    else:
                        result[cat] = torch.zeros(0, 0, dtype=torch.long)
                        result[f'{prefix}_plen'] = torch.zeros(0, dtype=torch.long)
                        result[f'{prefix}_phone_durs'] = torch.zeros(0, 0, dtype=torch.float32)
                        result[f'{prefix}_phone_durs_z'] = torch.zeros(0, 0, dtype=torch.float32)
                    result[f'{prefix}_dur'] = torch.tensor(vc[f'{prefix}_dur'], dtype=torch.float32)
            else:
                for prefix in ('v', 'c'):
                    result[f'{prefix}_phones'] = torch.zeros(0, 0, dtype=torch.long)
                    result[f'{prefix}_plen'] = torch.zeros(0, dtype=torch.long)
                    result[f'{prefix}_phone_durs'] = torch.zeros(0, 0, dtype=torch.float32)
                    result[f'{prefix}_phone_durs_z'] = torch.zeros(0, 0, dtype=torch.float32)
                    result[f'{prefix}_dur'] = torch.zeros(0, dtype=torch.float32)
        return result
