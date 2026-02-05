import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pickle
import lmdb
from src.data_sources import get_labels_for_source, DATA_SOURCE_HANDLERS

data_sources = list(DATA_SOURCE_HANDLERS.keys())

def process_labels(labels: list):
    unique_labels = sorted(set(labels))
    str2int = {label: i for i, label in enumerate(unique_labels)}
    int2str = {i: label for label, i in str2int.items()}

    return str2int, int2str

def parse_identifier(identifier: str):
    """
    Parse an identifier string into speaker_id and utterance_id.
    Expected format: <speaker_id>_<dataset_name>_<utterance_id>
    Example: YBAA_arctic_a0503.wav -> ('YBAA', 'a0503.wav')
    
    Args:
        identifier: String identifier in the format <speaker_id>_<dataset_name>_<utterance_id>
    
    Returns:
        speaker_id: The speaker identifier.
        utterance_id: The utterance identifier.
    """
    parts = identifier.split('_')
    speaker_id = parts[0]
    utterance_id = Path('_'.join(parts[2:])).stem   # In case utterance_id contains underscores
    return speaker_id, utterance_id

def process_identifiers(identifiers: list):
    """
    Create separate mappings for speaker IDs and utterance IDs.
    
    Args:
        identifiers: List of string identifiers in format <speaker_id>_<dataset_name>_<utterance_id>
    
    Returns:
        speaker_str2int: Dictionary mapping speaker IDs to integers.
        speaker_int2str: Dictionary mapping integers to speaker IDs.
        utterance_str2int: Dictionary mapping utterance IDs to integers.
        utterance_int2str: Dictionary mapping integers to utterance IDs.
    """
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

def collate_fn(batch):
    """
    Simple collate function for dictionary batches.
    Pads sequences and stacks tensors.
    """
    collated = {}
    for key in batch[0].keys():
        items = [sample[key] for sample in batch]
        # Skip string items - don't collate them, just keep as list
        if isinstance(items[0], str):
            collated[key] = items
        elif items[0].dim() > 0:
            collated[key] = torch.nn.utils.rnn.pad_sequence(items, batch_first=True)
        else:
            collated[key] = torch.stack(items)
    return collated

def subset_lmdb(env, keys_to_use):
    with env.begin() as txn:
        lmdb_keys = [key for key in txn.cursor().iternext(values=False) if key.decode('utf-8') in keys_to_use]
        return lmdb_keys

class DatasetLMDB(Dataset):
    """
    Custom Dataset for data stored in lmdb format
    Input: .lmdb path, audio data source filepath
    """
    def __init__(self, lmdb_path, data_source, split, items, test_size=0.2, random_seed=42):
        possible_splits = ['Train', 'Development', 'Test']
        possible_items = ['key', 'beats', 'envelope_spectrum', 'feats', 'intervals', 'envelope', 'spectrum_freq_bins', 'dur']
        assert data_source in set(data_sources), "Invalid data source. Check /data directory for available options."
        assert split in possible_splits, f"Invalid split. Must be one of {possible_splits}"
        assert set(items).issubset(possible_items), f"Invalid items. Must be one of {possible_items}"

        self.lmdb_path = lmdb_path
        self.split = split
        self.random_seed = random_seed
        self.test_size = test_size
        self.items = items

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        # Get labels for the specified data source
        self.labels = get_labels_for_source(data_source, split, test_size, random_seed) #Dict{identifier (as in lmdb): label}
        self.lmdb_keys = subset_lmdb(self.env, self.labels.keys())

        self.labels_map = process_labels(list(self.labels.values()))
        self.labels_str2int = self.labels_map[0]
        self.labels_int2str = self.labels_map[1]

        # Process identifiers into speaker and utterance mappings
        identifiers = [key.decode('utf-8') for key in self.lmdb_keys]
        speaker_str2int, speaker_int2str, utterance_str2int, utterance_int2str = process_identifiers(identifiers)
        self.speaker_str2int = speaker_str2int
        self.speaker_int2str = speaker_int2str
        self.utterance_str2int = utterance_str2int
        self.utterance_int2str = utterance_int2str


    def __len__(self):
        return len(self.lmdb_keys)

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            entry = pickle.loads(txn.get(self.lmdb_keys[idx]))
        result = {}
        
        key = entry['key']

        for item in self.items:
            result[item] = torch.tensor(np.array(entry[item], dtype=np.float32))  

        label = self.labels[key]
        label = self.labels_str2int.get(label)
        result['label'] = torch.tensor(label)

        # Add speaker ID and identifier as integers
        speaker_id, utterance_id = parse_identifier(key)
        result['identifier'] = key
        result['speaker_id'] = torch.tensor(self.speaker_str2int[speaker_id])
        result['utterance_id'] = torch.tensor(self.utterance_str2int[utterance_id])

        return result



