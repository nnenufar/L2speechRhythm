import torch
from transformers import AutoFeatureExtractor
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pickle
import lmdb
from src.data_sources import get_labels_for_source, DATA_SOURCE_HANDLERS

data_sources = list(DATA_SOURCE_HANDLERS.keys())

def process_labels(labels: list):
    """
    Process labels for classification or regression.
    For classification: creates string-to-int and int-to-string mappings.
    For regression (continuous values): creates identity mappings.
    """
    unique_labels = sorted(set(labels))
    
    # Check if labels are numeric (regression) or categorical (classification)
    # If all labels are numeric and there are many unique values, treat as regression
    try:
        numeric_labels = [float(l) for l in unique_labels]
        is_regression = True
    except (ValueError, TypeError):
        is_regression = False
    
    if is_regression:
        # For regression, map each unique value to itself (as float)
        str2int = {label: float(label) for label in unique_labels}
        int2str = {float(label): label for label in unique_labels}
    else:
        # For classification, create integer mappings
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

def processor_ssl(batch):
    """
    Collate function for SSL models (WavLM, etc.).
    Processes waveforms using the HuggingFace feature extractor.
    
    Args:
        batch: List of sample dicts from the dataset
    
    Returns:
        collated: Dict with processed waveforms and other collated fields
    """
    processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large",
                                                     cache_dir="/home/joao.lima/.cache/huggingface/hub/",
                                                     local_files_only=True)
    
    # batch is a list of sample dicts
    waveforms = [sample['waveform'] for sample in batch]
    
    # Process waveforms with the feature extractor
    processed = processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Build output dict with processed waveforms and other fields
    collated = {
        'waveform': processed.input_values,  # (B, T)
    }
    
    # Handle attention mask if present
    if hasattr(processed, 'attention_mask') and processed.attention_mask is not None:
        collated['attention_mask'] = processed.attention_mask
    
    # Collate other fields from the batch
    for key in batch[0].keys():
        if key == 'waveform':
            continue  # Already processed
        items = [sample[key] for sample in batch]
        if isinstance(items[0], str):
            collated[key] = items
        elif isinstance(items[0], torch.Tensor):
            if items[0].dim() > 0:
                collated[key] = torch.nn.utils.rnn.pad_sequence(items, batch_first=True)
            else:
                collated[key] = torch.stack(items)
        elif isinstance(items[0], np.ndarray):
            collated[key] = torch.tensor(np.stack(items))
        else:
            collated[key] = items
    
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
    def __init__(self, lmdb_path, data_source, split, items, test_size=0.2, random_seed=42,
                 target_mean=None, target_std=None,
                 external_utterance_str2int=None, external_utterance_int2str=None):
        possible_splits = ['Train', 'Development', 'Test']
        possible_items = ['key', 'beats', 'envelope_spectrum', 'feats', 'intervals', 'envelope', 'spectrum_freq_bins', 'dur', 'waveform', 'f0', 'voiced_mask', 'egemaps', 'f0_wavelet']
        assert data_source in set(data_sources), "Invalid data source. Check /data directory for available options."
        assert split in possible_splits, f"Invalid split. Must be one of {possible_splits}"
        assert set(items).issubset(possible_items), f"Invalid items. Must be one of {possible_items}"

        self.lmdb_path = lmdb_path
        self.split = split
        self.random_seed = random_seed
        self.test_size = test_size
        self.items = items
        
        # Target normalization for regression tasks
        self.target_mean = target_mean
        self.target_std = target_std

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        # Get labels for the specified data source
        self.labels = get_labels_for_source(data_source, split, test_size, random_seed) #Dict{identifier (as in lmdb): label}
        self.lmdb_keys = subset_lmdb(self.env, self.labels.keys())

        self.labels_map = process_labels(list(self.labels.values()))
        self.labels_str2int = self.labels_map[0]
        self.labels_int2str = self.labels_map[1]

        # Process identifiers into speaker and utterance mappings
        identifiers = [key.decode('utf-8') for key in self.lmdb_keys]

        if external_utterance_str2int is not None and external_utterance_int2str is not None:
            self.utterance_str2int = external_utterance_str2int
            self.utterance_int2str = external_utterance_int2str
        else:
            _, _, utterance_str2int, utterance_int2str = process_identifiers(identifiers)
            self.utterance_str2int = utterance_str2int
            self.utterance_int2str = utterance_int2str

        # Speaker mappings are always computed from the local split
        speaker_ids = [parse_identifier(id)[0] for id in identifiers]
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
            if item == "waveform":
                result[item] = np.array(entry[item], dtype=np.float32)
            else:
                result[item] = torch.tensor(np.array(entry[item], dtype=np.float32))

        label = self.labels[key]
        label = self.labels_str2int.get(label)
        
        # For regression: normalize target if stats are provided, use float
        # For classification: use integer label
        if self.target_mean is not None and self.target_std is not None:
            # Regression with normalization
            label = (float(label) - self.target_mean) / (self.target_std + 1e-8)
            result['label'] = torch.tensor(label, dtype=torch.float32)
        elif isinstance(label, float):
            # Regression without normalization
            result['label'] = torch.tensor(label, dtype=torch.float32)
        else:
            # Classification
            result['label'] = torch.tensor(label, dtype=torch.long)

        # Add speaker ID and identifier as integers
        speaker_id, utterance_id = parse_identifier(key)
        result['identifier'] = key
        result['speaker_id'] = torch.tensor(self.speaker_str2int[speaker_id])
        result['utterance_id'] = torch.tensor(self.utterance_str2int.get(utterance_id, 0))

        return result



