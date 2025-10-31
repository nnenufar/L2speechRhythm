import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
import pickle
import lmdb

data_sources = ['MSP', 'globo', 'mtedx']

def process_labels(labels: list):
    unique_labels = sorted(set(labels))
    str2int = {label: i for i, label in enumerate(unique_labels)}
    int2str = {i: label for label, i in str2int.items()}

    return str2int, int2str

def collate_fn(batch):
    """
    Simple collate function for dictionary batches.
    Pads sequences and stacks tensors.
    """
    # Get all keys from first item
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        items = [item[key] for item in batch]
        if items[0].dim() > 0:
            # Pad sequences
            collated[key] = pad_sequence(items, batch_first=True, padding_value=0.0)
        else:
            # Stack scalars
            collated[key] = torch.stack(items, dim=0)
    
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
        possible_items = ['key', 'beats', 'envelope_spectrum', 'feats', 'intervals']
        assert data_source in set(data_sources), "Invalid data source. Check /data directory for available options."
        assert split in possible_splits, f"Invalid split. Must be one of {possible_splits}"
        assert set(items).issubset(possible_items), f"Invalid items. Must be one of {possible_items}"

        self.lmdb_path = lmdb_path
        self.split = split
        self.random_seed = random_seed
        self.test_size = test_size
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.items = items

        if data_source == 'MSP':
            labels_to_use = ['A', 'H', 'N', 'S']
            # Load the corresponding labels from a separate file.
            labels_df = pd.read_csv(f'data/{data_source}/labels_consensus.csv')
            labels_df = labels_df[labels_df['EmoClass'].isin(labels_to_use)]
            dev_df = labels_df[labels_df['Split_Set'] == 'Development']

            dev_files, test_files = train_test_split(
                dev_df['FileName'].values,
                test_size=self.test_size,
                random_state=self.random_seed,
                stratify=dev_df['EmoClass'].values
            )
    
            if self.split is not None:
                labels_df = labels_df[labels_df['Split_Set'] == self.split] # Train files
                if self.split == 'Development':
                    labels_df = labels_df[labels_df['FileName'].isin(dev_files)]
                elif self.split == 'Test':
                    labels_df = dev_df[dev_df['FileName'].isin(test_files)]

            self.labels = labels_df.set_index('FileName')['EmoClass'].to_dict()

            self.lmdb_keys = subset_lmdb(self.env, self.labels.keys())

        if data_source == 'globo':
            train_files, testValid_files = train_test_split(
                os.listdir(f'data/{data_source}/wavs'),
                test_size=self.test_size, # Default 80% train, 20% Valid+Test
                random_state=self.random_seed
            )
            valid_files, test_files = train_test_split(
                testValid_files,
                test_size=0.5, # 10% valid, 10% test
                random_state=self.random_seed
            )

            split_map = {"Train": train_files,
                         "Development": valid_files,
                         "Test": test_files}
            
            keys = split_map.get(self.split)
            self.labels = {key:"J" for key in keys} # All files are assigned the same label
            self.lmdb_keys = subset_lmdb(self.env, self.labels.keys())

        if data_source == 'mtedx':
            base_path = 'data/mtedx/'

            train_files = os.listdir(base_path + 'train/wav')
            valid_files = os.listdir(base_path + 'valid/wav')
            test_files = os.listdir(base_path + 'test/wav')

            split_map = {"Train": train_files,
                         "Development": valid_files,
                         "Test": test_files}
            
            keys = split_map.get(self.split)
            self.labels = {key:"P" for key in keys} # All files are assigned the same label
            self.lmdb_keys = subset_lmdb(self.env, self.labels.keys())


        self.labels_map = process_labels(list(self.labels.values()))
        self.labels_str2int = self.labels_map[0]
        self.labels_int2str = self.labels_map[1]

    def __len__(self):
        return len(self.lmdb_keys)

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            entry = pickle.loads(txn.get(self.lmdb_keys[idx]))
        result = {}
        
        key = entry['key']
        if 'feats' in self.items:
            feats = np.array(entry['feats'], dtype=np.float32)
            result['feats'] = torch.tensor(feats)
        if 'intervals' in self.items:
            intervals = np.array(entry['intervals'], dtype=np.float32)
            result['intervals'] = torch.tensor(intervals)
        if 'envelope_spectrum' in self.items:
            result['envelope_spectrum'] = torch.tensor(entry['envelope_spectrum'], dtype=torch.float32)
            result['spectrum_freq_bins'] = torch.tensor(entry['spectrum_freq_bins'], dtype=torch.float32)     

        label = self.labels[key]
        label = self.labels_str2int.get(label)
        result['label'] = torch.tensor(label)

        return result



