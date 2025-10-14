import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import pickle
import lmdb

data_sources = ['MSP']

def process_labels(labels: list):
    unique_labels = sorted(set(labels))
    str2int = {label: i for i, label in enumerate(unique_labels)}
    int2str = {i: label for label, i in str2int.items()}

    return str2int, int2str

def collate_fn(batch):
    """
    Padding function to handle variable sequence size
    """
    feats, intervals, labels = zip(*batch)
    feats_padded = pad_sequence(feats, batch_first = True, padding_value=0.0)
    intervals_padded = pad_sequence(intervals, batch_first = True, padding_value=0.0)

    labels = torch.stack(labels, dim=0)

    return feats_padded, intervals_padded, labels

class DatasetLMDB(Dataset):
    """
    Custom Dataset for data stored in lmdb format
    Input: .lmdb path, audio data source filepath
    """
    def __init__(self, lmdb_path, data_source):
        assert data_source in set(data_sources), "Invalid data source. Check data directory for available options."

        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        if data_source == 'MSP':
            labels_to_use = ['A', 'H', 'N', 'S']
            # Load the corresponding labels from a separate file.
            self.labels = pd.read_csv(f'data/{data_source}/labels_consensus.csv')
            self.labels = self.labels[self.labels['EmoClass'].isin(labels_to_use)]
            self.labels = self.labels.set_index('FileName')['EmoClass'].to_dict()

            with self.env.begin() as txn:
                self.lmdb_keys = [key for key in txn.cursor().iternext(values=False) if key.decode('utf-8') in self.labels.keys()]

        # Not all audios have an associated label entry.
        # Intersect keys to ensure we only use data that is present in both files.

        self.labels_map = process_labels(list(self.labels.values()))
        self.labels_str2int = self.labels_map[0]
        self.labels_int2str = self.labels_map[1]
        print(f'Label mapping: {self.labels_str2int}\n')

    def __len__(self):
        return len(self.lmdb_keys)

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            entry = pickle.loads(txn.get(self.lmdb_keys[idx]))
        
        key = entry['key']
        label = self.labels[key]
        label = self.labels_str2int.get(label)

        intervals = np.array(entry['intervals'], dtype = np.float32)
        feats = np.array(entry['feats'], dtype = np.float32)

        return torch.tensor(feats), torch.tensor(intervals), torch.tensor(label)



