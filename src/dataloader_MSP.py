import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

def process_labels(labels: list):
    unique_labels = sorted(set(labels))
    str2int = {label: i for i, label in enumerate(unique_labels)}
    int2str = {i: label for label, i in str2int.items()}

    return str2int, int2str

def collate_fn(batch):
    """
    Padding function to handle variable sequence size
    """
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first = True, padding_value=0.0)
    labels = torch.stack(labels, dim=0)

    return sequences_padded, labels

class NpzDataset(Dataset):
    """
    Custom Dataset for data stored in a single .npz archive.
    Input: .npz timestamps filepath, MSP labels_consensus.csv filepath.
    """
    def __init__(self, npz_path, labels_path):
        # Load the entire .npz file into memory. It behaves like a dictionary.
        self.sequences_data = np.load(npz_path)

        # Load the corresponding labels from a separate file.
        self.labels = pd.read_csv(labels_path)
        self.labels = self.labels.set_index('FileName')['EmoClass'].to_dict()

        # Not all audios have an associated label entry.
        # Intersect keys to ensure we only use data that is present in both files.
        npz_keys = set(self.sequences_data.keys())
        label_keys = set(self.labels.keys())

        self.sequence_keys = sorted(list(npz_keys.intersection(label_keys)))

        self.labels_map = process_labels(list(self.labels.values()))
        self.labels_str2int = self.labels_map[0]
        self.labels_int2str = self.labels_map[1]
        print(f'Label mapping: {self.labels_str2int}\n')

        assert len(self.sequence_keys) == len(self.labels), \
            f"The number of sequences and labels must be the same. There are {len(self.sequence_keys)} sequences and {len(self.labels)} labels."

    def __len__(self):
        return len(self.sequence_keys)

    def __getitem__(self, idx):
        key = self.sequence_keys[idx]
        sequence = self.sequences_data[key]
        label = self.labels[key]
        label = self.labels_str2int.get(label)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
### Example usage
#dataset = NpzDataset(npz_path, labels_path)
#dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, collate_fn = collate_fn)

