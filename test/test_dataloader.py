from src.dataloaders import DatasetLMDB, collate_fn
from torch.utils.data import DataLoader
from src import train_utils
from collections import Counter
import torch

items = ['envelope_derivative']
splits = ['Train', 'Development', 'Test']

print("="*70)
print("Dataset Statistics")
print("="*70)

for split in splits:
    print(f"\n{'='*70}")
    print(f"Split: {split}")
    print("="*70)
    
    dataset = DatasetLMDB('data/speechocean/rtm_feats.lmdb', data_source='speechocean_custom', split=split, items=items)
    
    print(f"Number of samples: {len(dataset)}")
    print(f"Unique speakers: {len(dataset.speaker_str2int)}")
    print(f"Unique utterances: {len(dataset.utterance_str2int)}")
    print(f"Label mapping: {dataset.labels_str2int}")
    
    # Class distribution
    str_counts = Counter(dataset.labels.values())
    print(f"Class distribution: {dict(str_counts)}")

print("\n" + "="*70)
print("Batch Test (Train split)")
print("="*70)

collateFunc = collate_fn

# Test batch loading with Train split
dataset = DatasetLMDB('data/speechocean/rtm_feats.lmdb', data_source='speechocean_custom', split='Train', items=items)
#sampler = train_utils.create_weighted_sampler(dataset, str_counts)
str_counts = Counter(dataset.labels.values())

# Note: Weighted sampler is not supported for regression tasks
# For regression, just use shuffle=True
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collateFunc, shuffle=True) #sampler=sampler
batch = next(iter(dataloader))
print(dataset[50])

# Batch class distribution
batch_labels_int = batch['label']
ints, counts = torch.unique(batch_labels_int, return_counts=True)
print("Batch class distribution (int):", {int(i.item()): int(c.item()) for i, c in zip(ints, counts)})

for item in items:
    print(f"{item} shape: {batch[item].shape} type: {batch[item].dtype}")
print(f"Label shape: {batch['label'].shape} type: {batch['label'].dtype}")
