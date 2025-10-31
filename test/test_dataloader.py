from src.dataloaders import DatasetLMDB, collate_fn
from torch.utils.data import DataLoader

items = ['feats', 'intervals', 'envelope_spectrum']

dataset = DatasetLMDB('data/globo/rtm_feats.lmdb', data_source='globo', split='Development', items=items)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size = 32, shuffle=True, collate_fn=collate_fn)
batch = next(iter(dataloader))

for item in items:
    print(f"{item} shape: {batch[item].shape} type: {batch[item].dtype}")
print(f"Label shape: {batch['label'].shape} type: {batch['label'].dtype}")

print(dataset[1]['spectrum_freq_bins'])