from src.dataloaders import DatasetLMDB, collate_fn
from torch.utils.data import DataLoader

dataset = DatasetLMDB('data/MSP/rtm_feats.lmdb', data_source='MSP')
print(f'Feats type: {dataset[0][0].dtype}\nIntervals type: {dataset[0][1].dtype}\nLabels type: {dataset[0][2].dtype}\n')
print(f'Label mapping: {dataset.labels_int2str}')

dataloader = DataLoader(dataset, batch_size = 32, collate_fn = collate_fn, shuffle=True)
batch = next(iter(dataloader))

print("Batch type:", type(batch))
if isinstance(batch, (list, tuple)):
    for i, item in enumerate(batch):
        print(f"Item {i} shape/type:", getattr(item, 'shape', type(item)))
        print(item[-1])
else:
    print("Batch:", batch)