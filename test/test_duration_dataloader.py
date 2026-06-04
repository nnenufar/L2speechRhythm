from src.dataloaders import DatasetLMDB, collate_fn
from torch.utils.data import DataLoader
from collections import Counter
import json
import torch

with open('data/speechocean/vc_features.json', 'r') as f:
    vc_data = json.load(f)

vc_features = vc_data['samples']
num_tokens = len(vc_data['vocab'])

items = ['v_phones', 'v_plen', 'v_dur', 'c_phones', 'c_plen', 'c_dur']
splits = ['Train', 'Development', 'Test']

print("=" * 70)
print("Duration Regressor — Dataset Statistics")
print("=" * 70)
print(f"VC tokens in vocab: {num_tokens}")

for split in splits:
    print(f"\n{'=' * 70}")
    print(f"Split: {split}")
    print("=" * 70)

    dataset = DatasetLMDB('data/speechocean/rtm_feats.lmdb', data_source='speechocean',
                          split=split, items=items, vc_features=vc_features)

    print(f"Number of samples: {len(dataset)}")
    print(f"Unique speakers: {len(dataset.speaker_str2int)}")
    print(f"Unique utterances: {len(dataset.utterance_str2int)}")
    print(f"Label mapping: {dataset.labels_str2int}")

    str_counts = Counter(dataset.labels.values())
    print(f"Class distribution: {dict(str_counts)}")

    # Check VC coverage
    vc_present = sum(1 for k in dataset.lmdb_keys if k.decode('utf-8') in vc_features)
    print(f"VC features coverage: {vc_present}/{len(dataset)}")

    # Check a sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    for item in items:
        print(f"  {item}: shape={sample[item].shape}, dtype={sample[item].dtype}")

print("\n" + "=" * 70)
print("Batch Test (Train split)")
print("=" * 70)

dataset = DatasetLMDB('data/speechocean/rtm_feats.lmdb', data_source='speechocean',
                      split='Train', items=items, vc_features=vc_features)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
batch = next(iter(dataloader))

for item in items:
    print(f"{item}: shape={batch[item].shape}, type={batch[item].dtype}")
print(f"label: shape={batch['label'].shape}, type={batch['label'].dtype}")

# Quick model forward test
from src.models import DurationRegressor
model = DurationRegressor(num_tokens=num_tokens)
model.eval()
with torch.no_grad():
    out = model(batch)
print(f"\nModel output shape: {out.shape}")
print(f"Sample predictions: {out[:5].tolist()}")
