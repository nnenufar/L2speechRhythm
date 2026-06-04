from src.dataloaders import DatasetLMDB, collate_fn, ContrastiveBatchSampler
from torch.utils.data import DataLoader
from collections import Counter
import json
import torch

with open('data/arctic/vc_durations.json', 'r') as f:
    vc_durations = json.load(f)

items = ['v_dur', 'c_dur']
splits = ['Train', 'Development']

print("=" * 70)
print("Duration Contrastive — Dataset Statistics")
print("=" * 70)
print(f"VC duration samples: {len(vc_durations)}")

for split in splits:
    print(f"\n{'=' * 70}")
    print(f"Split: {split}")
    print("=" * 70)

    dataset = DatasetLMDB('data/arctic/rtm_feats_bark_f0_egemaps.lmdb',
                          data_source='arctic_contrastive',
                          split=split, items=items, vc_features=vc_durations)

    print(f"Number of samples: {len(dataset)}")
    print(f"Label mapping: {dataset.labels_str2int}")

    str_counts = Counter(dataset.labels.values())
    print(f"Class distribution: {dict(str_counts)}")

    vc_present = sum(1 for k in dataset.lmdb_keys
                     if k.decode('utf-8').replace('.WAV', '').replace('.wav', '') in vc_durations)
    print(f"VC duration coverage: {vc_present}/{len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    for item in items:
        print(f"  {item}: shape={sample[item].shape}, dtype={sample[item].dtype}")
        if sample[item].numel() > 0:
            print(f"    values: {sample[item][:5].tolist()}")

print("\n" + "=" * 70)
print("Contrastive Batch Test (Train split)")
print("=" * 70)

dataset = DatasetLMDB('data/arctic/rtm_feats_bark_f0_egemaps.lmdb',
                      data_source='arctic_contrastive',
                      split='Train', items=items, vc_features=vc_durations)

sampler = ContrastiveBatchSampler(
    dataset=dataset,
    csv_path='data/arctic/arctic_metadata.csv',
    utterances_per_batch=2,
    l1_per_utterance=2,
    l2_per_utterance=2,
    shuffle=True,
)
dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
batch = next(iter(dataloader))

print(batch)

for item in items:
    print(f"{item}: shape={batch[item].shape}, type={batch[item].dtype}")
print(f"label: shape={batch['label'].shape}, type={batch['label'].dtype}")
print(f"utterance_id: {batch['utterance_id'].unique().tolist()}")
print(f"L1/L2: {(batch['label']==0).sum().item()}/{(batch['label']==1).sum().item()}")

print("\n" + "=" * 70)
print("Model Forward Test")
print("=" * 70)

from src.models import DurationContrastiveModel
model = DurationContrastiveModel()
model.eval()
with torch.no_grad():
    out = model(batch['v_dur'], batch['c_dur'])
print(f"Output shape: {out.shape}")
print(f"Output norm (first 3): {out.norm(dim=-1)[:3].tolist()}")
