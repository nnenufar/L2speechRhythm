from src.train_utils import compute_dataset_statistics
from src.dataloaders import DatasetLMDB
import json
from pathlib import Path

DATASET_PATH = 'data/arctic/rtm_feats.lmdb'
DATA_SOURCE = 'arctic'
ITEMS = ['dur']
SPLIT = 'Train'

def main():
    dataset_path = DATASET_PATH
    data_source = DATA_SOURCE
    items = ITEMS
    split = SPLIT

    dataset = DatasetLMDB(dataset_path, data_source=data_source, split=split, items=items)

    stats = compute_dataset_statistics(dataset, items)
    out_path = Path(DATASET_PATH).parent / f'stats_{split}_{items[0]}.json'
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()