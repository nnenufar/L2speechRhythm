from src.train_utils import compute_dataset_statistics
from src.dataloaders import DatasetLMDB
import json
from pathlib import Path

DATASET_PATH = 'data/speechocean/rtm_feats.lmdb'
DATA_SOURCE = 'speechocean'
ITEMS = ['label']
SPLIT = 'Train'

def main():
    #TODO: add normalization for pitch?
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