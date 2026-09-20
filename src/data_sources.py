import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_speechocean_data_custom_split(split, test_size=0.20, random_seed=42, label_column='fluency'):
    df = pd.read_csv('data/speechocean/speechocean_metadata.csv')

    missing = df.index[df[label_column].isna()]
    if len(missing) > 0:
        print(f"Dropping {len(missing)} entries with missing {label_column} scores")
        df = df.drop(index=missing)

    df[label_column] = df[label_column].astype(float)
    all_identifiers = df['identifier'].values
    all_scores = df[label_column].values

    score_counts = pd.Series(all_scores).value_counts().to_dict()
    min_for_stratify = max(int(np.ceil(2.0 / test_size)), 4)
    rare_scores = {score for score, count in score_counts.items() if count < min_for_stratify}
    stratify_labels = np.array([-1.0 if score in rare_scores else score for score in all_scores])

    if rare_scores:
        print(f"Binning {len(rare_scores)} rare score classes for stratification: "
              f"{sorted(rare_scores)}")

    train_ids, temp_ids, train_scores, temp_scores = train_test_split(
        all_identifiers, all_scores,
        test_size=test_size, random_state=random_seed, stratify=stratify_labels,
    )

    temp_stratify = np.array([-1.0 if score in rare_scores else score for score in temp_scores])
    dev_ids, test_ids, _, _ = train_test_split(
        temp_ids, temp_scores,
        test_size=0.50, random_state=random_seed, stratify=temp_stratify,
    )

    split_map = {
        'Train': train_ids,
        'Development': dev_ids,
        'Test': test_ids,
    }
    print(f"Speechocean custom splits — Train: {len(train_ids)}, Dev: {len(dev_ids)}, "
          f"Test: {len(test_ids)}")

    files_for_split = split_map.get(split)
    labels_dict = df.set_index('identifier')[label_column].to_dict()
    return {key: labels_dict[key] for key in files_for_split}


DATA_SOURCE_HANDLERS = {
    'speechocean_custom': get_speechocean_data_custom_split,
}


def get_labels_for_source(data_source, split, test_size, random_seed, **kwargs):
    if data_source not in DATA_SOURCE_HANDLERS:
        raise ValueError(f"Unknown data source: {data_source}")
    return DATA_SOURCE_HANDLERS[data_source](
        split, test_size, random_seed, **kwargs
    )
