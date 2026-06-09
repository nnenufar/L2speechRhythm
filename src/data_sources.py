import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_arctic_data(split, test_size, random_seed, **kwargs):
    """Handle Arctic dataset configuration (combines arctic_cmu and arctic_l2)"""

    df = pd.read_csv('data/arctic/arctic_metadata.csv') # CSV created using utils/create_arctic_table.py

    df_n = df[df['cond'] == 'n']
    df_nn_ibt = df[(df['cond'] == 'nn') & (df['ibt'].notna())]
    df_nn_noibt = df[(df['cond'] == 'nn') & (df['ibt'].isna())]

    # num_spks_total = df['spkID'].nunique()
    # print(f"Total number of speakers : {num_spks_total}")
    # num_spks_native = df[df['cond'] == 'n']['spkID'].nunique()
    # print(f"Total number of native speakers : {num_spks_native}")
    # num_spks_nonNative = df[df['cond'] == 'nn']['spkID'].nunique()
    # print(f"Total number of non-native speakers : {num_spks_nonNative}")
    # df_with_ibt = df[(df['ibt'].notna()) & (df['cond']=='nn')].copy()
    # print(f"Number of non-native speakers with IBT scores: {df_with_ibt['spkID'].nunique()}")
    # unique_ibt_scores = df_with_ibt['ibt'].nunique()    
    # print(f'Number of unique ibt scores: {unique_ibt_scores}')

    native_speakers = df_n['spkID'].unique().tolist()
    native_test_speakers = df_n.loc[df_n['spkID'].isin([native_speakers.pop()]), 'identifier'].unique() # Only 1 N speaker for test
    native_dev_speakers = df_n.loc[df_n['spkID'].isin([native_speakers.pop()]), 'identifier'].unique() # Only 1 N speaker for dev
    native_train_speakers = df_n.loc[df_n['spkID'].isin(native_speakers), 'identifier'].unique() # Rest for training

    nonNat_train_speakers = df_nn_noibt['identifier'].unique() # All NNs without ibt to train
    nonNat_ibt_spks = df_nn_ibt['spkID'].unique().tolist()
    num_ibt_spks = len(nonNat_ibt_spks)
    nonNat_dev_speakers = df_nn_ibt.loc[df_nn_ibt['spkID'].isin([nonNat_ibt_spks.pop()]), 'identifier'].unique()
    nonNat_test_speakers = df_nn_ibt.loc[df_nn_ibt['spkID'].isin(nonNat_ibt_spks), 'identifier'].unique() # Half NN speakers w/ ibt for test # Half NN speakers w/ ibt for dev

    train = np.concatenate((native_train_speakers, nonNat_train_speakers))
    dev = np.concatenate((native_dev_speakers, nonNat_dev_speakers))
    test = np.concatenate((native_test_speakers, nonNat_test_speakers))

    split_map = {
        "Train": train,
        "Development": dev,
        "Test": test
    }

    files_for_split = split_map.get(split)
    labels_dict = df.set_index('identifier')['l1'].to_dict()
    return {key: labels_dict[key] for key in files_for_split}

def get_arctic_data_regression(split, test_size=0.10, random_seed=42, **kwargs):
    """
    Handle Arctic dataset configuration (only arctic_l2)
    Splits are separated for a regression with the ibt scores.
    Only includes samples that have an ibt score.
    
    Uses stratified splitting based on IBT scores to ensure
    similar distributions across train/dev/test splits.
    """

    df = pd.read_csv('data/arctic/arctic_metadata.csv')

    # Only non-native speakers with ibt scores
    df_nn_ibt = df[(df['cond'] == 'nn') & (df['ibt'].notna())].copy()
    
    all_identifiers = df_nn_ibt['identifier'].values
    all_ibt_scores = df_nn_ibt['ibt'].values
    
    # First split: 90% train, 10% temp (for dev+test)
    train_ids, temp_ids, train_scores, temp_scores = train_test_split(
        all_identifiers,
        all_ibt_scores,
        test_size=0.10,
        random_state=random_seed,
        stratify=all_ibt_scores
    )
    
    # Second split: 50/50 of temp -> 5% dev, 5% test
    dev_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=random_seed,
        stratify=temp_scores
    )
    
    split_map = {
        "Train": train_ids,
        "Development": dev_ids,
        "Test": test_ids
    }
    
    files_for_split = split_map.get(split)
    # Return ibt scores as labels for regression
    labels_dict = df_nn_ibt.set_index('identifier')['ibt'].to_dict()
    return {key: labels_dict[key] for key in files_for_split}


    """Handle mTEDx dataset configuration"""
    base_path = 'data/mtedx/'
    
    train_files = os.listdir(base_path + 'splits/train/wav')
    valid_files = os.listdir(base_path + 'splits/valid/wav')
    test_files = os.listdir(base_path + 'splits/test/wav')

    split_map = {
        "Train": train_files,
        "Development": valid_files,
        "Test": test_files
    }
    
    keys = split_map.get(split)
    return {key: "P" for key in keys}

def get_arctic_data_contrastive(split, test_size=0.05, random_seed=42, **kwargs):
    """
    Arctic samples for contrastive pretraining, split at the utterance level.
    Labels: "L1" for native (cond='n'), "L2" for non-native (cond='nn').
    test_size: fraction of UTTERANCES (not samples) held out for validation.
    """
    df = pd.read_csv('data/arctic/arctic_metadata.csv')

    all_utt_ids = sorted(df['uttID'].unique())
    rng = np.random.RandomState(random_seed)
    rng.shuffle(all_utt_ids)

    n_val = max(1, int(len(all_utt_ids) * test_size))
    val_utt_ids = set(all_utt_ids[:n_val])

    if split == 'Train':
        utt_filter = lambda utt: utt not in val_utt_ids
    elif split == 'Development':
        utt_filter = lambda utt: utt in val_utt_ids
    else:
        return {}

    labels_dict = {}
    for _, row in df.iterrows():
        if not utt_filter(row['uttID']):
            continue
        if row['cond'] == 'n':
            labels_dict[row['identifier']] = 'L1'
        elif row['cond'] == 'nn':
            labels_dict[row['identifier']] = 'L2'
    return labels_dict

def get_speechocean_data(split, test_size=0.05, random_seed=42, label_column='fluency'):
    """
    Speechocean dataset for fluency/prosodic score regression.
    Uses the predefined train/test split from the original dataset (split column in CSV).
    The training portion is subdivided into Train/Development via stratified splitting.
    Returns {identifier: score (float)}.
    """
    df = pd.read_csv('data/speechocean/speechocean_metadata.csv')

    na_indices = df.index[df[label_column].isna()]
    if len(na_indices) > 0:
        print(f"Dropping {len(na_indices)} entries with missing {label_column} scores "
              f"at rows: {na_indices.tolist()}")
        df = df.drop(index=na_indices)

    df[label_column] = df[label_column].astype(float)

    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Expected both 'train' and 'test' entries in split column. "
            f"Got {len(train_df)} train, {len(test_df)} test."
        )

    test_ids = test_df['identifier'].values

    train_ids = train_df['identifier'].values
    train_scores = train_df[label_column].values

    score_counts = pd.Series(train_scores).value_counts().to_dict()
    min_for_stratify = max(int(np.ceil(2.0 / test_size)), 4)
    rare_scores = {s for s, c in score_counts.items() if c < min_for_stratify}
    stratify_labels = np.array([-1.0 if s in rare_scores else s for s in train_scores])

    if rare_scores:
        print(f"Binning {len(rare_scores)} rare score classes for stratification: "
              f"{sorted(rare_scores)} (total {int((stratify_labels == -1.0).sum())} samples)")

    train_sub_ids, dev_ids, _, _ = train_test_split(
        train_ids, train_scores,
        test_size=test_size, random_state=random_seed, stratify=stratify_labels
    )

    split_map = {
        "Train": train_sub_ids,
        "Development": dev_ids,
        "Test": test_ids
    }

    print(f"Speechocean splits — Train: {len(train_sub_ids)}, Dev: {len(dev_ids)}, "
          f"Test: {len(test_ids)}")

    files_for_split = split_map.get(split)
    labels_dict = df.set_index('identifier')[label_column].to_dict()
    return {key: labels_dict[key] for key in files_for_split}


def get_speechocean_data_customSplit(split, test_size=0.20, random_seed=42, label_column='fluency'):
    """
    Speechocean dataset for fluency/prosodic score regression.
    Uses random stratified splits over all available data (ignores predefined split column).
    Returns {identifier: score (float)}.
    """
    df = pd.read_csv('data/speechocean/speechocean_metadata.csv')

    na_indices = df.index[df[label_column].isna()]
    if len(na_indices) > 0:
        print(f"Dropping {len(na_indices)} entries with missing {label_column} scores "
              f"at rows: {na_indices.tolist()}")
        df = df.drop(index=na_indices)

    df[label_column] = df[label_column].astype(float)

    all_identifiers = df['identifier'].values
    all_scores = df[label_column].values

    assert len(all_identifiers) == len(all_scores), \
        f"Identifier/scores length mismatch: {len(all_identifiers)} vs {len(all_scores)}"

    score_counts = pd.Series(all_scores).value_counts().to_dict()
    min_for_stratify = max(int(np.ceil(2.0 / test_size)), 4)
    rare_scores = {s for s, c in score_counts.items() if c < min_for_stratify}
    stratify_labels = np.array([-1.0 if s in rare_scores else s for s in all_scores])

    if rare_scores:
        print(f"Binning {len(rare_scores)} rare score classes for stratification: "
              f"{sorted(rare_scores)} (total {int((stratify_labels == -1.0).sum())} samples)")

    train_ids, temp_ids, train_scores, temp_scores = train_test_split(
        all_identifiers, all_scores,
        test_size=test_size, random_state=random_seed, stratify=stratify_labels
    )

    temp_stratify = np.array([-1.0 if s in rare_scores else s for s in temp_scores])
    dev_ids, test_ids, _, _ = train_test_split(
        temp_ids, temp_scores,
        test_size=0.50, random_state=random_seed, stratify=temp_stratify
    )

    split_map = {
        "Train": train_ids,
        "Development": dev_ids,
        "Test": test_ids
    }

    print(f"Speechocean custom splits — Train: {len(train_ids)}, Dev: {len(dev_ids)}, "
          f"Test: {len(test_ids)}")

    files_for_split = split_map.get(split)
    labels_dict = df.set_index('identifier')[label_column].to_dict()
    return {key: labels_dict[key] for key in files_for_split}


# Registry of data source handlers
DATA_SOURCE_HANDLERS = {
    'arctic': get_arctic_data,
    'arctic_regression': get_arctic_data_regression,
    'arctic_contrastive': get_arctic_data_contrastive,
    'speechocean': get_speechocean_data,
    'speechocean_custom': get_speechocean_data_customSplit,
}

def get_labels_for_source(data_source, split, test_size, random_seed, **kwargs):
    """Get labels for a specific data source"""
    if data_source not in DATA_SOURCE_HANDLERS:
        raise ValueError(f"Unknown data source: {data_source}")

    return DATA_SOURCE_HANDLERS[data_source](split, test_size, random_seed, **kwargs)