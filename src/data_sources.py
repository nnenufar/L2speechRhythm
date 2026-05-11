import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_arctic_data(split, test_size, random_seed):
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

def get_arctic_data_regression(split, test_size=0.10, random_seed=42):
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


def get_msp_data(split, test_size, random_seed):
    """Handle MSP dataset configuration"""
    labels_to_use = ['A', 'H', 'N', 'S']
    labels_df = pd.read_csv('data/MSP/labels_consensus.csv')
    labels_df = labels_df[labels_df['EmoClass'].isin(labels_to_use)]
    dev_df = labels_df[labels_df['Split_Set'] == 'Development']

    dev_files, test_files = train_test_split(
        dev_df['FileName'].values,
        test_size=test_size,
        random_state=random_seed,
        stratify=dev_df['EmoClass'].values
    )

    if split is not None:
        labels_df = labels_df[labels_df['Split_Set'] == split]
        if split == 'Development':
            labels_df = labels_df[labels_df['FileName'].isin(dev_files)]
        elif split == 'Test':
            labels_df = dev_df[dev_df['FileName'].isin(test_files)]

    return labels_df.set_index('FileName')['EmoClass'].to_dict()


def get_globo_data(split, test_size, random_seed):
    """Handle Globo dataset configuration"""
    train_files, testValid_files = train_test_split(
        os.listdir('data/globo/wavs'),
        test_size=test_size,
        random_state=random_seed
    )
    valid_files, test_files = train_test_split(
        testValid_files,
        test_size=0.5,
        random_state=random_seed
    )

    split_map = {
        "Train": train_files,
        "Development": valid_files,
        "Test": test_files
    }
    
    keys = split_map.get(split)
    return {key: "J" for key in keys}


def get_mtedx_data(split, test_size, random_seed):
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


# Registry of data source handlers
DATA_SOURCE_HANDLERS = {
    'MSP': get_msp_data,
    'globo': get_globo_data,
    'mtedx': get_mtedx_data,
    'arctic': get_arctic_data,
    'arctic_regression': get_arctic_data_regression,
}

def get_labels_for_source(data_source, split, test_size, random_seed):
    """Get labels for a specific data source"""
    if data_source not in DATA_SOURCE_HANDLERS:
        raise ValueError(f"Unknown data source: {data_source}")
    
    return DATA_SOURCE_HANDLERS[data_source](split, test_size, random_seed)