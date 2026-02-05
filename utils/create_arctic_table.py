import pandas as pd
from pathlib import Path
import os

# Assuming there are symlinks inside data/arctic pointing to audio files in other locations
# And that audio folders are arranged following the structure <spkID>/*.wav

# We will leave out some speakers from the CMU dataset that are non-native

base_path = 'data/arctic'

paths = [p for p in os.popen(
    'find "{}" -follow -type f -name "*.wav"'.format(base_path)
).read().splitlines()]

df_wavs = pd.DataFrame({
    "path": paths,
})

# Extract speaker ID (spkID)
df_wavs["spkID"] = df_wavs["path"].str.split("/").str[-2]

# Build identifier: "<spkID>_<filename>"
df_wavs["identifier"] = df_wavs["spkID"] + "_" + df_wavs["path"].str.split("/").str[-1]

df_metadata = pd.read_csv("data/arctic/spks.csv")

df_merged = df_wavs.merge(df_metadata, on="spkID", how="left")

df_merged = df_merged[~df_merged["spkID"].isin(["ahw", "aup", "axb", "fem", "gka", "rxr", "slp", "jmk", "awb", "ksp", "suitcase_corpus"])]

# Check for NaNs in the 'cond' column
nan_count = df_merged['cond'].isna().sum()
if nan_count > 0:
    print(f"Warning: Found {nan_count} NaN values in 'cond' column")
    print("Rows with NaN in 'cond':")
    print(df_merged[df_merged['cond'].isna()])
else:
    print("No NaN values found in 'cond' column")

df_merged.to_csv("data/arctic/arctic_metadata.csv", index=False)


