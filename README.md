Speech rhythm analysis tools

# Code purpose and functionalities

The ``vowel_beat_detector`` submodule provides rhythmic feature extraction tools (check the submodule's README.md for more details). The source code at root provides deep learning resources to process these features and perform feature learning and classification tasks.

# Quick start

1. Clone this repository and set up the environment
    ```
    cd <path to cloned repo>
    conda env create -f environment.yml
    conda activate rtm
    ```

2. Download datasets and extract features
    ```
    python vowel_beat_detector/src/main.py \
    -in <dataset_path> \
    -out data/<dataset_name> \
    -sr 16000
    ```
    Our experiments use the mTEDx_pt and g_neutral_speech_male datasets and automatically resample all audios to 16kHz. More detailed insctructions on how to download and process the datasets can be found in [TODO: add data processing scripts]

    The features will be extracted into a ``.lmdb`` file. Each entry in the file can be read similarly to a python dictionary where item corresponds to a type of feature. 

3. Set up the dataloader

    If you're implementing your own dataset, you will need to build a pytorch-style dataloader for it then add the dataset name in the training script.

    If you're using the already implemented datasets, simply use the dataset name in the training config file.

    The adopted approach is to specify samples IDs with data_sources.py then feed them to the DatasetLMDB class and create a dataloader from it.

    data_sources looks at the CSV file created with create_arctic_data.py

4. Config file
    Specify all desired training and model parameters in a ``.json`` file inside ``/config``

5. Run training script
    ```
    python -m src.train --config <config_path>
    ```

6. Follow experiment
    Training and evaluation metrics, as well as checkpoints, will be saved under ``/exp`` with the ``exp_name`` as defined in the used config file.


---

# Reproducing experiments

This section describes how to reproduce the experiments from the thesis. All experiments use the Speechocean dataset (performers reading a common English text, scored on fluency and prosodic control).

## Experiment A overview

| Sub-experiment | Model | Input | Target | Config |
|---|---|---|---|---|
| A1c (envelope) | RhythmRegressor | Amplitude envelope | fluency | `exp_A1c_env_fluency_expand.json` |
| A1c (envelope) | RhythmRegressor | Amplitude envelope | prosodic | `exp_A1c_env_prosody_expand.json` |
| A2c (duration) | DurationRegressor | V/C interval durs + phone IDs | fluency | `exp_A2c_dur_fluency.json` |
| A2c (duration) | DurationRegressor | V/C interval durs + phone IDs | prosodic | `exp_A2c_dur_prosody.json` |
| A2Zc (duration Z) | DurationRegressor | V/C interval durs (z-scored) + phone IDs | fluency | `exp_A2c_durZ_fluency.json` |
| A2Zc (duration Z) | DurationRegressor | V/C interval durs (z-scored) + phone IDs | prosodic | `exp_A2c_durZ_prosody.json` |
| A3c (env. derivative) | RhythmRegressor | Envelope derivative | fluency | `exp_A3c_envRate_fluency_expand.json` |
| A3c (env. derivative) | RhythmRegressor | Envelope derivative | prosodic | `exp_A3c_envRate_prosody_expand.json` |

## Prerequisites

Before running any experiment, ensure the following data is available:

| Path | Description |
|---|---|
| `<speechocean_wavs_dir>/` | Directory tree with 16 kHz mono WAVs, organised as `<speakerID>/<uttID>.WAV` |
| `<scores.json>` | JSON mapping `uttID` → `{fluency, prosodic}` scores |
| `<train/utt2spk>` | Kaldi-format utt2spk file for the training set |
| `<test/utt2spk>` | Kaldi-format utt2spk file for the test set |
| `data/speechocean/alignments/` | MFA forced alignments, one `.TextGrid` per utterance, in speaker subdirs (A2/A2Z only) |

## Step-by-step: Experiment A

### Step 1 — Environment

```bash
conda env create -f environment.yml
conda activate rtm
```

### Step 2 — Extract rhythmic features

Run the `vowel_beat_detector` submodule on the Speechocean audio. This produces a single LMDB database with amplitude envelopes, beat locations, and other per-sample features.

```bash
python vowel_beat_detector/src/main.py \
    -in <speechocean_wavs_dir> \
    -out data/speechocean \
    -sr 16000
```

Output: `data/speechocean/rtm_feats.lmdb`

### Step 3 — Build the Speechocean metadata CSV

Maps LMDB keys to fluency/prosodic scores and assigns each utterance to a train/test split.

```bash
python utils/create_speechocean_table.py \
    --lmdb data/speechocean/rtm_feats.lmdb \
    --scores <scores.json> \
    --train_utt2spk <train/utt2spk> \
    --test_utt2spk <test/utt2spk> \
    --output data/speechocean/speechocean_metadata.csv
```

Output: `data/speechocean/speechocean_metadata.csv`

### Step 4 — Build VC features (duration experiments only)

Required for A2 and A2Z. Skip to Step 5 if only running A1/A3.

First, run MFA forced alignment. Launch `slurm/mfa_align.sh` on a cluster, or run manually. This produces `data/speechocean/phone_alignments.json`.

Then build the remaining VC data files with the convenience script:

```bash
bash utils/build_vc_pipeline.sh data/speechocean
```

Outputs (under `data/speechocean/`):
- `vc_alignments.json` — V/C segment boundaries, phone durations, z-scored durations
- `phone_duration_stats.json` — per-phone mean/std used for z-scoring
- `vc_features.json` — tokenised phone sequences + interval durations consumed by the `DurationRegressor`

### Step 5 — Verify the dataloader (optional)

```bash
python test/test_dataloader.py
```

For duration experiments:

```bash
python test/test_duration_dataloader.py
```

### Step 6 — Train the models

Each sub-experiment has its own config in `config/`. Training uses random stratified splits (80/10/10), with early stopping on validation Spearman r (patience=50). All configs use `"collate_fn": "pad"` (no WavLM).

To launch training on a cluster, use `slurm/exp_A.sh` as a template — edit the `--config` path to point to your target config.

**Rhythm Regressor — envelope (A1)**

```bash
python -m src.train --config config/exp_A1c_env_fluency_expand.json
python -m src.train --config config/exp_A1c_env_prosody_expand.json
```

**Rhythm Regressor — envelope derivative (A3)**

```bash
python -m src.train --config config/exp_A3c_envRate_fluency_expand.json
python -m src.train --config config/exp_A3c_envRate_prosody_expand.json
```

**Duration Regressor — raw phone durations (A2)**

```bash
python -m src.train --config config/exp_A2c_dur_fluency.json
python -m src.train --config config/exp_A2c_dur_prosody.json
```

**Duration Regressor — z-scored phone durations (A2Z)**

```bash
python -m src.train --config config/exp_A2c_durZ_fluency.json
python -m src.train --config config/exp_A2c_durZ_prosody.json
```

### Step 7 — Where outputs are saved

All results land under `exp/<exp_name>/` with the following structure:

| Path | Contents |
|---|---|
| `checkpoints/<timestamp>/` | Best-model checkpoints (`best_model_epochN.pth`) |
| `plots/` | Training curves (loss, RMSE, Pearson r) |
| `dev_results/<timestamp>/` | `dev_results_<timestamp>.json` with final dev metrics |
| `eval/` | Per-sample predictions CSV + summary JSON (generated at dev time) |

Each checkpoint contains model weights, optimizer state, and the `utterance_str2int` mapping needed for inference.

### Step 8 — Run inference on a trained checkpoint

```bash
python -m src.eval_regression \
    --config config/exp_A1c_env_fluency_expand.json \
    --checkpoint exp/exp_A1c_env_fluency_expand/checkpoints/<ts>/best_model_epochN.pth \
    --split Test
```

Per-sample predictions (`predictions_Test_<ts>.csv`) and aggregate metrics (`summary_Test_<ts>.json`) are saved under `exp/<exp_name>/eval/`. On a cluster, use `slurm/eval_regression.sh` as a template.

### Notes

- **Split strategy**: The `speechocean_custom` data source uses random stratified splits on fluency/prosodic scores (80/10/10). The original `speechocean` data source uses the predefined train/test split from the utt2spk files.
- **No pretrained encoder**: The A1 and A3 configs do not specify a `pretrained_checkpoint`, so the `RhythmRegressor` trains its encoder from scratch. To use a contrastively pretrained encoder (experiment B1), add `"pretrained_checkpoint": "<path>"` to `model_params`.
- **Gradient clipping**: hardcoded to `max_norm=5.0` (`src/train.py:275`).
- **WavLM cache path**: hardcoded in `src/dataloaders.py:117` to `/home/joao.lima/.cache/huggingface/hub/`. Change this if running on a different machine (not needed for experiment A, which does not use WavLM).
- **W&B logging**: defaults to `"mode": "offline"`. Set `"mode": "online"` in the `wandb` config block to log to a server.