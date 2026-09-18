# Automated Assessment of L2 Speech Rhythm Using Low-Frequency Amplitude Modulations

This repository contains the code to reproduce the experiments performed in the "Automated Assessment of L2 Speech Rhythm Using Low-Frequency Amplitude Modulations" paper (IEEE SLT 2026). The supported models are:

- `RhythmRegressor`: amplitude envelope or envelope-derivative input
  (`env`, `envRate`)
- `DurationRegressor`: vocalic/intervocalic phone-duration input (`dur`, `durZ`)

All eight consolidated experiment configurations are in `config/consolidate/`.
The corresponding final test-set metrics and predictions are in
`exp/consolidate/<experiment>/eval/`.

## Repository layout

```text
config/consolidate/         Final experiment configs
exp/consolidate/            Final test metrics, predictions, and HPO best params
src/                        Model, dataloader, training, HPO, and inference code
slurm/                      Slurm templates for training, HPO, evaluation, and MFA
speech_feature_extractor/   Feature extraction submodule
utils/                      Speechocean metadata and VC-feature preparation tools
test/                       Dataloader smoke tests
environment.yml             Python environment
```

## 1. Environment

Initialize the feature-extractor submodule:

```bash
git submodule update --init --recursive
```

Create and activate the main environment:

```bash
conda env create -f environment.yml
conda activate rtm
```

The consolidated code requires Python 3.10, PyTorch, NumPy, Pandas, SciPy,
scikit-learn, Matplotlib, lmdb, Optuna, and wandb. Forced alignment for
`dur`/`durZ` preparation requires `montreal-forced-aligner`.

## 2. Required data

The training and evaluation code expects:

| Path | Description |
|---|---|
| `data/speechocean/rtm_feats.lmdb` | LMDB with envelope and envelope-derivative sequences |
| `data/speechocean/speechocean_metadata.csv` | Identifier -> fluency/prosodic score |
| `data/speechocean/vc_features.json` | Tokenized V/C intervals and phone durations for `dur`/`durZ` |

The first two files can be created from Speechocean WAVs, score files, and
Kaldi-style `utt2spk` files as described below.

## 3. Build Speechocean metadata

Extract the envelope/envelope-derivative LMDB used by the consolidated models:

```bash
bash utils/extract_features.sh /path/to/speechocean/wavs data/speechocean bark
```

Then create the metadata CSV:

```bash
python utils/create_speechocean_table.py \
    --lmdb data/speechocean/rtm_feats.lmdb \
    --scores /path/to/scores.json \
    --train_utt2spk /path/to/train_utt2spk \
    --test_utt2spk /path/to/test_utt2spk \
    --output data/speechocean/speechocean_metadata.csv
```

## 4. Build duration features

Only required for `dur` and `durZ` experiments.

1. Prepare MFA input and run forced alignment:

```bash
sbatch slurm/mfa_align.sh
```

2. Build V/C alignments, z-scored phone durations, and tokenized VC features:

```bash
bash utils/build_vc_pipeline.sh data/speechocean
```

This creates `vc_alignments.json`, `phone_duration_stats.json`, and
`vc_features.json`.

## 5. Verify dataloaders

```bash
python -m test.test_dataloader
python -m test.test_duration_dataloader
```

## 6. Train a single experiment

```bash
python -m src.train --config config/consolidate/<config>.json
```

On Slurm:

```bash
sbatch slurm/train.sh config/consolidate/<config>.json
```

Training uses MSE, gradient clipping at 5.0, and early stopping on validation
Spearman r. Outputs are written under `exp/<exp_name>/`.

## 7. Run hyperparameter search

```bash
python -m src.hpo \
    --config config/consolidate/<config>.json \
    --n_trials 80
```

On Slurm:

```bash
sbatch slurm/hpo.sh config/consolidate/<config>.json 80
```

HPO selects the best trial on validation Spearman r and automatically runs
test-set inference with the best checkpoint.

## 8. Evaluate a trained checkpoint

```bash
python -m src.eval_regression \
    --config config/consolidate/<config>.json \
    --checkpoint /path/to/best_model_epochN.pth \
    --split Test
```

On Slurm:

```bash
sbatch slurm/eval.sh \
    config/consolidate/<config>.json \
    /path/to/best_model_epochN.pth \
    Test
```

## 9. Consolidated results

The final configurations and their result directories are:

| Input | Fluency | Prosodic |
|---|---|---|
| Envelope | `env_fluency_lstm_bi` | `env_prosody_lstm_bi` |
| Envelope derivative | `envRate_fluency_lstm_bi` | `envRate_prosody_lstm_bi` |
| Phone durations | `dur_fluency_lstm_bi` | `dur_prosody_lstm_bi` |
| Z-scored phone durations | `durZ_fluency_lstm_bi` | `durZ_prosody_lstm_bi` |

For each experiment, the final metrics are in:

```text
exp/consolidate/<experiment>/eval/summary_Test_*.json
exp/consolidate/<experiment>/eval/predictions_Test_*.csv
```

Fluency experiments also include `hpo_best.json` with the selected
hyperparameters. Prosody experiments reuse the corresponding fluency
hyperparameters.
