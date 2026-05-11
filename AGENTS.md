# AGENTS.md — Speech Rhythm Classifier

## Environment

- **Conda env**: `conda env create -f environment.yml && conda activate rtm`
- Python 3.10, PyTorch (pip-installed via conda env)
- No linter, formatter, typechecker, or CI configured. No `setup.py` / `pyproject.toml`.

## Commands

```bash
# Training (the only entrypoint)
python -m src.train --config config/<name>.json [--save_ckpt]

# Single-epoch smoke test: set "num_epochs": 1 in config

# Inference
python -m src.inference \
    --config config/<name>.json \
    --checkpoint <path.pth> \
    --exp_id <id> \
    --split Test

# Basic dataloader sanity check
python test/test_dataloader.py

# Compute feature statistics (for normalizing f0, etc.)
python src/compute_stats.py

# Build ARCTIC metadata CSV from audio symlinks
python utils/create_arctic_table.py
```

## Architecture

- **Two model types**: `cnn` (CNN_MLP — Conv1D + optional LSTM + optional attention pooling) and `ssl` (WAV_LM — frozen WavLM-large + attention pooling). See `src/models.py:22-26`.
- **Two task types**: `classification` (CrossEntropy, macro F1) and `regression` (MSE, Pearson r). Controlled by `model_params.task` in the config. See `src/task_spec.py`.
- **Collate functions must match model_type**: use `"collate_fn": "pad"` for cnn models and `"collate_fn": "ssl"` for WavLM models. Mismatch will error.
- **Data**: LMDB database with pickled dicts per sample. DatasetLMDB handles label assignment via `DATA_SOURCE_HANDLERS` in `src/data_sources.py`. Supported sources: `arctic`, `arctic_regression`, `MSP`, `globo`, `mtedx`.
- **Feature extraction** before training is done via the `vowel_beat_detector` submodule (not included; see README).

## Config format

JSON files in `config/`. Required keys: `exp_name`, `batch_size`, `collate_fn`, `model_type`, `model_params`, `dataset_params`, `training_params`. Optional: `wandb` block (default mode: `"offline"`).

## Gotchas

- **WavLM cache path is hardcoded** in `src/dataloaders.py:117` to `/home/joao.lima/.cache/huggingface/hub/`. Change this before running on a different machine.
- **Config, exp, inference, stats, slurm, and data directories are all gitignored.** A fresh clone has no config files. Use existing config names as templates (files visible in `config/` are untracked).
- **`processor_ssl` hardcodes `sampling_rate=16000`.** Audio must be 16 kHz.
- **Gradient clipping is hardcoded** to `max_norm=1.0` (`src/train.py:249`).
- **W&B default is `"offline"`** — no network calls unless you set `"mode": "online"` in the wandb config block.
- **Identifier parsing** (`src/dataloaders.py:39`) expects format `<speakerID>_<datasetName>_<utteranceID>`.
- **`--save_ckpt` flag** must be passed to save periodic checkpoints; without it, only the best model is saved.
