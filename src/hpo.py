"""Optuna hyperparameter search over the supervised training pipeline.

Runs a config-driven sweep: each trial deep-copies a base config, samples the
search space defined in its ``"hpo"`` block, and calls ``src.train.train`` with a
per-trial experiment name. The objective is validation Spearman r.

Usage:
    python -m src.hpo --config config/<name>.json [--n_trials N]
        [--study-name NAME] [--storage sqlite:///path/to.db | none]
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

import optuna
from src.train import train as train_supervised
from src.task_spec import get_task_spec
from src.inference import run_test_inference


def set_nested(d, dotted_path, value):
    """Set a value at a dotted path, creating intermediate dicts as needed."""
    keys = dotted_path.split(".")
    node = d
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def sample_from_spec(trial, name, spec):
    """Draw a single hyperparameter value from an ``"hpo"`` search-space spec."""
    spec_type = spec.get("type", "float")
    if spec_type == "float":
        return trial.suggest_float(
            name, spec["low"], spec["high"], log=spec.get("log", False)
        )
    if spec_type == "int":
        return trial.suggest_int(
            name, spec["low"], spec["high"],
            log=spec.get("log", False), step=spec.get("step", 1),
        )
    if spec_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown hpo search-space type: {spec_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search over a supervised training config."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON configuration file.")
    parser.add_argument("--n_trials", type=int, default=None,
                        help="Number of trials (overrides config 'hpo.n_trials').")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Optuna study name (overrides config 'hpo.study_name').")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL. 'none' disables persistence; "
                             "defaults to exp/<exp_name>/hpo.db.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_config = json.load(f)

    hpo_config = base_config.get("hpo", {})
    base_exp_name = base_config.get("exp_name", "hpo")

    # Objective direction and metric: follow the task spec unless the config overrides.
    task = base_config["model_params"].get("task", "classification")
    task_spec = get_task_spec(task)
    higher_is_better = task_spec.higher_is_better
    best_metric_key = task_spec.best_metric_key
    best_metric_label = task_spec.best_metric_label
    direction = hpo_config.get("direction", "maximize" if higher_is_better else "minimize")

    n_trials = args.n_trials or hpo_config.get("n_trials", 20)
    study_name = args.study_name or hpo_config.get("study_name", base_exp_name)

    out_dir = Path("exp") / base_exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Storage: CLI arg wins, then config, then a local SQLite DB for resume.
    storage = None
    if args.storage is not None:
        storage = None if args.storage.lower() == "none" else args.storage
    elif hpo_config.get("storage"):
        storage = hpo_config["storage"]
    else:
        storage = f"sqlite:///{out_dir / 'hpo.db'}"

    sampler_name = hpo_config.get("sampler", "tpe")
    seed = hpo_config.get("seed")
    if sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler = None

    pruner_name = hpo_config.get("pruner", "median")
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    else:  # "none" or unknown
        pruner = None

    def objective(trial):
        config = deepcopy(base_config)

        for path, spec in hpo_config.get("search_space", {}).items():
            set_nested(config, path, sample_from_spec(trial, path, spec))

        # Each trial lives under exp/<exp_name>/trials/trial<N>/. The nested path is
        # handled by setup_experiment_dir's mkdir(parents=True, exist_ok=True).
        config["exp_name"] = f"{base_exp_name}/trials/trial{trial.number}"

        # Avoid one wandb run per trial unless explicitly requested.
        if not hpo_config.get("wandb", False):
            config["wandb"] = {**config.get("wandb", {}), "enabled": False}

        value = train_supervised(config, trial=trial, save_ckpt=False)
        if value is None or not np.isfinite(value):
            value = -float("inf") if direction == "maximize" else float("inf")
        return float(value)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    print("=" * 70)
    print(f"Study '{study_name}' finished ({direction}): "
          f"best value = {best.value:.6f} (trial #{best.number})")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("=" * 70)

    with open(out_dir / "hpo_best.json", "w") as f:
        json.dump({
            "study_name": study_name,
            "direction": direction,
            "best_metric_key": best_metric_key,
            "best_metric_label": best_metric_label,
            "best_trial_number": best.number,
            "best_value": best.value,
            "best_params": best.params,
        }, f, indent=2)

    study.trials_dataframe().to_csv(out_dir / "hpo_results.csv", index=False)
    print(f"Saved hpo_best.json and hpo_results.csv to {out_dir}")

    # Run inference on the test set with the best model (selected on dev set). The best
    # checkpoint lives under the best trial's dir; glob it since the timestamp subdir and
    # epoch number vary.
    best_trial_dir = out_dir / "trials" / f"trial{best.number}"
    best_ckpts = sorted(best_trial_dir.glob("checkpoints/*/best_model_epoch*.pth"))
    if not best_ckpts:
        print(f"[WARN] No best checkpoint found under {best_trial_dir}; "
              f"skipping test-set inference.")
        return

    best_ckpt = best_ckpts[-1]  # save_checkpoint(is_best=True) keeps exactly one

    # Reconstruct the best-trial config (base + best params) so the model/dataset can be
    # rebuilt for inference. Results go under exp/<exp_name>/, not the nested trial dir.
    best_config = deepcopy(base_config)
    for path, value in best.params.items():
        set_nested(best_config, path, value)
    best_config["exp_name"] = base_exp_name

    print("=" * 70)
    print(f"Running test-set inference with best model (trial #{best.number})")
    print(f"Checkpoint: {best_ckpt}")
    print("=" * 70)
    run_test_inference(best_config, best_ckpt)


if __name__ == "__main__":
    main()
