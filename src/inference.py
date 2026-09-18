"""Standalone inference on a trained checkpoint (used after HPO finishes).

Loads a checkpoint, rebuilds the model + requested split the same way
``src.train.train`` does, runs inference, and writes per-sample predictions
(CSV) and summary metrics (JSON) under ``exp/<exp_name>/eval/``.
"""

import json
from datetime import datetime
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.dataloaders import DatasetLMDB, DatasetRhythmFeatures
from src.evaluation import collect_regression_predictions
from src.task_spec import get_task_spec
from src.train import COLLATE_FUNC_MAPPING, MODEL_MAPPING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_classification_predictions(model, dataloader, device):
    """Run inference on a classification dataset and collect per-sample predictions."""
    model.eval()
    identifiers, labels, preds = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            outputs = model(batch_dev)
            predicted = torch.argmax(outputs, dim=1)
            identifiers.extend(batch['identifier'])
            labels.extend(batch_dev['label'].cpu().numpy().tolist())
            preds.extend(predicted.cpu().numpy().tolist())

    labels = np.array(labels)
    preds = np.array(preds)
    accuracy = 100 * float((preds == labels).mean())
    f1 = float(f1_score(labels, preds, average='macro'))
    return identifiers, labels, preds, {
        'accuracy': accuracy,
        'f1': f1,
        'num_samples': len(labels),
    }


def run_test_inference(config, checkpoint_path, split="Test", output_dir=None):
    """Rebuild the model + split from ``config`` and run inference from ``checkpoint_path``.

    Writes ``predictions_<split>_<ts>.csv`` and ``summary_<split>_<ts>.json`` under
    ``output_dir`` (defaults to ``exp/<exp_name>/eval``). Returns the summary dict.
    """
    exp_name = config.get("exp_name", "unnamed")
    task = config["model_params"].get("task", "classification")
    task_spec = get_task_spec(task)
    is_regression = task_spec.is_regression

    target_mean = config["model_params"].get("target_mean")
    target_std = config["model_params"].get("target_std")

    # VC features / vocab for duration_regressor.
    vc_features = None
    num_tokens = None
    max_phones = None
    if config.get("model_type") == "duration_regressor":
        vc_path = config["dataset_params"].get(
            "vc_features_path", "data/speechocean/vc_features.json"
        )
        with open(vc_path) as f:
            vc_data = json.load(f)
        vc_features = vc_data["samples"]
        num_tokens = len(vc_data["vocab"])
        max_v = max(max(len(p) for p in s.get("v_phones", [])) for s in vc_features.values())
        max_c = max(max(len(p) for p in s.get("c_phones", [])) for s in vc_features.values())
        max_phones = max(max_v, max_c)

    dataset_params = config["dataset_params"].copy()
    dataset_params.pop("vc_features_path", None)
    if is_regression and target_mean is not None and target_std is not None:
        dataset_params["target_mean"] = target_mean
        dataset_params["target_std"] = target_std

    is_rhythm_feature_mlp = config.get("model_type") == "rhythm_feature_mlp"

    # Build the Train split first to recover utterance / feature statistics, then the
    # requested split against those stats (identical to src.train.train).
    lmdb_env = None
    if is_rhythm_feature_mlp:
        train_dataset = DatasetRhythmFeatures(**dataset_params, split="Train")
    else:
        lmdb_env = lmdb.open(dataset_params["lmdb_path"], readonly=True, lock=False,
                             readahead=False, meminit=False)
        train_dataset = DatasetLMDB(**dataset_params, split="Train",
                                    vc_features=vc_features, env=lmdb_env)

    utterance_embed_dim = config["model_params"].get("utterance_embed_dim", 0)
    train_utterance_str2int = train_dataset.utterance_str2int if utterance_embed_dim > 0 else None
    train_utterance_int2str = train_dataset.utterance_int2str if utterance_embed_dim > 0 else None

    if is_rhythm_feature_mlp:
        split_dataset = DatasetRhythmFeatures(
            **dataset_params, split=split,
            external_utterance_str2int=train_utterance_str2int,
            external_utterance_int2str=train_utterance_int2str,
            external_feature_mean=train_dataset.feature_mean,
            external_feature_std=train_dataset.feature_std,
        )
    else:
        split_dataset = DatasetLMDB(
            **dataset_params, split=split,
            external_utterance_str2int=train_utterance_str2int,
            external_utterance_int2str=train_utterance_int2str,
            vc_features=vc_features, env=lmdb_env,
        )

    collate_func = COLLATE_FUNC_MAPPING.get(config.get("collate_fn", "pad"))
    loader = DataLoader(split_dataset, batch_size=config["batch_size"],
                        collate_fn=collate_func, shuffle=False)

    model_kwargs = {**config["model_params"]}
    if num_tokens is not None:
        model_kwargs["num_tokens"] = num_tokens
    if max_phones is not None:
        model_kwargs["max_phones"] = max_phones
    if utterance_embed_dim > 0:
        model_kwargs["num_utterances"] = len(train_utterance_str2int)

    model = MODEL_MAPPING.get(config["model_type"])(**model_kwargs).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    checkpoint_epoch = checkpoint.get("epoch")

    if output_dir is None:
        output_dir = Path("exp") / exp_name / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    if is_regression:
        results = collect_regression_predictions(model, loader, device)
        m = results["metrics"]
        df = pd.DataFrame({
            "identifier": results["identifiers"],
            "ground_truth": results["targets"],
            "prediction": results["preds"],
        })
        summary = {
            "exp_name": exp_name,
            "split": split,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "num_samples": m["num_samples"],
            "rmse": m["rmse"],
            "mae": m["mae"],
            "pearson_r": m["pearson_r"],
            "spearman_r": m["spearman_r"],
            "target_mean": m["target_mean"],
            "target_std": m["target_std"],
        }
    else:
        identifiers, labels, preds, m = _collect_classification_predictions(
            model, loader, device
        )
        df = pd.DataFrame({
            "identifier": identifiers,
            "ground_truth": labels,
            "prediction": preds,
            "ground_truth_label": [split_dataset.labels_int2str.get(int(l), "") for l in labels],
            "prediction_label": [split_dataset.labels_int2str.get(int(p), "") for p in preds],
        })
        summary = {
            "exp_name": exp_name,
            "split": split,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "num_samples": m["num_samples"],
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "label_mapping": {str(k): v for k, v in split_dataset.labels_int2str.items()},
        }

    csv_path = output_dir / f"predictions_{split}_{ts}.csv"
    summary_path = output_dir / f"summary_{split}_{ts}.json"
    df.to_csv(csv_path, index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Test inference complete: {len(df)} samples")
    print(f"  predictions -> {csv_path}")
    print(f"  summary     -> {summary_path}")
    if is_regression:
        print(f"  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
              f"Pearson r={m['pearson_r']:.4f}  Spearman r={m['spearman_r']:.4f}")
    else:
        print(f"  Accuracy={m['accuracy']:.2f}%  F1={m['f1']:.4f}")

    return summary
