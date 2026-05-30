from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def dataset_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = dict(config.get("dataset", {}) or {})
    return {
        "quick": bool(config.get("quick", False)),
        "seed": int(dataset.get("seed", 42)),
        "cache_dir": dataset.get("cache_dir"),
        "max_base_records": dataset.get("max_base_records"),
        "include_clean_hard": bool(dataset.get("include_clean_hard", True)),
        "allow_synthetic_fallback": dataset.get("allow_synthetic_fallback"),
    }


def training_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    training = dict(config.get("training", {}) or {})
    return {
        "output_dir": training.get("output_dir", "artifacts/modernbert-token-classifier"),
        "strict": bool(training.get("strict", False)),
        "model_name": training.get("model_name", "answerdotai/ModernBERT-base"),
        "epochs": float(training.get("epochs", 1.0)),
        "batch_size": int(training.get("batch_size", 2)),
        "max_length": int(training.get("max_length", 2048)),
        "learning_rate": float(training.get("learning_rate", 2e-5)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.06)),
        "fp16": training.get("fp16"),
        "early_stopping_patience": training.get("early_stopping_patience"),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", False)),
        "use_class_weights": bool(training.get("use_class_weights", True)),
    }


def baseline_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = dict(config.get("evaluation", {}) or {})
    return {
        "include_lettucedetect_fallback": bool(evaluation.get("include_lettucedetect_fallback", False)),
        "require_real_lettucedetect": bool(evaluation.get("require_real_lettucedetect", False)),
        "include_attention_lookback": bool(evaluation.get("include_attention_lookback", True)),
        "require_attention_lookback": bool(evaluation.get("require_attention_lookback", False)),
        "attention_lookback_model_name": evaluation.get("attention_lookback_model_name", "distilgpt2"),
        "attention_lookback_train_limit": int(evaluation.get("attention_lookback_train_limit", 240)),
        "attention_lookback_max_length": int(evaluation.get("attention_lookback_max_length", 384)),
        "attention_lookback_max_answer_tokens": int(evaluation.get("attention_lookback_max_answer_tokens", 96)),
    }


def output_dirs(config: Mapping[str, Any]) -> tuple[Path, Path]:
    outputs = dict(config.get("outputs", {}) or {})
    artifacts_dir = Path(outputs.get("artifacts_dir", "artifacts"))
    results_dir = Path(outputs.get("results_dir", "results"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir, results_dir


def table_rows(table: Any) -> list[dict[str, Any]]:
    if hasattr(table, "to_dict"):
        return list(table.to_dict(orient="records"))
    return list(table)


def write_table(table: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(table, "to_csv"):
        table.to_csv(path, index=False)
        return path
    rows = table_rows(table)
    if path.suffix == ".csv":
        import pandas as pd

        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def dataset_statistics(dataset: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for split, records in dataset.items():
        by_type: dict[str, int] = {}
        for record in records:
            key = str(record.get("corruption_type", "unknown"))
            by_type[key] = by_type.get(key, 0) + 1
        for corruption_type, count in sorted(by_type.items()):
            rows.append({"split": split, "corruption_type": corruption_type, "n": count})
    return rows


def corruption_audit(dataset: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for split, records in dataset.items():
        for record in records:
            labels = list(record.get("labels") or [])
            for label in labels:
                start = int(label["start"])
                end = int(label["end"])
                output = str(record.get("output", ""))
                rows.append({
                    "split": split,
                    "id": record.get("id"),
                    "corruption_type": record.get("corruption_type"),
                    "corruption_style": record.get("corruption_style", ""),
                    "label_type": label.get("label_type"),
                    "span_length": end - start,
                    "relative_start": start / max(1, len(output)),
                    "inserted_at_end": int(end >= len(output) - 1),
                    "meta": label.get("meta", ""),
                })
    return rows
