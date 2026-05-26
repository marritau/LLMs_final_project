"""Notebook-facing facade API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .baselines import (
    ImprovedEnsembleDetector,
    LookBackLensStyleBaseline,
    always_clean_predict,
    keyword_tool_action_predict,
    lettucedetect_predict,
    nli_sentence_verifier_predict,
)
from .corruption import build_corrupted_dataset
from .data import (
    flatten_splits,
    load_toolace_rows,
    normalize_toolace_rows,
    read_jsonl,
    split_by_source_id,
    synthetic_toolace_records,
    write_jsonl,
)
from .metrics import choose_best_threshold, evaluate_predictions, per_type_metrics, sentence_metrics


def prepare_dataset(
    quick: bool = False,
    seed: int = 42,
    cache_dir: str | Path | None = "data/processed",
    max_base_records: int | None = None,
    allow_synthetic_fallback: bool | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Prepare clean/corrupted RAGTruth-style records.

    `quick=True` uses a tiny offline sample so the submitted notebook has a fast
    smoke-test path. `quick=False` loads ToolACE from Hugging Face.
    """

    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path and not quick:
        cached = _try_read_cached_splits(cache_path)
        if cached is not None and not _is_synthetic_dataset(cached):
            return cached

    if allow_synthetic_fallback is None:
        allow_synthetic_fallback = quick

    if quick:
        base_records = synthetic_toolace_records()
    else:
        try:
            rows = load_toolace_rows(max_records=max_base_records)
            base_records = normalize_toolace_rows(rows, max_records=max_base_records)
        except Exception:
            if not allow_synthetic_fallback:
                raise
            base_records = synthetic_toolace_records()

    if max_base_records is not None:
        base_records = base_records[:max_base_records]
    corrupted = build_corrupted_dataset(base_records, include_clean=True)
    splits = split_by_source_id(corrupted, seed=seed)

    if cache_path and not quick:
        for split, records in splits.items():
            write_jsonl(records, cache_path / f"{split}.jsonl")
    return splits


def run_baselines(
    quick: bool = False,
    dataset: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    dataset = dataset or prepare_dataset(quick=quick)
    validation_records = _non_empty_split(dataset, "validation")
    test_records = _non_empty_split(dataset, "test")

    methods: dict[str, Any] = {
        "always_clean": always_clean_predict,
        "keyword_tool_action": keyword_tool_action_predict,
    }

    lookback = LookBackLensStyleBaseline().fit(validation_records)
    methods["lookback_lens_style"] = lookback.predict

    if not quick:
        methods["nli_sentence_verifier"] = nli_sentence_verifier_predict
        methods["lettucedetect"] = lettucedetect_predict

    rows = []
    predictions_by_method = {}
    for method_name, predict_fn in methods.items():
        validation_predictions = predict_fn(validation_records)
        validation_scores = [float(prediction.get("score", 0.0)) for prediction in validation_predictions]
        validation_gold = [int(len(record.get("labels", [])) > 0) for record in validation_records]
        threshold = 0.5 if method_name == "always_clean" else choose_best_threshold(validation_gold, validation_scores)

        predictions = predict_fn(test_records)
        predictions_by_method[method_name] = predictions
        evaluated = evaluate_predictions(test_records, predictions, threshold=threshold)
        row = {"method": method_name, **{f"sentence_{k}": v for k, v in evaluated["sentence"].items()}}
        row.update({f"span_{k}": v for k, v in evaluated["span"].items()})
        rows.append(row)

    return {
        "records": test_records,
        "metrics": _to_table(rows),
        "predictions": predictions_by_method,
    }


def train_best_model(
    quick: bool = False,
    dataset: dict[str, list[dict[str, Any]]] | None = None,
    output_dir: str | Path = "artifacts/modernbert-token-classifier",
) -> ImprovedEnsembleDetector | Path:
    """Train the best available detector.

    Quick mode returns a fitted lightweight ensemble. Full mode attempts to
    fine-tune the transformer token classifier and falls back to the ensemble if
    model training dependencies or GPU resources are unavailable.
    """

    dataset = dataset or prepare_dataset(quick=quick)
    validation_records = _non_empty_split(dataset, "validation")
    if quick:
        return ImprovedEnsembleDetector().fit(validation_records)

    try:
        from .modeling import train_token_classifier

        return train_token_classifier(
            train_records=_non_empty_split(dataset, "train"),
            validation_records=validation_records,
            output_dir=output_dir,
        )
    except Exception:
        return ImprovedEnsembleDetector().fit(validation_records)


def predict_spans(
    records: Sequence[Mapping[str, Any]],
    model_path: str | Path | ImprovedEnsembleDetector | None = None,
) -> list[dict[str, Any]]:
    if isinstance(model_path, ImprovedEnsembleDetector):
        return model_path.predict(records)
    if model_path is not None:
        from .modeling import predict_with_token_classifier

        return predict_with_token_classifier(records, model_path)
    return ImprovedEnsembleDetector().fit(records).predict(records)


def evaluate_experiment(
    quick: bool = False,
    model_path: str | Path | None = None,
    seed: int = 42,
    max_base_records: int | None = None,
) -> dict[str, Any]:
    dataset = prepare_dataset(quick=quick, seed=seed, max_base_records=max_base_records)
    test_records = _non_empty_split(dataset, "test")
    validation_records = _non_empty_split(dataset, "validation")

    baseline_result = run_baselines(quick=quick, dataset=dataset)

    detector_or_path: ImprovedEnsembleDetector | Path
    if model_path is not None:
        detector_or_path = Path(model_path)
        improved_threshold = 0.5
    else:
        detector_or_path = train_best_model(quick=quick, dataset=dataset)
        improved_threshold = detector_or_path.threshold if isinstance(detector_or_path, ImprovedEnsembleDetector) else 0.5

    improved_predictions = predict_spans(test_records, detector_or_path)
    improved_eval = evaluate_predictions(test_records, improved_predictions, threshold=improved_threshold)

    rows = _table_to_rows(baseline_result["metrics"])
    improved_row = {
        "method": "improved_ensemble" if isinstance(detector_or_path, ImprovedEnsembleDetector) else "modernbert_token_classifier",
        **{f"sentence_{k}": v for k, v in improved_eval["sentence"].items()},
        **{f"span_{k}": v for k, v in improved_eval["span"].items()},
    }
    rows.append(improved_row)

    validation_scores = [prediction["score"] for prediction in ImprovedEnsembleDetector().fit(validation_records).predict(validation_records)]
    validation_gold = [int(len(record.get("labels", [])) > 0) for record in validation_records]
    validation_summary = sentence_metrics(validation_gold, validation_scores, choose_best_threshold(validation_gold, validation_scores))

    return {
        "dataset": dataset,
        "test_records": test_records,
        "sentence_metrics": _to_table([{key: row[key] for key in row if key.startswith("sentence_") or key == "method"} for row in rows]),
        "span_metrics": _to_table([{key: row[key] for key in row if key.startswith("span_") or key == "method"} for row in rows]),
        "per_type_metrics": _to_table(per_type_metrics(test_records, improved_predictions, threshold=improved_threshold)),
        "all_metrics": _to_table(rows),
        "predictions": {"improved": improved_predictions, **baseline_result["predictions"]},
        "validation_summary": validation_summary,
    }


def _try_read_cached_splits(cache_path: Path) -> dict[str, list[dict[str, Any]]] | None:
    paths = {split: cache_path / f"{split}.jsonl" for split in ("train", "validation", "test")}
    if not all(path.exists() for path in paths.values()):
        return None
    return {split: read_jsonl(path) for split, path in paths.items()}


def _is_synthetic_dataset(dataset: Mapping[str, list[dict[str, Any]]]) -> bool:
    records = flatten_splits(dataset)
    if not records:
        return False
    return all(str(record.get("source_id", "")).startswith("synthetic-") for record in records)


def _non_empty_split(dataset: Mapping[str, list[dict[str, Any]]], split: str) -> list[dict[str, Any]]:
    records = list(dataset.get(split, []))
    if records:
        return records
    return flatten_splits(dataset)


def _to_table(rows: list[dict[str, Any]]) -> Any:
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except Exception:
        return rows


def _table_to_rows(table: Any) -> list[dict[str, Any]]:
    if hasattr(table, "to_dict"):
        return list(table.to_dict(orient="records"))
    return list(table)
