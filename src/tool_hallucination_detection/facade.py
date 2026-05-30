"""Notebook-facing facade API."""
from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any, Mapping, Sequence

from .baselines import (
    ImprovedEnsembleDetector,
    AttentionLookBackLensBaseline,
    LookBackLensStyleBaseline,
    RandomPriorBaseline,
    TfidfLogRegBaseline,
    always_clean_predict,
    always_hallucinated_predict,
    random_balanced_predict,
    keyword_tool_action_predict,
    lettucedetect_predict,
    nli_sentence_verifier_predict,
    value_checker_predict,
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
    include_clean_hard: bool = True,
    allow_synthetic_fallback: bool | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Prepare clean/corrupted RAGTruth-style records."""
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path and not quick:
        cached = _try_read_cached_splits(cache_path)
        if (
            cached is not None
            and not _is_synthetic_dataset(cached)
            and (not include_clean_hard or any(record.get("corruption_type") == "clean_hard" for record in flatten_splits(cached)))
        ):
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

    corrupted = build_corrupted_dataset(base_records, include_clean=True, include_clean_hard=include_clean_hard)
    splits = split_by_source_id(corrupted, seed=seed)
    if cache_path and not quick:
        for split, records in splits.items():
            write_jsonl(records, cache_path / f"{split}.jsonl")
    return splits


def run_baselines(
    quick: bool = False,
    dataset: dict[str, list[dict[str, Any]]] | None = None,
    include_lettucedetect_fallback: bool = False,
    require_real_lettucedetect: bool = False,
    lettuce_model_path: str | None = None,
    include_attention_lookback: bool = True,
    require_attention_lookback: bool = False,
    attention_lookback_model_name: str = "distilgpt2",
    attention_lookback_train_limit: int = 240,
    attention_lookback_max_length: int = 384,
    attention_lookback_max_answer_tokens: int = 96,
) -> dict[str, Any]:
    dataset = dataset or prepare_dataset(quick=quick)
    train_records = _non_empty_split(dataset, "train")
    validation_records = _non_empty_split(dataset, "validation")
    test_records = _non_empty_split(dataset, "test")

    rows = []
    predictions_by_method = {}
    thresholds_by_method = {}
    availability_rows = []

    methods: dict[str, Any] = {
        "always_clean": always_clean_predict,
        "always_hallucinated": always_hallucinated_predict,
        "random_balanced": random_balanced_predict,
        "random_prior": RandomPriorBaseline().fit(validation_records).predict,
        "keyword_tool_action": keyword_tool_action_predict,
        "value_checker": value_checker_predict,
        "tfidf_logreg": TfidfLogRegBaseline().fit(train_records).predict,
    }

    # Lexical proxy is kept as a sanity baseline, not as the main LookBackLens result.
    lookback_proxy = LookBackLensStyleBaseline().fit(validation_records)
    methods["lookback_lens_style_proxy"] = lookback_proxy.predict

    if include_attention_lookback and not quick:
        try:
            attention_lookback = AttentionLookBackLensBaseline(
                model_name=attention_lookback_model_name,
                max_train_records=attention_lookback_train_limit,
                max_length=attention_lookback_max_length,
                max_answer_tokens=attention_lookback_max_answer_tokens,
            ).fit(train_records)
            methods["attention_lookback_lens_adapted"] = attention_lookback.predict
        except Exception as exc:
            availability_rows.append({"method": "attention_lookback_lens_adapted", "status": "unavailable", "details": repr(exc)})
            if require_attention_lookback:
                raise

    if not quick:
        methods["nli_sentence_verifier"] = nli_sentence_verifier_predict
        methods["lettucedetect_real"] = lambda records: lettucedetect_predict(
            records,
            model_path=lettuce_model_path,
            fallback_to_keyword=include_lettucedetect_fallback,
        )

    for method_name, predict_fn in methods.items():
        try:
            validation_predictions = predict_fn(validation_records)
            predictions = predict_fn(test_records)
        except Exception as exc:
            availability_rows.append({"method": method_name, "status": "unavailable", "details": repr(exc)})
            if method_name == "lettucedetect_real" and require_real_lettucedetect:
                raise
            if method_name == "attention_lookback_lens_adapted" and require_attention_lookback:
                raise
            continue

        validation_scores = [float(prediction.get("score", 0.0)) for prediction in validation_predictions]
        validation_gold = [int(len(record.get("labels", [])) > 0) for record in validation_records]
        threshold = 0.5 if method_name == "always_clean" else choose_best_threshold(validation_gold, validation_scores, metric="macro_f1")

        actual_method = str(predictions[0].get("method", method_name)) if predictions else method_name
        predictions_by_method[actual_method] = predictions
        evaluated = evaluate_predictions(test_records, predictions, threshold=threshold)
        thresholds_by_method[actual_method] = threshold
        row = {"method": actual_method, **{f"sentence_{k}": v for k, v in evaluated["sentence"].items()}}
        row.update({f"span_{k}": v for k, v in evaluated["span"].items()})
        rows.append(row)
        if "fallback" in actual_method:
            status = "fallback"
        elif actual_method == "lettucedetect_real":
            status = "real"
        elif actual_method == "attention_lookback_lens_adapted":
            status = "attention_adapted"
        else:
            status = "ok"
        availability_rows.append({"method": actual_method, "status": status, "details": ""})

    return {
        "records": test_records,
        "metrics": _to_table(rows),
        "predictions": predictions_by_method,
        "thresholds": thresholds_by_method,
        "availability": _to_table(availability_rows),
    }


def train_best_model(
    quick: bool = False,
    dataset: dict[str, list[dict[str, Any]]] | None = None,
    output_dir: str | Path = "artifacts/modernbert-token-classifier",
    strict: bool = False,
    model_name: str = "answerdotai/ModernBERT-base",
    epochs: float = 1.0,
    batch_size: int = 2,
    max_length: int = 2048,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    fp16: bool | None = None,
    early_stopping_patience: int | None = None,
    gradient_checkpointing: bool = False,
    use_class_weights: bool = True,
) -> ImprovedEnsembleDetector | Path:
    """Train the best available detector.

    Quick mode returns a fitted lightweight ensemble. Full mode attempts to
    fine-tune the weighted transformer token classifier and falls back to the
    ensemble when strict=False.
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
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            max_length=max_length,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            fp16=fp16,
            early_stopping_patience=early_stopping_patience,
            gradient_checkpointing=gradient_checkpointing,
            use_class_weights=use_class_weights,
        )
    except Exception as exc:
        if strict:
            raise
        detector = ImprovedEnsembleDetector().fit(validation_records)
        setattr(detector, "training_error", repr(exc))
        warnings.warn(
            "Token-classifier training failed; falling back to the lightweight ensemble. "
            f"Original error: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return detector


def predict_spans(
    records: Sequence[Mapping[str, Any]],
    model_path: str | Path | ImprovedEnsembleDetector | None = None,
    span_threshold: float = 0.5,
    max_length: int = 2048,
    score_aggregator: str = "max_token_prob",
) -> list[dict[str, Any]]:
    if isinstance(model_path, ImprovedEnsembleDetector):
        return model_path.predict(records)
    if model_path is not None:
        from .modeling import predict_with_token_classifier
        return predict_with_token_classifier(
            records,
            model_path,
            threshold=span_threshold,
            max_length=max_length,
            score_aggregator=score_aggregator,
        )
    return ImprovedEnsembleDetector().fit(records).predict(records)


def evaluate_experiment(
    quick: bool = False,
    model_path: str | Path | None = None,
    seed: int = 42,
    max_base_records: int | None = None,
    include_clean_hard: bool = True,
    strict_training: bool = False,
    output_dir: str | Path = "artifacts/modernbert-token-classifier",
    model_name: str = "answerdotai/ModernBERT-base",
    epochs: float = 1.0,
    batch_size: int = 2,
    max_length: int = 2048,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    fp16: bool | None = None,
    early_stopping_patience: int | None = None,
    gradient_checkpointing: bool = False,
    score_aggregator: str = "max_token_prob",
    cache_dir: str | Path | None = "data/processed",
    use_class_weights: bool = True,
    include_lettucedetect_fallback: bool = False,
    require_real_lettucedetect: bool = False,
    lettuce_model_path: str | None = None,
    include_attention_lookback: bool = True,
    require_attention_lookback: bool = False,
    attention_lookback_model_name: str = "distilgpt2",
    attention_lookback_train_limit: int = 240,
    attention_lookback_max_length: int = 384,
    attention_lookback_max_answer_tokens: int = 96,
) -> dict[str, Any]:
    dataset = prepare_dataset(
        quick=quick,
        seed=seed,
        max_base_records=max_base_records,
        cache_dir=cache_dir,
        include_clean_hard=include_clean_hard,
    )
    test_records = _non_empty_split(dataset, "test")
    validation_records = _non_empty_split(dataset, "validation")

    baseline_result = run_baselines(
        quick=quick,
        dataset=dataset,
        include_lettucedetect_fallback=include_lettucedetect_fallback,
        require_real_lettucedetect=require_real_lettucedetect,
        lettuce_model_path=lettuce_model_path,
        include_attention_lookback=include_attention_lookback,
        require_attention_lookback=require_attention_lookback,
        attention_lookback_model_name=attention_lookback_model_name,
        attention_lookback_train_limit=attention_lookback_train_limit,
        attention_lookback_max_length=attention_lookback_max_length,
        attention_lookback_max_answer_tokens=attention_lookback_max_answer_tokens,
    )

    if model_path is not None:
        detector_or_path: ImprovedEnsembleDetector | Path = Path(model_path)
    else:
        detector_or_path = train_best_model(
            quick=quick,
            dataset=dataset,
            output_dir=output_dir,
            strict=strict_training,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            max_length=max_length,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            fp16=fp16,
            early_stopping_patience=early_stopping_patience,
            gradient_checkpointing=gradient_checkpointing,
            use_class_weights=use_class_weights,
        )

    threshold_info: dict[str, float]
    if isinstance(detector_or_path, ImprovedEnsembleDetector):
        improved_sentence_threshold = detector_or_path.threshold
        improved_span_threshold = detector_or_path.threshold
        threshold_info = {
            "sentence_threshold": float(improved_sentence_threshold),
            "span_threshold": float(improved_span_threshold),
        }
    else:
        from .modeling import tune_token_classifier_thresholds
        threshold_info = tune_token_classifier_thresholds(
            validation_records=validation_records,
            model_path=detector_or_path,
            max_length=max_length,
            sentence_metric="macro_f1",
            span_metric="char_f1",
            score_aggregator=score_aggregator,
        )
        improved_sentence_threshold = float(threshold_info["sentence_threshold"])
        improved_span_threshold = float(threshold_info["span_threshold"])

    improved_predictions = predict_spans(
        test_records,
        detector_or_path,
        span_threshold=improved_span_threshold,
        max_length=max_length,
        score_aggregator=score_aggregator,
    )
    improved_eval = evaluate_predictions(test_records, improved_predictions, threshold=improved_sentence_threshold)

    rows = _table_to_rows(baseline_result["metrics"])
    improved_method = "hybrid_value_ensemble" if isinstance(detector_or_path, ImprovedEnsembleDetector) else "modernbert_token_classifier"
    improved_row = {
        "method": improved_method,
        **{f"sentence_{k}": v for k, v in improved_eval["sentence"].items()},
        **{f"span_{k}": v for k, v in improved_eval["span"].items()},
    }
    rows.append(improved_row)

    final_predictions = improved_predictions
    if not isinstance(detector_or_path, ImprovedEnsembleDetector):
        value_validation_predictions = value_checker_predict(validation_records)
        value_test_predictions = value_checker_predict(test_records)
        modernbert_validation_predictions = predict_spans(
            validation_records,
            detector_or_path,
            span_threshold=improved_span_threshold,
            max_length=max_length,
            score_aggregator=score_aggregator,
        )
        validation_gold = [int(len(record.get("labels", [])) > 0) for record in validation_records]
        hybrid_validation_scores = [
            max(float(neural.get("score", 0.0)), float(value.get("score", 0.0)))
            for neural, value in zip(modernbert_validation_predictions, value_validation_predictions)
        ]
        hybrid_sentence_threshold = choose_best_threshold(validation_gold, hybrid_validation_scores, metric="macro_f1")
        hybrid_predictions = _combine_predictions(
            improved_predictions,
            value_test_predictions,
            method="hybrid_modernbert_value_checker",
        )
        hybrid_eval = evaluate_predictions(test_records, hybrid_predictions, threshold=hybrid_sentence_threshold)
        rows.append({
            "method": "hybrid_modernbert_value_checker",
            **{f"sentence_{k}": v for k, v in hybrid_eval["sentence"].items()},
            **{f"span_{k}": v for k, v in hybrid_eval["span"].items()},
        })
        threshold_info["value_checker_threshold"] = choose_best_threshold(
            validation_gold,
            [float(prediction.get("score", 0.0)) for prediction in value_validation_predictions],
            metric="macro_f1",
        )
        threshold_info["hybrid_sentence_threshold"] = float(hybrid_sentence_threshold)
        final_predictions = hybrid_predictions

    validation_scores = [prediction["score"] for prediction in ImprovedEnsembleDetector().fit(validation_records).predict(validation_records)]
    validation_gold = [int(len(record.get("labels", [])) > 0) for record in validation_records]
    validation_summary = sentence_metrics(validation_gold, validation_scores, choose_best_threshold(validation_gold, validation_scores, metric="macro_f1"))

    thresholds = {**baseline_result.get("thresholds", {}), improved_method: improved_sentence_threshold}
    thresholds[f"{improved_method}__sentence"] = improved_sentence_threshold
    thresholds[f"{improved_method}__span"] = improved_span_threshold
    if "hybrid_sentence_threshold" in threshold_info:
        thresholds["hybrid_modernbert_value_checker"] = float(threshold_info["hybrid_sentence_threshold"])

    return {
        "dataset": dataset,
        "test_records": test_records,
        "sentence_metrics": _to_table([{key: row[key] for key in row if key.startswith("sentence_") or key == "method"} for row in rows]),
        "span_metrics": _to_table([{key: row[key] for key in row if key.startswith("span_") or key == "method"} for row in rows]),
        "per_type_metrics": _to_table(per_type_metrics(
            test_records,
            final_predictions,
            threshold=float(threshold_info.get("hybrid_sentence_threshold", improved_sentence_threshold)),
        )),
        "all_metrics": _to_table(rows),
        "predictions": {"improved": final_predictions, **baseline_result["predictions"]},
        "validation_summary": validation_summary,
        "thresholds": thresholds,
        "threshold_info": threshold_info,
        "baseline_availability": baseline_result.get("availability"),
        "training_error": getattr(detector_or_path, "training_error", None),
        "model_path": str(detector_or_path) if isinstance(detector_or_path, Path) else None,
    }


def export_splits(dataset: Mapping[str, list[dict[str, Any]]], output_dir: str | Path = "artifacts/ragtruth_toolace") -> dict[str, str]:
    output_dir = Path(output_dir)
    paths = {}
    for split, records in dataset.items():
        path = write_jsonl(records, output_dir / f"{split}.jsonl")
        paths[split] = str(path)
    return paths


def _combine_predictions(
    left_predictions: Sequence[Mapping[str, Any]],
    right_predictions: Sequence[Mapping[str, Any]],
    method: str,
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for left, right in zip(left_predictions, right_predictions):
        spans = list(left.get("spans", [])) + list(right.get("spans", []))
        score = max(float(left.get("score", 0.0)), float(right.get("score", 0.0)))
        combined.append({"spans": spans, "score": score, "method": method})
    return combined


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
