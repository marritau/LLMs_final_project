"""Sentence-level and span-level evaluation metrics."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import coerce_labels, sentence_label


def sentence_metrics(gold: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> dict[str, float]:
    """Binary sample-level metrics.

    The positive class means: this answer contains at least one hallucinated span.
    In this project the dataset usually contains 1 clean and 3 corrupted variants per
    base dialogue, so positive-class F1 alone can be misleading. We therefore report
    macro-F1, balanced accuracy, PR-AUC, ROC-AUC, and confusion counts as well.
    """
    gold_arr = np.asarray(gold, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    if gold_arr.size == 0:
        return _empty_sentence_metrics(threshold)

    pred_arr = (score_arr >= threshold).astype(int)

    tp = int(((gold_arr == 1) & (pred_arr == 1)).sum())
    fp = int(((gold_arr == 0) & (pred_arr == 1)).sum())
    fn = int(((gold_arr == 1) & (pred_arr == 0)).sum())
    tn = int(((gold_arr == 0) & (pred_arr == 0)).sum())

    pos_precision = _safe_div(tp, tp + fp)
    pos_recall = _safe_div(tp, tp + fn)
    pos_f1 = _f1(pos_precision, pos_recall)

    neg_precision = _safe_div(tn, tn + fn)
    neg_recall = _safe_div(tn, tn + fp)  # specificity
    neg_f1 = _f1(neg_precision, neg_recall)

    accuracy = _safe_div(tp + tn, len(gold_arr))
    balanced_accuracy = (pos_recall + neg_recall) / 2
    macro_precision = (pos_precision + neg_precision) / 2
    macro_recall = (pos_recall + neg_recall) / 2
    macro_f1 = (pos_f1 + neg_f1) / 2

    metrics = {
        "n": float(len(gold_arr)),
        "positive_rate": float(gold_arr.mean()),
        "predicted_positive_rate": float(pred_arr.mean()),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": pos_precision,
        "recall": pos_recall,
        "f1": pos_f1,
        "negative_precision": neg_precision,
        "negative_recall": neg_recall,
        "negative_f1": neg_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "specificity": neg_recall,
        "threshold": float(threshold),
        "roc_auc": _roc_auc(gold_arr, score_arr),
        "pr_auc": _pr_auc(gold_arr, score_arr),
        "brier": _brier(gold_arr, score_arr),
        "ece_10": _ece(gold_arr, score_arr, n_bins=10),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }
    return metrics


def choose_best_threshold(gold: Sequence[int], scores: Sequence[float], metric: str = "macro_f1") -> float:
    """Pick a validation threshold.

    `macro_f1` is the default because positive-class F1 can reward an
    always-hallucinated classifier on the 1-clean/3-corrupted dataset.
    """
    unique_scores = sorted(set(float(score) for score in scores))
    candidates = {0.5, 1.0}
    for score in unique_scores:
        candidates.add(score + 1e-9)
    for left, right in zip(unique_scores, unique_scores[1:]):
        candidates.add((left + right) / 2)
    if unique_scores:
        candidates.add(min(unique_scores) - 1e-9)
        candidates.add(max(unique_scores) + 1e-9)

    best_threshold = 0.5
    best_value = -1.0
    for threshold in sorted(candidates):
        metrics = sentence_metrics(gold, scores, threshold)
        value = float(metrics.get(metric, metrics.get("f1", 0.0)))
        if value > best_value:
            best_value = value
            best_threshold = threshold
    return float(best_threshold)


def span_metrics(
    gold_spans: Sequence[Sequence[Mapping[str, Any]]],
    predicted_spans: Sequence[Sequence[Mapping[str, Any]]],
    output_lengths: Sequence[int] | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    if output_lengths is None:
        max_end = 0
        for spans in list(gold_spans) + list(predicted_spans):
            for span in spans:
                max_end = max(max_end, int(span["end"]))
        output_lengths = [max_end + 1 for _ in gold_spans]

    char_tp = char_fp = char_fn = 0
    exact_tp = exact_fp = exact_fn = 0
    relaxed_tp = relaxed_fp = relaxed_fn = 0

    for gold, pred, length in zip(gold_spans, predicted_spans, output_lengths):
        gold_ranges = [_span_tuple(span) for span in gold]
        pred_ranges = [_span_tuple(span) for span in pred]
        gold_mask = _mask(gold_ranges, length)
        pred_mask = _mask(pred_ranges, length)

        char_tp += int(np.logical_and(gold_mask, pred_mask).sum())
        char_fp += int(np.logical_and(~gold_mask, pred_mask).sum())
        char_fn += int(np.logical_and(gold_mask, ~pred_mask).sum())

        gold_set = set(gold_ranges)
        pred_set = set(pred_ranges)
        exact_tp += len(gold_set & pred_set)
        exact_fp += len(pred_set - gold_set)
        exact_fn += len(gold_set - pred_set)

        matched_gold: set[int] = set()
        matched_pred: set[int] = set()
        for pred_index, pred_span in enumerate(pred_ranges):
            best_gold_index = None
            best_iou = 0.0
            for gold_index, gold_span in enumerate(gold_ranges):
                if gold_index in matched_gold:
                    continue
                score = _iou(pred_span, gold_span)
                if score > best_iou:
                    best_iou = score
                    best_gold_index = gold_index
            if best_gold_index is not None and best_iou >= iou_threshold:
                matched_gold.add(best_gold_index)
                matched_pred.add(pred_index)
        relaxed_tp += len(matched_pred)
        relaxed_fp += len(pred_ranges) - len(matched_pred)
        relaxed_fn += len(gold_ranges) - len(matched_gold)

    char_precision = _safe_div(char_tp, char_tp + char_fp)
    char_recall = _safe_div(char_tp, char_tp + char_fn)
    exact_precision = _safe_div(exact_tp, exact_tp + exact_fp)
    exact_recall = _safe_div(exact_tp, exact_tp + exact_fn)
    relaxed_precision = _safe_div(relaxed_tp, relaxed_tp + relaxed_fp)
    relaxed_recall = _safe_div(relaxed_tp, relaxed_tp + relaxed_fn)

    return {
        "char_precision": char_precision,
        "char_recall": char_recall,
        "char_f1": _f1(char_precision, char_recall),
        "exact_span_precision": exact_precision,
        "exact_span_recall": exact_recall,
        "exact_span_f1": _f1(exact_precision, exact_recall),
        "relaxed_span_precision": relaxed_precision,
        "relaxed_span_recall": relaxed_recall,
        "relaxed_span_f1": _f1(relaxed_precision, relaxed_recall),
    }


def evaluate_predictions(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    gold = [sentence_label(record) for record in records]
    scores = [float(prediction.get("score", 0.0)) for prediction in predictions]
    gold_spans = [record.get("labels", []) for record in records]
    # Apply the same sentence-level threshold to spans. This keeps the binary and span
    # views consistent for methods that return a score and candidate spans.
    predicted_spans = [
        prediction.get("spans", []) if float(prediction.get("score", 0.0)) >= threshold else []
        for prediction in predictions
    ]
    lengths = [len(str(record.get("output", ""))) for record in records]
    return {
        "sentence": sentence_metrics(gold, scores, threshold),
        "span": span_metrics(gold_spans, predicted_spans, lengths),
    }


def per_type_metrics(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    threshold: float = 0.5,
) -> list[dict[str, float | str]]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_type[str(record.get("corruption_type", "unknown"))].append(index)

    rows: list[dict[str, float | str]] = []
    for corruption_type, indices in sorted(by_type.items()):
        subset_records = [records[index] for index in indices]
        subset_predictions = [predictions[index] for index in indices]
        evaluated = evaluate_predictions(subset_records, subset_predictions, threshold)
        row: dict[str, float | str] = {"corruption_type": corruption_type, "n": len(indices)}
        row.update({f"sentence_{key}": value for key, value in evaluated["sentence"].items()})
        row.update({f"span_{key}": value for key, value in evaluated["span"].items()})
        rows.append(row)
    return rows


def labels_to_char_spans(labels: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    return [_span_tuple(label) for label in labels]


def _mask(spans: Sequence[tuple[int, int]], length: int) -> np.ndarray:
    mask = np.zeros(max(0, int(length)), dtype=bool)
    for start, end in spans:
        mask[max(0, start) : min(len(mask), end)] = True
    return mask


def _span_tuple(span: Mapping[str, Any]) -> tuple[int, int]:
    return int(span["start"]), int(span["end"])


def _iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return _safe_div(intersection, union)


def _roc_auc(gold: np.ndarray, scores: np.ndarray) -> float:
    if len(set(gold.tolist())) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(gold, scores))
    except Exception:
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        pos = gold == 1
        n_pos = int(pos.sum())
        n_neg = int((~pos).sum())
        return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pr_auc(gold: np.ndarray, scores: np.ndarray) -> float:
    if len(set(gold.tolist())) < 2:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(gold, scores))
    except Exception:
        return float("nan")


def _brier(gold: np.ndarray, scores: np.ndarray) -> float:
    scores = np.clip(scores.astype(float), 0.0, 1.0)
    return float(np.mean((scores - gold.astype(float)) ** 2)) if len(gold) else float("nan")


def _ece(gold: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    if len(gold) == 0:
        return float("nan")
    scores = np.clip(scores.astype(float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            mask = (scores >= left) & (scores <= right)
        else:
            mask = (scores >= left) & (scores < right)
        if not mask.any():
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(gold[mask].mean())
        ece += float(mask.mean()) * abs(confidence - accuracy)
    return float(ece)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def _empty_sentence_metrics(threshold: float) -> dict[str, float]:
    keys = [
        "n", "positive_rate", "predicted_positive_rate", "accuracy", "balanced_accuracy",
        "precision", "recall", "f1", "negative_precision", "negative_recall", "negative_f1",
        "macro_precision", "macro_recall", "macro_f1", "specificity", "threshold",
        "roc_auc", "pr_auc", "brier", "ece_10", "tp", "fp", "fn", "tn",
    ]
    values = {key: float("nan") for key in keys}
    values["threshold"] = float(threshold)
    values["n"] = 0.0
    return values


def coerce_gold_spans(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [label.to_dict() for label in coerce_labels(record.get("labels"))]
