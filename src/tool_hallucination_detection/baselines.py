"""Baseline detectors for the hallucination benchmark."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import choose_best_threshold
from .schema import sentence_label


Prediction = dict[str, Any]

SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
HIGH_SIGNAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bpast few months\b",
        r"\bremained stable\b",
        r"\bbook a flight\b",
        r"\breserve a hotel\b",
        r"\bproceed with the payment\b",
        r"\bsend a confirmation email\b",
        r"\badd this to your calendar\b",
        r"\bcomplete the next external action\b",
        r"\bcontradicts the tool output\b",
    ]
]
CONFLICT_PAIRS = [
    ("sunny", "rainy"),
    ("up", "down"),
    ("open", "closed"),
    ("available", "unavailable"),
    ("approved", "rejected"),
    ("success", "failure"),
    ("true", "false"),
]


def always_clean_predict(records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
    return [{"spans": [], "score": 0.0, "method": "always_clean"} for _ in records]


def keyword_tool_action_predict(records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
    """Fast sanity baseline for the injected hallucination types."""

    return [_keyword_prediction(record) for record in records]


def nli_sentence_verifier_predict(
    records: Sequence[Mapping[str, Any]],
    model_name: str = "MoritzLaurer/deberta-v3-base-mnli-fever-anli",
    fallback_to_keyword: bool = True,
) -> list[Prediction]:
    """Optional open-source NLI baseline.

    If the model is unavailable, the function falls back to the deterministic
    keyword baseline by default so the notebook remains runnable.
    """

    try:
        from transformers import pipeline
    except Exception:
        if fallback_to_keyword:
            return keyword_tool_action_predict(records)
        raise

    try:
        classifier = pipeline("text-classification", model=model_name, top_k=None, truncation=True)
    except Exception:
        if fallback_to_keyword:
            return keyword_tool_action_predict(records)
        raise

    predictions: list[Prediction] = []
    for record in records:
        spans = []
        max_score = 0.0
        context = f"Question: {record['query']}\nTool output: {record['context']}"
        for start, end, sentence in _iter_sentence_spans(str(record["output"])):
            if not sentence.strip():
                continue
            result = classifier({"text": context, "text_pair": sentence})[0]
            label_scores = {item["label"].lower(): float(item["score"]) for item in result}
            contradiction = max(
                [score for label, score in label_scores.items() if "contradiction" in label],
                default=0.0,
            )
            neutral = max([score for label, score in label_scores.items() if "neutral" in label], default=0.0)
            score = max(contradiction, neutral * 0.65)
            if score >= 0.5:
                spans.append(_span(start, end, sentence, score, "nli_sentence_verifier"))
            max_score = max(max_score, score)
        predictions.append({"spans": spans, "score": max_score, "method": "nli_sentence_verifier"})
    return predictions


def lettucedetect_predict(
    records: Sequence[Mapping[str, Any]],
    model_path: str | None = None,
    fallback_to_keyword: bool = True,
) -> list[Prediction]:
    """Run LettuceDetect when it is available."""

    try:
        from lettucedetect.models.inference import HallucinationDetector
    except Exception:
        if fallback_to_keyword:
            return keyword_tool_action_predict(records)
        raise

    try:
        detector = HallucinationDetector(model_path=model_path) if model_path else HallucinationDetector()
    except Exception:
        if fallback_to_keyword:
            return keyword_tool_action_predict(records)
        raise

    predictions: list[Prediction] = []
    for record in records:
        raw = detector.predict(
            context=str(record["context"]),
            question=str(record["query"]),
            answer=str(record["output"]),
            output_format="spans",
        )
        spans = []
        for item in raw:
            start = int(_get_field(item, "start"))
            end = int(_get_field(item, "end"))
            confidence = float(_get_field(item, "hallucination_score", _get_field(item, "confidence", 0.5)))
            text = str(record["output"])[start:end]
            spans.append(_span(start, end, text, confidence, "lettucedetect"))
        score = max([span["confidence"] for span in spans], default=0.0)
        predictions.append({"spans": spans, "score": score, "method": "lettucedetect"})
    return predictions


@dataclass
class LookBackLensStyleBaseline:
    """Small reproducible approximation of LookBackLens-style scoring.

    Full LookBackLens uses attention maps from a decoder LLM. This class keeps a
    compatible experiment slot for the course project and can optionally train a
    logistic regression over sentence support features. It is fast enough for
    quick runs and gives a transparent baseline when attention extraction is too
    expensive for a notebook smoke test.
    """

    threshold: float = 0.5
    weights: np.ndarray | None = None

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "LookBackLensStyleBaseline":
        features = []
        labels = []
        for record in records:
            record_labels = record.get("labels", [])
            for start, end, sentence in _iter_sentence_spans(str(record["output"])):
                features.append(_support_features(record, sentence))
                labels.append(int(_overlaps_any(start, end, record_labels)))

        if len(set(labels)) < 2:
            self.weights = np.zeros(4)
            self.threshold = 0.5
            return self

        try:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(np.asarray(features), np.asarray(labels))
            scores = clf.predict_proba(np.asarray(features))[:, 1]
            self.threshold = choose_best_threshold(labels, scores)
            self.weights = np.concatenate([clf.intercept_, clf.coef_.ravel()])
        except Exception:
            self.weights = np.asarray([-1.0, -2.0, 1.2, 0.8, 0.6])
            scores = [_sigmoid(float(np.dot(self.weights, [1.0, *feat]))) for feat in features]
            self.threshold = choose_best_threshold(labels, scores)
        return self

    def predict(self, records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        predictions: list[Prediction] = []
        for record in records:
            spans = []
            max_score = 0.0
            for start, end, sentence in _iter_sentence_spans(str(record["output"])):
                features = _support_features(record, sentence)
                score = self._score(features)
                if score >= self.threshold:
                    spans.append(_span(start, end, sentence, score, "lookback_lens_style"))
                max_score = max(max_score, score)
            predictions.append({"spans": spans, "score": max_score, "method": "lookback_lens_style"})
        return predictions

    def _score(self, features: Sequence[float]) -> float:
        if self.weights is None:
            support, novelty, action, conflict = features
            return max(0.0, min(1.0, 0.15 + 0.55 * novelty + 0.25 * action + 0.20 * conflict - 0.25 * support))
        return _sigmoid(float(np.dot(self.weights, [1.0, *features])))


@dataclass
class ImprovedEnsembleDetector:
    threshold: float = 0.5
    lookback: LookBackLensStyleBaseline | None = None

    def fit(self, validation_records: Sequence[Mapping[str, Any]]) -> "ImprovedEnsembleDetector":
        self.lookback = LookBackLensStyleBaseline().fit(validation_records)
        scores = [self._score_record(record) for record in validation_records]
        gold = [sentence_label(record) for record in validation_records]
        self.threshold = choose_best_threshold(gold, scores)
        return self

    def predict(self, records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        predictions = []
        for record in records:
            keyword = _keyword_prediction(record)
            lookback_prediction = (self.lookback or LookBackLensStyleBaseline()).predict([record])[0]
            score = self._score_record(record, keyword, lookback_prediction)
            spans = _merge_spans(keyword["spans"] or lookback_prediction["spans"], str(record["output"]))
            if score < self.threshold:
                spans = []
            predictions.append({"spans": spans, "score": score, "method": "improved_ensemble"})
        return predictions

    def _score_record(
        self,
        record: Mapping[str, Any],
        keyword: Prediction | None = None,
        lookback_prediction: Prediction | None = None,
    ) -> float:
        keyword = keyword or _keyword_prediction(record)
        lookback_prediction = lookback_prediction or (self.lookback or LookBackLensStyleBaseline()).predict([record])[0]
        tool_rule_score = _tool_rule_score(record)
        return float(
            min(
                1.0,
                0.65 * float(lookback_prediction.get("score", 0.0))
                + 0.20 * float(keyword.get("score", 0.0))
                + 0.15 * tool_rule_score,
            )
        )


def _keyword_prediction(record: Mapping[str, Any]) -> Prediction:
    output = str(record["output"])
    spans = []
    max_score = 0.0
    for start, end, sentence in _iter_sentence_spans(output):
        score = _sentence_heuristic_score(record, sentence)
        if score >= 0.5:
            spans.append(_span(start, end, sentence, score, "keyword_tool_action"))
        max_score = max(max_score, score)
    return {"spans": _merge_spans(spans, output), "score": max_score, "method": "keyword_tool_action"}


def _sentence_heuristic_score(record: Mapping[str, Any], sentence: str) -> float:
    score = 0.0
    if any(pattern.search(sentence) for pattern in HIGH_SIGNAL_PATTERNS):
        score = max(score, 0.92)
    support, novelty, action, conflict = _support_features(record, sentence)
    if action:
        score = max(score, 0.75)
    if conflict:
        score = max(score, 0.70)
    if novelty > 0.80 and len(_tokens(sentence)) >= 8:
        score = max(score, 0.55)
    return score


def _support_features(record: Mapping[str, Any], sentence: str) -> list[float]:
    context_tokens = set(_tokens(str(record.get("context", ""))) + _tokens(str(record.get("query", ""))))
    sentence_tokens = _tokens(sentence)
    if not sentence_tokens:
        return [1.0, 0.0, 0.0, 0.0]
    overlap = sum(1 for token in sentence_tokens if token in context_tokens)
    support = overlap / len(sentence_tokens)
    novelty = 1.0 - support
    action = float(_has_missing_tool_action(sentence, str(record.get("tools", ""))))
    conflict = float(_has_known_conflict(sentence, str(record.get("context", ""))))
    return [support, novelty, action, conflict]


def _tool_rule_score(record: Mapping[str, Any]) -> float:
    output = str(record.get("output", ""))
    tools = str(record.get("tools", ""))
    if any(_has_missing_tool_action(sentence, tools) for _, _, sentence in _iter_sentence_spans(output)):
        return 0.9
    if _has_known_conflict(output, str(record.get("context", ""))):
        return 0.7
    return 0.0


def _iter_sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    for match in SENTENCE_RE.finditer(text):
        sentence = match.group(0)
        if sentence.strip():
            spans.append((match.start(), match.end(), sentence))
    return spans or [(0, len(text), text)]


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def _has_missing_tool_action(sentence: str, tools: str) -> bool:
    lowered_sentence = sentence.casefold()
    lowered_tools = tools.casefold()
    for hint, pattern in [
        ("flight", "book a flight"),
        ("hotel", "reserve a hotel"),
        ("payment", "payment"),
        ("email", "confirmation email"),
        ("calendar", "calendar"),
    ]:
        if pattern in lowered_sentence and hint not in lowered_tools:
            return True
    return "external action" in lowered_sentence


def _has_known_conflict(answer: str, context: str) -> bool:
    answer_lower = answer.casefold()
    context_lower = context.casefold()
    for left, right in CONFLICT_PAIRS:
        if left in context_lower and right in answer_lower:
            return True
        if right in context_lower and left in answer_lower:
            return True
    return False


def _overlaps_any(start: int, end: int, labels: Sequence[Mapping[str, Any]]) -> bool:
    for label in labels:
        if max(start, int(label["start"])) < min(end, int(label["end"])):
            return True
    return False


def _span(start: int, end: int, text: str, confidence: float, method: str) -> dict[str, Any]:
    return {
        "start": int(start),
        "end": int(end),
        "text": text,
        "confidence": float(confidence),
        "label_type": method,
    }


def _get_field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _merge_spans(spans: Sequence[Mapping[str, Any]], output: str) -> list[dict[str, Any]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda span: (int(span["start"]), int(span["end"])))
    merged: list[dict[str, Any]] = []
    for span in ordered:
        start, end = int(span["start"]), int(span["end"])
        confidence = float(span.get("confidence", 0.5))
        if not merged or start > int(merged[-1]["end"]):
            merged.append(_span(start, end, output[start:end], confidence, str(span.get("label_type", "merged"))))
        else:
            merged[-1]["end"] = max(int(merged[-1]["end"]), end)
            merged[-1]["text"] = output[int(merged[-1]["start"]) : int(merged[-1]["end"])]
            merged[-1]["confidence"] = max(float(merged[-1].get("confidence", 0.0)), confidence)
    return merged


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
