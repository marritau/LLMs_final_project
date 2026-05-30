"""Baseline detectors for the hallucination benchmark."""
from __future__ import annotations

import hashlib
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
STRUCTURED_VALUE_RE = re.compile(
    r"\b(?:"
    r"(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2}|"
    r"\d{1,2}:\d{2}|"
    r"[-+]?\d+(?:\.\d+)?(?:\s*(?:USD|EUR|GBP|CNY|RUB|km|miles?|kg|lbs?|cm|mm|C|F|percent|%))?|"
    r"[A-Z]{2,}[-_]?[A-Z0-9]{2,}"
    r")\b"
)
QUOTED_VALUE_RE = re.compile(r"""[\"']([^\"'{}\[\]:,]{2,100})[\"']""")
ENTITY_VALUE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|[A-Z]{2,}(?:\s+[A-Z0-9]{2,}){0,2})\b")
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
    ("sunny", "rainy"), ("up", "down"), ("open", "closed"),
    ("available", "unavailable"), ("approved", "rejected"),
    ("success", "failure"), ("true", "false"),
    ("active", "inactive"), ("enabled", "disabled"), ("paid", "unpaid"),
    ("above", "below"), ("before", "after"), ("increase", "decrease"),
]
ENTITY_STOPWORDS = {
    "A", "An", "And", "Answer", "Available", "Context", "I", "No", "Question",
    "The", "This", "Tool", "User", "You",
}


def always_clean_predict(records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
    return [{"spans": [], "score": 0.0, "method": "always_clean"} for _ in records]



def always_hallucinated_predict(records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
    """Trivial majority-class baseline: every answer is hallucinated.

    This is important because the generated dataset is usually 75% positive
    (three corrupted variants and one clean variant per source dialogue).
    """
    predictions: list[Prediction] = []
    for record in records:
        output = str(record.get("output", ""))
        spans = [_full_output_span(output, 1.0, "always_hallucinated")] if output else []
        predictions.append({"spans": spans, "score": 1.0, "method": "always_hallucinated"})
    return predictions


def random_balanced_predict(records: Sequence[Mapping[str, Any]], seed: int = 42) -> list[Prediction]:
    """Deterministic random baseline with uniform scores."""
    predictions: list[Prediction] = []
    for record in records:
        score = _stable_random_score(record, seed=seed)
        output = str(record.get("output", ""))
        spans = [_full_output_span(output, score, "random_balanced")] if output and score >= 0.5 else []
        predictions.append({"spans": spans, "score": score, "method": "random_balanced"})
    return predictions


def keyword_tool_action_predict(records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
    """Fast sanity baseline for injected hallucination types."""
    return [_keyword_prediction(record) for record in records]


def value_checker_predict(records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
    """Detect structured values/entities in the answer that conflict with tool output.

    This neural-symbolic baseline targets the hardest project class:
    short ``tool_contradiction`` spans such as changed numbers, dates, status
    values, units, codes, and named entities.
    """
    return [_value_checker_prediction(record) for record in records]


def nli_sentence_verifier_predict(
    records: Sequence[Mapping[str, Any]],
    model_name: str = "MoritzLaurer/deberta-v3-base-mnli-fever-anli",
    fallback_to_keyword: bool = True,
) -> list[Prediction]:
    """Optional open-source NLI baseline."""
    try:
        from transformers import pipeline
    except Exception:
        if fallback_to_keyword:
            return keyword_tool_action_predict(records)
        raise

    try:
        classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            truncation=True,
            device=0 if _cuda_available() else -1,
        )
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
            result = classifier({"text": context, "text_pair": sentence})
            label_scores = _normalize_pipeline_scores(result)
            contradiction = max([score for label, score in label_scores.items() if "contradiction" in label], default=0.0)
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
    fallback_to_keyword: bool = False,
) -> list[Prediction]:
    """Run the real LettuceDetect baseline.

    When ``fallback_to_keyword`` is False, loading/import errors are raised so the
    final table cannot silently report a keyword method as LettuceDetect.
    """
    try:
        from lettucedetect.models.inference import HallucinationDetector
    except Exception:
        if fallback_to_keyword:
            preds = keyword_tool_action_predict(records)
            for pred in preds:
                pred["method"] = "lettucedetect_fallback_keyword"
            return preds
        raise

    try:
        detector = HallucinationDetector(method="transformer", model_path=model_path) if model_path else HallucinationDetector(method="transformer")
    except Exception:
        if fallback_to_keyword:
            preds = keyword_tool_action_predict(records)
            for pred in preds:
                pred["method"] = "lettucedetect_fallback_keyword"
            return preds
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
            spans.append(_span(start, end, text, confidence, "lettucedetect_real"))
        score = max([span["confidence"] for span in spans], default=0.0)
        predictions.append({"spans": spans, "score": score, "method": "lettucedetect_real"})
    return predictions


@dataclass
class RandomPriorBaseline:
    """Random baseline whose expected positive rate is fitted on validation data."""
    prior: float = 0.5
    seed: int = 42

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "RandomPriorBaseline":
        if records:
            self.prior = float(np.mean([sentence_label(record) for record in records]))
        return self

    def predict(self, records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        predictions: list[Prediction] = []
        # A uniform score can be thresholded on validation; the default span decision
        # uses the fitted prior only for an interpretable random positive rate.
        cutoff = 1.0 - float(self.prior)
        for record in records:
            score = _stable_random_score(record, seed=self.seed + 17)
            output = str(record.get("output", ""))
            spans = [_full_output_span(output, score, "random_prior")] if output and score >= cutoff else []
            predictions.append({"spans": spans, "score": score, "method": "random_prior"})
        return predictions


@dataclass
class TfidfLogRegBaseline:
    """Sentence-level TF-IDF + Logistic Regression baseline."""
    classifier: Any | None = None
    vectorizer: Any | None = None
    threshold: float = 0.5
    seed: int = 42

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "TfidfLogRegBaseline":
        texts = [_format_record_text(record) for record in records]
        labels = [sentence_label(record) for record in records]
        if len(set(labels)) < 2 or not texts:
            self.classifier = None
            self.vectorizer = None
            self.threshold = 0.5
            return self
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion

        word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=40000)
        char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, max_features=60000)
        self.vectorizer = FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])
        features = self.vectorizer.fit_transform(texts)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.seed)
        clf.fit(features, labels)
        scores = clf.predict_proba(features)[:, 1]
        self.classifier = clf
        self.threshold = choose_best_threshold(labels, scores, metric="macro_f1")
        return self

    def predict(self, records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        texts = [_format_record_text(record) for record in records]
        if self.classifier is not None and self.vectorizer is not None and texts:
            scores = self.classifier.predict_proba(self.vectorizer.transform(texts))[:, 1]
        else:
            scores = np.zeros(len(records), dtype=float)
        predictions: list[Prediction] = []
        for record, score in zip(records, scores):
            output = str(record.get("output", ""))
            spans = [_full_output_span(output, float(score), "tfidf_logreg")] if output and float(score) >= self.threshold else []
            predictions.append({"spans": spans, "score": float(score), "method": "tfidf_logreg"})
        return predictions


@dataclass
class LookBackLensStyleBaseline:
    """Lightweight lexical proxy kept only as an extra sanity baseline.

    This class does not use attention maps, so it is not reported as the main
    LookBackLens baseline in the final method discussion. The attention-based
    adaptation is implemented by AttentionLookBackLensBaseline below.
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
            self.weights = np.zeros(5)
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
                    spans.append(_span(start, end, sentence, score, "lookback_lens_style_proxy"))
                max_score = max(max_score, score)
            predictions.append({"spans": spans, "score": max_score, "method": "lookback_lens_style_proxy"})
        return predictions

    def _score(self, features: Sequence[float]) -> float:
        if self.weights is None:
            support, novelty, action, conflict = features
            return max(0.0, min(1.0, 0.15 + 0.55 * novelty + 0.25 * action + 0.20 * conflict - 0.25 * support))
        return _sigmoid(float(np.dot(self.weights, [1.0, *features])))


@dataclass
class AttentionLookBackLensBaseline:
    """Adapted attention-based LookBackLens baseline for fixed answers.

    Original LookBackLens uses attention traces during autoregressive generation.
    Our benchmark already contains fixed clean/corrupted answers, so this class uses
    teacher forcing: it feeds ``Question + Context + Answer`` into a small causal LM
    and extracts attention from answer tokens back to context tokens. A logistic
    classifier is then trained on these attention features.
    """
    model_name: str = "distilgpt2"
    max_length: int = 384
    max_answer_tokens: int = 96
    max_train_records: int = 240
    seed: int = 42
    classifier: Any | None = None
    threshold: float = 0.5
    _tokenizer: Any | None = None
    _model: Any | None = None

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "AttentionLookBackLensBaseline":
        train_records = list(records)
        if self.max_train_records and len(train_records) > self.max_train_records:
            # Deterministic subsample to keep the baseline feasible in Colab.
            scored = [(_stable_random_score(record, seed=self.seed + 101), record) for record in train_records]
            train_records = [record for _, record in sorted(scored, key=lambda item: item[0])[: self.max_train_records]]

        features = [self._features(record) for record in train_records]
        labels = [sentence_label(record) for record in train_records]
        if len(set(labels)) < 2:
            self.classifier = None
            self.threshold = 0.5
            return self

        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.seed)
            clf.fit(np.asarray(features, dtype=float), np.asarray(labels, dtype=int))
            scores = clf.predict_proba(np.asarray(features, dtype=float))[:, 1]
            self.classifier = clf
            self.threshold = choose_best_threshold(labels, scores, metric="macro_f1")
        except Exception:
            self.classifier = None
            scores = [self._fallback_score(feat) for feat in features]
            self.threshold = choose_best_threshold(labels, scores, metric="macro_f1")
        return self

    def predict(self, records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        predictions: list[Prediction] = []
        for record in records:
            features = self._features(record)
            score = self._score_features(features)
            output = str(record.get("output", ""))
            spans = [_full_output_span(output, score, "attention_lookback_lens_adapted")] if output and score >= self.threshold else []
            predictions.append({"spans": spans, "score": score, "method": "attention_lookback_lens_adapted"})
        return predictions

    def _score_features(self, features: Sequence[float]) -> float:
        arr = np.asarray([features], dtype=float)
        if self.classifier is not None:
            try:
                return float(self.classifier.predict_proba(arr)[0, 1])
            except Exception:
                pass
        return self._fallback_score(features)

    def _fallback_score(self, features: Sequence[float]) -> float:
        # features: context_attention, answer_attention, lookback_ratio, lexical_support, novelty, length_norm
        context_attention, answer_attention, lookback_ratio, lexical_support, novelty, length_norm = features
        risk = 1.2 * (1.0 - lookback_ratio) + 0.7 * novelty + 0.2 * length_norm - 0.4 * lexical_support
        return float(max(0.0, min(1.0, risk / 2.0)))

    def _features(self, record: Mapping[str, Any]) -> list[float]:
        try:
            return self._attention_features(record)
        except Exception:
            # If attention extraction fails for one example, fall back to transparent
            # lexical support features so the whole run does not become unusable.
            support, novelty, action, conflict = _support_features(record, str(record.get("output", "")))
            return [support, 1.0 - support, support, support, novelty, min(1.0, len(_tokens(str(record.get("output", "")))) / 80.0)]

    def _ensure_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                output_attentions=True,
                attn_implementation="eager",
            )
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, output_attentions=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._model.eval()

    def _attention_features(self, record: Mapping[str, Any]) -> list[float]:
        import torch
        self._ensure_model()
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        question = str(record.get("query", ""))
        context = str(record.get("context", ""))
        answer = str(record.get("output", ""))
        prefix_before_context = f"Question: {question}\nContext: "
        middle = "\nAnswer: "
        full_text = prefix_before_context + context + middle + answer
        context_start = len(prefix_before_context)
        context_end = context_start + len(context)
        answer_start = context_end + len(middle)
        answer_end = answer_start + len(answer)

        encoded = tokenizer(
            full_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}

        context_indices = [i for i, (s, e) in enumerate(offsets) if e > context_start and s < context_end and e > s]
        answer_indices = [i for i, (s, e) in enumerate(offsets) if e > answer_start and s < answer_end and e > s]
        if self.max_answer_tokens and len(answer_indices) > self.max_answer_tokens:
            answer_indices = answer_indices[-self.max_answer_tokens :]
        if not context_indices or not answer_indices:
            support, novelty, _action, _conflict = _support_features(record, answer)
            return [support, 1.0 - support, support, support, novelty, min(1.0, len(_tokens(answer)) / 80.0)]

        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True, use_cache=False)
        attentions = outputs.attentions
        selected_layers = attentions[-4:] if len(attentions) >= 4 else attentions

        context_mass_values = []
        prev_answer_mass_values = []
        ratio_values = []
        context_tensor = torch.tensor(context_indices, device=device, dtype=torch.long)
        answer_set = set(answer_indices)
        for attn in selected_layers:
            # attn shape: [batch, heads, seq, seq]
            attn0 = attn[0].float()
            for pos in answer_indices:
                if pos <= 0:
                    continue
                head_rows = attn0[:, pos, :]
                context_mass = head_rows.index_select(-1, context_tensor).sum(dim=-1).mean().item()
                prev_answer_indices = [idx for idx in answer_indices if idx < pos and idx in answer_set]
                if prev_answer_indices:
                    prev_tensor = torch.tensor(prev_answer_indices, device=device, dtype=torch.long)
                    prev_answer_mass = head_rows.index_select(-1, prev_tensor).sum(dim=-1).mean().item()
                else:
                    prev_answer_mass = 0.0
                denom = context_mass + prev_answer_mass + 1e-8
                ratio = context_mass / denom
                context_mass_values.append(context_mass)
                prev_answer_mass_values.append(prev_answer_mass)
                ratio_values.append(ratio)

        support, novelty, _action, _conflict = _support_features(record, answer)
        length_norm = min(1.0, len(_tokens(answer)) / 80.0)
        return [
            float(np.mean(context_mass_values)) if context_mass_values else 0.0,
            float(np.mean(prev_answer_mass_values)) if prev_answer_mass_values else 0.0,
            float(np.mean(ratio_values)) if ratio_values else 0.0,
            float(support),
            float(novelty),
            float(length_norm),
        ]


@dataclass
class ImprovedEnsembleDetector:
    threshold: float = 0.5
    lookback: LookBackLensStyleBaseline | None = None
    value_threshold: float = 0.5

    def fit(self, validation_records: Sequence[Mapping[str, Any]]) -> "ImprovedEnsembleDetector":
        self.lookback = LookBackLensStyleBaseline().fit(validation_records)
        scores = [self._score_record(record) for record in validation_records]
        gold = [sentence_label(record) for record in validation_records]
        self.threshold = choose_best_threshold(gold, scores)
        value_scores = [float(prediction.get("score", 0.0)) for prediction in value_checker_predict(validation_records)]
        self.value_threshold = choose_best_threshold(gold, value_scores, metric="macro_f1")
        return self

    def predict(self, records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        predictions = []
        for record in records:
            keyword = _keyword_prediction(record)
            value_prediction = _value_checker_prediction(record)
            lookback_prediction = (self.lookback or LookBackLensStyleBaseline()).predict([record])[0]
            score = self._score_record(record, keyword, lookback_prediction, value_prediction)
            spans = _merge_spans(
                [
                    *keyword.get("spans", []),
                    *lookback_prediction.get("spans", []),
                    *value_prediction.get("spans", []),
                ],
                str(record["output"]),
            )
            if score < self.threshold and float(value_prediction.get("score", 0.0)) < self.value_threshold:
                spans = []
            predictions.append({"spans": spans, "score": score, "method": "hybrid_value_ensemble"})
        return predictions

    def _score_record(
        self,
        record: Mapping[str, Any],
        keyword: Prediction | None = None,
        lookback_prediction: Prediction | None = None,
        value_prediction: Prediction | None = None,
    ) -> float:
        keyword = keyword or _keyword_prediction(record)
        lookback_prediction = lookback_prediction or (self.lookback or LookBackLensStyleBaseline()).predict([record])[0]
        value_prediction = value_prediction or _value_checker_prediction(record)
        tool_rule_score = _tool_rule_score(record)
        return float(min(
            1.0,
            0.50 * float(lookback_prediction.get("score", 0.0))
            + 0.20 * float(keyword.get("score", 0.0))
            + 0.20 * float(value_prediction.get("score", 0.0))
            + 0.10 * tool_rule_score,
        ))


def _full_output_span(output: str, confidence: float, method: str) -> dict[str, Any]:
    return {
        "start": 0,
        "end": len(output),
        "text": output,
        "confidence": float(confidence),
        "label_type": method,
    }


def _stable_random_score(record: Mapping[str, Any], seed: int = 42) -> float:
    key = f"{seed}|{record.get('id', '')}|{record.get('source_id', '')}|{record.get('output', '')}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:16]
    return int(digest, 16) / float(16 ** 16 - 1)


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


def _value_checker_prediction(record: Mapping[str, Any]) -> Prediction:
    output = str(record.get("output", ""))
    context = str(record.get("context", ""))
    query = str(record.get("query", ""))
    tool_call = str(record.get("tool_call", ""))

    context_values = {_normalize_value(item["text"]) for item in _extract_values(context)}
    allowed_query_values = {_normalize_value(item["text"]) for item in _extract_values(query + " " + tool_call)}
    output_values = _extract_values(output)

    spans: list[dict[str, Any]] = []
    max_score = 0.0
    for value in output_values:
        normalized = _normalize_value(value["text"])
        if not normalized or normalized in context_values or normalized in allowed_query_values:
            continue
        if _is_hard_clean_context(output, value["start"]):
            continue
        score = 0.90 if value["kind"] in {"structured", "quoted"} else 0.72
        spans.append(_span(value["start"], value["end"], value["text"], score, "value_checker"))
        max_score = max(max_score, score)

    for start, end, sentence in _iter_sentence_spans(output):
        if _has_known_conflict(sentence, context) and not _is_hard_clean_context(output, start):
            score = 0.78
            spans.append(_span(start, end, sentence, score, "value_checker"))
            max_score = max(max_score, score)

    return {"spans": _merge_spans(spans, output), "score": max_score, "method": "value_checker"}


def _extract_values(text: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []

    def add(start: int, end: int, value: str, kind: str) -> None:
        value = value.strip().strip(".,;:()[]{}")
        if not value or any(max(start, s) < min(end, e) for s, e in occupied):
            return
        if kind == "entity" and value in ENTITY_STOPWORDS:
            return
        values.append({"start": start, "end": end, "text": value, "kind": kind})
        occupied.append((start, end))

    for regex, kind in ((QUOTED_VALUE_RE, "quoted"), (STRUCTURED_VALUE_RE, "structured"), (ENTITY_VALUE_RE, "entity")):
        for match in regex.finditer(text):
            if regex is QUOTED_VALUE_RE:
                start, end = match.start(1), match.end(1)
                value = match.group(1)
            else:
                start, end = match.start(), match.end()
                value = match.group(0)
            add(start, end, value, kind)
    return sorted(values, key=lambda item: (item["start"], item["end"]))


def _normalize_value(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().strip(".,;:()[]{}")).casefold()
    normalized = normalized.replace(",", "")
    normalized = normalized.replace(" percent", "%")
    return normalized


def _is_hard_clean_context(output: str, start: int) -> bool:
    window = output[max(0, start - 80) : start + 80].casefold()
    return any(pattern in window for pattern in ("cannot ", "can't ", "not provide", "does not provide", "no ", "not available"))


def _format_record_text(record: Mapping[str, Any]) -> str:
    return (
        f"Question: {record.get('query', '')}\n"
        f"Tools: {record.get('tools', '')}\n"
        f"Context: {record.get('context', '')}\n"
        f"Answer: {record.get('output', '')}"
    )


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
    return {"start": int(start), "end": int(end), "text": text, "confidence": float(confidence), "label_type": method}


def _normalize_pipeline_scores(result: Any) -> dict[str, float]:
    items: list[Mapping[str, Any]] = []
    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            if "label" in value and "score" in value:
                items.append(value)
            return
        if isinstance(value, list):
            for nested in value:
                collect(nested)
    collect(result)
    return {str(item["label"]).lower(): float(item["score"]) for item in items}


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


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False
