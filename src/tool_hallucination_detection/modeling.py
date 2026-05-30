"""Optional transformer token-classification model for span hallucination detection."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-base"


def format_model_input(record: Mapping[str, Any]) -> tuple[str, int]:
    prefix = f"Question: {record['query']}\nContext: {record['context']}\nAnswer: "
    return prefix + str(record["output"]), len(prefix)


def tokenize_with_span_labels(record: Mapping[str, Any], tokenizer: Any, max_length: int = 2048) -> dict[str, Any]:
    """Tokenize full input and create token labels over answer characters.

    Important implementation details:
    - Query/context tokens are ignored with label -100.
    - Only answer tokens can contribute to the loss.
    - `offset_mapping` is popped before collation. Keeping it in the dataset
      causes DataCollatorForTokenClassification to fail on variable nested lists.
    """
    full_text, answer_start = format_model_input(record)
    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    offset_mapping = encoded.pop("offset_mapping")

    gold_spans = [
        (answer_start + int(label["start"]), answer_start + int(label["end"]))
        for label in record.get("labels", [])
    ]

    labels = []
    for start, end in offset_mapping:
        if end <= answer_start or start == end:
            labels.append(-100)
        elif any(max(start, gold_start) < min(end, gold_end) for gold_start, gold_end in gold_spans):
            labels.append(1)
        else:
            labels.append(0)

    encoded["labels"] = labels
    return encoded


class WeightedTokenClassificationTrainerMixin:
    """Trainer mixin with class-weighted token-level cross entropy.

    Hallucinated tokens are much rarer than normal answer tokens. Weighted loss
    prevents the classifier from minimizing loss by mostly predicting the
    negative class. The ignore index -100 masks question/context tokens.
    """

    def __init__(self, *args: Any, class_weights: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, num_items_in_batch: Any = None):
        import torch
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train_token_classifier(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    output_dir: str | Path = "artifacts/modernbert-token-classifier",
    model_name: str = DEFAULT_MODEL_NAME,
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
    positive_class_weight_cap: float = 12.0,
) -> Path:
    """Fine-tune a token classifier on answer spans."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    class WeightedTrainer(WeightedTokenClassificationTrainerMixin, Trainer):
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = Dataset.from_list(list(train_records)).map(
        lambda record: tokenize_with_span_labels(record, tokenizer, max_length=max_length),
        remove_columns=list(train_records[0].keys()),
    )
    validation_dataset = Dataset.from_list(list(validation_records)).map(
        lambda record: tokenize_with_span_labels(record, tokenizer, max_length=max_length),
        remove_columns=list(train_records[0].keys()),
    )

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    output_dir = Path(output_dir)
    args = TrainingArguments(**_training_args_kwargs(
        TrainingArguments=TrainingArguments,
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
    ))
    callbacks = []
    if early_stopping_patience is not None and int(early_stopping_patience) > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(early_stopping_patience)))

    base_trainer_kwargs = _trainer_kwargs(
        Trainer=Trainer,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        callbacks=callbacks,
    )

    if use_class_weights:
        weights = _estimate_class_weights(train_dataset, positive_cap=positive_class_weight_cap)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        trainer = WeightedTrainer(class_weights=class_weights, **base_trainer_kwargs)
    else:
        trainer = Trainer(**base_trainer_kwargs)

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def _estimate_class_weights(dataset: Any, positive_cap: float = 12.0) -> list[float]:
    counts = {0: 0, 1: 0}
    for row in dataset:
        for label in row.get("labels", []):
            if int(label) in counts:
                counts[int(label)] += 1
    neg = max(1, counts[0])
    pos = max(1, counts[1])
    total = neg + pos
    weight_neg = total / (2.0 * neg)
    weight_pos = min(total / (2.0 * pos), float(positive_cap))
    # Normalize so the negative class stays close to 1.0 and only the positive
    # upweighting changes substantially.
    if weight_neg > 0:
        weight_pos = weight_pos / weight_neg
        weight_neg = 1.0
    return [float(weight_neg), float(weight_pos)]


def _training_args_kwargs(
    TrainingArguments: Any,
    output_dir: Path,
    batch_size: int,
    epochs: float,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    fp16: bool | None = None,
) -> dict[str, Any]:
    """Build TrainingArguments kwargs across transformers versions."""
    import torch
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "save_strategy": "epoch",
        "logging_steps": 20,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "report_to": [],
        "remove_unused_columns": False,
        "fp16": bool(torch.cuda.is_available()) if fp16 is None else bool(fp16),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "warmup_ratio": float(warmup_ratio),
        "lr_scheduler_type": "linear",
        "save_total_limit": 2,
        "seed": 42,
        "data_seed": 42,
    }
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _trainer_kwargs(
    Trainer: Any,
    model: Any,
    args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    data_collator: Any,
    callbacks: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build Trainer kwargs across transformers versions."""
    signature = inspect.signature(Trainer.__init__)
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    if callbacks:
        kwargs["callbacks"] = list(callbacks)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def predict_with_token_classifier(
    records: Sequence[Mapping[str, Any]],
    model_path: str | Path,
    threshold: float = 0.5,
    max_length: int = 2048,
    score_aggregator: str = "max_token_prob",
) -> list[dict[str, Any]]:
    """Predict hallucinated answer spans with a fine-tuned token classifier."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForTokenClassification.from_pretrained(str(model_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    predictions = []
    for record in records:
        full_text, answer_start = format_model_input(record)
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        token_spans = []
        answer_probabilities = []
        for (start, end), probability in zip(offsets, probs):
            if end <= answer_start or start == end:
                continue
            probability = float(probability)
            answer_probabilities.append(probability)
            if probability >= threshold:
                token_spans.append((start - answer_start, end - answer_start, probability))
        spans = _aggregate_token_spans(token_spans, str(record["output"]))
        predictions.append({
            "spans": spans,
            "score": _aggregate_answer_score(answer_probabilities, score_aggregator),
            "method": "modernbert_token_classifier",
        })
    return predictions


def tune_token_classifier_thresholds(
    validation_records: Sequence[Mapping[str, Any]],
    model_path: str | Path,
    max_length: int = 2048,
    sentence_metric: str = "macro_f1",
    span_metric: str = "char_f1",
    score_aggregator: str = "max_token_prob",
    span_threshold_grid: Sequence[float] = tuple(np.round(np.linspace(0.10, 0.90, 9), 2)),
) -> dict[str, float]:
    """Tune sentence and span thresholds separately on validation data."""
    from .metrics import choose_best_threshold, evaluate_predictions
    from .schema import sentence_label

    score_predictions = predict_with_token_classifier(
        validation_records,
        model_path,
        threshold=0.5,
        max_length=max_length,
        score_aggregator=score_aggregator,
    )
    gold = [sentence_label(record) for record in validation_records]
    scores = [float(prediction.get("score", 0.0)) for prediction in score_predictions]
    sentence_threshold = choose_best_threshold(gold, scores, metric=sentence_metric)

    best_span_threshold = 0.5
    best_span_value = -1.0
    for threshold in span_threshold_grid:
        predictions = predict_with_token_classifier(
            validation_records,
            model_path,
            threshold=float(threshold),
            max_length=max_length,
            score_aggregator=score_aggregator,
        )
        metrics = evaluate_predictions(validation_records, predictions, threshold=sentence_threshold)["span"]
        value = float(metrics.get(span_metric, 0.0))
        if value > best_span_value:
            best_span_value = value
            best_span_threshold = float(threshold)

    return {
        "sentence_threshold": float(sentence_threshold),
        "span_threshold": float(best_span_threshold),
        "score_aggregator": score_aggregator,
        f"validation_{span_metric}": float(best_span_value),
    }


def _aggregate_token_spans(token_spans: Sequence[tuple[int, int, float]], output: str) -> list[dict[str, Any]]:
    if not token_spans:
        return []
    merged: list[dict[str, Any]] = []
    for start, end, probability in token_spans:
        start = max(0, int(start))
        end = min(len(output), int(end))
        if not merged or start > int(merged[-1]["end"]) + 1:
            merged.append({
                "start": start,
                "end": end,
                "text": output[start:end],
                "confidence": float(probability),
                "label_type": "modernbert_token_classifier",
            })
        else:
            merged[-1]["end"] = max(int(merged[-1]["end"]), end)
            merged[-1]["text"] = output[int(merged[-1]["start"]) : int(merged[-1]["end"])]
            merged[-1]["confidence"] = max(float(merged[-1]["confidence"]), float(probability))
    return merged


def _aggregate_answer_score(probabilities: Sequence[float], score_aggregator: str = "max_token_prob") -> float:
    if not probabilities:
        return 0.0
    values = np.asarray(list(probabilities), dtype=float)
    if score_aggregator == "topk_mean_prob_3":
        k = min(3, len(values))
        return float(np.mean(np.sort(values)[-k:]))
    if score_aggregator == "topk_mean_prob_5":
        k = min(5, len(values))
        return float(np.mean(np.sort(values)[-k:]))
    if score_aggregator == "span_sum_score":
        positives = values[values >= 0.5]
        if len(positives) == 0:
            return float(np.max(values))
        return float(min(1.0, float(np.mean(positives)) * np.log1p(len(positives)) / np.log1p(8)))
    return float(np.max(values))
