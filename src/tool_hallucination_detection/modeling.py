"""Optional transformer token-classification model.

The full model is meant for Colab/GPU runs. Imports are intentionally local so
the repository remains usable for quick offline checks.
"""

from __future__ import annotations

from pathlib import Path
import inspect
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-base"


def format_model_input(record: Mapping[str, Any]) -> tuple[str, int]:
    prefix = f"Question: {record['query']}\nContext: {record['context']}\nAnswer: "
    return prefix + str(record["output"]), len(prefix)


def tokenize_with_span_labels(
    record: Mapping[str, Any],
    tokenizer: Any,
    max_length: int = 4096,
) -> dict[str, Any]:
    full_text, answer_start = format_model_input(record)
    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    labels = []
    gold_spans = [
        (answer_start + int(label["start"]), answer_start + int(label["end"]))
        for label in record.get("labels", [])
    ]
    for start, end in encoded["offset_mapping"]:
        if end <= answer_start or start == end:
            labels.append(-100)
        elif any(max(start, gold_start) < min(end, gold_end) for gold_start, gold_end in gold_spans):
            labels.append(1)
        else:
            labels.append(0)
    encoded["labels"] = labels
    return encoded


def train_token_classifier(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    output_dir: str | Path = "artifacts/modernbert-token-classifier",
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: float = 1.0,
    batch_size: int = 2,
    max_length: int = 2048,
) -> Path:
    """Fine-tune a token classifier on answer spans."""

    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = Dataset.from_list(list(train_records)).map(
        lambda record: tokenize_with_span_labels(record, tokenizer, max_length=max_length),
        remove_columns=list(train_records[0].keys()),
    )
    validation_dataset = Dataset.from_list(list(validation_records)).map(
        lambda record: tokenize_with_span_labels(record, tokenizer, max_length=max_length),
        remove_columns=list(validation_records[0].keys()),
    )

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)
    output_dir = Path(output_dir)
    args = TrainingArguments(**_training_args_kwargs(
        TrainingArguments=TrainingArguments,
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
    ))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def _training_args_kwargs(
    TrainingArguments: Any,
    output_dir: Path,
    batch_size: int,
    epochs: float,
) -> dict[str, Any]:
    """Build TrainingArguments kwargs across transformers versions."""

    signature = inspect.signature(TrainingArguments.__init__)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "logging_steps": 20,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "report_to": [],
        "remove_unused_columns": False,
    }
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def predict_with_token_classifier(
    records: Sequence[Mapping[str, Any]],
    model_path: str | Path,
    threshold: float = 0.5,
    max_length: int = 2048,
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
        for (start, end), probability in zip(offsets, probs):
            if end <= answer_start or start == end:
                continue
            if float(probability) >= threshold:
                token_spans.append((start - answer_start, end - answer_start, float(probability)))
        spans = _aggregate_token_spans(token_spans, str(record["output"]))
        predictions.append(
            {
                "spans": spans,
                "score": max([span["confidence"] for span in spans], default=float(np.max(probs) if len(probs) else 0.0)),
                "method": "modernbert_token_classifier",
            }
        )
    return predictions


def _aggregate_token_spans(token_spans: Sequence[tuple[int, int, float]], output: str) -> list[dict[str, Any]]:
    if not token_spans:
        return []
    merged: list[dict[str, Any]] = []
    for start, end, probability in token_spans:
        start = max(0, int(start))
        end = min(len(output), int(end))
        if not merged or start > int(merged[-1]["end"]) + 1:
            merged.append(
                {
                    "start": start,
                    "end": end,
                    "text": output[start:end],
                    "confidence": float(probability),
                    "label_type": "modernbert_token_classifier",
                }
            )
        else:
            merged[-1]["end"] = max(int(merged[-1]["end"]), end)
            merged[-1]["text"] = output[int(merged[-1]["start"]) : int(merged[-1]["end"])]
            merged[-1]["confidence"] = max(float(merged[-1]["confidence"]), float(probability))
    return merged
