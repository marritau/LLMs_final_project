"""Shared record and label helpers.

The project uses dictionaries for easy JSONL export, but these helpers keep the
RAGTruth-style span contract explicit and testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SpanLabel:
    start: int
    end: int
    text: str
    label_type: str
    meta: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HallucinationRecord:
    id: str
    source_id: str
    split: str
    corruption_type: str
    query: str
    tools: str
    tool_call: str
    context: str
    output: str
    labels: list[SpanLabel] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["labels"] = [label.to_dict() for label in self.labels]
        return data


def coerce_label(label: SpanLabel | Mapping[str, Any]) -> SpanLabel:
    if isinstance(label, SpanLabel):
        return label
    return SpanLabel(
        start=int(label["start"]),
        end=int(label["end"]),
        text=str(label["text"]),
        label_type=str(label["label_type"]),
        meta=str(label.get("meta", "")),
    )


def coerce_labels(labels: list[SpanLabel | Mapping[str, Any]] | None) -> list[SpanLabel]:
    return [coerce_label(label) for label in labels or []]


def sentence_label(record: Mapping[str, Any]) -> int:
    return int(len(record.get("labels") or []) > 0)


def validate_labels(record: Mapping[str, Any]) -> None:
    """Validate non-empty, non-overlapping spans and exact text offsets."""

    output = str(record.get("output", ""))
    labels = sorted(coerce_labels(record.get("labels")), key=lambda item: item.start)
    previous_end = -1
    for label in labels:
        if label.start < 0 or label.end <= label.start:
            raise ValueError(f"Invalid span bounds in record {record.get('id')}: {label}")
        if label.end > len(output):
            raise ValueError(f"Span exceeds output length in record {record.get('id')}: {label}")
        if label.start < previous_end:
            raise ValueError(f"Overlapping spans in record {record.get('id')}: {label}")
        if output[label.start : label.end] != label.text:
            raise ValueError(
                f"Span text mismatch in record {record.get('id')}: "
                f"expected {label.text!r}, got {output[label.start:label.end]!r}"
            )
        previous_end = label.end


def ensure_record_dict(record: HallucinationRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, HallucinationRecord):
        return record.to_dict()
    data = dict(record)
    data["labels"] = [coerce_label(label).to_dict() for label in data.get("labels", [])]
    return data
