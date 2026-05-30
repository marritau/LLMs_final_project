"""ToolACE loading, normalization, splitting, and JSONL export."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import validate_labels


def load_toolace_rows(split: str = "train", max_records: int | None = None) -> list[dict[str, Any]]:
    """Load raw ToolACE rows from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the `datasets` package to load ToolACE.") from exc

    dataset = load_dataset("Team-ACE/ToolACE", split=split)
    if max_records is not None:
        dataset = dataset.select(range(min(max_records, len(dataset))))
    return [dict(row) for row in dataset]


def normalize_toolace_row(row: Mapping[str, Any], row_index: int | str) -> dict[str, Any] | None:
    """Convert one ToolACE row into the project record shape."""
    conversations = list(row.get("conversations") or [])
    if not conversations:
        return None

    user_turn = next((turn for turn in conversations if _turn_role(turn) == "user"), None)
    tool_indices = [i for i, turn in enumerate(conversations) if _turn_role(turn) == "tool"]
    if user_turn is None or not tool_indices:
        return None

    first_tool_index = tool_indices[0]
    tool_call_turn = None
    for turn in reversed(conversations[:first_tool_index]):
        if _turn_role(turn) == "assistant":
            tool_call_turn = turn
            break
    if tool_call_turn is None:
        return None

    final_answer_turn = None
    for turn in conversations[tool_indices[-1] + 1 :]:
        if _turn_role(turn) == "assistant":
            final_answer_turn = turn
            break
    if final_answer_turn is None:
        return None

    output = _clean_text(_turn_value(final_answer_turn))
    if not output:
        return None

    return {
        "id": f"toolace-{row_index}",
        "source_id": f"toolace-{row_index}",
        "split": "unsplit",
        "corruption_type": "base",
        "query": _clean_text(_turn_value(user_turn)),
        "tools": _clean_text(row.get("system", row.get("tools", ""))),
        "tool_call": _clean_text(_turn_value(tool_call_turn)),
        "context": _clean_text(_turn_value(conversations[first_tool_index])),
        "output": output,
        "labels": [],
    }


def normalize_toolace_rows(rows: Iterable[Mapping[str, Any]], max_records: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        record = normalize_toolace_row(row, index)
        if record is not None:
            records.append(record)
        if max_records is not None and len(records) >= max_records:
            break
    return records


def split_by_source_id(
    records: list[dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    """Split records while keeping variants of the same source together."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_id"])].append(record)

    source_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(source_ids)

    n_total = len(source_ids)
    n_train = int(round(n_total * train_ratio))
    n_validation = int(round(n_total * validation_ratio))
    if n_total >= 3:
        n_validation = max(1, n_validation)
        n_train = min(max(1, n_train), n_total - n_validation - 1)

    train_ids = set(source_ids[:n_train])
    validation_ids = set(source_ids[n_train : n_train + n_validation])

    splits = {"train": [], "validation": [], "test": []}
    for source_id in source_ids:
        split = "train" if source_id in train_ids else "validation" if source_id in validation_ids else "test"
        for record in grouped[source_id]:
            cloned = dict(record)
            cloned["split"] = split
            splits[split].append(cloned)
    return splits


def flatten_splits(splits: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [record for split in ("train", "validation", "test") for record in splits.get(split, [])]


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write records with both internal `labels` and assignment-facing `hallucination_labels`.

    The code uses `labels` internally for brevity, while the project PDF names the
    RAGTruth-style field `hallucination_labels`. Keeping both fields makes the
    exported JSONL unambiguous for grading and Hugging Face publication.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            output_record = _with_hallucination_label_alias(record)
            validate_labels(output_record)
            file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [_with_hallucination_label_alias(json.loads(line)) for line in file if line.strip()]


def _with_hallucination_label_alias(record: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(record)
    labels = list(data.get("labels") or data.get("hallucination_labels") or [])
    data["labels"] = labels
    data["hallucination_labels"] = labels
    return data


def _clean_text(value: Any) -> str:
    return " ".join(str(value).split())


def _turn_role(turn: Mapping[str, Any]) -> str:
    return str(turn.get("from", turn.get("role", turn.get("speaker", "")))).casefold()


def _turn_value(turn: Mapping[str, Any]) -> Any:
    return turn.get("value", turn.get("content", turn.get("text", "")))


def synthetic_toolace_records() -> list[dict[str, Any]]:
    """Small offline sample used by quick smoke runs."""
    return [
        {
            "id": "synthetic-weather",
            "source_id": "synthetic-weather",
            "split": "unsplit",
            "corruption_type": "base",
            "query": "Help me check the weather in Beijing.",
            "tools": "Available tools: Weather_API(location). Calendar_API(date).",
            "tool_call": '[Weather_API(location="Beijing")]',
            "context": '{"location": "Beijing", "weather": "sunny", "temperature": "26 C"}',
            "output": "The weather in Beijing is sunny with a temperature of 26 C.",
            "labels": [],
        },
        {
            "id": "synthetic-market",
            "source_id": "synthetic-market",
            "split": "unsplit",
            "corruption_type": "base",
            "query": "Get the top market trends in the US.",
            "tools": "Available tools: Market Trends API(trend_type, country).",
            "tool_call": '[Market Trends API(trend_type="MARKET_INDEXES", country="us")]',
            "context": '{"country": "US", "trend": "S&P 500 is up", "change": "1.2%"}',
            "output": "The top US market trend is that the S&P 500 is up by 1.2%.",
            "labels": [],
        },
        {
            "id": "synthetic-quotes",
            "source_id": "synthetic-quotes",
            "split": "unsplit",
            "corruption_type": "base",
            "query": "Find me a quote about inspiration.",
            "tools": "Available tools: Quotes by Keywords(word).",
            "tool_call": '[Quotes by Keywords(word="inspiration")]',
            "context": '{"quotes": [{"text": "The only way to do great work is to love what you do.", "author": "Steve Jobs"}]}',
            "output": 'A relevant quote is "The only way to do great work is to love what you do" by Steve Jobs.',
            "labels": [],
        },
    ]
