"""Synthetic hallucination injection for tool-calling dialogues."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .schema import SpanLabel, validate_labels


QUOTED_VALUE_RE = re.compile(r'"([^"{}[\]:,]{2,80})"')
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9&+.-]{2,}\b")

CONTRADICTION_REPLACEMENTS = {
    "sunny": "rainy",
    "rainy": "sunny",
    "cloudy": "clear",
    "clear": "cloudy",
    "up": "down",
    "down": "up",
    "open": "closed",
    "closed": "open",
    "available": "unavailable",
    "unavailable": "available",
    "approved": "rejected",
    "rejected": "approved",
    "success": "failure",
    "failure": "success",
    "true": "false",
    "false": "true",
}

MISSING_TOOL_ACTIONS = [
    ("flight", " Would you like me to book a flight for you now?"),
    ("hotel", " I can also reserve a hotel room for you."),
    ("payment", " I can proceed with the payment immediately."),
    ("email", " I can send a confirmation email to everyone involved."),
    ("calendar", " I can add this to your calendar right away."),
]


def build_corrupted_dataset(base_records: Iterable[Mapping[str, Any]], include_clean: bool = True) -> list[dict[str, Any]]:
    """Create clean and three hallucinated variants for every usable base record."""

    generated: list[dict[str, Any]] = []
    for base in base_records:
        if include_clean:
            clean = _clone_base(base, "clean")
            validate_labels(clean)
            generated.append(clean)

        for corruption_type, builder in (
            ("tool_contradiction", inject_tool_contradiction),
            ("overgeneration", inject_overgeneration),
            ("missing_tool", inject_missing_tool),
        ):
            corrupted = builder(base)
            if corrupted is None:
                continue
            corrupted["corruption_type"] = corruption_type
            corrupted["id"] = f"{base['source_id']}::{corruption_type}"
            validate_labels(corrupted)
            generated.append(corrupted)
    return generated


def inject_tool_contradiction(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Replace an answer value supported by the tool output with a conflicting one."""

    output = str(record["output"])
    context = str(record["context"])

    for candidate in _candidate_values(context):
        match = _find_case_insensitive(output, candidate)
        if match is None:
            continue
        replacement = _replacement_for(candidate)
        if replacement.casefold() == candidate.casefold():
            continue
        start, end = match
        new_output = output[:start] + replacement + output[end:]
        label = SpanLabel(
            start=start,
            end=start + len(replacement),
            text=replacement,
            label_type="tool_contradiction",
            meta=f"Replaced tool-supported value {candidate!r} with conflicting value {replacement!r}.",
        )
        return _variant(record, new_output, [label])

    fallback = " This contradicts the tool output by reporting a different result."
    start = len(output)
    new_output = output + fallback
    label = SpanLabel(
        start=start,
        end=len(new_output),
        text=fallback,
        label_type="tool_contradiction",
        meta="Fallback contradiction injected because no copied tool value was found in the final answer.",
    )
    return _variant(record, new_output, [label])


def inject_overgeneration(record: Mapping[str, Any]) -> dict[str, Any]:
    output = str(record["output"])
    addition = " The tool output also shows that this pattern has remained stable for the past few months."
    start = len(output)
    new_output = output + addition
    label = SpanLabel(
        start=start,
        end=len(new_output),
        text=addition,
        label_type="overgeneration",
        meta="Added unsupported historical information that is absent from the tool response.",
    )
    return _variant(record, new_output, [label])


def inject_missing_tool(record: Mapping[str, Any]) -> dict[str, Any]:
    output = str(record["output"])
    tools_text = str(record.get("tools", "")).casefold()
    addition = next(
        (sentence for tool_hint, sentence in MISSING_TOOL_ACTIONS if tool_hint not in tools_text),
        " I can complete the next external action for you now.",
    )
    start = len(output)
    new_output = output + addition
    label = SpanLabel(
        start=start,
        end=len(new_output),
        text=addition,
        label_type="missing_tool",
        meta="Added an action that requires a tool not listed as available.",
    )
    return _variant(record, new_output, [label])


def _clone_base(record: Mapping[str, Any], corruption_type: str) -> dict[str, Any]:
    cloned = {
        "id": f"{record['source_id']}::{corruption_type}",
        "source_id": str(record["source_id"]),
        "split": str(record.get("split", "unsplit")),
        "corruption_type": corruption_type,
        "query": str(record["query"]),
        "tools": str(record.get("tools", "")),
        "tool_call": str(record.get("tool_call", "")),
        "context": str(record["context"]),
        "output": str(record["output"]),
        "labels": [],
    }
    return cloned


def _variant(record: Mapping[str, Any], output: str, labels: list[SpanLabel]) -> dict[str, Any]:
    data = _clone_base(record, str(record.get("corruption_type", "corrupted")))
    data["output"] = output
    data["labels"] = [label.to_dict() for label in labels]
    return data


def _candidate_values(context: str) -> list[str]:
    candidates: list[str] = []
    for regex in (QUOTED_VALUE_RE, NUMBER_RE, WORD_RE):
        for match in regex.finditer(context):
            value = match.group(1) if regex is QUOTED_VALUE_RE else match.group(0)
            value = value.strip()
            if _useful_candidate(value) and value.casefold() not in {item.casefold() for item in candidates}:
                candidates.append(value)
    return sorted(candidates, key=len, reverse=True)


def _useful_candidate(value: str) -> bool:
    lowered = value.casefold()
    if len(value) < 2:
        return False
    if lowered in {"name", "text", "result", "results", "value", "quotes", "output", "data"}:
        return False
    return True


def _replacement_for(value: str) -> str:
    lowered = value.casefold()
    if lowered in CONTRADICTION_REPLACEMENTS:
        return _match_case(value, CONTRADICTION_REPLACEMENTS[lowered])
    if NUMBER_RE.fullmatch(value):
        return _change_number(value)
    if len(value) <= 5:
        return f"not {value}"
    return f"incorrect {value}"


def _change_number(value: str) -> str:
    has_percent = value.endswith("%")
    numeric_part = value[:-1] if has_percent else value
    try:
        number = float(numeric_part)
    except ValueError:
        return f"not {value}"
    changed = number + 1 if number >= 0 else number - 1
    if numeric_part.isdigit():
        rendered = str(int(changed))
    else:
        rendered = f"{changed:.1f}".rstrip("0").rstrip(".")
    return rendered + ("%" if has_percent else "")


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def _find_case_insensitive(text: str, needle: str) -> tuple[int, int] | None:
    match = re.search(re.escape(needle), text, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.start(), match.end()
