"""Harder synthetic hallucination injection for tool-calling dialogues.

The assignment requires automatic corruption of ToolACE examples with three hallucination types:
1. contradiction between answer and tool output;
2. overgeneration unsupported by the tool output;
3. missing-tool actions.

This module keeps exact character spans but makes corruptions less template-like than a simple
single appended phrase. The goal is a more useful controlled benchmark, not random noise.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping

from .schema import SpanLabel, validate_labels

# Word-boundary regexes. These must contain the regex token \b, not the backspace character.
QUOTED_VALUE_RE = re.compile(r"""[\"']([^\"'{}\[\]:,]{2,100})[\"']""")
DATE_RE = re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2}\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
NUMBER_RE = re.compile(r"\b[-+]?\d+(?:\.\d+)?%?\b")
CODE_RE = re.compile(r"\b[A-Z]{2,}[-_]?[A-Z0-9]{2,}\b")
UNIT_VALUE_RE = re.compile(
    r"\b[-+]?\d+(?:\.\d+)?\s*(?:USD|EUR|GBP|CNY|RUB|km|miles?|kg|lbs?|cm|mm|C|F|percent|%)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9&+.-]{2,}\b")

CONTRADICTION_REPLACEMENTS = {
    "sunny": "rainy", "rainy": "sunny", "cloudy": "clear", "clear": "cloudy",
    "snowy": "hot", "stormy": "calm", "hot": "cold", "cold": "hot",
    "warm": "freezing", "freezing": "warm", "high": "low", "low": "high",
    "up": "down", "down": "up", "increase": "decrease", "decrease": "increase",
    "increased": "decreased", "decreased": "increased", "open": "closed", "closed": "open",
    "available": "unavailable", "unavailable": "available", "active": "inactive", "inactive": "active",
    "enabled": "disabled", "disabled": "enabled", "approved": "rejected", "rejected": "approved",
    "accepted": "denied", "denied": "accepted", "success": "failure", "failure": "success",
    "successful": "failed", "failed": "successful", "true": "false", "false": "true",
    "yes": "no", "no": "yes", "valid": "invalid", "invalid": "valid",
    "confirmed": "cancelled", "cancelled": "confirmed", "completed": "pending", "pending": "completed",
    "paid": "unpaid", "unpaid": "paid", "on": "off", "off": "on",
    "above": "below", "below": "above", "greater": "less", "less": "greater",
    "before": "after", "after": "before", "earlier": "later", "later": "earlier",
    "at least": "at most", "at most": "at least",
}

ENTITY_REPLACEMENTS = [
    "London", "Tokyo", "Paris", "Berlin", "New York", "Toronto", "Sydney", "Beijing",
    "Madrid", "Rome", "Singapore", "Dubai", "Boston", "Chicago", "Amsterdam",
]

MISSING_TOOL_ACTIONS = [
    ("flight", " I can also book a flight for you now.", {"book", "flight", "ticket", "reservation"}),
    ("hotel", " I can reserve a hotel room for you as the next step.", {"hotel", "reservation", "booking"}),
    ("payment", " I can proceed with the payment immediately.", {"payment", "pay", "checkout", "purchase"}),
    ("email", " I can send a confirmation email to everyone involved.", {"email", "mail", "send"}),
    ("calendar", " I can add this event to your calendar right away.", {"calendar", "event", "schedule"}),
    ("sms", " I can send an SMS notification to the customer.", {"sms", "text", "message"}),
    ("refund", " I can issue the refund directly from here.", {"refund", "return", "reimburse"}),
    ("call", " I can call the customer and confirm the details.", {"call", "phone", "dial"}),
    ("ride", " I can order a ride to the destination for you.", {"ride", "taxi", "car"}),
    ("restaurant", " I can make a restaurant reservation for tonight.", {"restaurant", "dining", "reservation"}),
    ("cancel", " I can cancel the existing booking for you.", {"cancel", "delete", "void"}),
]

OVERGENERATION_TEMPLATES = [
    ("unsupported_history", " Historical records show that this has been the usual outcome for the past few months."),
    ("unsupported_trend", " The trend is expected to remain stable throughout next week."),
    ("unsupported_cause", " This happened because demand increased sharply in the surrounding area."),
    ("unsupported_user_reaction", " Users were also satisfied with the result according to previous feedback."),
    ("unsupported_risk", " There is no risk of delay or failure based on the available records."),
    ("unsupported_recommendation", " Therefore, the best recommendation is to proceed without any further checks."),
    ("unsupported_forecast", " The same result is very likely to continue for the rest of the month."),
    ("unsupported_external_confirmation", " This was also independently confirmed by an external monitoring system."),
    ("unsupported_comparison", " Compared with similar cases, this is one of the best outcomes recorded recently."),
    ("unsupported_guarantee", " The tool output guarantees that no additional intervention will be required."),
]

HARD_CLEAN_TEMPLATES = [
    ("context_limitation", "The tool output does not provide enough evidence for historical trends."),
    ("forecast_limitation", "I cannot infer a monthly forecast from this tool output alone."),
    ("external_confirmation_limitation", "The available tool result does not independently confirm external follow-up actions."),
    ("payment_refusal", "I cannot proceed with a payment because no payment tool is available."),
    ("email_refusal", "I cannot send a confirmation email because no email-sending tool is available."),
    ("booking_refusal", "I cannot book travel from here because no booking tool is available."),
]

FALLBACK_CONTRADICTIONS = [
    ("fallback_status", " However, the returned status should be interpreted as failed rather than successful."),
    ("fallback_value", " However, the answer reports a different value from the one returned by the tool."),
    ("fallback_opposite", " However, this states the opposite of what the tool output supports."),
]

STOP_CANDIDATES = {
    "api", "args", "argument", "arguments", "assistant", "call", "content", "data", "date", "details",
    "false", "function", "id", "input", "json", "message", "name", "none", "null", "object", "output",
    "parameter", "parameters", "query", "request", "response", "result", "results", "return", "returned",
    "schema", "status", "string", "success", "system", "text", "tool", "tools", "true", "type", "user",
    "value", "values", "weather", "location", "city", "country",
}


def build_corrupted_dataset(
    base_records: Iterable[Mapping[str, Any]],
    include_clean: bool = True,
    include_clean_hard: bool = True,
) -> list[dict[str, Any]]:
    """Create clean and three span-labeled hallucinated variants for every usable base record."""
    generated: list[dict[str, Any]] = []
    for base in base_records:
        if include_clean:
            clean = _clone_base(base, "clean")
            validate_labels(clean)
            generated.append(clean)
        if include_clean_hard:
            hard_clean = inject_hard_clean(base)
            validate_labels(hard_clean)
            generated.append(hard_clean)
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


def inject_hard_clean(record: Mapping[str, Any]) -> dict[str, Any]:
    """Add a supported refusal/limitation sentence with no hallucinated label.

    These examples make keyword baselines less brittle: words such as "book",
    "payment", or "email" are not hallucinations when the answer explicitly says
    that the action cannot be performed with the available tools.
    """
    output = str(record["output"])
    tools_text = str(record.get("tools", "")).casefold()
    templates = []
    for style, sentence in HARD_CLEAN_TEMPLATES:
        if "payment" in style and "payment" in tools_text:
            continue
        if "email" in style and ("email" in tools_text or "mail" in tools_text):
            continue
        if "booking" in style and any(word in tools_text for word in ("book", "booking", "flight", "hotel")):
            continue
        templates.append((style, sentence))
    style, addition = _choice(templates or HARD_CLEAN_TEMPLATES[:3], record, "clean_hard")
    start, new_output, inserted = _insert_extra(output, addition, record, "clean_hard_insert")
    hard_clean = _clone_base(record, "clean_hard")
    hard_clean["output"] = new_output
    hard_clean["corruption_style"] = style
    hard_clean["labels"] = []
    hard_clean["hallucination_labels"] = []
    hard_clean["meta"] = f"hard clean limitation inserted at {start}: {inserted!r}"
    return hard_clean


def inject_tool_contradiction(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Substitute a tool-supported answer span with a conflicting value.

    Preference order: dates/times/numbers/codes/quoted values copied from the tool output into the answer.
    These are harder than appending a generic contradiction because the hallucinated span can be short.
    """
    output = str(record["output"])
    context = str(record["context"])
    candidates = _candidate_values(context)

    for candidate in candidates:
        match = _find_case_insensitive(output, candidate)
        if match is None:
            continue
        replacement, style = _replacement_for(candidate, candidates, record)
        if not replacement or replacement.casefold() == candidate.casefold():
            continue
        start, end = match
        new_output = output[:start] + replacement + output[end:]
        label = SpanLabel(
            start=start,
            end=start + len(replacement),
            text=replacement,
            label_type="tool_contradiction",
            meta=(
                f'{{"style": "{style}", "source_value": {candidate!r}, '
                f'"new_value": {replacement!r}}}'
            ),
        )
        return _variant(record, new_output, [label], corruption_style=style)

    # Controlled fallback: still span-labeled, but reported as a fallback style in the audit.
    style, addition = _choice(FALLBACK_CONTRADICTIONS, record, "contradiction_fallback")
    start, new_output, inserted = _insert_extra(output, addition, record, "contradiction_insert")
    label = SpanLabel(
        start=start,
        end=start + len(inserted),
        text=inserted,
        label_type="tool_contradiction",
        meta=f'{{"style": "{style}", "fallback": true}}',
    )
    return _variant(record, new_output, [label], corruption_style=style)


def inject_overgeneration(record: Mapping[str, Any]) -> dict[str, Any]:
    """Insert plausible but unsupported information absent from the tool output."""
    output = str(record["output"])
    style, addition = _choice(OVERGENERATION_TEMPLATES, record, "overgeneration")
    start, new_output, inserted = _insert_extra(output, addition, record, "overgeneration_insert")
    label = SpanLabel(
        start=start,
        end=start + len(inserted),
        text=inserted,
        label_type="overgeneration",
        meta=f'{{"style": "{style}", "reason": "claim_absent_from_tool_response"}}',
    )
    return _variant(record, new_output, [label], corruption_style=style)


def inject_missing_tool(record: Mapping[str, Any]) -> dict[str, Any]:
    """Insert an action that would require a missing tool.

    The chosen action is filtered against available tool text, so we avoid asking an email tool to send email if
    an email-sending tool is actually available.
    """
    output = str(record["output"])
    tools_text = str(record.get("tools", "")).casefold()
    query_text = str(record.get("query", "")).casefold()

    unavailable = []
    for style, sentence, keywords in MISSING_TOOL_ACTIONS:
        if not any(keyword in tools_text for keyword in keywords):
            # prefer actions related to the query, but keep all unavailable actions as fallback choices
            relevance = int(any(keyword in query_text for keyword in keywords))
            unavailable.append((relevance, style, sentence))

    if unavailable:
        unavailable = sorted(unavailable, key=lambda item: (-item[0], item[1]))
        _, style, addition = _choice(unavailable[:4] if len(unavailable) >= 4 else unavailable, record, "missing_tool")
    else:
        style, addition = "generic_external_action", " I can complete the next external action for you now."

    start, new_output, inserted = _insert_extra(output, addition, record, "missing_tool_insert")
    label = SpanLabel(
        start=start,
        end=start + len(inserted),
        text=inserted,
        label_type="missing_tool",
        meta=f'{{"style": "{style}", "reason": "required_tool_unavailable"}}',
    )
    return _variant(record, new_output, [label], corruption_style=style)


def _clone_base(record: Mapping[str, Any], corruption_type: str) -> dict[str, Any]:
    return {
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
        "hallucination_labels": [],
        "corruption_style": "",
    }


def _variant(
    record: Mapping[str, Any],
    output: str,
    labels: list[SpanLabel],
    corruption_style: str = "",
) -> dict[str, Any]:
    data = _clone_base(record, str(record.get("corruption_type", "corrupted")))
    data["output"] = output
    label_dicts = [label.to_dict() for label in labels]
    data["labels"] = label_dicts
    data["hallucination_labels"] = label_dicts
    data["corruption_style"] = corruption_style
    return data


def _candidate_values(context: str) -> list[str]:
    """Extract potential facts from tool output in a priority order."""
    candidates: list[str] = []
    seen = set()
    regexes = (UNIT_VALUE_RE, DATE_RE, TIME_RE, QUOTED_VALUE_RE, CODE_RE, NUMBER_RE, WORD_RE)
    for regex in regexes:
        for match in regex.finditer(context):
            value = match.group(1) if regex is QUOTED_VALUE_RE else match.group(0)
            value = value.strip().strip(".,;:()[]{}")
            lowered = value.casefold()
            if _useful_candidate(value) and lowered not in seen:
                candidates.append(value)
                seen.add(lowered)
    # Longer and more structured candidates first; this reduces partial replacements.
    return sorted(candidates, key=lambda v: (not _is_structured(v), -len(v)))


def _useful_candidate(value: str) -> bool:
    lowered = value.casefold()
    if len(value) < 2:
        return False
    if lowered in STOP_CANDIDATES:
        return False
    if lowered.startswith(("http", "www")):
        return False
    if len(value) > 100:
        return False
    return True


def _replacement_for(value: str, candidates: list[str] | None, record: Mapping[str, Any]) -> tuple[str, str]:
    lowered = value.casefold()
    if DATE_RE.fullmatch(value):
        return _change_date(value), "date_substitution"
    if TIME_RE.fullmatch(value):
        return _change_time(value), "time_substitution"
    if UNIT_VALUE_RE.fullmatch(value):
        return _change_unit_value(value), "unit_substitution"
    if NUMBER_RE.fullmatch(value):
        return _change_number(value), "number_substitution"
    if lowered in CONTRADICTION_REPLACEMENTS:
        return _match_case(value, CONTRADICTION_REPLACEMENTS[lowered]), "status_or_boolean_flip"
    if CODE_RE.fullmatch(value):
        return _change_code(value), "code_substitution"
    if _looks_like_entity(value):
        return _entity_replacement(value, record), "entity_substitution"

    # Cross-field value swap is used only when the alternative is not identical and not a stopword.
    for other in candidates or []:
        if other.casefold() != lowered and _useful_candidate(other) and not NUMBER_RE.fullmatch(other):
            if abs(len(other) - len(value)) <= max(8, len(value)):
                return _match_case(value, other), "cross_field_value_swap"

    if len(value) <= 6:
        return f"not {value}", "negation_substitution"
    return f"different {value}", "generic_value_substitution"


def _change_number(value: str) -> str:
    has_percent = value.endswith("%")
    numeric_part = value[:-1] if has_percent else value
    try:
        number = float(numeric_part)
    except ValueError:
        return f"not {value}"
    if number == 0:
        changed = 1.0
    else:
        # A noticeable but plausible perturbation.
        sign = 1 if _stable_int(value, "number") % 2 == 0 else -1
        changed = number + sign * max(1.0, abs(number) * 0.15)
    rendered = str(int(round(changed))) if re.fullmatch(r"[-+]?\d+", numeric_part) else f"{changed:.2f}".rstrip("0").rstrip(".")
    return rendered + ("%" if has_percent else "")


def _change_unit_value(value: str) -> str:
    match = re.match(r"(?P<num>[-+]?\d+(?:\.\d+)?)(?P<space>\s*)(?P<unit>[A-Za-z%]+)", value)
    if match is None:
        return _change_number(value)
    unit = match.group("unit")
    replacements = {
        "usd": "EUR", "eur": "USD", "gbp": "EUR", "cny": "USD", "rub": "EUR",
        "km": "miles", "mile": "km", "miles": "km", "kg": "lb", "lb": "kg", "lbs": "kg",
        "cm": "in", "mm": "cm", "c": "F", "f": "C", "percent": "%", "%": "percent",
    }
    new_unit = replacements.get(unit.casefold(), "units")
    return f"{match.group('num')}{match.group('space')}{new_unit}"


def _change_date(value: str) -> str:
    sep = "/" if "/" in value else "-"
    parts = value.split(sep)
    try:
        if len(parts[0]) == 4:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            d = min(28, d + 3)
            return f"{y:04d}{sep}{m:02d}{sep}{d:02d}"
        d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
        d = min(28, d + 3)
        return f"{d:02d}{sep}{m:02d}{sep}{y:04d}"
    except Exception:
        return value + " later"


def _change_time(value: str) -> str:
    try:
        hour, minute = value.split(":")
        new_hour = (int(hour) + 3) % 24
        return f"{new_hour:02d}:{int(minute):02d}"
    except Exception:
        return "23:59"


def _change_code(value: str) -> str:
    if len(value) < 3:
        return value + "X"
    last = value[-1]
    replacement = "X" if last != "X" else "Y"
    return value[:-1] + replacement


def _entity_replacement(value: str, record: Mapping[str, Any]) -> str:
    pool = [item for item in ENTITY_REPLACEMENTS if item.casefold() != value.casefold()]
    return _match_case(value, _choice(pool, record, f"entity::{value}"))


def _looks_like_entity(value: str) -> bool:
    tokens = value.split()
    if not tokens or len(tokens) > 4:
        return False
    if any(token[:1].isupper() for token in tokens):
        return True
    if value.isupper() and len(value) >= 3:
        return True
    return False


def _is_structured(value: str) -> bool:
    return bool(DATE_RE.fullmatch(value) or TIME_RE.fullmatch(value) or NUMBER_RE.fullmatch(value) or CODE_RE.fullmatch(value))


def _insert_extra(output: str, addition: str, record: Mapping[str, Any], salt: str) -> tuple[int, str, str]:
    """Insert text with a controlled position distribution.

    Approximate distribution: append 40%, after first sentence 30%, prepend 10%,
    replace first sentence 20%. This makes position-based shortcuts less useful.
    """
    inserted = addition.strip()
    modes = ["append", "append", "append", "append", "after_first_sentence", "after_first_sentence", "after_first_sentence", "prepend", "replace_first_sentence", "replace_first_sentence"]
    mode = _choice(modes, record, salt)
    if mode == "after_first_sentence":
        match = re.search(r"[.!?]\s+", output)
        if match is not None and match.end() < len(output):
            start = match.end()
            prefix = output[:start]
            suffix = output[start:]
            text = prefix + inserted + " " + suffix
            return start, text, inserted
    if mode == "prepend":
        return 0, inserted + " " + output, inserted
    if mode == "replace_first_sentence":
        match = re.match(r"\s*[^.!?]+[.!?]?", output)
        if match is not None and match.end() > match.start():
            start = match.start()
            suffix = output[match.end():]
            text = output[:start] + inserted + suffix
            return start, text, inserted
    separator = "" if not output or output[-1].isspace() else " "
    start = len(output) + len(separator)
    return start, output + separator + inserted, inserted


def _choice(items: list[Any], record: Mapping[str, Any], salt: str) -> Any:
    key = f"{record.get('source_id', '')}|{record.get('output', '')}|{salt}".encode("utf-8")
    index = int(hashlib.sha256(key).hexdigest()[:8], 16) % len(items)
    return items[index]


def _stable_int(value: str, salt: str) -> int:
    key = f"{value}|{salt}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _find_case_insensitive(text: str, needle: str) -> tuple[int, int] | None:
    match = re.search(re.escape(needle), text, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.start(), match.end()
