from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC."""

    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_json_list(value: str | None) -> tuple[list[Any], bool]:
    """Return a JSON list and whether the source value satisfied that contract."""

    if not value:
        return [], True
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return [], False
    return (parsed, True) if isinstance(parsed, list) else ([], False)


def tool_calls_from_messages(messages: Iterable[Any]) -> list[str]:
    names: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        parts = message.get("parts", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, Mapping) or part.get("type") != "tool_call":
                continue
            name = part.get("name")
            names.append(str(name) if name else "<unnamed>")
    return names


def _event_class(status_code: int | None, tool_names: list[str]) -> str:
    if status_code not in (None, 0, 1):
        return "error"
    if tool_names:
        return "tool_call"
    return "final_or_text"


def _route_key(event_class: str, tool_names: list[str]) -> str:
    if event_class != "tool_call":
        return event_class
    if len(tool_names) == 1:
        return f"tool:{tool_names[0]}"
    return "tool:<multi>"


def _peak_positive_interval_concurrency(
    intervals: Iterable[tuple[datetime, datetime]],
) -> int:
    events: list[tuple[datetime, int]] = []
    for start, end in intervals:
        if end <= start:
            continue
        events.extend(((start, 1), (end, -1)))
    active = 0
    peak = 0
    for _timestamp, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def extract_session_features(
    row: Mapping[str, Any],
    *,
    source_file: str,
    source_row: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract content-free span and session features from one public trace row."""

    session_id = str(row["session_id"])
    harness = str(row["harness"])
    benchmark = str(row["benchmark"])
    models = [str(model) for model in (row.get("models") or [])]
    spans = sorted(
        row.get("spans") or [],
        key=lambda span: (parse_timestamp(span["start_time"]), str(span.get("span_id", ""))),
    )
    if not spans:
        raise ValueError(f"session {session_id!r} has no spans")

    first_start = min(parse_timestamp(span["start_time"]) for span in spans)
    last_end = max(parse_timestamp(span["end_time"]) for span in spans)
    running_end: datetime | None = None
    previous_start: datetime | None = None
    previous_end: datetime | None = None
    span_features: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    intervals: list[tuple[datetime, datetime]] = []

    for span_index, span in enumerate(spans):
        start = parse_timestamp(span["start_time"])
        end = parse_timestamp(span["end_time"])
        intervals.append((start, end))
        attributes = span.get("attributes") or {}
        status = span.get("status") or {}
        status_code = status.get("code")
        output_messages, output_json_valid = parse_json_list(
            attributes.get("gen_ai.output.messages")
        )
        input_messages, input_json_valid = parse_json_list(
            attributes.get("gen_ai.input.messages")
        )
        tool_definitions, tool_definitions_json_valid = parse_json_list(
            attributes.get("gen_ai.tool.definitions")
        )
        tool_names = tool_calls_from_messages(output_messages)
        tool_counts.update(tool_names)
        event_class = _event_class(status_code, tool_names)
        route_key = _route_key(event_class, tool_names)
        route_counts[route_key] += 1
        finish_reasons = [
            str(reason) for reason in (attributes.get("gen_ai.response.finish_reasons") or [])
        ]
        input_text = attributes.get("gen_ai.input.messages") or ""
        output_text = attributes.get("gen_ai.output.messages") or ""
        tool_definitions_text = attributes.get("gen_ai.tool.definitions") or ""
        frontier_gap = None if running_end is None else (start - running_end).total_seconds()
        span_features.append(
            {
                "source_file": source_file,
                "source_row": source_row,
                "session_id": session_id,
                "benchmark": benchmark,
                "harness": harness,
                "models": "|".join(models),
                "session_span_count": len(spans),
                "span_index": span_index,
                "span_id": str(span.get("span_id", "")),
                "span_type": str(span.get("type", "")),
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "start_offset_s": (start - first_start).total_seconds(),
                "ready_offset_s": (end - first_start).total_seconds(),
                "duration_ms": (end - start).total_seconds() * 1_000.0,
                "inter_start_s": (
                    None if previous_start is None else (start - previous_start).total_seconds()
                ),
                "gap_from_previous_end_s": (
                    None if previous_end is None else (start - previous_end).total_seconds()
                ),
                "frontier_gap_s": frontier_gap,
                "overlaps_active_span": bool(frontier_gap is not None and frontier_gap < 0),
                "status_code": status_code,
                "input_tokens": attributes.get("gen_ai.usage.input_tokens"),
                "output_tokens": attributes.get("gen_ai.usage.output_tokens"),
                "input_chars": len(input_text),
                "output_chars": len(output_text),
                "tool_definition_chars": len(tool_definitions_text),
                "input_message_count": len(input_messages),
                "output_message_count": len(output_messages),
                "tool_definition_count": len(tool_definitions),
                "tool_call_count": len(tool_names),
                "unique_tool_call_count": len(set(tool_names)),
                "tool_names": json.dumps(tool_names, separators=(",", ":")),
                "event_class": event_class,
                "route_key": route_key,
                "finish_reasons": "|".join(finish_reasons),
                "input_json_valid": input_json_valid,
                "output_json_valid": output_json_valid,
                "tool_definitions_json_valid": tool_definitions_json_valid,
            }
        )
        previous_start = start
        previous_end = end
        running_end = end if running_end is None else max(running_end, end)

    route_total = sum(route_counts.values())
    route_probabilities = [count / route_total for count in route_counts.values()]
    route_entropy_bits = -sum(p * math.log2(p) for p in route_probabilities)
    route_concentration = sum(p * p for p in route_probabilities)
    session_summary = {
        "source_file": source_file,
        "source_row": source_row,
        "session_id": session_id,
        "benchmark": benchmark,
        "harness": harness,
        "models": "|".join(models),
        "span_count": len(spans),
        "session_start": first_start.isoformat(),
        "session_end": last_end.isoformat(),
        "session_duration_s": (last_end - first_start).total_seconds(),
        "tool_call_spans": sum(feature["event_class"] == "tool_call" for feature in span_features),
        "tool_call_count": sum(feature["tool_call_count"] for feature in span_features),
        "unique_tool_count": len(tool_counts),
        "route_count": len(route_counts),
        "dominant_route_share": max(route_probabilities),
        "route_concentration": route_concentration,
        "effective_route_count": 1.0 / route_concentration,
        "route_entropy_bits": route_entropy_bits,
        "peak_recorded_span_concurrency": _peak_positive_interval_concurrency(intervals),
        "overlapping_span_starts": sum(
            bool(feature["overlaps_active_span"]) for feature in span_features
        ),
        "invalid_json_fields": sum(
            not feature[field]
            for feature in span_features
            for field in (
                "input_json_valid",
                "output_json_valid",
                "tool_definitions_json_valid",
            )
        ),
        "nonpositive_duration_spans": sum(feature["duration_ms"] <= 0 for feature in span_features),
        "failed_status_spans": sum(
            feature["status_code"] not in (None, 0, 1) for feature in span_features
        ),
        "total_input_tokens": sum(feature["input_tokens"] or 0 for feature in span_features),
        "total_output_tokens": sum(feature["output_tokens"] or 0 for feature in span_features),
    }
    return span_features, session_summary
