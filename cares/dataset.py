from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .jsonio import load_json

REACTIONS: tuple[str, ...] = ("pivot", "verbal", "behavioral", "ambient")

REACTED_TO: tuple[str, ...] = ("pivot", "verbal", "behavioral")

REACTION_TAG = {
    "pivot": "PIVOT",
    "verbal": "VERBAL",
    "behavioral": "BEHAVIORAL",
    "ambient": "AMBIENT",
}


def load_scenarios(path: str | Path) -> list[dict] | None:
    data = load_json(path)
    if data is None:
        return None
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected scenario format: {type(data).__name__}")


def index_by_id(scenarios: list[dict]) -> dict[str, dict]:
    return {s["scenario_id"]: s for s in scenarios}


def scn_subject(scn: dict) -> str:
    subject = scn.get("subject")
    if subject is None:
        subject = (scn.get("llm_output") or {}).get("subject")
    return (subject or "").strip()


def scn_events(scn: dict) -> list[dict]:
    out = []
    for event in scn.get("events") or []:
        if not event.get("event_id"):
            continue
        canonical = {
            "event_id": event["event_id"],
            "reaction": event.get("reaction"),
            "is_rare": bool(event.get("is_rare", False)),
        }
        if event.get("register"):
            canonical["register"] = event["register"]
        if event.get("slot"):
            canonical["slot"] = event["slot"]
        out.append(canonical)
    return out


def scn_speakers(scn: dict) -> dict:
    speakers = scn.get("speakers")
    if isinstance(speakers, dict) and speakers.get("A") and speakers.get("B"):
        return {"A": speakers["A"], "B": speakers["B"]}
    return {"A": scn.get("role_a"), "B": scn.get("role_b")}


def n_reacted_to(scn: dict) -> int:
    return sum(1 for e in scn_events(scn) if e.get("reaction") in REACTED_TO)


def stratum_key(scn: dict) -> str:
    return f"r{n_reacted_to(scn)}"


def has_rare(scn: dict) -> bool:
    return bool(scn.get("has_rare_event")) or any(e.get("is_rare") for e in scn_events(scn))


def has_pivot(scn: dict) -> bool:
    return any(e.get("reaction") == "pivot" for e in scn_events(scn))


def normalize_scenario(scn: dict) -> dict:
    role_a = scn.get("role_a")
    role_b = scn.get("role_b")
    if role_a is None or role_b is None:
        speakers = scn.get("speakers") or {}
        role_a = role_a if role_a is not None else speakers.get("A")
        role_b = role_b if role_b is not None else speakers.get("B")

    theme = scn.get("theme")
    if isinstance(theme, dict):
        theme_title = scn.get("template_title") or theme.get("title", "")
        theme_description = scn.get("template_description") or theme.get("description", "")
        theme_label = theme.get("title")
    else:
        theme_title = scn.get("template_title") or (theme or "")
        theme_description = scn.get("template_description", "")
        theme_label = theme

    events = scn_events(scn)
    return {
        "scenario_id": scn.get("scenario_id"),
        "template_id": scn.get("template_id"),
        "category": scn.get("category"),
        "theme_title": theme_title,
        "theme_description": theme_description,
        "theme_label": theme_label,
        "scene": scn.get("scene"),
        "role_a": role_a,
        "role_b": role_b,
        "gender_a": scn.get("gender_a"),
        "gender_b": scn.get("gender_b"),
        "subject": scn_subject(scn),
        "events": events,
        "has_rare_event": scn.get("has_rare_event", any(e["is_rare"] for e in events)),
        "scenario_idx_in_template": scn.get("scenario_idx_in_template"),
    }


PROSODY_TAG_PATTERN = re.compile(r"\[[^\]]+\]")


def strip_prosody_tags(text: str) -> str:
    return PROSODY_TAG_PATTERN.sub(" ", text).strip()


def utterances(timeline: list[dict]) -> list[dict]:
    return [item for item in timeline if item.get("type") == "utterance"]


def timeline_events(timeline: list[dict]) -> list[dict]:
    return [item for item in timeline if item.get("type") == "event"]


_TIMELINE_WRAPPERS = ("dialogue", "llm_output", "output", "result", "generation")


def find_timeline(obj: Any) -> list[dict] | None:
    if isinstance(obj, list):
        return obj if obj and isinstance(obj[0], dict) and "type" in obj[0] else None
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("timeline"), list):
        return obj["timeline"]
    for key in _TIMELINE_WRAPPERS:
        value = obj.get(key)
        if isinstance(value, dict) and isinstance(value.get("timeline"), list):
            return value["timeline"]
        if isinstance(value, list) and value and isinstance(value[0], dict) and "type" in value[0]:
            return value
    return None


def restrict_to_splits(records: list[dict], spec: str | None) -> list[dict]:
    if not spec:
        return records
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    return [r for r in records if r.get("split") in wanted]
