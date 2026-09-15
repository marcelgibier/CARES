from __future__ import annotations

from pathlib import Path

from .config import SCENARIO_BANKS
from .jsonio import load_json
from .log import get_logger

log = get_logger(__name__)

REQUIRED_BANKS = ("relations", "scenes", "events_per_scene", "general_events", "rare_events")

SHARED_NAMESPACE = "shared"


def event_id(category: str, name: str) -> str:
    return f"{category}/{name}"


def event_name(event_id_str: str) -> str:
    return event_id_str.split("/", 1)[1] if "/" in event_id_str else event_id_str


def load_banks(path: str | Path = SCENARIO_BANKS, *, max_events: int = 3) -> dict:
    banks = load_json(path)
    if banks is None:
        raise FileNotFoundError(f"Banks not found: {path}")
    for key in REQUIRED_BANKS:
        if key not in banks:
            raise ValueError(f"Missing bank: '{key}' in {path}")

    banks.setdefault("vocal_events", [])
    banks.setdefault("shared_events", {})

    for scene in banks["scenes"]:
        if scene not in banks["events_per_scene"]:
            raise ValueError(f"Scene '{scene}' without events_per_scene")
        if not banks["events_per_scene"][scene]:
            raise ValueError(f"Scene '{scene}' has an empty bank")

    known_scenes = set(banks["scenes"])
    for name, scenes in banks["shared_events"].items():
        if not scenes:
            raise ValueError(f"Shared sound '{name}' without a scene: move it to "
                             f"general_events if it holds everywhere.")
        unknown = [s for s in scenes if s not in known_scenes]
        if unknown:
            raise ValueError(f"Shared sound '{name}': unknown scene(s) {unknown}")
        if len(scenes) == 1:
            raise ValueError(f"Shared sound '{name}': a single scene ({scenes[0]}), "
                             f"move it to events_per_scene.")
        clash = [s for s in scenes if name in banks["events_per_scene"].get(s, [])]
        if clash:
            raise ValueError(f"Shared sound '{name}' also declared in "
                             f"events_per_scene for {clash}")

    for scene in banks["scenes"]:
        n_general = sum(1 for name in banks["general_events"]
                        if scene not in GENERAL_EVENT_FORBIDDEN_SCENES.get(name, ()))
        n_eligible = (len(banks["events_per_scene"][scene])
                      + len(shared_for_scene(banks, scene)) + n_general)
        if n_eligible < max_events:
            raise ValueError(
                f"Scene '{scene}': only {n_eligible} non-vocal events "
                f"(scene + shared + general), at least {max_events} are needed "
                f"to fill up to {max_events} ordinary events.")

    check_forbidden_scenes(banks, path)
    return banks


def shared_for_scene(banks: dict, scene: str) -> list[str]:
    return [name for name, scenes in (banks.get("shared_events") or {}).items()
            if scene in scenes]


def scene_pool(banks: dict, scene: str) -> list[str]:
    return (
        [event_id(scene, name) for name in banks["events_per_scene"][scene]]
        + [event_id(SHARED_NAMESPACE, name) for name in shared_for_scene(banks, scene)]
        + [event_id("general", name) for name in banks["general_events"]
           if scene not in GENERAL_EVENT_FORBIDDEN_SCENES.get(name, ())]
    )


RARE_EVENT_FORBIDDEN_SCENES: dict[str, set[str]] = {
    "window_smashing": {"beach", "rural", "nighttime_nature", "public_park", "street_traffic"},
    "fire_alarm": {"beach", "rural", "nighttime_nature", "public_park", "street_traffic", "car"},
    "crash_vehicle": {"beach", "rural", "nighttime_nature", "public_park"},
    "animal_growling": {"office", "shopping_mall", "airport", "train_station"},
    "firework_pop": {"office", "car", "shopping_mall", "airport"},
    "siren_police": {"beach", "nighttime_nature"},
    "siren_ambulance": {"beach", "nighttime_nature"},
}

GENERAL_EVENT_FORBIDDEN_SCENES: dict[str, set[str]] = {
    "egg_crack": {"street_traffic", "public_park", "shopping_mall", "beach", "car",
                  "construction_site", "nighttime_nature", "airport", "train_station",
                  "office"},
    "spray_paint_shake": {"car", "airport", "cafe_restaurant", "shopping_mall", "office",
                          "beach", "train_station", "nighttime_nature"},
    "balloon_pop": {"construction_site", "nighttime_nature", "rural", "car",
                    "street_traffic"},
    "cards_shuffle": {"construction_site", "street_traffic", "shopping_mall"},
    "baby_crying": {"construction_site", "nighttime_nature"},
}


def check_forbidden_scenes(banks: dict, path: str | Path = "") -> None:
    check_rare_forbidden_scenes(banks, path)
    known_scenes = set(banks["scenes"])
    known_general = set(banks["general_events"])
    if not known_general & set(GENERAL_EVENT_FORBIDDEN_SCENES):
        return
    absent = sorted(set(GENERAL_EVENT_FORBIDDEN_SCENES) - known_general)
    if absent:
        log.warning("GENERAL_EVENT_FORBIDDEN_SCENES: %s absent from %s, "
                    "constraint has no effect (rename?)", ", ".join(absent), path)
    for name, scenes in GENERAL_EVENT_FORBIDDEN_SCENES.items():
        if name not in known_general:
            continue
        unknown = sorted(scenes - known_scenes)
        if unknown:
            raise ValueError(
                f"GENERAL_EVENT_FORBIDDEN_SCENES: '{name}' forbids scenes that are "
                f"absent from {path}: {unknown}. Align those names on 'scenes' in "
                f"cares/banks.py, or remove them from the table.")


def check_rare_forbidden_scenes(banks: dict, path: str | Path = "") -> None:
    known_rares = set(banks["rare_events"])
    if not known_rares & set(RARE_EVENT_FORBIDDEN_SCENES):
        return
    known_scenes = set(banks["scenes"])
    for rare, scenes in RARE_EVENT_FORBIDDEN_SCENES.items():
        if rare not in known_rares:
            raise ValueError(
                f"RARE_EVENT_FORBIDDEN_SCENES names a rare event absent from "
                f"{path}: '{rare}'. Align the table on rare_events in "
                f"cares/banks.py (rename), or drop that entry.")
        unknown = sorted(scenes - known_scenes)
        if unknown:
            raise ValueError(
                f"RARE_EVENT_FORBIDDEN_SCENES: '{rare}' forbids scenes that are "
                f"absent from {path}: {unknown}. Align those names on 'scenes' in "
                f"cares/banks.py, or remove them from the table.")


def rare_pool_for_scene(rare_pool: list[str], scene: str) -> list[str]:
    return [rid for rid in rare_pool
            if scene not in RARE_EVENT_FORBIDDEN_SCENES.get(event_name(rid), set())]


def describe(banks: dict) -> str:
    shared = banks.get("shared_events") or {}
    return (f"{len(banks['relations'])} relations, {len(banks['scenes'])} scenes, "
            f"{sum(len(v) for v in banks['events_per_scene'].values())} scene events, "
            f"{len(shared)} shared events ({sum(len(s) for s in shared.values())} "
            f"attachments), {len(banks['general_events'])} general events, "
            f"{len(banks.get('vocal_events', []))} vocal events (rendered as prosody), "
            f"{len(banks['rare_events'])} rare events")
