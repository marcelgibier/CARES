from __future__ import annotations

import random
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from .log import get_logger

log = get_logger(__name__)

SPLITS: tuple[str, ...] = ("train", "dev", "test")

DEFAULT_SPLIT_SIZES: dict[str, int] = {"train": 800, "dev": 100, "test": 100}

SPLIT_SEED = 20260815


def parse_sizes(spec: str, n_templates: int) -> dict[str, int] | None:
    if not spec or spec.lower() in ("none", "off"):
        return None
    parts = spec.split("/")
    if len(parts) != 3:
        raise ValueError(f"--split expects 'train/dev/test' or 'none', got {spec!r}")

    values: list[int] = []
    for part in parts:
        part = part.strip()
        if part.endswith("%"):
            values.append(round(n_templates * float(part[:-1]) / 100))
        else:
            values.append(int(part))
    sizes = dict(zip(SPLITS, values, strict=True))

    total = sum(sizes.values())
    if total != n_templates:
        raise ValueError(f"--split: {total} templates requested for {n_templates} "
                         f"available ({sizes}).")
    if any(v < 0 for v in sizes.values()):
        raise ValueError(f"--split: negative count ({sizes}).")
    return sizes


def _proportional_shares(group_size: int, remaining: dict[str, int]) -> dict[str, int]:
    total_remaining = sum(remaining.values())
    if total_remaining <= 0:
        return {s: 0 for s in SPLITS}

    exact = {s: group_size * remaining[s] / total_remaining for s in SPLITS}
    shares = {s: min(remaining[s], int(exact[s])) for s in SPLITS}

    leftover = group_size - sum(shares.values())
    by_fraction = sorted(SPLITS, key=lambda s: (-(exact[s] - int(exact[s])), SPLITS.index(s)))
    for split in by_fraction:
        if leftover <= 0:
            break
        if shares[split] < remaining[split]:
            shares[split] += 1
            leftover -= 1
    return shares


def assign_templates(templates: list[dict], sizes: dict[str, int],
                     seed: int = SPLIT_SEED) -> dict[str, str]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for template in templates:
        groups[(template.get("category"), template.get("theme"))].append(template)

    rng = random.Random(seed)
    remaining = dict(sizes)
    assignment: dict[str, str] = {}

    for key in sorted(groups, key=lambda k: (str(k[0]), str(k[1]))):
        members = sorted(groups[key], key=lambda t: t["template_id"])
        rng.shuffle(members)
        shares = _proportional_shares(len(members), remaining)
        cursor = 0
        for split in SPLITS:
            for template in members[cursor:cursor + shares[split]]:
                assignment[template["template_id"]] = split
            cursor += shares[split]
            remaining[split] -= shares[split]

    missing = [t["template_id"] for t in templates if t["template_id"] not in assignment]
    if missing:  # pragma: no cover
        raise RuntimeError(f"{len(missing)} templates left unassigned (e.g. {missing[:3]}).")
    return assignment


def describe_template_assignment(templates: list[dict], assignment: dict[str, str]) -> list[str]:
    by_split: Counter = Counter(assignment.values())
    themes: dict[str, set] = defaultdict(set)
    categories: dict[str, set] = defaultdict(set)
    for template in templates:
        split = assignment[template["template_id"]]
        themes[split].add(template.get("theme"))
        categories[split].add(template.get("category"))

    n_themes = len({t.get("theme") for t in templates})
    lines = [f"Template split ({len(templates)} in total):"]
    for split in SPLITS:
        lines.append(f"  {split:5s} : {by_split[split]:4d} templates | "
                     f"{len(themes[split]):3d}/{n_themes} themes | "
                     f"{len(categories[split])} categories")
    return lines


def split_report(todos: list[dict]) -> dict:
    report: dict[str, dict] = {}
    for split in SPLITS:
        members = [t for t in todos if t.get("split") == split]
        if not members:
            continue
        scenes: Counter = Counter(t["scene"] for t in members)
        comps: Counter = Counter(
            (t["n_pivot"], t["n_verbal"], t["n_behavioral"], t["n_ambient"]) for t in members)
        reactions: Counter = Counter()
        for todo in members:
            for event in todo["events"]:
                reactions[event["reaction"]] += 1
        n_events = sum(reactions.values())
        report[split] = {
            "n_scenarios": len(members),
            "n_templates": len({t["template"]["template_id"] for t in members}),
            "n_themes": len({t["template"].get("theme") for t in members}),
            "n_events": n_events,
            "scenes": {"n": len(scenes), "min": min(scenes.values()), "max": max(scenes.values())},
            "compositions": {"n": len(comps), "min": min(comps.values()),
                             "max": max(comps.values())},
            "reactions": dict(reactions),
            "reaction_share": {r: n / n_events for r, n in reactions.items()} if n_events else {},
            "n_rare": sum(1 for t in members if t["has_rare_event"]),
        }
    return report


def describe_split_report(report: dict) -> list[str]:
    if not report:
        return ["Split: none (monolithic dataset)"]
    lines = ["Train / dev / test split:"]
    for split, stats in report.items():
        lines.append(
            f"  {split:5s} : {stats['n_scenarios']:5d} scenarios | "
            f"{stats['n_templates']:4d} templates | {stats['n_themes']:3d} themes | "
            f"scenes {stats['scenes']['min']}-{stats['scenes']['max']} | "
            f"compositions {stats['compositions']['n']}/30 | "
            f"rare {stats['n_rare']}")
        shares = stats["reaction_share"]
        lines.append("          sounds: " + "  ".join(
            f"{r}={stats['reactions'].get(r, 0)} ({100 * shares.get(r, 0):.1f}%)"
            for r in ("pivot", "verbal", "behavioral", "ambient")))
    return lines


def check_disjoint(todos: list[dict]) -> list[str]:
    by_split: dict[str, set] = defaultdict(set)
    for todo in todos:
        split = todo.get("split")
        if split:
            by_split[split].add(todo["template"]["template_id"])

    problems = []
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            shared = by_split.get(a, set()) & by_split.get(b, set())
            if shared:
                problems.append(f"{len(shared)} template(s) shared between {a} and {b}: "
                                f"{sorted(shared)[:3]}")
    return problems


DEFAULT_FILE_RATIOS: dict[str, float] = {"train": 0.8, "dev": 0.1, "test": 0.1}

_ACTIVE_RATIOS: dict[str, float] = dict(DEFAULT_FILE_RATIOS)


def parse_ratios(spec: str | None) -> dict[str, float]:
    if not spec:
        return dict(DEFAULT_FILE_RATIOS)
    parts = spec.split("/")
    if len(parts) != 3:
        raise ValueError(f"--audio-ratios expects 'train/dev/test', got {spec!r}")
    values = [float(p) for p in parts]
    if any(v < 0 for v in values):
        raise ValueError(f"--audio-ratios: negative share ({spec!r})")
    total = sum(values)
    if total <= 0:
        raise ValueError("--audio-ratios: zero sum")
    return {s: v / total for s, v in zip(SPLITS, values, strict=True)}


def set_file_ratios(ratios: dict[str, float]) -> None:
    global _ACTIVE_RATIOS
    _ACTIVE_RATIOS = dict(ratios)


def partition_files(files: list[Path], ratios: dict[str, float] | None = None,
                    salt: str = "") -> dict[str, list[Path]]:
    ratios = ratios or _ACTIVE_RATIOS
    ordered = sorted(files, key=lambda p: (zlib.crc32(f"{salt}|{p.name}".encode()), p.name))
    n = len(ordered)
    out: dict[str, list[Path]] = {s: [] for s in SPLITS}
    if n == 0:
        return out

    exact = {s: n * ratios.get(s, 0.0) for s in SPLITS}
    counts = {s: int(exact[s]) for s in SPLITS}
    leftover = n - sum(counts.values())
    for split in sorted(SPLITS, key=lambda s: (-(exact[s] - int(exact[s])), SPLITS.index(s))):
        if leftover <= 0:
            break
        counts[split] += 1
        leftover -= 1

    wanted = [s for s in SPLITS if ratios.get(s, 0.0) > 0]
    if n >= len(wanted):
        for split in wanted:
            if counts[split] == 0:
                donor = max(SPLITS, key=lambda s: counts[s])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[split] += 1
    else:
        counts = {s: (1 if i < n else 0) for i, s in enumerate(wanted)}
        counts = {s: counts.get(s, 0) for s in SPLITS}

    cursor = 0
    for split in SPLITS:
        out[split] = ordered[cursor:cursor + counts[split]]
        cursor += counts[split]
    return out


def check_pool(files: list[Path], label: str, splits_needed: tuple[str, ...] = SPLITS,
               ratios: dict[str, float] | None = None) -> tuple[bool, str]:
    if not files:
        return False, f"{label}: no file"
    partition = partition_files(files, ratios, salt=label)
    empty = [s for s in splits_needed if not partition[s]]
    if empty:
        return False, (f"{label}: {len(files)} file(s), nothing for {', '.join(empty)} "
                       f"-> disjoint partition impossible")
    return True, f"{label}: {len(files)} files -> " + " ".join(
        f"{s}={len(partition[s])}" for s in SPLITS)
