from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable

from .banks import event_id, rare_pool_for_scene, scene_pool
from .config import REACTION_REGISTERS
from .jsonio import load_json
from .log import get_logger
from .splits import SPLITS

log = get_logger(__name__)

DIST_SEED = 42

N_SCENARIOS_PER_TEMPLATE = 10

MIN_EVENTS_PER_SCENARIO = 0
MAX_EVENTS_PER_SCENARIO = 3

RARE_EVENT_RATE = 0.05

DEFAULT_EVENT_DIST = "compositions"

Composition = tuple[int, int, int, int]


def build_compositions() -> list[Composition]:
    combos: list[Composition] = []
    for total in range(MIN_EVENTS_PER_SCENARIO, MAX_EVENTS_PER_SCENARIO + 1):
        for n_p in range(0, min(total, 1) + 1):
            for n_v in range(0, total - n_p + 1):
                for n_b in range(0, total - n_p - n_v + 1):
                    combos.append((n_p, n_v, n_b, total - n_p - n_v - n_b))
    return combos


ALL_COMPOSITIONS = build_compositions()


def composition_key(comp: Composition) -> str:
    n_p, n_v, n_b, n_a = comp
    return f"P{n_p}V{n_v}B{n_b}A{n_a}"


def quota_counts(n_total: int, k: int) -> list[int]:
    base, rem = divmod(n_total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def weighted_quota(n_total: int, weights: list[float]) -> list[int]:
    total = sum(weights)
    if total <= 0:
        return quota_counts(n_total, len(weights))
    exact = [n_total * w / total for w in weights]
    counts = [int(x) for x in exact]
    leftover = n_total - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - int(exact[i])), i))
    for i in order[:leftover]:
        counts[i] += 1
    return counts


def build_composition_sequence(n_total: int, strategy: str = DEFAULT_EVENT_DIST,
                               seed: int = DIST_SEED,
                               weights: Callable[[Composition], float] | None = None,
                               ) -> list[Composition]:
    rng = random.Random(seed)
    pool: list[Composition] = []

    if strategy == "compositions":
        counts = (weighted_quota(n_total, [weights(c) for c in ALL_COMPOSITIONS])
                  if weights else quota_counts(n_total, len(ALL_COMPOSITIONS)))
        for comp, count in zip(ALL_COMPOSITIONS, counts, strict=True):
            pool.extend([comp] * count)
    elif strategy == "counts":
        if weights is not None:
            raise ValueError("Composition weighting only applies to the "
                             "'compositions' strategy.")
        by_count: dict[int, list[Composition]] = defaultdict(list)
        for comp in ALL_COMPOSITIONS:
            by_count[sum(comp)].append(comp)
        totals = sorted(by_count)
        for total, n_for_total in zip(totals, quota_counts(n_total, len(totals)), strict=True):
            comps = by_count[total]
            for comp, count in zip(comps, quota_counts(n_for_total, len(comps)), strict=True):
                pool.extend([comp] * count)
    else:
        raise ValueError(f"Unknown distribution strategy: {strategy!r}")

    rng.shuffle(pool)
    return pool


def build_scene_sequence(n_total: int, scenes: list[str],
                         seed: int = DIST_SEED + 1) -> list[str]:
    rng = random.Random(seed)
    pool: list[str] = []
    for scene, count in zip(scenes, quota_counts(n_total, len(scenes)), strict=True):
        pool.extend([scene] * count)
    rng.shuffle(pool)
    return pool


def load_reaction_registers(path=None) -> dict[str, list[str]]:
    data = load_json(path or REACTION_REGISTERS) or {}
    return {reaction: [r["key"] for r in regs]
            for reaction, regs in (data.get("registers") or {}).items()}


def register_move(reaction: str, key: str, path=None) -> str:
    data = load_json(path or REACTION_REGISTERS) or {}
    for r in (data.get("registers") or {}).get(reaction, []):
        if r["key"] == key:
            return r["move"]
    return ""


def register_required_tags(reaction: str, key: str, path=None) -> tuple[str, ...]:
    data = load_json(path or REACTION_REGISTERS) or {}
    for r in (data.get("registers") or {}).get(reaction, []):
        if r["key"] == key:
            return tuple(r.get("required_tags") or ())
    return ()


def register_needs_delivery(reaction: str, key: str, path=None) -> bool:
    return bool(register_required_tags(reaction, key, path))


PLACEMENT_SLOTS: tuple[str, ...] = ("early", "middle", "late")

PIVOT_SLOTS: tuple[str, ...] = ("early", "middle")

SLOT_WORDS = {
    "early": "in the FIRST THIRD of the conversation",
    "middle": "in the MIDDLE THIRD of the conversation",
    "late": "in the LAST THIRD of the conversation",
}


class RegisterCycle:
    def __init__(self, registers: dict[str, list[str]], rng: random.Random) -> None:
        self.registers = {k: list(v) for k, v in registers.items()}
        self.rng = rng
        self._queues: dict[str, list[str]] = {}

    def next(self, reaction: str) -> str | None:
        pool = self.registers.get(reaction)
        if not pool:
            return None
        queue = self._queues.get(reaction)
        if not queue:
            queue = list(pool)
            self.rng.shuffle(queue)
            self._queues[reaction] = queue
        return queue.pop()


GENDER_WORDS = {"F": "a woman", "M": "a man"}

GENDER_PAIRS: tuple[tuple[str, str], ...] = (("F", "F"), ("F", "M"),
                                             ("M", "F"), ("M", "M"))


def build_gender_sequence(n_total: int, seed: int = DIST_SEED + 5) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pool: list[tuple[str, str]] = []
    for pair, count in zip(GENDER_PAIRS, quota_counts(n_total, len(GENDER_PAIRS)),
                           strict=True):
        pool.extend([pair] * count)
    rng.shuffle(pool)
    return pool


def build_relation_sequence(n_total: int, relations: list[list[str]],
                            seed: int = DIST_SEED + 2) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pool: list[tuple[str, str]] = []
    for relation, count in zip(relations, quota_counts(n_total, len(relations)), strict=True):
        for _ in range(count):
            if rng.random() < 0.5:
                pool.append((relation[1], relation[0]))
            else:
                pool.append((relation[0], relation[1]))
    rng.shuffle(pool)
    return pool


def assign_rare_flags(comps: list[Composition], target_rate: float,
                      seed: int = DIST_SEED + 3) -> list[bool]:
    rng = random.Random(seed)
    n_target = int(round(len(comps) * target_rate))
    eligible = [i for i, comp in enumerate(comps) if comp[0] == 1]
    if n_target > len(eligible):
        log.warning("assign_rare_flags: target %d > eligible %d; capping.",
                    n_target, len(eligible))
        n_target = len(eligible)
    rng.shuffle(eligible)
    chosen = set(eligible[:n_target])
    return [i in chosen for i in range(len(comps))]


DEFAULT_EVENT_PAIRING = "random"


class EventAllocator:
    def __init__(self, banks: dict, seed: int = DIST_SEED + 4, *,
                 pairing: str = DEFAULT_EVENT_PAIRING,
                 targets: dict[str, float] | None = None) -> None:
        if pairing not in ("random", "balanced"):
            raise ValueError(f"Unknown pairing: {pairing!r}")
        self.rng = random.Random(seed)
        self.tie_rng = random.Random(seed + 1000)
        self.slot_rng = random.Random(seed + 3000)
        self.registers = RegisterCycle(load_reaction_registers(),
                                       random.Random(seed + 2000))
        self.pairing = pairing
        self.targets = targets or {"pivot": 0.125, "verbal": 0.291667,
                                   "behavioral": 0.291667, "ambient": 0.291667}
        self.pools = {scene: scene_pool(banks, scene) for scene in banks["scenes"]}
        self.rare_pool = [event_id("rare_events", e) for e in banks["rare_events"]]
        self.rare_pool_by_scene = {
            scene: rare_pool_for_scene(self.rare_pool, scene) for scene in banks["scenes"]
        }
        self.usage: Counter = Counter()
        self.pair_usage: Counter = Counter()

    def _deficit(self, event: str, reaction: str) -> float:
        placed = self.usage[event]
        share = self.pair_usage[(event, reaction)] / placed if placed else 0.0
        return share - self.targets.get(reaction, 0.0)

    def draw_for_scenario(self, scene: str, n_pivot: int, n_verbal: int,
                          n_behavioral: int, n_ambient: int,
                          has_rare: bool) -> list[dict]:
        if n_pivot not in (0, 1):
            raise ValueError(f"n_pivot must be 0 or 1, got {n_pivot}")
        if has_rare and n_pivot != 1:
            raise ValueError("has_rare requires n_pivot == 1")

        total = n_pivot + n_verbal + n_behavioral + n_ambient
        used: set[str] = set()

        rare_pivot: str | None = None
        if has_rare:
            compatible = self.rare_pool_by_scene.get(scene)
            if compatible is None:
                compatible = self.rare_pool
            elif not compatible:
                raise ValueError(
                    f"Scene '{scene}': no plausible rare event "
                    f"(RARE_EVENT_FORBIDDEN_SCENES forbids them all). Allow one, "
                    f"or remove this scene from the rare draw.")
            rare_pivot = self.rng.choice(compatible)
            used.add(rare_pivot)
            self.usage[rare_pivot] += 1

        n_normal = total - (1 if has_rare else 0)
        pool = [e for e in self.pools[scene] if e not in used]
        if n_normal > len(pool):
            raise ValueError(f"Pool of '{scene}' too small ({len(pool)}) to draw "
                             f"{n_normal} ordinary events.")
        normals = self.rng.sample(pool, n_normal) if n_normal > 0 else []
        for event in normals:
            self.usage[event] += 1

        slots = ["pivot"] * (0 if has_rare else n_pivot)
        slots += ["verbal"] * n_verbal + ["behavioral"] * n_behavioral + ["ambient"] * n_ambient
        pairs = self._pair(normals, slots)

        out: list[dict] = []
        if has_rare:
            out.append({"event_id": rare_pivot, "reaction": "pivot", "is_rare": True,
                        "register": self.registers.next("pivot")})
            self.pair_usage[(rare_pivot, "pivot")] += 1
        for event, reaction in pairs:
            out.append({"event_id": event, "reaction": reaction, "is_rare": False,
                        "register": self.registers.next(reaction)})
            self.pair_usage[(event, reaction)] += 1
        self._assign_slots(out)
        return out

    def _assign_slots(self, events: list[dict]) -> None:
        n = len(events)
        if not n:
            return
        order = list(range(n))
        self.slot_rng.shuffle(order)
        shuffled = [events[i] for i in order]

        rank = {s: i for i, s in enumerate(PLACEMENT_SLOTS)}
        if n >= len(PLACEMENT_SLOTS):
            bands = list(PLACEMENT_SLOTS)
        else:
            picked = self.slot_rng.sample(list(PLACEMENT_SLOTS), n)
            bands = sorted(picked, key=lambda s: rank[s])

        for i in range(n - 1, -1, -1):
            if shuffled[i]["reaction"] != "pivot" or bands[i] in PIVOT_SLOTS:
                continue
            for j in range(i):
                if shuffled[j]["reaction"] != "pivot" and bands[j] in PIVOT_SLOTS:
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
                    break

        for i, event in enumerate(shuffled):
            if event["reaction"] == "pivot" and bands[i] not in PIVOT_SLOTS:
                free = [s for s in PIVOT_SLOTS if s not in bands]
                bands[i] = free[0] if free else self.slot_rng.choice(list(PIVOT_SLOTS))

        for event, slot in zip(shuffled, bands, strict=True):
            event["slot"] = slot
        events[:] = shuffled

    def _pair(self, events: list[str], slots: list[str]) -> list[tuple[str, str]]:
        if self.pairing == "random":
            return list(zip(events, slots, strict=True))

        order = sorted(range(len(slots)), key=lambda i: self.targets.get(slots[i], 0.0))
        remaining = list(events)
        paired: list[tuple[str, str]] = [("", "")] * len(slots)
        for i in order:
            reaction = slots[i]
            best = min(remaining,
                       key=lambda e: (self._deficit(e, reaction), self.tie_rng.random()))
            remaining.remove(best)
            paired[i] = (best, reaction)
        return paired


def reaction_targets(comps: list[Composition]) -> dict[str, float]:
    counts = dict(zip(("pivot", "verbal", "behavioral", "ambient"),
                      [sum(c[i] for c in comps) for i in range(4)], strict=True))
    total = sum(counts.values())
    return {r: n / total for r, n in counts.items()} if total else {}


COMPOSITION_INDEX = {"pivot": 0, "verbal": 1, "behavioral": 2, "ambient": 3}


def composition_weights(boosts: dict[str, float]) -> Callable[[Composition], float] | None:
    active = {k: v for k, v in (boosts or {}).items() if v != 1.0}
    if not active:
        return None
    unknown = set(active) - set(COMPOSITION_INDEX)
    if unknown:
        raise ValueError(f"Unknown class(es) for weighting: {sorted(unknown)}")

    def weight(comp: Composition) -> float:
        value = 1.0
        for reaction, factor in active.items():
            value *= factor ** comp[COMPOSITION_INDEX[reaction]]
        return value

    return weight


def parse_boosts(spec: str) -> dict[str, float]:
    boosts: dict[str, float] = {}
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"--test-boost expects 'class=factor', got {chunk!r}")
        name, _, factor = chunk.partition("=")
        boosts[name.strip()] = float(factor)
    return boosts


def _sequences(n_total: int, banks: dict, event_dist: str, seed_offset: int,
               weights: Callable[[Composition], float] | None):
    comps = build_composition_sequence(n_total, strategy=event_dist,
                                       seed=DIST_SEED + seed_offset, weights=weights)
    scenes = build_scene_sequence(n_total, banks["scenes"], seed=DIST_SEED + 1 + seed_offset)
    relations = build_relation_sequence(n_total, banks["relations"],
                                        seed=DIST_SEED + 2 + seed_offset)
    rare_flags = assign_rare_flags(comps, RARE_EVENT_RATE, seed=DIST_SEED + 3 + seed_offset)
    genders = build_gender_sequence(n_total, seed=DIST_SEED + 5 + seed_offset)
    return comps, scenes, relations, rare_flags, genders


def build_todos(templates: list[dict], banks: dict,
                event_dist: str = DEFAULT_EVENT_DIST,
                pairing: str = DEFAULT_EVENT_PAIRING,
                assignment: dict[str, str] | None = None,
                test_boosts: dict[str, float] | None = None) -> tuple[list[dict], Counter]:
    allocator = EventAllocator(banks, pairing=pairing, targets=None)

    if assignment is None:
        lots = [("", templates, 0, None)]
    else:
        lots = []
        for offset, split in enumerate(SPLITS):
            members = [t for t in templates if assignment.get(t["template_id"]) == split]
            weights = composition_weights(test_boosts) if split == "test" else None
            lots.append((split, members, 100 * (offset + 1), weights))

    all_comps: list[Composition] = []
    prepared = []
    for split, members, offset, weights in lots:
        if not members:
            continue
        n_total = len(members) * N_SCENARIOS_PER_TEMPLATE
        sequences = _sequences(n_total, banks, event_dist, offset, weights)
        all_comps.extend(sequences[0])
        prepared.append((split, members, sequences))
    allocator.targets = reaction_targets(all_comps) or allocator.targets

    todos: list[dict] = []
    index = 0
    for split, members, (comps, scenes, relations, rare_flags, genders) in prepared:
        for i, template in enumerate(members):
            for s in range(N_SCENARIOS_PER_TEMPLATE):
                local = i * N_SCENARIOS_PER_TEMPLATE + s
                n_p, n_v, n_b, n_a = comps[local]
                scene = scenes[local]
                role_a, role_b = relations[local]
                gender_a, gender_b = genders[local]
                has_rare = rare_flags[local]
                todo = {
                    "scenario_id": f"{template['template_id']}__s{s + 1:02d}",
                    "template": template,
                    "scene": scene,
                    "role_a": role_a,
                    "role_b": role_b,
                    "gender_a": gender_a,
                    "gender_b": gender_b,
                    "n_pivot": n_p,
                    "n_verbal": n_v,
                    "n_behavioral": n_b,
                    "n_ambient": n_a,
                    "has_rare_event": has_rare,
                    "events": allocator.draw_for_scenario(scene, n_p, n_v, n_b, n_a, has_rare),
                    "scenario_idx_in_template": s + 1,
                    "global_index": index,
                }
                if split:
                    todo["split"] = split
                todos.append(todo)
                index += 1
    return todos, allocator.usage


def event_balance(todos: list[dict]) -> dict:
    usage: Counter = Counter()
    by_event: dict[str, Counter] = {}
    marginal: Counter = Counter()

    for todo in todos:
        for event in todo["events"]:
            event_id_str = event["event_id"]
            usage[event_id_str] += 1
            if event_id_str.startswith("rare_events/"):
                continue
            by_event.setdefault(event_id_str, Counter())[event["reaction"]] += 1
            marginal[event["reaction"]] += 1

    total = sum(marginal.values())
    deviations = [abs(counts[r] / sum(counts.values()) - marginal[r] / total)
                  for counts in by_event.values()
                  for r in ("pivot", "verbal", "behavioral", "ambient")] if total else [0.0]

    def group(prefix: str, keep: bool) -> list[int]:
        return sorted(n for e, n in usage.items() if e.startswith(prefix) is keep
                      and not e.startswith("rare_events/"))

    general = group("general/", True)
    scene = group("general/", False)
    ordered = sorted(n for e, n in usage.items() if not e.startswith("rare_events/"))
    usage_stats = ({"min": ordered[0], "median": ordered[len(ordered) // 2],
                    "max": ordered[-1]} if ordered else {"min": 0, "median": 0, "max": 0})
    return {
        "n_placements": sum(usage.values()),
        "n_event_ids": len(usage),
        "usage": usage_stats,
        "general": {"n_ids": len(general),
                    "mean_usage": sum(general) / len(general) if general else 0,
                    "share_of_placements": sum(general) / max(1, sum(ordered))},
        "scene": {"n_ids": len(scene),
                  "mean_usage": sum(scene) / len(scene) if scene else 0},
        "pairing_deviation_pts": {
            "mean": 100 * sum(deviations) / len(deviations),
            "max": 100 * max(deviations),
        },
        "reaction_marginal": {r: marginal[r] / total for r in marginal} if total else {},
        "usage_per_event": dict(usage.most_common()),
    }


def describe_balance(balance: dict) -> list[str]:
    general, scene = balance["general"], balance["scene"]
    ratio = general["mean_usage"] / scene["mean_usage"] if scene["mean_usage"] else float("inf")
    return [
        f"Sound balance: {balance['n_placements']} placements over "
        f"{balance['n_event_ids']} ids",
        f"  Usage: min={balance['usage']['min']} median={balance['usage']['median']} "
        f"max={balance['usage']['max']}",
        f"  General: {general['n_ids']} ids, {general['mean_usage']:.0f} placements/sound, "
        f"{100 * general['share_of_placements']:.0f}% of placements",
        f"  Scene: {scene['n_ids']} ids, {scene['mean_usage']:.0f} placements/sound "
        f"(general/scene ratio: {ratio:.1f}x)",
        f"  Pairing id x type: deviation from the marginal "
        f"{balance['pairing_deviation_pts']['mean']:.2f} pts on average, "
        f"{balance['pairing_deviation_pts']['max']:.2f} pts at worst",
    ]


def describe_distribution(todos: list[dict], strategy: str) -> list[str]:
    if not todos:
        return [f"Target distribution: no scenario (strategy={strategy})"]

    comp_count: Counter = Counter()
    count_count: Counter = Counter()
    reaction_count: Counter = Counter()
    scene_count: Counter = Counter()
    for todo in todos:
        comp = (todo["n_pivot"], todo["n_verbal"], todo["n_behavioral"], todo["n_ambient"])
        comp_count[composition_key(comp)] += 1
        count_count[sum(comp)] += 1
        for event in todo["events"]:
            reaction_count[event["reaction"]] += 1
        scene_count[todo["scene"]] += 1

    n = len(todos)
    n_pivot = sum(1 for t in todos if t["n_pivot"] >= 1)
    n_rare = sum(1 for t in todos if t["has_rare_event"])
    counts = ", ".join(f"{k}:{count_count[k]}" for k in sorted(count_count))
    return [
        f"Target distribution: {n} scenarios (strategy={strategy})",
        f"  Compositions ({len(ALL_COMPOSITIONS)} expected): "
        f"min={min(comp_count.values())}, max={max(comp_count.values())}",
        f"  Event count: {{{counts}}}",
        f"  Reactions: {dict(reaction_count.most_common())}",
        f"  Scenes: min={min(scene_count.values())}, max={max(scene_count.values())}",
        f"  With pivot: {n_pivot} ({100 * n_pivot / max(n, 1):.1f}%)",
        f"  Rare events: {n_rare} ({100 * n_rare / max(n, 1):.1f}%)",
    ]
