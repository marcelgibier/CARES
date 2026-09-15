from __future__ import annotations

import argparse
import os
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ...config import DEFAULT_CLAUDE_MODEL, Paths
from ...dataset import load_scenarios, restrict_to_splits, strip_prosody_tags
from ...jsonio import load_json, save_json
from ...llm import ClaudeChat
from ...llm.claude_batch import POLL_INTERVAL, run_batch
from ...log import get_logger
from ...runner import Checkpoint, run_async, run_pool

log = get_logger(__name__)

MIN_TURNS = 8
MAX_TURNS = 13

GEN_MAX_TOKENS = 2500
GEN_MAX_RETRIES = 5

SAVE_EVERY = 50
DEFAULT_CONCURRENCY = 8
HTTP_TIMEOUT = 600.0


WORN_PHRASINGS: tuple[tuple[str, str], ...] = (
    ("what was I saying", r"what was i saying"),
    ("where was I", r"where was i\s*[?.!]"),
    ("where were we", r"where were we\s*[?.!]"),
    ("what were we talking about", r"what were we talking about"),
    ("what was the last thing I said", r"what was the last thing i (said|was saying)"),
    ("I lost my train of thought", r"lost my train of thought"),
    ("I lost my place", r"lost my place"),
    ("I've completely lost it", r"completely lost it"),
    ("sorry, I blanked", r"i blanked"),
)


FORBIDDEN_TEXT_PATTERNS = [
    (re.compile(r"—"), "em dash (—)"),
    (re.compile(r";"), "semicolon (;)"),
    (re.compile(r"[()]"), "parenthesis"),
    (re.compile(r"\b(uh|um|er|erm|hmm)\b", re.IGNORECASE),
     "hesitation filler (uh/um/er/erm/hmm)"),
    (re.compile(r"\bare you (still |)there\b", re.IGNORECASE), "phone call marker"),
    (re.compile(r"\bspeakerphone\b", re.IGNORECASE), "speakerphone forbidden"),
    (re.compile(r"\bhold on,?\s+i'?m on (a|the|another) call\b", re.IGNORECASE),
     "phone call marker"),
    (re.compile(r"\bcan you hear me\b", re.IGNORECASE), "phone call marker"),
    (re.compile(r"^(A|B)\s*:", re.IGNORECASE), "speaker label in the text"),
    (re.compile(r"\*[^*]+\*"), "stage direction (asterisks)"),
    (re.compile(r"^\s*(\[[^\]]+\]\s*)+$"), "utterance with no speech (tags only)"),
]

FORBIDDEN_TEXT_PATTERNS += [
    (re.compile(pattern, re.IGNORECASE), f"worn phrasing '{phrase}'")
    for phrase, pattern in WORN_PHRASINGS
]

MAX_WORDS_PER_TURN = 60

MAX_CONSECUTIVE_TURNS = 3


WEAK_TAG_WORDS: tuple[str, ...] = (
    "slightly", "slight", "a bit", "a little", "a touch", "somewhat",
    "faintly", "faint", "mildly", "mild", "subtly", "subtle", "barely",
    "lightly", "vaguely", "marginally", "half", "sort of", "kind of",
)

_WEAK_TAG_RE = re.compile(
    r"\b(?:" + "|".join(w.replace(" ", r"\s+") for w in WEAK_TAG_WORDS) + r")\b",
    re.IGNORECASE,
)
_PROSODY_TAG_RE = re.compile(r"\[([^\]]*)\]")


def _clean_tag(inner: str) -> tuple[str, bool]:
    shout = bool(_VOLUME_TAG_RE.search(inner))
    inner = _MANNER_TAG_RE.sub(
        " ", _VOLUME_TAG_RE.sub(" ", _WEAK_TAG_RE.sub(" ", inner)))
    inner = re.sub(r"\s+", " ", inner).strip(" ,;")
    inner = re.sub(r"\s*,\s*", ", ", inner)
    words = [w for w in re.findall(r"[a-z']+", inner.lower())]
    if words and all(w in _TAG_FUNCTION_WORDS for w in words):
        return "", shout
    return inner, shout


def _strengthen_one(text: str) -> str:
    out: list[str] = []
    pos, shout = 0, False
    for m in _PROSODY_TAG_RE.finditer(text):
        run = text[pos:m.start()]
        out.append(_shout_span(run) if shout else run)
        shout = False
        inner, is_volume = _clean_tag(m.group(1))
        if inner:
            out.append(f"[{inner}]")
        shout = is_volume
        pos = m.end()
    run = text[pos:]
    out.append(_shout_span(run) if shout else run)

    joined = "".join(out)
    if joined == text:
        return text
    return re.sub(r"[ \t]{2,}", " ", joined).strip()


VOLUME_TAG_WORDS: tuple[str, ...] = (
    "raising voice", "raised voice", "raises voice", "voice raised",
    "shouting", "shouts", "shouted", "yelling", "yells", "yelled",
    "loudly", "louder", "loud", "booming", "projecting",
    "calling out", "over the noise", "at volume",
)

_VOLUME_TAG_RE = re.compile(
    r"\b(?:" + "|".join(w.replace(" ", r"\s+") for w in VOLUME_TAG_WORDS) + r")\b",
    re.IGNORECASE,
)

MANNER_TAG_WORDS: tuple[str, ...] = (
    "again", "back to normal", "as before", "once more",
    "softer", "warmer", "quieter", "flatter", "gentler", "slower",
    "warming", "refocusing", "softening", "steadying",
    "flat", "clipped", "curt", "brisk",
    "unbothered", "patient", "careful", "detached",
)

_TAG_FUNCTION_WORDS = frozenset("""
a an the to it its of in on at for with and or but as so then now
is was be been that this these those there here up out back down off over
all one own more less very quite still just about into from
""".split())


_MANNER_TAG_RE = re.compile(
    r"\b(?:" + "|".join(w.replace(" ", r"\s+") for w in MANNER_TAG_WORDS) + r")\b",
    re.IGNORECASE,
)

VOCAL_TAG_SPELLINGS: tuple[str, ...] = (
    "[laughs]", "[laughs harder]", "[starts laughing]", "[chuckles]", "[giggles]",
    "[sighs]", "[exhales]", "[gasps]", "[groans]", "[snorts]", "[sniffs]",
    "[clears throat]", "[crying]", "[sobs]", "[gulps]", "[swallows]", "[coughs]",
    "[yawns]",
)
_SENTENCE_END_RE = re.compile(r"[.!?]")


def _shout_span(rest: str) -> str:
    lead = len(rest) - len(rest.lstrip())
    head, body = rest[:lead], rest[lead:]
    m = _SENTENCE_END_RE.search(body)
    if not m:
        return head + body.upper()
    end = m.end()
    sentence, tail = body[:end], body[end:]
    if sentence.endswith("."):
        sentence = sentence[:-1] + "!"
    return head + sentence.upper() + tail


def repair_item_types(timeline: list[dict]) -> list[dict]:
    for item in timeline:
        if isinstance(item, dict):
            fixed = normalize_item_type(item)
            if fixed in ("utterance", "event"):
                item["type"] = fixed
    return timeline


def strengthen_prosody_tags(timeline: list[dict]) -> list[dict]:
    for item in timeline:
        if item.get("type") == "utterance" and isinstance(item.get("text"), str):
            item["text"] = _strengthen_one(item["text"])
    return timeline


TIMELINE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["utterance", "event"]},
        "speaker": {"type": "string", "enum": ["A", "B"]},
        "text": {"type": "string"},
        "event_id": {"type": "string"},
    },
    "required": ["type"],
}

DIALOGUE_SCHEMA = {
    "type": "object",
    "properties": {"timeline": {"type": "array", "items": TIMELINE_ITEM_SCHEMA}},
    "required": ["timeline"],
    "additionalProperties": False,
}

DIALOGUE_TOOL = {
    "name": "emit_dialogue",
    "description": ("Emit the final dialogue as an ordered timeline of "
                    "utterances and acoustic events, conforming to the schema."),
    "input_schema": DIALOGUE_SCHEMA,
}


@dataclass
class TimelineStats:
    n_utterances: int = 0
    events: list[tuple[int, str]] = field(default_factory=list)
    speakers: set[str] = field(default_factory=set)


_UTTERANCE_KEYS = ("speaker", "text")


def normalize_item_type(item: dict) -> str | None:
    declared = item.get("type")
    if declared in ("utterance", "event"):
        return declared
    looks_utterance = all(k in item for k in _UTTERANCE_KEYS)
    looks_event = "event_id" in item
    if looks_utterance and not looks_event:
        return "utterance"
    if looks_event and not looks_utterance:
        return "event"
    return declared


def scan_timeline(
    timeline: Any,
    check_event: Callable[[int, str], str | None],
) -> tuple[str | None, TimelineStats]:
    stats = TimelineStats()
    if not isinstance(timeline, list):
        return "timeline is not a list", stats
    if len(timeline) == 0:
        return "empty timeline", stats

    last_speaker = None
    consecutive = 0

    for i, item in enumerate(timeline):
        if not isinstance(item, dict):
            return f"timeline[{i}] not a dict", stats
        item_type = normalize_item_type(item)

        if item_type == "utterance":
            speaker = item.get("speaker")
            if speaker not in ("A", "B"):
                return f"timeline[{i}]: invalid speaker {speaker!r}", stats
            text = (item.get("text") or "").strip()
            if not text:
                return f"timeline[{i}]: empty text", stats
            for pattern, label in FORBIDDEN_TEXT_PATTERNS:
                if pattern.search(text):
                    return f"timeline[{i}]: {label} in '{text[:60]}...'", stats
            spoken = strip_prosody_tags(text)
            if not spoken:
                return f"timeline[{i}]: no speech (tags only)", stats
            n_words = len(spoken.split())
            if n_words < 1:
                return f"timeline[{i}]: text too short", stats
            if n_words > MAX_WORDS_PER_TURN:
                return f"timeline[{i}]: text too long ({n_words} words)", stats
            if speaker == last_speaker:
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_TURNS:
                    return (f"timeline[{i}]: {MAX_CONSECUTIVE_TURNS} consecutive "
                            f"utterances from the same speaker ({speaker})"), stats
            else:
                consecutive = 1
                last_speaker = speaker
            stats.n_utterances += 1
            stats.speakers.add(speaker)

        elif item_type == "event":
            reason = check_event(i, item.get("event_id"))
            if reason:
                return reason, stats
            stats.events.append((i, item["event_id"]))

        else:
            return f"timeline[{i}]: invalid type {item_type!r}", stats

    return None, stats


def max_turns_for(events: list[dict], base: int = MAX_TURNS) -> int:
    if not any(e.get("reaction") == "pivot" for e in events):
        return base
    return base + len(events) + 1


def check_turn_counts(stats: TimelineStats, min_turns: int, max_turns: int) -> str | None:
    if stats.n_utterances < min_turns:
        return f"too few utterances: {stats.n_utterances} (min {min_turns})"
    if stats.n_utterances > max_turns:
        return f"too many utterances: {stats.n_utterances} (max {max_turns})"
    return None


class Variant(Protocol):
    name: str
    mode: str
    system_prompt: str

    def build_prompt(self, scenario: dict) -> tuple[str, Any]:
        pass

    def validate(self, parsed: Any, context: Any) -> tuple[bool, str]:
        pass

    def final_record(self, scenario: dict, timeline: list[dict]) -> dict:
        pass


def raw_record(scenario_id: str, timeline: list | None, raw: str, status: str) -> dict:
    return {
        "scenario_id": scenario_id,
        "timeline": timeline,
        "raw_generation": raw,
        "generation_status": status,
    }


def _postprocess_for(variant: Variant, context: Any) -> Callable[[Any], tuple[list | None, str]]:
    def postprocess(parsed: Any) -> tuple[list | None, str]:
        if not isinstance(parsed, dict):
            return None, "root is not a dict"
        ok, reason = variant.validate(parsed, context)
        return (parsed["timeline"], "") if ok else (None, reason)
    return postprocess


def _cached_is_valid(variant: Variant, scenario: dict, record: Any) -> bool:
    if not record or not record.get("timeline"):
        return False
    try:
        _, context = variant.build_prompt(scenario)
        ok, _ = variant.validate({"timeline": record["timeline"]}, context)
    except Exception:  # noqa: BLE001
        return False
    return ok


async def _generate(variant: Variant, scenarios: list[dict], chat: ClaudeChat,
                    cache: Checkpoint, concurrency: int) -> dict:
    log.info("Resume: %d dialogues already cached", len(cache))
    pending = [s for s in scenarios
               if not _cached_is_valid(variant, s, cache.get(s["scenario_id"]))]
    log.info("To generate: %d dialogues (concurrency=%d)", len(pending), concurrency)
    if not pending:
        return cache.data

    async def worker(scenario: dict) -> None:
        sid = scenario["scenario_id"]
        try:
            user, context = variant.build_prompt(scenario)
        except ValueError as exc:
            log.warning("  skip '%s' : %s", sid, exc)
            await cache.record(sid, raw_record(sid, None, "", f"build_prompt_failed: {exc}"))
            return
        try:
            result = await chat.generate(user, _postprocess_for(variant, context), label=sid)
        except Exception as exc:  # noqa: BLE001
            log.error("  unhandled failure on '%s': %s", sid, exc)
            await cache.record(sid, raw_record(sid, None, "", f"unhandled: {exc}"))
            return
        if not result.ok:
            log.warning("  failure on '%s': %s", sid, result.status)
        await cache.record(sid, raw_record(sid, result.value, result.raw, result.status))

    await run_pool(pending, worker, concurrency=concurrency, desc="Generating dialogues")
    cache.flush()
    n_ok = sum(1 for r in cache.data.values() if r.get("timeline"))
    log.info("Generation done: %d valid dialogues out of %d processed", n_ok, len(cache))
    return cache.data


def _run_batch(variant: Variant, scenarios: list[dict], args: argparse.Namespace,
               cache: Checkpoint, paths: Paths) -> dict:
    from anthropic import Anthropic

    client_kwargs: dict[str, Any] = {"timeout": HTTP_TIMEOUT, "max_retries": 2}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = Anthropic(**client_kwargs)

    chat = _build_chat(variant, args)
    by_id = {s["scenario_id"]: s for s in scenarios}

    def build_params(scenario: dict) -> dict:
        user, _ = variant.build_prompt(scenario)
        return chat.message_params(user, "tool")

    def parse_result(sid: str, parsed: Any, raw: str) -> tuple[list | None, str]:
        from ...jsonparse import parse_json_loose

        if parsed is None and raw:
            parsed = parse_json_loose(raw)
        if parsed is None:
            return None, "batch:no_json"
        scenario = by_id.get(sid)
        try:
            _, context = variant.build_prompt(scenario)
        except Exception as exc:  # noqa: BLE001
            return None, f"batch:build_prompt_failed: {exc}"
        ok, reason = variant.validate(parsed, context)
        if ok:
            return parsed["timeline"], "ok (batch)"
        return None, f"batch:validation_failed: {reason}"

    results = run_batch(
        client=client,
        items=scenarios,
        item_id=lambda s: s["scenario_id"],
        build_params=build_params,
        results=cache.data,
        results_path=paths.raw_dialogues(variant.mode),
        state_path=paths.batches_state(variant.mode),
        needs_work=lambda s, rec: not _cached_is_valid(variant, s, rec),
        tool_name=DIALOGUE_TOOL["name"],
        parse_result=parse_result,
        make_record=lambda sid, value, raw, status: raw_record(sid, value, raw, status),
        wait=not args.batch_no_wait,
        interval=args.batch_poll_interval,
        desc="dialogues",
    )
    if args.batch_no_wait:
        log.info("Batch(es) submitted. Re-run with --batch (without --batch-no-wait) "
                 "to wait, collect and finalize.")
    return results


def warn_if_struct_mode_ignored(args: argparse.Namespace) -> None:
    if args.batch and args.struct_mode != "auto":
        log.warning("--struct-mode %s ignored under --batch: submission always "
                    "goes through the tool.", args.struct_mode)


def _build_chat(variant: Variant, args: argparse.Namespace) -> ClaudeChat:
    return ClaudeChat(
        model=args.model,
        system=variant.system_prompt,
        tool=DIALOGUE_TOOL,
        api_key=args.api_key,
        base_url=args.base_url,
        max_tokens=GEN_MAX_TOKENS,
        temperature=args.temperature,
        top_p=args.top_p,
        struct_mode=args.struct_mode,
        max_retries=getattr(args, "max_retries", GEN_MAX_RETRIES),
        timeout=HTTP_TIMEOUT,
        prompt_cache=not args.no_prompt_cache,
    )


def finalize(variant: Variant, results: dict, scenarios: list[dict], paths: Paths) -> None:
    by_id = {s["scenario_id"]: s for s in scenarios}
    final: list[dict] = []
    n_skipped = 0

    for sid in sorted(by_id):
        record = results.get(sid)
        if not record or not record.get("timeline"):
            n_skipped += 1
            continue
        final.append(variant.final_record(by_id[sid], record["timeline"]))

    output = paths.dialogues(variant.mode)
    save_json(output, final)
    log.info("Final dataset (%s): %d dialogues -> %s", variant.name, len(final), output)
    log.info("  Scenarios without a valid dialogue: %d", n_skipped)


def load_regen_ids(path: str | Path) -> set[str] | None:
    data = load_json(Path(path))
    if data is None:
        return None
    if isinstance(data, dict):
        data = list(data)
    if not isinstance(data, list):
        return None
    return {str(x) for x in data if x}


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default=None,
                        help="Scenarios JSON (default: scenarios.json of the data-dir)")
    parser.add_argument("--model", default=DEFAULT_CLAUDE_MODEL,
                        help=f"Claude model (default: {DEFAULT_CLAUDE_MODEL})")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (otherwise ANTHROPIC_API_KEY)")
    parser.add_argument("--base-url", default=None,
                        help="Alternative base URL (compatible proxy)")
    parser.add_argument("--only-split", default=None,
                        help="Process only these splits, e.g. 'test' or 'dev,test'.")
    parser.add_argument("--regen-from", default=None,
                        help="JSON of scenario_id to REGENERATE: their dialogues "
                             "are dropped from the cache, then produced again, as "
                             "with `filter_output/scenarios_to_regen.json`. This "
                             "DESTROYS paid generations, and a dialogue rejected "
                             "by the filter is structurally valid: without this "
                             "option it would be kept as is.")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS,
                        help=f"Turn cap (default: {MAX_TURNS}; a pivot scene also "
                             f"gets one turn per event). Raising it accepts longer "
                             f"dialogues instead of losing them, which is often the "
                             f"right call on a remainder, since overflowing scenes "
                             f"are mostly PIVOT scenes. Report it in the paper if "
                             f"part of the dataset was generated under another cap.")
    parser.add_argument("--max-retries", type=int, default=GEN_MAX_RETRIES,
                        help=f"Attempts PER MODE before giving up on a scenario "
                             f"(default: {GEN_MAX_RETRIES}). Under --struct-mode "
                             f"auto Claude does two, so the real budget is twice "
                             f"that and every attempt is billed. No effect with "
                             f"--batch, which retries nothing.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug: process only N scenarios (drawn at random)")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Shuffle seed, to make --limit reproducible")
    parser.add_argument("--struct-mode", choices=["auto", "tool", "prompt"], default="auto",
                        help="auto = tool then fall back on prompt")
    parser.add_argument("--finalize-only", action="store_true",
                        help="No generation, rebuild the dataset from the cache")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Drop cached dialogues without a timeline, then re-run")
    parser.add_argument("--no-prompt-cache", action="store_true",
                        help="Disable cache_control on the system prompt")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Send only to a model that accepts it")
    parser.add_argument("--top-p", type=float, default=None, help="Same as temperature")
    parser.add_argument("--batch", action="store_true",
                        help="Generate through the Batch API (~50%% cheaper)")
    parser.add_argument("--batch-no-wait", action="store_true",
                        help="With --batch: submit without blocking on polling")
    parser.add_argument("--batch-poll-interval", type=int, default=POLL_INTERVAL,
                        help="Interval (s) between two batch checks")


def run_variant(variant: Variant, args: argparse.Namespace,
                select: Callable[[list[dict]], list[dict]] | None = None) -> int:
    paths = Paths.resolve(args.data_dir)
    input_path = args.input or paths.scenarios
    scenarios = load_scenarios(input_path)
    if scenarios is None:
        log.error("Input file not found: %s", input_path)
        return 1

    if select is not None:
        scenarios = select(scenarios)
        if not scenarios:
            return 1

    if getattr(args, "only_split", None):
        before = len(scenarios)
        scenarios = restrict_to_splits(scenarios, args.only_split)
        log.info("--only-split %s: %d scenarios out of %d", args.only_split,
                 len(scenarios), before)
        if not scenarios:
            log.error("No scenario in that split or those splits.")
            return 1

    random.Random(args.shuffle_seed).shuffle(scenarios)
    if args.limit:
        scenarios = scenarios[: args.limit]
    log.info("Scenarios loaded: %d (variant %s)", len(scenarios), variant.name)

    cache = Checkpoint(paths.raw_dialogues(variant.mode), save_every=SAVE_EVERY)

    if args.finalize_only:
        if not cache.data:
            log.error("No %s found.", paths.raw_dialogues(variant.mode))
            return 1
        finalize(variant, cache.data, scenarios, paths)
        return 0

    if not (args.api_key or os.getenv("ANTHROPIC_API_KEY")):
        log.error("Missing API key: pass --api-key or set ANTHROPIC_API_KEY.")
        return 2

    if args.retry_failed:
        removed = cache.drop(lambda r: bool(r.get("timeline")))
        log.info("--retry-failed: %d failures dropped from the cache", removed)

    if getattr(args, "regen_from", None):
        to_redo = load_regen_ids(args.regen_from)
        if to_redo is None:
            log.error("--regen-from: file not found or unreadable: %s",
                      args.regen_from)
            return 1
        present = {r.get("scenario_id") for r in cache.data.values()} & to_redo
        backup = paths.raw_dialogues(variant.mode).with_suffix(".json.before_regen")
        save_json(backup, cache.data)
        log.info("--regen-from: cache backed up -> %s", backup)
        removed = cache.drop(lambda r: r.get("scenario_id") not in to_redo)
        log.info("--regen-from: %d id(s) requested, %d present in the cache, "
                 "%d dialogue(s) dropped - they will be REGENERATED and re-billed",
                 len(to_redo), len(present), removed)
        if not removed:
            log.warning("  no dialogue dropped: was the filter re-run since the "
                        "last generation?")

    warn_if_struct_mode_ignored(args)
    if args.batch:
        results = _run_batch(variant, scenarios, args, cache, paths)
    else:
        chat = _build_chat(variant, args)
        results = run_async(_generate(variant, scenarios, chat, cache, args.concurrency))

    finalize(variant, results, scenarios, paths)
    return 0
