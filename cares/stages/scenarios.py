from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from typing import Any

from ..allocation import (
    DEFAULT_EVENT_DIST,
    DEFAULT_EVENT_PAIRING,
    GENDER_WORDS,
    MAX_EVENTS_PER_SCENARIO,
    N_SCENARIOS_PER_TEMPLATE,
    build_todos,
    composition_key,
    describe_balance,
    describe_distribution,
    event_balance,
    parse_boosts,
)
from ..banks import describe, load_banks
from ..config import DEFAULT_CLAUDE_MODEL, SCENARIO_BANKS, Paths
from ..jsonio import load_json, save_json
from ..jsonparse import parse_json_loose
from ..llm import ClaudeChat, OpenAIChat, StructuredChat
from ..llm.claude_batch import POLL_INTERVAL, run_batch
from ..log import get_logger
from ..runner import Checkpoint, run_async, run_pool
from ..splits import (
    DEFAULT_SPLIT_SIZES,
    assign_templates,
    check_disjoint,
    describe_split_report,
    describe_template_assignment,
    parse_sizes,
    split_report,
)

log = get_logger(__name__)

GEN_TEMPERATURE = 0.9
GEN_TOP_P = 0.95
GEN_MAX_NEW_TOKENS = 800
GEN_MAX_RETRIES = 5

SAVE_EVERY = 50
DEFAULT_CONCURRENCY = 64
HTTP_TIMEOUT = 600.0


SYSTEM_PROMPT = """You are designing realistic short audio-scene scenarios for \
a two-speaker dialogue TTS dataset. The scene, the two speakers, and the \
acoustic events are PRE-ALLOCATED and given to you. Your only job is to produce \
the SUBJECT of the conversation: what the two adults are talking about. You \
answer with strict JSON only."""


USER_PROMPT = """Design the SUBJECT of ONE audio-scene conversation based on \
the following theme and pre-allocated parameters.

THEME CATEGORY : {category}
THEME          : {theme}
TEMPLATE TITLE : {template_title}
TEMPLATE DESC. : {template_description}

═══════════════════════════════════════════════════════════════════
CRITICAL RULE — DISCUSSANTS, NOT ACTORS
═══════════════════════════════════════════════════════════════════
The two speakers DISCUSS the theme, they do NOT enact the typical situation.
  • Theme about acupuncture: NOT patient + acupuncturist in a treatment room. \
INSTEAD two adults talking about one of them trying acupuncture last week.
  • Theme about disciplining children: NOT parent + child. INSTEAD two adult \
parents (or two adults, period) discussing how to handle discipline.
  • Theme about a raise negotiation: NOT employee + manager mid-meeting. \
INSTEAD two adults discussing how the negotiation went.

Consequence: the LOCATION is decoupled from the theme. People discuss anything \
anywhere. BOTH SPEAKERS ARE ALWAYS ADULTS. No children. No professional \
in-role encounters (doctor/patient, teacher/student, manager-employee).

═══════════════════════════════════════════════════════════════════
PRE-ALLOCATED PARAMETERS (FIXED — do not change)
═══════════════════════════════════════════════════════════════════
- Scene/Location : {scene}
- Speaker A      : {role_a} ({gender_a})
- Speaker B      : {role_b} ({gender_b})

Sounds occurring in this scene (CONTEXT ONLY — do NOT describe, list or recap \
them in the subject; how the speakers react to them is written later):
{sounds_block}
{pivot_block}

═══════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════
Produce ONE conversation SUBJECT: a brief, concrete description of what \
speakers A and B are fundamentally talking about — the spine of the scene.

Requirements for "subject":
  - 6 to 30 words.
  - A DESCRIPTION of the topic, NOT a spoken line, NOT a quote, NOT meta-talk.
  - Coherent with the theme, the speakers' relation, and the scene.
  - Where it fits naturally, ground the subject in concrete NAMED ENTITIES — a \
date or time ("last Tuesday", "in March"), a place ("in Lisbon", "on Oak \
Street"), a person's name ("Sarah", "my brother Tom"), an amount or number \
("40 euros", "three weeks"), or a brand/product. Use one or two only when \
plausible; do NOT force them or pile them up.
  - The theme should surface NATURALLY (a personal anecdote, a question, a \
complaint, an observation), not as a structured exposition.

EXAMPLES of good subjects:
  ✓ "two friends catching up while one recounts trying acupuncture last \
Tuesday for her back pain and whether it helped"
  ✓ "a couple comparing notes after their IKEA mattress order arrived damaged"
  ✓ "two colleagues weighing whether the new 40-euro monthly parking permit is \
worth it"
  ✓ "two old friends planning a long weekend in Lisbon in early March, debating \
whether to book an Airbnb near Alfama"

EXAMPLES of BAD subjects (do NOT do this):
  ✗ "So I tried acupuncture last week..."   — a spoken line, not a subject
  ✗ "they talk about acupuncture"           — too vague
  ✗ "discusses the theme of this scenario"  — meta-talk, no real content
  ✗ "a coffee machine hisses while they..." — do NOT build the subject on the \
sounds

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT — strict JSON, no preamble, no postamble, no code fences
═══════════════════════════════════════════════════════════════════
{{
  "subject": "..."
}}"""


PIVOT_BLOCK_TEMPLATE = """
═══════════════════════════════════════════════════════════════════
NOTE — ONE SOUND WILL INTERRUPT AND BE ENGAGED WITH
═══════════════════════════════════════════════════════════════════
One of the sounds above ({pivot_id}) will later interrupt the conversation, \
and the speakers will react to it over several turns. That reaction is written \
later, in the dialogue step. For now, the SUBJECT you produce must be the \
natural conversation the speakers are having BEFORE/AROUND that interruption — \
do NOT describe or summarise the interrupting sound itself."""


LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"subject": {"type": "string"}},
    "required": ["subject"],
}


OPENAI_DEFAULT_BASE_URL = "http://localhost:8000/v1"

SUBJECT_TOOL = {
    "name": "emit_subject",
    "description": "Emit the conversation subject for this scenario.",
    "input_schema": LLM_OUTPUT_SCHEMA,
}


def validate_llm_output(output: Any) -> tuple[bool, str]:
    if not isinstance(output, dict):
        return False, f"non-dict output ({type(output).__name__})"
    subject = (output.get("subject") or "").strip()
    if not subject:
        return False, "empty subject"
    n_words = len(subject.split())
    if n_words < 4:
        return False, f"subject too short: '{subject}'"
    if n_words > 60:
        return False, f"subject too long ({n_words} words): '{subject[:60]}...'"
    return True, ""


def _postprocess(parsed: Any) -> tuple[dict | None, str]:
    ok, reason = validate_llm_output(parsed)
    return (parsed, "") if ok else (None, reason)


def _format_sounds_block(events: list[dict]) -> str:
    if not events:
        return "    (none)"
    return "\n".join(f"    - {e['event_id']}" for e in events)


def _pivot_event_id(events: list[dict]) -> str | None:
    return next((e["event_id"] for e in events if e["reaction"] == "pivot"), None)


def build_user_prompt(todo: dict) -> str:
    template = todo["template"]
    pivot_id = _pivot_event_id(todo["events"])
    user = USER_PROMPT.format(
        category=template["category"],
        theme=template["theme"],
        template_title=template["title"],
        template_description=template["description"],
        scene=todo["scene"],
        role_a=todo["role_a"],
        role_b=todo["role_b"],
        gender_a=GENDER_WORDS[todo["gender_a"]],
        gender_b=GENDER_WORDS[todo["gender_b"]],
        sounds_block=_format_sounds_block(todo["events"]),
        pivot_block=PIVOT_BLOCK_TEMPLATE.format(pivot_id=pivot_id) if pivot_id else "",
    )
    return user


def build_messages(todo: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(todo)},
    ]


def todo_record(todo: dict, value: dict | None, raw: str, status: str) -> dict:
    template = todo["template"]
    return {
        "scenario_id": todo["scenario_id"],
        "template_id": template["template_id"],
        "category": template["category"],
        "theme": template["theme"],
        "template_title": template["title"],
        "template_description": template["description"],
        "scenario_idx_in_template": todo["scenario_idx_in_template"],
        "scene": todo["scene"],
        "gender_a": todo["gender_a"],
        "gender_b": todo["gender_b"],
        "role_a": todo["role_a"],
        "role_b": todo["role_b"],
        "n_pivot": todo["n_pivot"],
        "n_verbal": todo["n_verbal"],
        "n_behavioral": todo["n_behavioral"],
        "n_ambient": todo["n_ambient"],
        "has_rare_event": todo["has_rare_event"],
        "split": todo.get("split"),
        "events": todo["events"],
        "llm_output": value,
        "raw_generation": raw,
        "generation_status": status,
    }


def _cache_is_valid(record: dict) -> bool:
    output = (record or {}).get("llm_output")
    return bool(output) and validate_llm_output(output)[0]


def _restrict_to_splits(todos: list[dict], spec: str | None) -> list[dict]:
    if not spec:
        return todos
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    return [t for t in todos if t.get("split") in wanted]


def _resolve_backend_defaults(args: argparse.Namespace) -> None:
    if args.backend == "claude":
        args.model = args.model or DEFAULT_CLAUDE_MODEL
        return
    args.base_url = args.base_url or OPENAI_DEFAULT_BASE_URL
    args.api_key = args.api_key or "EMPTY"


def _build_chat(args: argparse.Namespace) -> tuple[StructuredChat, Callable[[dict], Any]]:
    if args.backend == "claude":
        return ClaudeChat(
            model=args.model,
            system=SYSTEM_PROMPT,
            tool=SUBJECT_TOOL,
            api_key=args.api_key,
            base_url=args.base_url,
            max_tokens=GEN_MAX_NEW_TOKENS,
            temperature=args.temperature,
            top_p=args.top_p,
            struct_mode=args.struct_mode,
            max_retries=GEN_MAX_RETRIES,
            timeout=HTTP_TIMEOUT,
        ), build_user_prompt
    return OpenAIChat(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        schema=LLM_OUTPUT_SCHEMA,
        schema_name="scenario_subject",
        temperature=GEN_TEMPERATURE if args.temperature is None else args.temperature,
        top_p=GEN_TOP_P if args.top_p is None else args.top_p,
        max_tokens=GEN_MAX_NEW_TOKENS,
        struct_mode=args.struct_mode,
        max_retries=GEN_MAX_RETRIES,
        timeout=HTTP_TIMEOUT,
    ), build_messages


async def _generate(todos: list[dict], chat: StructuredChat, cache: Checkpoint,
                    concurrency: int, build_prompt: Callable[[dict], Any]) -> dict:
    log.info("Resume: %d scenarios already cached", len(cache))
    pending = cache.pending(todos, key=lambda t: t["scenario_id"], is_valid=_cache_is_valid)
    log.info("To generate: %d scenarios (concurrency=%d)", len(pending), concurrency)
    if not pending:
        return cache.data

    async def worker(todo: dict) -> None:
        sid = todo["scenario_id"]
        result = await chat.generate(build_prompt(todo), _postprocess, label=sid,
                                     max_retries=GEN_MAX_RETRIES)
        if not result.ok:
            log.warning("  failure '%s': %s", sid, result.status)
        await cache.record(sid, todo_record(todo, result.value, result.raw, result.status))

    await run_pool(pending, worker, concurrency=concurrency, desc="Subject generation")
    cache.flush()
    n_ok = sum(1 for r in cache.data.values() if r.get("llm_output"))
    log.info("Generation done: %d valid scenarios out of %d processed", n_ok, len(cache))
    return cache.data


def _run_batch(todos: list[dict], args: argparse.Namespace,
               cache: Checkpoint, paths: Paths) -> dict:
    from anthropic import Anthropic

    client_kwargs: dict[str, Any] = {"timeout": HTTP_TIMEOUT, "max_retries": 2}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = Anthropic(**client_kwargs)

    chat, _ = _build_chat(args)
    by_id = {t["scenario_id"]: t for t in todos}

    def parse_result(sid: str, parsed: Any, raw: str) -> tuple[dict | None, str]:
        if parsed is None and raw:
            parsed = parse_json_loose(raw)
        if parsed is None:
            return None, "batch:no_json"
        value, reason = _postprocess(parsed)
        return (value, "ok (batch)") if value else (None, f"batch:{reason}")

    def make_record(sid: str, value: Any, raw: str, status: str) -> dict:
        return todo_record(by_id[sid], value, raw, status)

    results = run_batch(
        client=client,
        items=todos,
        item_id=lambda t: t["scenario_id"],
        build_params=lambda t: chat.message_params(build_user_prompt(t), "tool"),
        results=cache.data,
        results_path=paths.raw_scenarios,
        state_path=paths.scenarios_batches_state,
        needs_work=lambda _t, rec: not _cache_is_valid(rec),
        tool_name=SUBJECT_TOOL["name"],
        parse_result=parse_result,
        make_record=make_record,
        wait=not args.batch_no_wait,
        interval=args.batch_poll_interval,
        desc="subjects",
    )
    if args.batch_no_wait:
        log.info("Batch(es) submitted. Re-run with --batch (without "
                 "--batch-no-wait) to wait, collect and finalize.")
    return results


def finalize(results: dict, paths: Paths) -> None:
    final: list[dict] = []
    comp_dist: Counter = Counter()
    count_dist: Counter = Counter()
    reaction_dist: Counter = Counter()
    scene_dist: Counter = Counter()
    relation_dist: Counter = Counter()
    event_usage: Counter = Counter()
    n_pivot_scenarios = 0
    n_rare = 0

    ordered = sorted(results.values(),
                     key=lambda r: (r.get("template_id", ""),
                                    r.get("scenario_idx_in_template", 0)))

    for record in ordered:
        llm_output = record.get("llm_output")
        if llm_output is None:
            continue

        comp = (record["n_pivot"], record["n_verbal"],
                record["n_behavioral"], record["n_ambient"])
        n_events = sum(comp)

        comp_dist[composition_key(comp)] += 1
        count_dist[n_events] += 1
        for event in record.get("events", []):
            reaction_dist[event["reaction"]] += 1
            event_usage[event["event_id"]] += 1
        scene_dist[record["scene"]] += 1
        relation_dist[(record["role_a"], record["role_b"])] += 1
        if comp[0] >= 1:
            n_pivot_scenarios += 1
        if record["has_rare_event"]:
            n_rare += 1

        entry = {
            "scenario_id": record["scenario_id"],
            "template_id": record["template_id"],
            "category": record["category"],
            "theme": {
                "title": record["template_title"],
                "description": record["template_description"],
            },
            "scene": record["scene"],
            "speakers": {"A": record["role_a"], "B": record["role_b"]},
            "gender_a": record.get("gender_a"),
            "gender_b": record.get("gender_b"),
            "subject": llm_output["subject"].strip(),
            "events": record.get("events", []),
            "has_rare_event": record["has_rare_event"],
            "metadata": {
                "n_events": n_events,
                "n_pivot": comp[0],
                "n_verbal": comp[1],
                "n_behavioral": comp[2],
                "n_ambient": comp[3],
                "n_reacted_to": comp[0] + comp[1] + comp[2],
                "scenario_idx_in_template": record["scenario_idx_in_template"],
            },
        }
        if record.get("split"):
            entry["split"] = record["split"]
        final.append(entry)

    n_ok = len(final)
    save_json(paths.scenarios, final)
    save_json(paths.distribution_log, {
        "n_scenarios": n_ok,
        "n_with_pivot": n_pivot_scenarios,
        "n_with_rare_event": n_rare,
        "rare_event_rate": n_rare / n_ok if n_ok else 0,
        "event_count_distribution": {str(k): count_dist[k] for k in sorted(count_dist)},
        "composition_distribution": dict(sorted(comp_dist.items())),
        "reaction_type_distribution": dict(reaction_dist.most_common()),
        "scene_distribution": dict(sorted(scene_dist.items())),
        "relation_distribution": {f"{a}|{b}": c for (a, b), c in sorted(relation_dist.items())},
        "event_usage": dict(event_usage.most_common()),
    })

    counts = ", ".join(f"{k}:{count_dist[k]}" for k in sorted(count_dist))
    log.info("Final dataset: %d scenarios -> %s", n_ok, paths.scenarios)
    log.info("  With pivot   : %d (%.1f%%)", n_pivot_scenarios,
             100 * n_pivot_scenarios / max(n_ok, 1))
    log.info("  Rare events  : %d (%.1f%%)", n_rare, 100 * n_rare / max(n_ok, 1))
    log.info("  Reactions    : %s", dict(reaction_dist.most_common()))
    log.info("  Event count  : {%s}", counts)
    log.info("  Scenes (top 5) : %s", scene_dist.most_common(5))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default=None,
                        help="Flat template JSON (default: templates_flat.json of the data-dir)")
    parser.add_argument("--banks", default=str(SCENARIO_BANKS),
                        help="Bank JSON (relations, scenes, events)")
    parser.add_argument("--backend", choices=["claude", "openai"], default="claude",
                        help="claude: Anthropic API (default). openai: "
                             "OpenAI-compatible server, a local vLLM for instance.")
    parser.add_argument("--base-url", default=None,
                        help="Default: nothing for claude, "
                             f"{OPENAI_DEFAULT_BASE_URL} for openai.")
    parser.add_argument("--model", default=None,
                        help=f"Required except with --finalize-only. Default for "
                             f"claude: {DEFAULT_CLAUDE_MODEL}.")
    parser.add_argument("--api-key", default=None,
                        help="Default: ANTHROPIC_API_KEY for claude, "
                             "'EMPTY' for openai.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Force the sampling. Not sent to Claude by default: "
                             "Opus 4.7+ models refuse it (400). "
                             f"Default on the openai side: {GEN_TEMPERATURE}.")
    parser.add_argument("--top-p", type=float, default=None, help="Same as --temperature.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None,
                        help=f"Debug: process N templates only (x {N_SCENARIOS_PER_TEMPLATE})")
    parser.add_argument("--event-dist", choices=["compositions", "counts"],
                        default=DEFAULT_EVENT_DIST,
                        help="Distribution of the event compositions")
    parser.add_argument("--event-pairing", choices=["random", "balanced"],
                        default=DEFAULT_EVENT_PAIRING,
                        help="Pairing of sound identifier x reaction type. 'balanced' "
                             "gives every sound the four types in the same proportions, "
                             "without changing its usage frequency; 'random' (default) "
                             "reproduces the historical draw.")
    parser.add_argument("--struct-mode",
                        choices=["auto", "response_format", "guided_json", "json_object", "none"],
                        default="auto")
    default_split = "/".join(str(DEFAULT_SPLIT_SIZES[s]) for s in ("train", "dev", "test"))
    parser.add_argument("--split", default=default_split,
                        help=f"TEMPLATE split 'train/dev/test' (default: {default_split}), "
                             f"in percentages ('80%%/10%%/10%%'), or 'none' for a "
                             f"monolithic dataset. The split is on templates and not on "
                             f"scenarios: two scenarios of one template share the subject.")
    parser.add_argument("--only-split", default=None,
                        help="Generate these splits only, e.g. 'test' or 'dev,test'. "
                             "The allocation stays WHOLE: the draw of the requested "
                             "split is the one it would have in a complete run. "
                             "The final dataset then holds these splits only; "
                             "re-running without the filter completes the rest.")
    parser.add_argument("--test-boost", default="",
                        help="Over-represent a class in the test split, e.g. 'behavioral=1.5' "
                             "or 'pivot=2,behavioral=1.5'. Weight applied per class slot.")
    parser.add_argument("--batch", action="store_true",
                        help="Generate through the Anthropic Batch API (~50%% cheaper, "
                             "results deferred up to 24 h). claude backend only; "
                             "forces --struct-mode tool.")
    parser.add_argument("--batch-no-wait", action="store_true",
                        help="With --batch: submit the batches and return, without "
                             "blocking on polling. Re-run --batch to collect.")
    parser.add_argument("--batch-poll-interval", type=int, default=POLL_INTERVAL,
                        help=f"Seconds between two polls (default: {POLL_INTERVAL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute the pre-allocation and its balance, then stop: "
                             "no model call, no cache touched. Use it to tune the banks "
                             "before paying for the generation.")
    parser.add_argument("--finalize-only", action="store_true",
                        help="No generation, rebuild the final files from the cache")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Drop the scenarios without a subject from the cache, then re-run")


def run(args: argparse.Namespace) -> int:
    paths = Paths.resolve(args.data_dir)
    cache = Checkpoint(paths.raw_scenarios, save_every=SAVE_EVERY)

    if args.finalize_only:
        if not cache.data:
            log.error("No %s found.", paths.raw_scenarios)
            return 1
        finalize(cache.data, paths)
        return 0

    banks = load_banks(args.banks, max_events=MAX_EVENTS_PER_SCENARIO)
    log.info("Banks loaded: %s", describe(banks))

    templates_path = args.input or paths.templates_flat
    templates = load_json(templates_path)
    if templates is None:
        log.error("Input file not found: %s", templates_path)
        return 1
    if not templates:
        log.error("No template in %s: nothing to generate.", templates_path)
        return 1
    if args.limit:
        templates = templates[: args.limit]
    log.info("Templates loaded: %d (-> %d scenarios)",
             len(templates), len(templates) * N_SCENARIOS_PER_TEMPLATE)

    try:
        sizes = parse_sizes(args.split, len(templates))
        boosts = parse_boosts(args.test_boost)
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    if boosts and not sizes:
        log.error("--test-boost only applies to the 'test' split: it has no effect "
                  "with --split none. Ask for a split, or drop --test-boost.")
        return 2

    assignment = None
    if sizes:
        assignment = assign_templates(templates, sizes)
        for line in describe_template_assignment(templates, assignment):
            log.info("%s", line)

    try:
        todos, _ = build_todos(templates, banks, event_dist=args.event_dist,
                               pairing=args.event_pairing, assignment=assignment,
                               test_boosts=boosts)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    todos_all = todos
    for line in describe_distribution(todos, args.event_dist):
        log.info("%s", line)
    log.info("  Pairing id x type: %s", args.event_pairing)

    balance = event_balance(todos)
    for line in describe_balance(balance):
        log.info("%s", line)

    report = split_report(todos)
    for line in describe_split_report(report):
        log.info("%s", line)
    problems = check_disjoint(todos)
    if problems:
        for problem in problems:
            log.error("LEAK: %s", problem)
        log.error("Split is not disjoint: generation interrupted.")
        return 2

    if assignment:
        save_json(paths.scenarios_dir / "splits.json", {
            "sizes": sizes,
            "test_boost": boosts,
            "report": report,
            "template_split": assignment,
        })

    if args.dry_run:
        preview = paths.scenarios_dir / "allocation_preview.json"
        save_json(preview, {
            "n_scenarios": len(todos),
            "event_dist": args.event_dist,
            "event_pairing": args.event_pairing,
            "split": args.split,
            "test_boost": boosts,
            "banks": str(args.banks),
            "balance": balance,
            "splits": report,
        })
        log.info("Pre-allocation written to %s (no model call).", preview)
        return 0

    todos = _restrict_to_splits(todos, args.only_split)
    if not todos:
        log.error("--only-split %s: no scenario. Available splits: %s",
                  args.only_split,
                  ", ".join(sorted({t.get("split") or "(none)" for t in todos_all})))
        return 2

    _resolve_backend_defaults(args)
    if not args.model:
        log.error("--model is required (except with --finalize-only)")
        return 2

    if args.retry_failed:
        removed = cache.drop(lambda r: bool(r.get("llm_output")))
        log.info("--retry-failed: %d failures dropped from the cache", removed)

    if args.batch:
        if args.backend != "claude":
            log.error("--batch only exists for --backend claude.")
            return 2
        if args.struct_mode not in ("auto", "tool"):
            log.warning("--batch ignores --struct-mode %s: forced to 'tool'.",
                        args.struct_mode)
        results = _run_batch(todos, args, cache, paths)
        if args.batch_no_wait:
            return 0
    else:
        chat, build_prompt = _build_chat(args)
        results = run_async(_generate(todos, chat, cache, args.concurrency, build_prompt))
    finalize(results, paths)
    return 0
