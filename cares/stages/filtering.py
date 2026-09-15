from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from typing import Any

from ..analysis.reaction_metrics import (
    compute_metrics,
    log_metrics,
    plot_confusion_matrix,
    safe_div,
)
from ..config import Paths
from ..dataset import (
    REACTIONS,
    has_pivot,
    has_rare,
    load_scenarios,
    scn_events,
    scn_speakers,
    scn_subject,
    stratum_key,
)
from ..embeddings import DEFAULT_EMBED_MODEL, cosine, load_encoder
from ..jsonio import load_json, save_json
from ..llm import GenerationResult, OpenAIChat
from ..log import get_logger
from ..runner import run_async, run_pool

log = get_logger(__name__)

DEFAULT_TAU = 0.60
DEFAULT_K_MAX = 5

GEN_TEMPERATURE = 0.1
GEN_TOP_P = 0.9
GEN_MAX_NEW_TOKENS = 1200
GEN_MAX_RETRIES = 5

SAVE_EVERY = 50
DEFAULT_CONCURRENCY = 32
HTTP_TIMEOUT = 600.0


SYSTEM_PROMPT = """You are an expert annotator of audio-scene transcripts. You \
are given the FULL transcript of a two-speaker scene, including every \
non-speech sound that occurs, each marked inline. For every sound, you decide \
HOW the speakers react to it, and you say what the conversation is about. You \
answer with strict JSON only."""

USER_PROMPT = """Below is the FULL transcript of a scene between two adults who \
are physically together at a location. Every spoken turn is shown, and every \
non-speech SOUND is marked inline as [SOUND: <id>]. All sounds that occur are \
already shown to you; you do not have to find them.

═══════════════════════════════════════════════════════════════════
SCENE CONTEXT
═══════════════════════════════════════════════════════════════════
Location  : {scene}
Speaker A : {role_a}
Speaker B : {role_b}

═══════════════════════════════════════════════════════════════════
FULL TRANSCRIPT (utterances and sounds, in order)
═══════════════════════════════════════════════════════════════════
{transcript}

═══════════════════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════════════════
Two things:

(1) SUBJECT — in one short phrase (10 to 25 words), say what the conversation \
is fundamentally about: the concrete situation, the people's specific concern, \
and where relevant the key specifics (a name, a place, a date, an amount). \
Write it the way a program description would: concrete and specific, not \
abstract. Do NOT write a generic topic label.

Good examples of the expected style and level of detail:
  ✓ "two classmates discussing how one of them has been coping since his \
father passed away last week, and whether returning to campus so soon was \
the right decision"
  ✓ "two old college friends talking about how one has been coping since his \
sister Sarah passed away last Tuesday, and the quiet ways their other \
friends have shown support"
  ✓ "two collaborators discussing how to support a grieving colleague after \
the death of his wife last month, and sharing personal memories of loss"
  ✓ "two neighbors on Oak Street discussing how to make their vacation homes \
appear occupied using timed lights and spare key drop-offs, after a recent \
break-in last month"

Bad examples (do NOT do this):
  ✗ "two people talking about grief"          — too generic, no specifics
  ✗ "a conversation about a recent loss"      — vague, could be anything
  ✗ "they discuss returning to campus"        — drops the concrete situation

(2) REACTION TYPE OF EACH SOUND — for EACH [SOUND: <id>] in the transcript, \
decide which ONE of four reaction types best describes how the speakers react \
to it, judging ONLY from what the transcript shows. Type EVERY sound shown, \
copying its id EXACTLY as written between [SOUND: and ].

The four types rest on two yes/no questions about the turns around the sound:
  • Is the sound REFERRED TO in words? (a speaker mentions it, asks what it \
was, comments on it — even obliquely, e.g. "what was that", "did you hear \
that", without naming exactly what it is)
  • Does the sound CHANGE THE FLOW of the talk? (someone is interrupted, \
pauses, loses the thread, or the conversation turns to the sound)

From those two axes:
  • PIVOT  — referred to AND the conversation turns to it: after the sound, \
the speakers engage with it and the talk is about it (and its aftermath) for \
several turns (about four or more). At most ONE sound in a scene is a pivot.
  • VERBAL — referred to but the flow does NOT change: a brief mention or \
question, then the conversation continues on its previous course.
  • BEHAVIORAL — NOT referred to in words, but the flow is DERAILED at the \
sound, and then the speaker returns to the same subject. The derail takes one \
of two forms, and BOTH count:
      (a) IN THE WORDS — a speaker breaks off an unfinished sentence, stumbles \
or repeats, or says they lost their thread ("what was I saying", "I blanked" — \
about their own thread, not the sound);
      (b) IN THE DELIVERY ONLY — the words run on unbroken, but a tag inside \
the line marks the break: a silence ([pauses]) or an involuntary sound the \
speaker makes ([gasps], [clears throat], [gulps], [swallows], [sniffs], \
[coughs]) sitting INSIDE a phrase rather than after a full stop. The \
transcript below keeps those tags, so you can see them. A tag placed mid-phrase \
at the sound IS a derail, even though the sentence itself is perfect — it is \
audible in the recording and it is why nothing shows in the wording.
A clean, coherent topic change with NEITHER a stumble NOR such a tag is NOT \
behavioral: treat that sound as AMBIENT.

MECHANICAL CHECK, DO IT BEFORE YOU DECIDE ANYTHING ELSE. For every sound, look \
at the FIRST utterance after [SOUND: ...]. If it contains [pauses], [gasps], \
[clears throat], [gulps], [swallows], [sniffs] or [coughs] anywhere other than \
at the very start of the line, that sound is BEHAVIORAL. Full stop. Do not \
weigh how fluent the sentence reads — fluency is exactly what this kind of \
reaction is supposed to preserve, and the words are MEANT to look untouched. \
The tag is not decoration: it is a hole or a caught breath in the recording, \
placed there because of the sound. Examples of what counts, all BEHAVIORAL:
    "it is just the [pauses] late ones, and only after midnight."
    "a nine-year-old can actually [pauses] picture that."
    "I would rather log it than start [gulps] something we cannot finish."
Read those again: nothing in the WORDS is broken, and all three are behavioral. \
Calling them ambient is the single most common mistake on this task.
  • AMBIENT — neither referred to nor reacted to: the conversation runs \
straight over it as if it had not happened.

CRITICAL — REACTION, NOT LOUDNESS. A sound is not a pivot or verbal just \
because it is loud, sudden, or dramatic. What matters is whether the speakers \
actually respond to it in the transcript. A loud sound the speakers talk \
straight past, with no reference and no change of flow, is AMBIENT.

Decision order for each sound: if the speakers clearly turn to it for several \
turns -> pivot. Else if it is mentioned in passing -> verbal. Else if the talk \
visibly adapts without any mention — in the WORDS or in a delivery tag at that \
moment — -> behavioral. Else -> ambient.

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT — strict JSON, no preamble, no code fences
═══════════════════════════════════════════════════════════════════
{{
  "subject": "<short phrase: what the conversation is about, 3-15 words>",
  "sounds": [
    {{"event_id": "<exact id copied from a [SOUND: ...] marker>",
      "reaction": "pivot" | "verbal" | "behavioral" | "ambient"}}
  ]
}}
List one entry in "sounds" for every [SOUND: ...] in the transcript. If the \
transcript contains no sound at all, return "sounds": []."""


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "sounds": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "reaction": {"type": "string", "enum": list(REACTIONS)},
                },
                "required": ["event_id", "reaction"],
            },
        },
    },
    "required": ["subject", "sounds"],
}


def build_transcript(timeline: list[dict]) -> str:
    lines = []
    for item in timeline:
        if item.get("type") == "utterance":
            lines.append(f"{item['speaker']}: {item['text']}")
        elif item.get("type") == "event":
            lines.append(f"[SOUND: {item['event_id']}]")
    return "\n".join(lines)


def transcript_event_ids(timeline: list[dict]) -> set[str]:
    return {it["event_id"] for it in timeline
            if it.get("type") == "event" and it.get("event_id")}


def build_messages(dialogue: dict, scenario: dict) -> list[dict]:
    speakers = scn_speakers(scenario)
    user = USER_PROMPT.format(
        scene=scenario.get("scene"),
        role_a=speakers["A"],
        role_b=speakers["B"],
        transcript=build_transcript(dialogue["timeline"]),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def normalize_inferred(parsed: Any, present_ids: set[str]) -> tuple[dict | None, str]:
    if not isinstance(parsed, dict):
        return None, "root is not a dict"
    subject = (parsed.get("subject") or "").strip()
    if not subject or len(subject.split()) < 2:
        return None, "subject empty or too short"
    sounds = parsed.get("sounds")
    if not isinstance(sounds, list):
        return None, "sounds is not a list"

    predicted: dict[str, str] = {}
    spurious: list[str] = []
    for i, sound in enumerate(sounds):
        if not isinstance(sound, dict):
            return None, f"sound {i} is not a dict"
        event_id = (sound.get("event_id") or "").strip()
        reaction = (sound.get("reaction") or "").strip().lower()
        if reaction not in REACTIONS:
            return None, f"sound {i}: invalid reaction {reaction!r}"
        if not event_id:
            return None, f"sound {i}: empty event_id"
        if event_id not in present_ids:
            spurious.append(event_id)
            continue
        predicted.setdefault(event_id, reaction)

    if present_ids and not predicted and not spurious:
        return None, "no sound typed although the transcript contains some"

    return {"subject": subject, "pred_by_id": predicted, "spurious": spurious}, ""


def delivery_tag_confirms(timeline: list[dict], event: dict) -> bool:
    from ..allocation import register_required_tags
    from .dialogues.grounded import check_required_tags

    if not register_required_tags(event.get("reaction") or "",
                                  event.get("register") or ""):
        return False
    ok, _ = check_required_tags(timeline, [event])
    return ok


def evaluate_reactions(scenario: dict, inferred: dict, encoder: Any, tau: float,
                       timeline: list[dict] | None = None) -> dict:
    gold_events = scn_events(scenario)
    predicted = inferred["pred_by_id"]

    subject_cos = cosine(encoder, scn_subject(scenario), inferred["subject"])
    topic_ok = subject_cos is not None and subject_cos >= tau

    pairs: list[tuple[str, str]] = []
    n_correct = 0
    n_missing = 0
    n_rescued = 0
    for event in gold_events:
        gold = event["reaction"]
        pred = predicted.get(event["event_id"])
        if (pred == "ambient" and gold == "behavioral" and timeline is not None
                and delivery_tag_confirms(timeline, event)):
            n_rescued += 1
            n_correct += 1
            pairs.append((gold, pred))
            continue
        if pred is None:
            pairs.append((gold, "missing"))
            n_missing += 1
        else:
            pairs.append((gold, pred))
            n_correct += int(pred == gold)

    n_sounds = len(gold_events)
    reactions_ok = n_missing == 0 and n_correct == n_sounds

    return {
        "recovered_subject": inferred["subject"],
        "subject_cos": subject_cos,
        "topic_ok": bool(topic_ok),
        "reactions_ok": bool(reactions_ok),
        "accepted": bool(topic_ok and reactions_ok),
        "n_sounds": n_sounds,
        "n_correct": n_correct,
        "n_missing": n_missing,
        "n_spurious": len(inferred.get("spurious", [])),
        "n_rescued_by_tag": n_rescued,
        "pairs": pairs,
        "tau": tau,
    }


def recompute_topic_ok(evaluation: dict, tau: float) -> bool:
    cos = evaluation.get("subject_cos")
    return bool(cos is not None and cos >= tau)


def recompute_accept(evaluation: dict, tau: float) -> bool:
    return bool(recompute_topic_ok(evaluation, tau) and evaluation.get("reactions_ok"))


class FilterPaths:
    def __init__(self, root):
        self.accepted = root / "accepted_dialogues.json"
        self.attempts = root / "attempts_log.json"
        self.regen = root / "scenarios_to_regen.json"
        self.stats = root / "filter_stats.json"
        self.raw_inferences = root / "filter_raw_inferences.json"
        self.metrics = root / "reaction_metrics.json"
        self.confusion_png = root / "confusion_matrix.png"
        self.confusion_pdf = root / "confusion_matrix.pdf"
        self.final_dataset = root / "dialogues_filtered.json"


async def run_one_pass(candidates: list[dict], scenarios_by_id: dict, chat: OpenAIChat,
                       encoder: Any, tau: float, concurrency: int, pass_index: int,
                       files: FilterPaths) -> tuple[dict, dict]:
    accepted = load_json(files.accepted, default={}) or {}
    attempts = load_json(files.attempts, default={}) or {}
    raw_inferences = load_json(files.raw_inferences, default={}) or {}

    log.info("[pass %d] already accepted: %d", pass_index, len(accepted))
    pending = [d for d in candidates if d["scenario_id"] not in accepted]
    log.info("[pass %d] candidates to test: %d", pass_index, len(pending))
    if not pending:
        return accepted, attempts

    save_lock = asyncio.Lock()
    counter = {"done": 0, "accepted": 0, "rejected": 0}

    def postprocess_for(present_ids: set[str]):
        def postprocess(parsed: Any) -> tuple[dict | None, str]:
            return normalize_inferred(parsed, present_ids)
        return postprocess

    async def worker(dialogue: dict) -> None:
        sid = dialogue["scenario_id"]
        scenario = scenarios_by_id.get(sid)
        if scenario is None:
            async with save_lock:
                counter["done"] += 1
            return

        present_ids = transcript_event_ids(dialogue["timeline"])
        failed = {"subject_cos": None, "topic_ok": False, "reactions_ok": False,
                  "accepted": False, "n_sounds": len(scn_events(scenario)),
                  "pairs": [], "tau": tau}
        try:
            result = await chat.generate(build_messages(dialogue, scenario),
                                         postprocess_for(present_ids),
                                         label=sid, max_retries=GEN_MAX_RETRIES)
            evaluation = (evaluate_reactions(scenario, result.value, encoder, tau,
                                             dialogue["timeline"])
                          if result.ok else failed)
        except Exception as exc:  # noqa: BLE001
            log.error("  unhandled failure on '%s': %s", sid, exc)
            result = GenerationResult(None, "", f"unhandled: {exc}")
            evaluation = failed

        async with save_lock:
            record = attempts.get(sid, {"n_attempts": 0, "history": []})
            record["n_attempts"] += 1
            record["history"].append({
                "pass": pass_index,
                "status": result.status,
                "subject_cos": evaluation.get("subject_cos"),
                "reactions_ok": evaluation.get("reactions_ok"),
                "n_rescued_by_tag": evaluation.get("n_rescued_by_tag") or 0,
                "accepted": evaluation.get("accepted", False),
            })
            attempts[sid] = record
            raw_inferences[sid] = {"inferred": result.value, "raw": result.raw,
                                   "status": result.status, "eval": evaluation}
            if evaluation.get("accepted"):
                accepted[sid] = dialogue
                counter["accepted"] += 1
            else:
                counter["rejected"] += 1
            counter["done"] += 1
            if counter["done"] % SAVE_EVERY == 0:
                save_json(files.accepted, accepted)
                save_json(files.attempts, attempts)
                save_json(files.raw_inferences, raw_inferences)

    await run_pool(pending, worker, concurrency=concurrency, desc=f"pass {pass_index}")

    save_json(files.accepted, accepted)
    save_json(files.attempts, attempts)
    save_json(files.raw_inferences, raw_inferences)
    log.info("[pass %d] accepted: %d, rejected: %d",
             pass_index, counter["accepted"], counter["rejected"])
    return accepted, attempts


def compute_regen_and_stats(scenarios: list[dict], accepted: dict, attempts: dict,
                            k_max: int, pass_index: int, files: FilterPaths) -> dict:
    accepted_ids = set(accepted)
    to_regen: list[str] = []
    abandoned: list[str] = []
    for scenario in scenarios:
        sid = scenario["scenario_id"]
        if sid in accepted_ids:
            continue
        n_attempts = attempts.get(sid, {}).get("n_attempts", 0)
        (abandoned if n_attempts >= k_max else to_regen).append(sid)
    save_json(files.regen, to_regen)

    by_total: Counter = Counter()
    by_accepted: Counter = Counter()
    by_first_try: Counter = Counter()
    by_attempts: Counter = Counter()
    rare_total = rare_accepted = nonrare_total = nonrare_accepted = 0
    pivot_total = pivot_accepted = 0

    for scenario in scenarios:
        sid = scenario["scenario_id"]
        stratum = stratum_key(scenario)
        rare, pivot = has_rare(scenario), has_pivot(scenario)
        by_total[stratum] += 1
        rare_total += int(rare)
        nonrare_total += int(not rare)
        pivot_total += int(pivot)
        if sid in accepted_ids:
            by_accepted[stratum] += 1
            rare_accepted += int(rare)
            nonrare_accepted += int(not rare)
            pivot_accepted += int(pivot)
            history = attempts.get(sid, {}).get("history", [])
            if history and history[0].get("accepted"):
                by_first_try[stratum] += 1
        by_attempts[stratum] += attempts.get(sid, {}).get("n_attempts", 0)

    n_rescued = 0
    for attempt in attempts.values():
        past = attempt.get("history") or []
        if past:
            n_rescued += past[-1].get("n_rescued_by_tag") or 0

    stats = {
        "pass_index": pass_index,
        "k_max": k_max,
        "n_rescued_by_tag": n_rescued,
        "n_scenarios_total": len(scenarios),
        "n_accepted": len(accepted_ids),
        "n_to_regen": len(to_regen),
        "n_abandoned": len(abandoned),
        "acceptance_rate_global": safe_div(len(accepted_ids), len(scenarios)),
        "rare_vs_nonrare": {
            "rare_total": rare_total,
            "rare_accepted": rare_accepted,
            "rare_acceptance_rate": safe_div(rare_accepted, rare_total),
            "nonrare_total": nonrare_total,
            "nonrare_accepted": nonrare_accepted,
            "nonrare_acceptance_rate": safe_div(nonrare_accepted, nonrare_total),
        },
        "pivot_scenarios": {
            "total": pivot_total,
            "accepted": pivot_accepted,
            "acceptance_rate": safe_div(pivot_accepted, pivot_total),
        },
        "per_stratum": {},
        "abandoned_ids": abandoned,
    }
    for stratum in sorted(by_total):
        total = by_total[stratum]
        stats["per_stratum"][stratum] = {
            "total": total,
            "accepted": by_accepted[stratum],
            "acceptance_rate": safe_div(by_accepted[stratum], total),
            "first_try_accepted": by_first_try[stratum],
            "first_try_rate": safe_div(by_first_try[stratum], total),
            "mean_attempts": safe_div(by_attempts[stratum], total),
        }
    save_json(files.stats, stats)
    return stats


def log_stats(stats: dict) -> None:
    log.info("=" * 64)
    log.info("FILTER STATS (pass %d)", stats["pass_index"])
    log.info("=" * 64)
    log.info("Total: %d  Accepted: %d (%.1f%%)  To regenerate: %d  Abandoned: %d",
             stats["n_scenarios_total"], stats["n_accepted"],
             (stats["acceptance_rate_global"] or 0) * 100,
             stats["n_to_regen"], stats["n_abandoned"])
    if stats.get("n_rescued_by_tag"):
        log.info("  including %d event(s) confirmed by their delivery tag, "
                 "against the judge (deterministic rule, not a vote)",
                 stats["n_rescued_by_tag"])
    pivot = stats["pivot_scenarios"]
    log.info("  pivot : %d/%d (%.1f%%)", pivot["accepted"], pivot["total"],
             (pivot["acceptance_rate"] or 0) * 100)
    log.info("  %-8s %5s %5s %6s %6s %6s", "stratum", "tot", "acc", "acc%", "1st%", "#try")
    for stratum, s in stats["per_stratum"].items():
        log.info("  %-8s %5d %5d %5.1f%% %5.1f%% %6.2f", stratum, s["total"], s["accepted"],
                 (s["acceptance_rate"] or 0) * 100, (s["first_try_rate"] or 0) * 100,
                 s["mean_attempts"] or 0)


CARRIED_FIELDS = ("template_id", "category", "theme", "theme_label", "metadata", "split")


def finalize_dataset(scenarios_by_id: dict, accepted: dict, output_path) -> list[dict]:
    final = []
    for sid, dialogue in accepted.items():
        scenario = scenarios_by_id.get(sid, {})
        entry = {
            "scenario_id": sid,
            "scene": scenario.get("scene"),
            "speakers": scn_speakers(scenario),
            "subject": scn_subject(scenario),
            "events": scn_events(scenario),
            "has_rare_event": has_rare(scenario),
            "timeline": dialogue.get("timeline"),
        }
        for field in CARRIED_FIELDS:
            value = dialogue.get(field) or scenario.get(field)
            if value:
                entry[field] = value
        theme = entry.get("theme")
        if "theme_label" not in entry and isinstance(theme, dict) and theme.get("title"):
            entry["theme_label"] = theme["title"]
        final.append(entry)
    save_json(output_path, final)
    log.info("Final dataset: %d dialogues -> %s", len(final), output_path)
    return final


def _report(scenarios: list[dict], raw_inferences: dict, tau: float,
            files: FilterPaths, no_plot: bool) -> None:
    metrics = compute_metrics(scenarios, raw_inferences, tau)
    save_json(files.metrics, metrics)
    log_metrics(metrics)
    if not no_plot:
        plot_confusion_matrix(metrics["confusion_matrix"],
                              files.confusion_png, files.confusion_pdf)


def resolve_api_key(explicit: str | None) -> str:
    import os

    return explicit or os.environ.get("OPENAI_API_KEY") or "EMPTY"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidates", default=None,
                        help="Candidate dialogues (default: raw_dialogues.json of the data-dir)")
    parser.add_argument("--scenarios", default=None,
                        help="Gold scenarios (default: scenarios.json of the data-dir)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default=None, help="required for an evaluation pass")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: $OPENAI_API_KEY when set, else "
                             "'EMPTY', which is what vLLM expects)")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["minimal", "low", "medium", "high"],
                        help="Reasoning effort, for an OpenAI model that takes one "
                             "(e.g. --model gpt-5.6-sol --reasoning-effort medium). "
                             "Setting it also marks the model as a reasoning one: "
                             "temperature and top_p are then NOT sent, and the output "
                             "budget goes to max_completion_tokens.")
    parser.add_argument("--service-tier", default=None,
                        help="OpenAI service tier (e.g. 'standard', 'flex', "
                             "'priority'). Passed to the API as is.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                        help="Cosine threshold for subject recovery")
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX,
                        help="Maximum number of attempts before a scenario is abandoned")
    parser.add_argument("--pass-index", type=int, default=0)
    parser.add_argument("--struct-mode",
                        choices=["auto", "response_format", "guided_json", "json_object"],
                        default="auto")
    parser.add_argument("--finalize", "--finalize-only", action="store_true",
                        help="Write the final dataset from the accepted dialogues")
    parser.add_argument("--final-output", default=None,
                        help="Filtered dataset (default: dialogues_filtered.json of filter_output)")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Recompute metrics and confusion matrix, without any LLM")
    parser.add_argument("--recompute-acceptance", action="store_true",
                        help="Re-evaluate acceptance at a new tau, without LLM or embeddings")
    parser.add_argument("--no-plot", action="store_true",
                        help="Do not plot the confusion matrix")


def run(args: argparse.Namespace) -> int:
    paths = Paths.resolve(args.data_dir)
    files = FilterPaths(paths.filter_dir)
    paths.filter_dir.mkdir(parents=True, exist_ok=True)

    scenarios_path = args.scenarios or paths.scenarios
    scenarios = load_scenarios(scenarios_path)
    if scenarios is None:
        log.error("Scenarios not found: %s", scenarios_path)
        return 1
    scenarios_by_id = {s["scenario_id"]: s for s in scenarios}

    candidates_path = args.candidates or paths.raw_dialogues("grounded")

    if args.finalize:
        accepted = load_json(files.accepted, default={}) or {}
        finalize_dataset(scenarios_by_id, accepted,
                         args.final_output or files.final_dataset)
        return 0

    if args.metrics_only:
        raw_inferences = load_json(files.raw_inferences, default={}) or {}
        _report(scenarios, raw_inferences, args.tau, files, args.no_plot)
        return 0

    if args.recompute_acceptance:
        raw_inferences = load_json(files.raw_inferences, default={}) or {}
        attempts = load_json(files.attempts, default={}) or {}
        candidates = load_scenarios(candidates_path) or []
        by_id = {c["scenario_id"]: c for c in candidates if c.get("timeline")}

        accepted: dict[str, dict] = {}
        n_reevaluated = 0
        for sid, record in raw_inferences.items():
            evaluation = record.get("eval")
            if not evaluation:
                continue
            n_reevaluated += 1
            is_accepted = recompute_accept(evaluation, args.tau)
            evaluation["topic_ok"] = recompute_topic_ok(evaluation, args.tau)
            evaluation["accepted"] = is_accepted
            evaluation["tau"] = args.tau
            if is_accepted and sid in by_id:
                accepted[sid] = by_id[sid]
        save_json(files.raw_inferences, raw_inferences)
        save_json(files.accepted, accepted)
        log.info("Re-evaluation (tau=%s) over %d inferences: %d accepted.",
                 args.tau, n_reevaluated, len(accepted))

        log_stats(compute_regen_and_stats(scenarios, accepted, attempts,
                                          args.k_max, args.pass_index, files))
        _report(scenarios, raw_inferences, args.tau, files, args.no_plot)
        return 0

    candidates = load_scenarios(candidates_path)
    if candidates is None:
        log.error("Candidates not found: %s", candidates_path)
        return 1
    candidates = [c for c in candidates if c.get("timeline")]
    log.info("Candidates: %d", len(candidates))
    log.info("Criterion: cos(subject)>=tau AND exact 4-class typing (tau=%s)", args.tau)

    if not args.model:
        log.error("--model is required for an evaluation pass")
        return 2

    log.info("Loading the %s embeddings ...", args.embed_model)
    encoder = load_encoder(args.embed_model)

    chat = OpenAIChat(
        base_url=args.base_url,
        model=args.model,
        api_key=resolve_api_key(args.api_key),
        reasoning_effort=args.reasoning_effort,
        service_tier=args.service_tier,
        schema=RESPONSE_SCHEMA,
        schema_name="reaction_typing",
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        max_tokens=GEN_MAX_NEW_TOKENS,
        struct_mode=args.struct_mode,
        max_retries=GEN_MAX_RETRIES,
        timeout=HTTP_TIMEOUT,
    )

    accepted, attempts = run_async(run_one_pass(
        candidates, scenarios_by_id, chat, encoder, args.tau,
        args.concurrency, args.pass_index, files))

    log_stats(compute_regen_and_stats(scenarios, accepted, attempts,
                                      args.k_max, args.pass_index, files))
    raw_inferences = load_json(files.raw_inferences, default={}) or {}
    _report(scenarios, raw_inferences, args.tau, files, args.no_plot)

    to_regen = load_json(files.regen, default=[]) or []
    if not to_regen:
        log.info("All accepted (or abandoned). Run --finalize.")
    else:
        log.info("%d to regenerate. Regenerate the dialogues of %s, then run again "
                 "with --pass-index %d.", len(to_regen), files.regen, args.pass_index + 1)
    return 0
