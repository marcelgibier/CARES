from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from ..alm import AudioChat, FakeAudioChat
from ..alm.base import strip_thinking
from ..config import SCENARIO_BANKS, Paths
from ..dataset import REACTIONS
from ..jsonio import load_json, save_json
from ..log import get_logger
from ..runner import Checkpoint

log = get_logger(__name__)

SAVE_EVERY = 10

GUARANTEED_MARGIN_DB = 6.0

MARGIN_BUCKETS = (0.0, 6.0, 10.0, 15.0, 1e9)

STOPWORDS = frozenset("""
a an the of in on at to and or with sound sounds noise noises someone somebody
something people person man woman audio faint distant loud short brief soft
""".split())

MATCH_MIN_OVERLAP = 0.34

REACTION_DEFINITIONS = """\
- pivot: the sound takes over the conversation. The speakers turn to it and it \
becomes the subject for several turns.
- verbal: one speaker acknowledges it in passing — a brief remark, a question, \
a reaction — and the conversation goes back to what it was about.
- behavioral: nobody mentions the sound, but it visibly disrupts the speaking. \
A speaker breaks off, loses the thread, or falters, then returns to the same \
subject.
- ambient: nobody reacts. The conversation continues exactly as if the sound \
were not there."""


REACTIONS_PROMPT = """\
This recording is a conversation between two people who are physically together \
in a real place.
{localisation}
Decide how the two speakers respond to that sound. Exactly one of:

{definitions}

Judge only from what you hear. The speakers never name the sound by its \
category, so a reaction may refer to it obliquely, or not at all.

Answer with strict JSON only, no other text:
{{"reaction": "pivot" | "verbal" | "behavioral" | "ambient"}}"""

LOCALISATION_TIME = """
A discrete sound occurs at about {t:.1f} seconds into the recording.
"""
LOCALISATION_TIME_ID = """
A discrete sound occurs at about {t:.1f} seconds into the recording. It is a \
{name}.
"""
LOCALISATION_NONE = """
One discrete sound occurs somewhere in the recording.
"""

MCQ_OPTIONS = 4

MCQ_SOUND_PROMPT = """\
This recording is a conversation between two people who are physically together \
in a real place.

A discrete sound occurs at about {t:.1f} seconds into the recording — a short, \
individual acoustic event, as opposed to the two voices and the continuous \
background ambience.

Which of these is it?
{options}

Judge only from what you hear at that moment. Exactly one answer is correct.

Answer with strict JSON only, no other text:
{{"answer": "<one of the options above, copied exactly>"}}"""

MCQ_SCENE_PROMPT = """\
This recording is a conversation between two people who are physically together \
in a real place.

Where are they? Judge from the acoustics and the background sounds of the \
location, not from what the speakers talk about.

{options}

Exactly one answer is correct.

Answer with strict JSON only, no other text:
{{"answer": "<one of the options above, copied exactly>"}}"""


SUMMARY_BUDGETS: tuple[int, ...] = (25, 75, 200)

SUMMARY_PROMPT = """\
This recording is a conversation between two people who are physically together \
in a real place.

Summarise what happens in this recording in about {budget} words.

Write plain prose, no lists and no headings. Mention whatever you judge worth \
mentioning — the people, what they discuss, the place, and anything you hear \
happening around them. Do not speculate about what you cannot hear."""

JUDGE_SYSTEM = """You decide whether a written summary of an audio recording \
mentions particular sounds. You answer only with the tool."""

JUDGE_PROMPT = """\
Here is a summary somebody wrote of an audio recording:

\"\"\"
{summary}
\"\"\"

For EACH sound below, decide what the summary does with it. Judge each one on \
its own. The default is ABSENT: most sounds in the list will not be in the \
summary at all, and that is the expected answer.

Sounds:
{sounds}

For each, choose exactly one verdict:

  HEARD     — the summary states that THIS sound source was audible, and \
names it or describes it unmistakably. "faint clicks - a camera shutter" is \
heard; "a fountain burbles" is heard.
  NATURE    — the summary does not get the label right, but it describes a \
sound event whose NATURE matches this one: the material, the action, or the \
mechanism. "papers rustle" for page turning. "a mechanical grinding" for a \
paper shredder. "a sharp crack" for a balloon pop. The name may be wrong or \
absent; what must be right is what the thing IS.
  DISCUSSED — the summary only reports that the PEOPLE mentioned it, commented \
on it or reacted to it. "they comment on the sudden noise", "discussing camera \
equipment", "she says the dog kept her awake". If the summary attributes it to \
what the speakers say rather than to what the recording contains, it is \
DISCUSSED, not heard.
  ABSENT    — anything else.

THE TEST THAT DECIDES NATURE, and it decides it alone. Before you mark \
NATURE, ask: would this same phrase be just as true if that sound had been \
REMOVED from the recording? If yes, the verdict is ABSENT. "background \
noise", "faint sounds", "the usual room tone", "distant traffic", "birds \
chirping", "the engine hums" are true of almost any recording, so they are \
ABSENT no matter what is on the list. A phrase earns NATURE only when \
deleting the sound would make that phrase false.

ONE PHRASE CANNOT CARRY TWO SOUNDS. If the same words are your evidence for \
two different sounds on the list, they are too vague for either one: mark \
both ABSENT, unless the phrase describes each of them separately. "papers \
rustle and faint background noises fill the space" cannot be both page \
turning and a paper shredder.

THE TEST THAT DECIDES FAMILY, and it can only take a verdict away, never give \
one. Before you write HEARD or NATURE, name to yourself the source that the \
phrase fits BEST — in three words, the way the writer would have said it. \
Then compare that name with the sound you are judging, and keep the verdict \
only if they are the SAME source. If the best fit is something else, this \
sound is ABSENT, whether or not that something else is itself on the list. \
"someone raps on the door" fits knocking at a door, so a hammer is ABSENT. \
"a tap runs into a sink" fits a running tap, so rain on a window is ABSENT. \
"footsteps crunch over gravel" fits footsteps, so a spade turning soil is \
ABSENT. If two sounds on the list fit equally well, neither of them is the \
best fit and both are ABSENT, as above. The same family is not the same \
source: what separates them is the mechanism. A shutter rolling down is not a \
camera shutter. A phone ringing is not a phone vibrating.

A phrase that gives only a material, an action or a mechanism, and points at \
no source in particular, has no better fit to lose to: judge it by the \
removal test above and nothing else.

THAT IS THE ONLY COMPARISON YOU MAKE, AND IT ONLY SUBTRACTS. You are still \
answering one question per sound — is THIS sound the best fit for something \
the summary says? — and never "which of these sounds look plausible?". The \
rest of the list can rule a sound out; it can never rule one in.

ALSO ALWAYS ABSENT: the topic of the conversation, when the summary never \
says it was heard.

For every sound you mark HEARD, NATURE or DISCUSSED you must quote EVIDENCE: \
the exact words from the summary, copied character for character, that carry \
your decision. If you cannot copy a phrase that names or describes that \
specific sound, the verdict is ABSENT. Leave the evidence empty for ABSENT."""

JUDGE_TOOL = {
    "name": "report_mentions",
    "description": "For each candidate sound, what the summary does with it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sound": {"type": "string"},
                        "verdict": {"type": "string",
                                    "enum": ["heard", "nature", "discussed",
                                             "absent"]},
                        "evidence": {"type": "string"},
                    },
                    "required": ["sound", "verdict", "evidence"],
                },
            },
        },
        "required": ["verdicts"],
    },
}


GROUNDING_PROMPT = """\
This recording is a conversation between two people who are physically together \
in a real place.

Listen to what happens at about {t:.1f} seconds.

Did a DISCRETE sound occur there — a short, individual acoustic event, as \
opposed to the two voices and the continuous background ambience of the place?

Judge ONLY from what you hear at that moment. The speakers may or may not \
react to it, and a reaction in the conversation is NOT evidence that a sound \
occurred: decide from the audio alone.

Answer with strict JSON only, no other text:
{{"sound_present": true}} or {{"sound_present": false}}"""


def _pretty(event_id: str) -> str:
    name = event_id.split("/", 1)[1] if "/" in event_id else event_id
    return name.replace("_", " ")


def event_margin_db(manifest: dict, event: dict) -> float | None:
    bed = (manifest.get("levels_lufs") or {}).get("background_applied")
    level = event.get("level_lufs")
    if bed is None or level is None:
        return None
    return float(level) - float(bed)


def load_manifests(scenes_dir: Path) -> list[dict]:
    out = []
    for path in sorted(scenes_dir.glob("*.manifest.json")):
        data = load_json(path)
        if isinstance(data, dict) and data.get("events_placed") is not None:
            data["_manifest_path"] = str(path)
            out.append(data)
    return out


def audio_path_for(manifest: dict, scenes_dir: Path) -> Path:
    declared = manifest.get("audio_path")
    if declared and Path(declared).exists():
        return Path(declared)
    return scenes_dir / f"{manifest['scenario_id']}.wav"


def build_grounding_items(manifests: list[dict], scenes_dir: Path,
                          counterfactual_dirs: list[Path]) -> list[dict]:
    by_sid = {m["scenario_id"]: m for m in manifests}
    items = []
    for rank, cf_dir in enumerate(counterfactual_dirs):
        for cf_path in sorted(cf_dir.glob("*.manifest.json")):
            cf = load_json(cf_path)
            if not isinstance(cf, dict):
                continue
            sid = cf.get("scenario_id")
            events = cf.get("events_placed") or []
            muted = [(i, e) for i, e in enumerate(events) if e.get("muted")]
            if sid not in by_sid or not muted:
                continue
            i, e = muted[0]
            ref = by_sid[sid]["events_placed"]
            if i >= len(ref) or e.get("start_s") is None:
                continue
            pair = f"{sid}::pair::{rank}"
            for present, audio in ((True, audio_path_for(by_sid[sid], scenes_dir)),
                                   (False, audio_path_for(cf, cf_dir))):
                items.append({
                    "item_id": f"{pair}::{'present' if present else 'muted'}",
                    "pair_id": pair,
                    "scenario_id": sid,
                    "scene": cf.get("scene"),
                    "audio": str(audio),
                    "task": "grounding",
                    "event_id": e["event_id"],
                    "start_s": ref[i].get("start_s"),
                    "margin_db": event_margin_db(by_sid[sid], ref[i]),
                    "distance_m": ref[i].get("distance_m"),
                    "reaction": ref[i].get("reaction"),
                    "gold": present,
                })
    return items


def _mcq_rng(item_id: str):
    import hashlib
    import random

    seed = int(hashlib.sha1(item_id.encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def _mcq_options(right: str, candidates: list[str], item_id: str) -> list[str]:
    rng = _mcq_rng(item_id)
    others = [c for c in sorted(set(candidates)) if c != right]
    rng.shuffle(others)
    options = [right] + others[: MCQ_OPTIONS - 1]
    rng.shuffle(options)
    return options


def build_mcq_sound_items(manifests: list[dict], scenes_dir: Path,
                          banks_path: str) -> list[dict]:
    from ..banks import load_banks, scene_pool

    banks = load_banks(banks_path or str(SCENARIO_BANKS))
    items = []
    for m in manifests:
        scene = m.get("scene")
        try:
            pool = [_pretty(e) for e in scene_pool(banks, scene)]
        except Exception:  # noqa: BLE001
            log.warning("[%s] scene '%s' is not in the bank: item skipped",
                        m.get("scenario_id"), scene)
            continue
        present = {_pretty(e["event_id"]) for e in m["events_placed"]}
        candidates = [c for c in pool if c not in present]
        for rank, e in enumerate(m["events_placed"]):
            right = _pretty(e["event_id"])
            if len(candidates) < MCQ_OPTIONS - 1:
                continue
            item_id = f"{m['scenario_id']}::mcq-sound::{rank}"
            items.append({
                "item_id": item_id,
                "scenario_id": m["scenario_id"],
                "scene": scene,
                "audio": str(audio_path_for(m, scenes_dir)),
                "task": "sounds-mcq",
                "t": e["start_s"],
                "gold": right,
                "options": _mcq_options(right, candidates, item_id),
                "reaction": e.get("reaction"),
                "margin_db": event_margin_db(m, e),
            })
    return items


def build_scene_items(manifests: list[dict], scenes_dir: Path,
                      banks_path: str) -> list[dict]:
    from ..banks import load_banks

    places = sorted(load_banks(banks_path or str(SCENARIO_BANKS))["scenes"])
    items = []
    for m in manifests:
        scene = m.get("scene")
        if scene not in places:
            continue
        item_id = f"{m['scenario_id']}::scene"
        items.append({
            "item_id": item_id,
            "scenario_id": m["scenario_id"],
            "scene": scene,
            "audio": str(audio_path_for(m, scenes_dir)),
            "task": "scene",
            "gold": scene.replace("_", " "),
            "options": _mcq_options(scene.replace("_", " "),
                                    [x.replace("_", " ") for x in places], item_id),
        })
    return items


def build_summary_items(manifests: list[dict], scenes_dir: Path,
                        budgets: tuple[int, ...] = SUMMARY_BUDGETS) -> list[dict]:
    items = []
    for m in manifests:
        gold = [{"event_id": e["event_id"], "name": _pretty(e["event_id"]),
                 "reaction": e.get("reaction"), "is_rare": _is_rare(e),
                 "margin_db": event_margin_db(m, e)}
                for e in m["events_placed"]]
        if not gold:
            continue
        for budget in budgets:
            items.append({
                "item_id": f"{m['scenario_id']}::summary::{budget}",
                "scenario_id": m["scenario_id"],
                "scene": m.get("scene"),
                "audio": str(audio_path_for(m, scenes_dir)),
                "task": "summary",
                "budget": budget,
                "gold": gold,
            })
    return items


def _is_rare(event: dict) -> bool:
    return ("rare_events/" in str(event.get("source_file") or "")
            or event.get("layer") == "rare"
            or bool(event.get("is_rare")))


def build_reaction_items(manifests: list[dict], scenes_dir: Path) -> list[dict]:
    items = []
    for m in manifests:
        audio = str(audio_path_for(m, scenes_dir))
        for i, e in enumerate(m["events_placed"]):
            if e.get("reaction") not in REACTIONS:
                continue
            items.append({
                "item_id": f"{m['scenario_id']}::reaction::{i}",
                "scenario_id": m["scenario_id"],
                "scene": m.get("scene"),
                "audio": audio,
                "task": "reactions",
                "event_id": e["event_id"],
                "start_s": e.get("start_s"),
                "margin_db": event_margin_db(m, e),
                "distance_m": e.get("distance_m"),
                "duration_s": e.get("duration_s"),
                "gold": e["reaction"],
            })
    return items


def _options_block(options: list[str]) -> str:
    return "\n".join(f"  - {o}" for o in options)


def reactions_prompt(item: dict, reveal: str) -> str:
    if reveal == "none" or item.get("start_s") is None:
        loc = LOCALISATION_NONE
    elif reveal == "time_id":
        loc = LOCALISATION_TIME_ID.format(t=item["start_s"], name=_pretty(item["event_id"]))
    else:
        loc = LOCALISATION_TIME.format(t=item["start_s"])
    return REACTIONS_PROMPT.format(localisation=loc, definitions=REACTION_DEFINITIONS)


_WORD_RE = re.compile(r"[a-z]+")


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2]:
        word = word[:-1]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def _bucket(margin_db: float | None) -> str:
    if margin_db is None:
        return "unknown"
    margin_db = round(margin_db, 2)
    for low, high in zip(MARGIN_BUCKETS, MARGIN_BUCKETS[1:], strict=False):
        if low <= margin_db < high:
            hi = "inf" if high > 1e8 else f"{high:g}"
            return f"{low:g}-{hi} dB"
    return "unknown"


def mcnemar_p(correct: int, reversed_: int) -> float:
    from math import comb

    n = correct + reversed_
    if n == 0:
        return 1.0
    k = max(correct, reversed_)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2**n)


def _mcq_answer(value: Any, options: list[str]) -> str | None:
    if isinstance(value, dict):
        for key in ("answer", "choice", "option", "sound", "scene", "location"):
            if key in value:
                value = value[key]
                break
    text = str(value or "").strip().lower()
    if not text:
        return None
    exact = [o for o in options if o.lower() == text]
    if exact:
        return exact[0]
    quoted = [o for o in options if o.lower() in text]
    return quoted[0] if len(quoted) == 1 else None


def score_mcq(records: list[dict]) -> dict:
    from collections import Counter

    n = correct = unreadable = 0
    by_position: Counter = Counter()
    by_answer: Counter = Counter()
    by_reaction: dict[str, list[int]] = {}
    for r in records:
        options = r.get("options") or []
        if not options:
            continue
        n += 1
        answer = _mcq_answer(r.get("prediction"), options)
        if answer is None:
            unreadable += 1
            continue
        by_position[options.index(answer)] += 1
        by_answer[answer] += 1
        ok = int(answer == r.get("gold"))
        correct += ok
        if r.get("reaction"):
            by_reaction.setdefault(r["reaction"], []).append(ok)
    readable = n - unreadable
    return {
        "n_items": n,
        "n_unreadable": unreadable,
        "accuracy": (correct / n) if n else None,
        "accuracy_on_readable": (correct / readable) if readable else None,
        "chance": 1.0 / MCQ_OPTIONS,
        "answer_position": {str(k): v for k, v in sorted(by_position.items())},
        "most_frequent_answers": by_answer.most_common(5),
        "by_reaction": {k: sum(v) / len(v) for k, v in sorted(by_reaction.items())},
    }


def describe_mcq(scores: dict, title: str) -> list[str]:
    lines = [f"{title}: {scores['n_items']} items, chance {scores['chance']:.2f}"]
    acc, readable = scores.get("accuracy"), scores.get("accuracy_on_readable")
    if acc is not None:
        detail = f"  (on readable {readable:.3f})" if readable is not None else ""
        lines.append(f"  accuracy {acc:.3f}{detail}")
    if scores["n_unreadable"]:
        lines.append(f"  unreadable or ambiguous answers: {scores['n_unreadable']}")
    lines.append(f"  chosen positions: {scores['answer_position']}"
                 "   <- a position-biased model shows up here")
    if scores["most_frequent_answers"]:
        top = ", ".join(f"{k} x{v}" for k, v in scores["most_frequent_answers"][:3])
        lines.append(f"  most frequent answers: {top}")
    if scores.get("by_reaction"):
        per = "  ".join(f"{k} {v:.2f}" for k, v in scores["by_reaction"].items())
        lines.append(f"  by reaction type: {per}")
    return lines


def describe_sounds_mcq(scores: dict) -> list[str]:
    return describe_mcq(scores, "SOUNDS-MCQ (which sound, 4 options)")


def describe_scene(scores: dict) -> list[str]:
    return describe_mcq(scores, "SCENE (which place, 4 options)")


_PUNCTUATION_EQUIVALENTS = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2032": "'", "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u00a0": " ", "\u202f": " ", "\u2026": "...",
})


def _normalise_punctuation(text: str) -> str:
    return text.translate(_PUNCTUATION_EQUIVALENTS)


def _evidence_in(evidence: Any, summary: str) -> bool:
    text = _normalise_punctuation(str(evidence or "").strip().lower())
    if len(text.split()) < 2:
        return False
    flat = " ".join(_normalise_punctuation(str(summary or "").lower()).split())
    return " ".join(text.split()) in flat


def read_judge_verdicts(verdicts: Any, names: list[str], summary: str,
                        item_id: str, raw: str, status: str) -> dict:
    heard, nature, discussed, no_evidence, off_list = [], [], [], [], []
    for v in verdicts or []:
        if not isinstance(v, dict):
            continue
        name, verdict = str(v.get("sound") or ""), str(v.get("verdict") or "")
        if name not in names:
            off_list.append(name)
            continue
        if verdict not in ("heard", "nature", "discussed"):
            continue
        if not _evidence_in(v.get("evidence"), summary):
            no_evidence.append(name)
            continue
        {"heard": heard, "nature": nature,
         "discussed": discussed}[verdict].append(name)
    for dropped, reason in ((off_list, "off-list"),
                            (no_evidence, "unquoted")):
        if dropped:
            log.warning("  '%s': %d %s verdict(s) discarded: %s",
                        item_id, len(dropped), reason, dropped[:3])
    return {
        "item_id": item_id, "heard": heard,
        "nature": [n for n in nature if n not in heard],
        "discussed": [n for n in discussed
                      if n not in heard and n not in nature],
        "off_list": off_list, "no_evidence": no_evidence,
        "n_candidates": len(names), "raw": raw, "status": status,
    }


def rejudge_from_raw(records: list[dict], cache: Checkpoint) -> int:
    from ..jsonparse import find_list, parse_json_loose

    by_id = {r["item_id"]: r for r in records}
    changed = 0
    for item_id, old in list(cache.data.items()):
        rec = by_id.get(item_id)
        if rec is None or not old.get("raw"):
            continue
        parsed = parse_json_loose(old["raw"])
        verdicts = find_list(parsed, ("verdicts",))
        if not isinstance(verdicts, list):
            log.warning("  '%s': unreadable raw text, verdict left unchanged", item_id)
            continue
        fresh = read_judge_verdicts(verdicts, [g["name"] for g in rec["gold"]],
                                    rec.get("summary") or "", item_id,
                                    old["raw"], old.get("status", ""))
        if any(fresh[k] != old.get(k) for k in
               ("heard", "nature", "discussed", "off_list", "no_evidence")):
            changed += 1
        cache.data[item_id] = fresh
    cache.flush()
    log.info("Judge re-read: %d verdict(s) changed out of %d, without a call",
             changed, len(cache.data))
    return changed


def judge_summaries(records: list[dict], args: argparse.Namespace,
                    cache: Checkpoint) -> int:
    from ..llm import ClaudeChat
    from ..runner import run_async, run_pool

    todo = [r for r in records
            if r.get("summary") and cache.get(r["item_id"]) is None]
    log.info("Judge: %d summary(ies) to adjudicate (%d already cached)",
             len(todo), len(records) - len(todo))
    if not todo:
        return 0

    chat = ClaudeChat(model=args.judge_model, system=JUDGE_SYSTEM, tool=JUDGE_TOOL,
                      api_key=args.judge_api_key, base_url=None, max_tokens=1000,
                      temperature=None, top_p=None, struct_mode="tool",
                      max_retries=3, timeout=120.0, prompt_cache=False)

    def postprocess(parsed):
        if not isinstance(parsed, dict) or not isinstance(parsed.get("verdicts"), list):
            return None, "'verdicts' missing or not a list"
        return parsed["verdicts"], ""

    async def one(rec):
        names = [g["name"] for g in rec["gold"]]
        prompt = JUDGE_PROMPT.format(summary=rec["summary"],
                                     sounds="\n".join(f"  - {n}" for n in names))
        res = await chat.generate(prompt, postprocess, label=rec["item_id"])
        if not res.ok:
            log.warning("  judge failed on '%s' (%s): item not adjudicated",
                        rec["item_id"], res.status)
            return
        await cache.record(rec["item_id"],
                           read_judge_verdicts(res.value, names, rec["summary"],
                                               rec["item_id"], res.raw, res.status))

    async def every_one():
        await run_pool(todo, one, concurrency=args.judge_concurrency)
        cache.flush()

    run_async(every_one())
    return len(todo)


def score_summary(records: list[dict]) -> dict:
    from collections import Counter

    by_budget: dict[int, dict] = {}
    by_sound: dict[tuple, dict] = {}
    for r in records:
        if not r.get("mentioned_names") and r.get("summary") is None:
            continue
        b = r.get("budget")
        d = by_budget.setdefault(b, {
            "present": Counter(), "mentioned": Counter(), "discussed": Counter(),
            "present_excl_rare": Counter(), "mentioned_excl_rare": Counter(),
            "n_scenes": 0, "n_words": [], "n_without": 0})
        d["n_scenes"] += 1
        if r.get("summary"):
            d["n_words"].append(len(str(r["summary"]).split()))
        mentioned = set(r.get("mentioned_names") or [])
        discussed = set(r.get("discussed_names") or [])
        if not mentioned:
            d["n_without"] += 1
        sid = r.get("scenario_id") or str(r.get("item_id", "")).split("::")[0]
        for g in r["gold"]:
            if not g.get("reaction"):
                continue
            key = (sid, g["name"])
            u = by_sound.setdefault(key, {"reaction": g["reaction"],
                                          "is_rare": bool(g.get("is_rare")),
                                          "budgets_mentioning": set()})
            if g["name"] in mentioned:
                u["budgets_mentioning"].add(b)
        for g in r["gold"]:
            if not g.get("reaction"):
                continue
            d["present"][g["reaction"]] += 1
            if g["name"] in discussed:
                d["discussed"][g["reaction"]] += 1
            if g["name"] in mentioned:
                d["mentioned"][g["reaction"]] += 1
            if not g.get("is_rare"):
                d["present_excl_rare"][g["reaction"]] += 1
                if g["name"] in mentioned:
                    d["mentioned_excl_rare"][g["reaction"]] += 1

    def parts(mentioned: Counter, present: Counter) -> dict:
        n_c, n_p = sum(mentioned.values()), sum(present.values())
        return {
            "share_mentioned": {k: v / n_c for k, v in sorted(mentioned.items())} if n_c else {},
            "share_present": {k: v / n_p for k, v in sorted(present.items())} if n_p else {},
            "recall_by_reaction": {k: mentioned[k] / v
                                   for k, v in sorted(present.items()) if v},
        }

    out = {"budgets": {}}
    for b in sorted(by_budget):
        d = by_budget[b]
        n_mentioned = sum(d["mentioned"].values())
        n_present = sum(d["present"].values())
        out["budgets"][str(b)] = {
            "n_scenes": d["n_scenes"],
            "median_words": (sorted(d["n_words"])[len(d["n_words"]) // 2]
                             if d["n_words"] else None),
            "over_budget_rate": (sum(n > 1.5 * b for n in d["n_words"]) / len(d["n_words"])
                                 if d["n_words"] else None),
            "n_summaries_mentioning_nothing": d["n_without"],
            "mention_rate": (n_mentioned / n_present) if n_present else None,
            "n_sounds": dict(sorted(d["present"].items())),
            "n_mentioned": dict(sorted(d["mentioned"].items())),
            "n_discussed_only": dict(sorted(d["discussed"].items())),
            **parts(d["mentioned"], d["present"]),
            "excluding_rare": parts(d["mentioned_excl_rare"], d["present_excl_rare"]),
        }

    present_u, mentioned_u, mentioned_hr, present_hr = (Counter() for _ in range(4))
    budgets_by_sound = Counter()
    for u in by_sound.values():
        present_u[u["reaction"]] += 1
        if not u["is_rare"]:
            present_hr[u["reaction"]] += 1
        if u["budgets_mentioning"]:
            mentioned_u[u["reaction"]] += 1
            budgets_by_sound[len(u["budgets_mentioning"])] += 1
            if not u["is_rare"]:
                mentioned_hr[u["reaction"]] += 1
    n_c, n_p = sum(mentioned_u.values()), sum(present_u.values())
    n_pairs = sum(k * v for k, v in budgets_by_sound.items())
    out["by_sound"] = {
        "n_sounds": n_p,
        "n_mentioned": n_c,
        "mention_rate": (n_c / n_p) if n_p else None,
        "budgets_per_mentioned_sound": (n_pairs / n_c) if n_c else None,
        "n_mentioned_by_n_budgets": dict(sorted(budgets_by_sound.items())),
        "n_sounds_by_reaction": dict(sorted(present_u.items())),
        "n_mentioned_by_reaction": dict(sorted(mentioned_u.items())),
        **parts(mentioned_u, present_u),
        "excluding_rare": parts(mentioned_hr, present_hr),
    }
    return out


def describe_summary(scores: dict) -> list[str]:
    lines = ["SUMMARY (free summary, mentions adjudicated by a text LLM)"]
    for b, d in scores["budgets"].items():
        if d["mention_rate"] is None:
            lines.append(f"  budget {b} words: no item")
            continue
        over = d.get("over_budget_rate") or 0.0
        alert = f"   WARNING {100*over:.0f} % go over 1.5x the budget" if over > 0.2 else ""
        lines.append(f"  budget {b} words: {d['n_scenes']} summaries, "
                     f"actual median {d['median_words']} words, "
                     f"mention rate {d['mention_rate']:.3f}{alert}")
        if d["n_summaries_mentioning_nothing"]:
            lines.append(f"     of which {d['n_summaries_mentioning_nothing']} summary(ies) "
                         "mention NO annotated sound")
        hr = d["excluding_rare"]
        for k in sorted(d["share_present"]):
            share, base = d["share_mentioned"].get(k, 0.0), d["share_present"][k]
            arrow = "+" if share > base else " "
            without = (f"   excl. rare {100*hr['share_mentioned'].get(k, 0.0):5.1f} % "
                       f"against {100*hr['share_present'].get(k, 0.0):5.1f} %"
                       if k == "pivot" and hr["share_present"] else "")
            lines.append(f"     {k:11s} {100*share:5.1f} % of the mentions "
                         f"against {100*base:5.1f} % of the sounds {arrow}"
                         f"   (recall {d['recall_by_reaction'].get(k, 0):.2f}){without}")
    lines.append("  read the middle column: a preference is the gap to the base rate, "
                 "not the mention rate alone.")
    lines.append("  every RARE sound is a pivot: the 'excl. rare' column says whether "
                 "the pivot gap survives without them — if it collapses, it was "
                 "acoustic salience.")
    return lines


def score_grounding(records: list[dict]) -> dict:
    by_pair: dict[str, dict] = {}
    hits = fa = n_present = n_muted = n_invalid = 0
    for rec in records:
        pred = rec.get("prediction")
        if pred not in (True, False):
            n_invalid += 1
        if rec["gold"]:
            n_present += 1
            hits += pred is True
        else:
            n_muted += 1
            fa += pred is True
        by_pair.setdefault(rec["pair_id"], {})[bool(rec["gold"])] = pred

    complete = [v for v in by_pair.values() if len(v) == 2]
    correct = sum(1 for v in complete if v.get(True) is True and v.get(False) is False)
    reversed_ = sum(1 for v in complete if v.get(True) is False and v.get(False) is True)
    always_yes = sum(1 for v in complete if v.get(True) is True and v.get(False) is True)
    always_no = sum(1 for v in complete if v.get(True) is False and v.get(False) is False)
    hit_rate = hits / n_present if n_present else 0.0
    fa_rate = fa / n_muted if n_muted else 0.0
    return {
        "n_pairs": len(complete),
        "n_items": len(records),
        "n_unparsable": n_invalid,
        "paired_accuracy": round(correct / len(complete), 4) if complete else 0.0,
        "hit_rate": round(hit_rate, 4),
        "false_alarm_rate": round(fa_rate, 4),
        "sensitivity": round(hit_rate - fa_rate, 4),
        "paired_table": {"correct": correct, "reversed": reversed_,
                         "always_yes": always_yes, "always_no": always_no},
        "always_yes_pairs": always_yes,
        "mcnemar_p": round(mcnemar_p(correct, reversed_), 4),
    }


def describe_grounding(scores: dict) -> list[str]:
    t = scores["paired_table"]
    p = scores["mcnemar_p"]
    discordant = t["correct"] + t["reversed"]
    verdict = ("discriminates" if p < 0.05 and t["correct"] > t["reversed"]
               else "does not discriminate")
    return [
        f"Grounding: {scores['n_pairs']} pairs (sound present / sound muted)",
        f"  paired accuracy {scores['paired_accuracy']:.3f}   (chance 0.25; "
        f"always answering yes gives 0.000)",
        f"  hits {scores['hit_rate']:.3f} | false alarms "
        f"{scores['false_alarm_rate']:.3f} | sensitivity "
        f"{scores['sensitivity']:+.3f}",
        f"  paired table: {t['correct']} correct | {t['reversed']} reversed | "
        f"{t['always_yes']} yes on both sides | {t['always_no']} no on both sides",
        f"  McNemar on the {discordant} discordant pairs: p = {p:.4f} "
        f"-> the model {verdict}",
    ]


def score_reactions(records: list[dict]) -> dict:
    matrix = {g: dict.fromkeys((*REACTIONS, "invalid"), 0) for g in REACTIONS}
    n_invalid = 0
    for rec in records:
        gold = rec["gold"]
        pred = rec.get("prediction")
        if pred not in REACTIONS:
            pred = "invalid"
            n_invalid += 1
        matrix[gold][pred] += 1

    per_class, recalls = {}, []
    for r in REACTIONS:
        tp = matrix[r][r]
        fn = sum(matrix[r].values()) - tp
        fp = sum(matrix[g][r] for g in REACTIONS if g != r)
        per_class[r] = _prf(tp, fp, fn)
        support = tp + fn
        if support:
            recalls.append(tp / support)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
    n = len(records)
    correct = sum(matrix[r][r] for r in REACTIONS)
    return {
        "n_events": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "balanced_accuracy": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "macro_f1": round(macro_f1, 4),
        "n_unparsable": n_invalid,
        "per_class": per_class,
        "confusion": matrix,
    }


TEXT_ONLY_CEILING_NOTE = (
    "  warning: this task is solved at ~100 % from the dialogue text ALONE "
    "when the event position is given. A high score is therefore no proof of "
    "audio understanding — cf. --task grounding."
)


def describe_reactions(scores: dict) -> list[str]:
    lines = [f"Reactions: {scores['n_events']} events | "
             f"macro-F1 {scores['macro_f1']:.3f} | "
             f"balanced accuracy {scores['balanced_accuracy']:.3f} | "
             f"raw {scores['accuracy']:.3f}"]
    if scores["n_unparsable"]:
        lines.append(f"  unreadable answers: {scores['n_unparsable']}")
    head = "  " + " ".join(f"{r[:5]:>7s}" for r in (*REACTIONS, "invalid"))
    lines.append("  truth \\ predicted" + head)
    for g in REACTIONS:
        row = " ".join(f"{scores['confusion'][g][p]:7d}"
                       for p in (*REACTIONS, "invalid"))
        lines.append(f"  {g:12s} {row}")
    lines.append(TEXT_ONLY_CEILING_NOTE)
    return lines


def label_from_text(raw: str) -> str | None:
    if not raw:
        return None
    found = {r for r in REACTIONS if re.search(rf"\b{r}\b", raw, re.IGNORECASE)}
    return found.pop() if len(found) == 1 else None


def presence_from_text(raw: str) -> bool | None:
    if not raw:
        return None
    yes = bool(re.search(r"\b(true|yes)\b", raw, re.IGNORECASE))
    no = bool(re.search(r"\b(false|no)\b", raw, re.IGNORECASE))
    return yes if yes != no else None


def _predicted_presence(value: Any) -> bool | None:
    if isinstance(value, dict):
        for key in ("sound_present", "present", "sound", "answer"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "oui"):
            return True
        if low in ("false", "no", "non"):
            return False
    return None


def _predicted_reaction(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("reaction")
    if isinstance(value, str) and value.strip().lower() in REACTIONS:
        return value.strip().lower()
    return None


def _is_cuda_poisoned(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return "CUDA" in text or "AcceleratorError" in type(exc).__name__


def evaluate(items: list[dict], chat: AudioChat, cache: Checkpoint,
             args: argparse.Namespace) -> bool:
    todo = [it for it in items if it["item_id"] not in cache.data]
    log.info("Items: %d in total, %d already cached, %d to query",
             len(items), len(items) - len(todo), len(todo))
    for n, item in enumerate(todo, 1):
        audio = Path(item["audio"])
        if not audio.exists():
            log.warning("  missing audio, item skipped: %s", audio)
            continue
        if item["task"] == "sounds-mcq":
            prompt = MCQ_SOUND_PROMPT.format(
                t=item["t"], options=_options_block(item["options"]))
        elif item["task"] == "scene":
            prompt = MCQ_SCENE_PROMPT.format(options=_options_block(item["options"]))
        elif item["task"] == "grounding":
            prompt = GROUNDING_PROMPT.format(t=item["start_s"])
        elif item["task"] == "summary":
            prompt = SUMMARY_PROMPT.format(budget=item["budget"])
        else:
            prompt = reactions_prompt(item, args.reveal)

        try:
            reply = chat.ask(audio, prompt)
        except Exception as exc:  # noqa: BLE001
            log.error("  model failure on '%s': %s: %s",
                      item["item_id"], type(exc).__name__, exc)
            cache.data[item["item_id"]] = {
                **item, "raw": "", "status": f"error: {type(exc).__name__}: {exc}",
                "error": True,
            }
            cache.flush()
            if _is_cuda_poisoned(exc):
                log.error("CUDA error: the GPU context is unusable, stopping. "
                          "%d answer(s) kept; running the same command again "
                          "resumes after this item (CUDA_LAUNCH_BLOCKING=1 to "
                          "locate the real assert site).", len(cache.data))
                return False
            continue
        record = {**item, "raw": reply.raw, "status": reply.status,
                  "run": {"reveal": args.reveal,
                          "max_new_tokens": args.max_new_tokens,
                          "model_id": args.model_id}}
        if item["task"] == "summary":
            record["summary"] = strip_thinking(reply.raw or "").strip()
            record["status"] = "ok_text" if record["summary"] else "empty_response"
        elif item["task"] in ("sounds-mcq", "scene"):
            pred = _mcq_answer(reply.value, item["options"])
            if pred is None:
                pred = _mcq_answer(reply.raw, item["options"])
                record["status"] = "ok_text" if pred is not None else record["status"]
            record["prediction"] = pred
        elif item["task"] == "grounding":
            pred = _predicted_presence(reply.value)
            if pred is None:
                pred = presence_from_text(reply.raw)
                record["status"] = "ok_text" if pred is not None else record["status"]
            record["prediction"] = pred
        else:
            pred = _predicted_reaction(reply.value)
            if pred is None:
                pred = label_from_text(reply.raw)
                record["status"] = "ok_text" if pred is not None else record["status"]
            record["prediction"] = pred
        cache.data[item["item_id"]] = record
        if n % SAVE_EVERY == 0:
            cache.flush()
            log.info("  %d/%d", n, len(todo))
    cache.flush()
    return True


def reparse_cache(cache: Checkpoint, items: list[dict]) -> int:
    from ..alm import read_reply

    by_id = {it["item_id"]: it for it in items}
    n = 0
    for item_id, rec in cache.data.items():
        item = by_id.get(item_id)
        if item is None or rec.get("error"):
            continue
        reply = read_reply(rec.get("raw") or "")
        rec["status"] = reply.status
        if item["task"] == "summary":
            rec["summary"] = strip_thinking(rec.get("raw") or "").strip()
        elif item["task"] in ("sounds-mcq", "scene"):
            rec["prediction"] = (_mcq_answer(reply.value, item["options"])
                                 or _mcq_answer(rec.get("raw") or "", item["options"]))
        elif item["task"] == "grounding":
            rec["prediction"] = (_predicted_presence(reply.value)
                                 or presence_from_text(rec.get("raw") or ""))
        else:
            rec["prediction"] = (_predicted_reaction(reply.value)
                                 or label_from_text(rec.get("raw") or ""))
        n += 1
    if n:
        cache.flush()
    return n


def build_chat(args: argparse.Namespace) -> AudioChat:
    if args.backend == "fake":
        return FakeAudioChat(malformed_rate=args.fake_malformed_rate)

    from ..alm import (
        CascadeChat,
        FlamingoChat,
        KimiAudioChat,
        MiDashengLMChat,
        MimoAudioChat,
        MossAudioChat,
        QwenOmniChat,
    )

    model_kw = {"model_id": args.model_id} if args.model_id else {}
    if args.backend == FlamingoChat.name:
        return FlamingoChat(device_map=args.device_map,
                            max_new_tokens=args.max_new_tokens, **model_kw)
    if args.backend == QwenOmniChat.name:
        return QwenOmniChat(device_map=args.device_map,
                            max_new_tokens=args.max_new_tokens,
                            attn_implementation=args.attn_implementation, **model_kw)
    if args.backend == MiDashengLMChat.name:
        return MiDashengLMChat(device_map=args.device_map,
                               max_new_tokens=args.max_new_tokens, **model_kw)
    if args.backend == MossAudioChat.name:
        return MossAudioChat(device_map=args.device_map,
                             max_new_tokens=args.max_new_tokens,
                             repo=args.alm_repo, **model_kw)
    if args.backend == MimoAudioChat.name:
        kw = dict(model_kw)
        if args.tokenizer_id:
            kw["tokenizer_id"] = args.tokenizer_id
        return MimoAudioChat(repo=args.alm_repo, **kw)
    if args.backend == KimiAudioChat.name:
        return KimiAudioChat(max_new_tokens=args.max_new_tokens, **model_kw)
    if args.backend == CascadeChat.name:
        kw = {}
        if args.asr_id:
            kw["asr_id"] = args.asr_id
        if args.model_id:
            kw["llm_id"] = args.model_id
        out_dir = Path(args.output_dir) if getattr(args, "output_dir", None) \
            else Paths.resolve(getattr(args, "data_dir", None)).eval_dir
        return CascadeChat(device_map=args.device_map,
                           max_new_tokens=args.max_new_tokens,
                           transcripts_path=out_dir / "transcripts_whisper.json", **kw)
    raise ValueError(f"Unknown backend: {args.backend}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    from ..alm import BACKENDS, MAX_NEW_TOKENS
    from ..config import SCENARIO_BANKS

    parser.add_argument("--scenes-dir", default=None,
                        help="Directory of the mixed scenes (default: audio_scenes "
                             "of the data-dir)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory of the results (default: eval_output of the "
                             "data-dir)")
    parser.add_argument("--task",
                        choices=["scene", "sounds-mcq", "reactions", "grounding",
                                 "summary", "all"],
                        default="all",
                        help="scene: which PLACE out of 4, a capacity control "
                             "independent of the taxonomy; sounds-mcq: WHICH of "
                             "4 sounds, distractors drawn from the scene pool; "
                             "reactions: which reaction type, to be published with "
                             "its text-only ceiling; grounding: did a sound occur at "
                             "t, on pairs where only the sound changes — the only "
                             "task that cannot be solved from the transcript. "
                             "'all' IS NOT EVERYTHING: it chains the first three and "
                             "leaves out grounding (which needs the muted mixes) and "
                             "summary (which calls a paid judge). With --dump-items, "
                             "where nothing is called, it does cover the five.")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(SUMMARY_BUDGETS),
                        help=f"--task summary: summary lengths to ask for "
                             f"(default: {' '.join(map(str, SUMMARY_BUDGETS))}). The "
                             f"text control (--backend whisper+llm) does not need the "
                             f"three: one budget answers its question, for a third of "
                             f"the compute time.")
    parser.add_argument("--judge-model", default="claude-sonnet-5",
                        help="Text model that adjudicates the summaries of --task "
                             "summary. It sees neither the audio nor the reaction "
                             "types: only the summary and the sound names.")
    parser.add_argument("--judge-api-key", default=None,
                        help="Anthropic key of the judge (else ANTHROPIC_API_KEY).")
    parser.add_argument("--judge-rule", choices=["strict", "nature"],
                        default="strict",
                        help="what counts as a mention. 'strict': the summary names "
                             "the sound or describes it unambiguously. 'nature': "
                             "adds the descriptions whose MATERIAL, ACTION or "
                             "MECHANISM matches, even under another name. The judge "
                             "always returns both, so changing rule replays through "
                             "--rejudge-from-raw, without a call.")
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--rejudge-from-raw", action="store_true",
                        help="Replays the READING of the verdicts on the raw text "
                             "already cached, without a single call to the judge. "
                             "Use after fixing the parsing or the quote check.")
    parser.add_argument("--no-judge", action="store_true",
                        help="--task summary: produce the summaries without "
                             "adjudicating them. Useful to split the two costs, or "
                             "to read the summaries before paying the judge.")
    parser.add_argument("--dump-items", action="store_true",
                        help="Writes the items (question, options, right answer) to "
                             "items_<task>.json and stops there: no model is loaded "
                             "or queried. Multiple-choice questions are drawn from a "
                             "seed derived from the item id, hence identical across "
                             "runs and machines — but freezing them in a file is what "
                             "makes the benchmark citable without replaying it.")
    parser.add_argument("--counterfactual-dirs", default=None,
                        help="grounding: directories of the MUTED mixes, comma "
                             "separated (produced by 'cares mix --mute-event-rank K "
                             "--loudness-from <scenes dir>'). Default: the audio_cf* "
                             "directories next to the scenes.")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="fake",
                        help="fake: deterministic answers without a GPU, to run the "
                             "harness in. Default: fake, so that forgetting the flag "
                             "does not load tens of GB of weights.")
    parser.add_argument("--model-id", default=None,
                        help="Weights to load. Default: the one of the chosen "
                             "backend. For whisper+llm this is the LLM (the ASR "
                             "comes from --asr-id).")
    parser.add_argument("--alm-repo", default=None,
                        help="moss-audio and mimo-audio: path of the cloned model "
                             "repository, to put on sys.path (their classes are not "
                             "in transformers).")
    parser.add_argument("--tokenizer-id", default=None,
                        help="mimo-audio: audio tokenizer kept apart from the weights "
                             "(default: XiaomiMiMo/MiMo-Audio-Tokenizer).")
    parser.add_argument("--asr-id", default=None,
                        help="whisper+llm: transcription model "
                             "(default: openai/whisper-large-v3).")
    parser.add_argument("--attn-implementation", default=None,
                        help="qwen3-omni: e.g. flash_attention_2 if installed.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--banks", default=str(SCENARIO_BANKS),
                        help="Banks: they supply the distractors of `sounds-mcq` "
                             "(pool of the place) and the list of places of `scene`. "
                             "Must be THE ONE USED TO GENERATE the dataset, else the "
                             "distractors come from another universe of sounds.")
    parser.add_argument("--reveal", choices=["none", "time", "time_id"], default="time",
                        help="What the model is told about the event to type. "
                             "'time' isolates reasoning from detection.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug: query only N items.")
    parser.add_argument("--reparse", action="store_true",
                        help="Replays reading and matching from the raw text in "
                             "cache (free), to benefit from an improved parser or "
                             "matcher without querying the model again.")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Drops the items in error (model crash) from the cache, "
                             "to retry them.")
    parser.add_argument("--scores-only", action="store_true",
                        help="Ask the model nothing: recompute the scores from the "
                             "cache.")
    parser.add_argument("--fake-malformed-rate", type=float, default=0.0,
                        help="fake backend: share of unreadable answers, to check "
                             "that they are counted and not fatal.")


def run(args: argparse.Namespace) -> int:
    if args.backend != "fake" and not args.scores_only:
        from ..alm import ensure_libstdcxx_on_path

        ensure_libstdcxx_on_path()

    paths = Paths.resolve(args.data_dir)
    scenes_dir = Path(args.scenes_dir) if args.scenes_dir else paths.scenes_dir
    out_dir = Path(args.output_dir) if args.output_dir else paths.eval_dir

    manifests = load_manifests(scenes_dir)
    if not manifests:
        log.error("No mixing manifest in %s: run 'cares mix' first.", scenes_dir)
        return 1
    log.info("Mixed scenes: %d in %s", len(manifests), scenes_dir)

    ALL_TASKS = ("scene", "sounds-mcq", "reactions", "grounding", "summary")
    DEFAULT_ALL = ("scene", "sounds-mcq", "reactions")
    if args.task != "all":
        tasks = (args.task,)
    elif args.dump_items:
        tasks = ALL_TASKS
    else:
        tasks = DEFAULT_ALL
        omitted = [t for t in ALL_TASKS if t not in tasks]
        log.warning("--task all only runs %s. NOT run: %s "
                    "(grounding needs muted mixes, summary calls a paid judge). "
                    "Run them by name.",
                    ", ".join(tasks), ", ".join(omitted))
    if args.counterfactual_dirs:
        cf_dirs = [Path(x.strip()) for x in args.counterfactual_dirs.split(",") if x.strip()]
    else:
        cf_dirs = sorted(scenes_dir.parent.glob("audio_cf*"))
    builders = {
        "summary": lambda m, d: build_summary_items(m, d, tuple(args.budgets)),
        "sounds-mcq": lambda m, d: build_mcq_sound_items(m, d, args.banks),
        "scene": lambda m, d: build_scene_items(m, d, args.banks),
        "reactions": build_reaction_items,
        "grounding": lambda m, d: build_grounding_items(m, d, cf_dirs),
    }
    scorers = {"reactions": score_reactions, "grounding": score_grounding,
               "sounds-mcq": score_mcq, "scene": score_mcq,
               "summary": score_summary}
    describers = {"reactions": describe_reactions, "grounding": describe_grounding,
                  "sounds-mcq": describe_sounds_mcq, "scene": describe_scene,
                  "summary": describe_summary}
    cf_dirs = [d for d in cf_dirs if d.is_dir() and any(d.glob("*.manifest.json"))]
    if "grounding" in tasks and not cf_dirs:
        missing = ("no muted mix found. Produce some with 'cares mix "
                   "--mute-event-rank K --output-dir <dir> --loudness-from "
                   "<scenes dir>'.")
        if args.task != "all":
            log.error("--task grounding: %s", missing)
            return 2
        log.warning("grounding skipped: %s", missing)
        tasks = tuple(t for t in tasks if t != "grounding")

    chat = None
    interrupted = False
    summary: dict[str, Any] = {"backend": args.backend, "scenes": len(manifests)}
    for task in tasks:
        items = builders[task](manifests, scenes_dir)
        if args.limit:
            items = items[: args.limit]
        if not items:
            log.warning("Task '%s': no item.", task)
            continue

        if args.dump_items:
            path = out_dir / f"items_{task}.json"
            save_json(path, items)
            log.info("Task '%s': %d items written -> %s", task, len(items), path)
            continue

        tag = f"{args.backend}_{task}"
        cache = Checkpoint(out_dir / f"raw_{tag}.json", save_every=SAVE_EVERY)
        if args.retry_errors:
            removed = cache.drop(lambda r: not r.get("error"))
            log.info("--retry-errors: %d item(s) in error removed from the cache",
                     removed)
        if args.reparse:
            n_reparsed = reparse_cache(cache, items)
            log.info("--reparse: %d answer(s) re-read from the raw text", n_reparsed)
        if not args.scores_only and not interrupted:
            if chat is None:
                chat = build_chat(args)
            if not evaluate(items, chat, cache, args):
                interrupted = True
        elif not cache.data:
            log.error("--scores-only: nothing cached for '%s'.", task)
            return 1

        records = [cache.data[it["item_id"]] for it in items
                   if it["item_id"] in cache.data]
        if not records:
            log.warning("Task '%s': no answer to score.", task)
            continue
        if task == "summary":
            if args.no_judge:
                log.warning("--no-judge: %d summary(ies) written, not adjudicated, "
                            "no score. A mention rate without a judge is 0 by "
                            "construction. Running again without --no-judge does "
                            "not query the audio model: the summaries are cached.",
                            len(records))
                continue
            judged = Checkpoint(out_dir / f"judged_{tag}.json", save_every=SAVE_EVERY)
            if args.rejudge_from_raw:
                rejudge_from_raw(records, judged)
            judge_summaries(records, args, judged)
            adjudicated = []
            for rec in records:
                verdict = judged.get(rec["item_id"])
                if verdict is None:
                    continue
                nature = list(verdict.get("nature") or [])
                rec["mentioned_names"] = list(verdict.get("heard") or [])
                if args.judge_rule == "nature":
                    rec["mentioned_names"] += [
                        n for n in nature if n not in rec["mentioned_names"]]
                rec["nature_names"] = nature
                rec["discussed_names"] = list(verdict.get("discussed") or [])
                adjudicated.append(rec)
            if (args.judge_rule == "nature"
                    and not any("nature" in (judged.get(r["item_id"]) or {})
                                for r in records)):
                log.error("--judge-rule nature on a cache judged BEFORE that "
                          "verdict: no record carries 'nature', the numbers "
                          "would be those of the strict rule. Judge again "
                          "(delete judged_%s_summary.json) — --rejudge-from-raw "
                          "is not enough, the PROMPT has changed.", tag)
                return 2
            if len(adjudicated) < len(records):
                log.warning("  %d summary(ies) not adjudicated, dropped from the "
                            "cross-tabulation", len(records) - len(adjudicated))
            records = adjudicated
            if not records:
                log.error("Task 'summary': no adjudicated summary, nothing to score.")
                continue
        scores = scorers[task](records)
        for line in describers[task](scores):
            log.info("%s", line)
        save_json(out_dir / f"scores_{tag}.json", scores)
        summary[task] = {k: v for k, v in scores.items()
                         if k not in ("per_scene", "confusion")}

    if args.dump_items:
        return 0
    path = out_dir / f"summary_{args.backend}.json"
    previous = load_json(path, default=None)
    if isinstance(previous, dict):
        merged = {**previous, **summary}
        kept = [k for k in previous
                if k not in summary and k not in ("backend", "scenes")]
        if kept:
            log.info("Summary: %d task(s) kept from a previous run: %s",
                     len(kept), ", ".join(sorted(kept)))
        summary = merged
    save_json(path, summary)
    log.info("Results written to %s", out_dir)
    if interrupted:
        log.error("Run interrupted by a GPU error: run the same command again to "
                  "continue (the offending item is skipped, --retry-errors to "
                  "retry it).")
        return 1
    return 0
