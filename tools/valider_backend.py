"""Does a new backend actually HEAR? To be run before any campaign.

Three open questions are asked about a scene whose place, sounds and dialogue
are known; the answers are checked for degeneracy (a token or a sentence
repeated forever) and for clues that the model really listened. The reply is
printed as well: the script makes the verdict easy, not automatic.

    python tools/valider_backend.py --backend moss-audio --data-dir other/data/data_test
    python tools/valider_backend.py --backend midasheng-lm --scenes 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: The questions, from the most open to the most targeted. The first one is the
#: model-card question: if it degenerates, nothing else is worth launching.
QUESTIONS = (
    "Caption the audio.",
    "Describe everything you hear in this recording, including the setting, "
    "the people talking, and any distinct sounds.",
    "What are the two people talking about?",
)

#: Beyond this, a reply is held degenerate. A healthy reply never repeats a
#: token more than a few times.
MAX_REPETITIONS = 8


def degeneracy_reason(text: str) -> str | None:
    """Say WHY a reply is degenerate, or None if it looks healthy."""
    t = (text or "").strip()
    if not t:
        return "empty reply"
    stripped = re.sub(r"\s+", "", t)
    if len(set(stripped)) <= 2 and len(stripped) > 20:
        return f"one or two repeated characters ({stripped[:12]}...)"
    words = re.findall(r"[a-zA-Z']+", t.lower())
    if not words:
        return "no alphabetic word"
    most_common, n = Counter(words).most_common(1)[0]
    if n > MAX_REPETITIONS and n > len(words) * 0.3:
        return f"the word '{most_common}' occurs {n} times out of {len(words)}"
    for size in (3, 4, 5):
        if len(words) > size * 4:
            grams = Counter(tuple(words[i:i + size])
                            for i in range(len(words) - size))
            gram, k = grams.most_common(1)[0]
            if k > MAX_REPETITIONS:
                return f"the run '{' '.join(gram)}' occurs {k} times"
    return None


def listening_clues(text: str, manifest: dict, dialogue: str) -> list[str]:
    """What, in the reply, suggests the model REALLY listened."""
    low = (text or "").lower()
    found = []
    place = str(manifest.get("scene") or "").replace("_", " ")
    if place and any(w in low for w in place.split() if len(w) > 3):
        found.append(f"names the place ({place})")
    for e in manifest.get("events_placed") or []:
        name = str(e.get("event_id", "")).split("/")[-1].replace("_", " ")
        words = [w for w in name.split() if len(w) > 3]
        if words and all(w in low for w in words):
            found.append(f"names the sound '{name}'")
    # Rare words of the dialogue: the strongest clue, and the hardest to fake.
    rare = [w for w in set(re.findall(r"[a-z]{6,}", dialogue.lower()))
            if w in low]
    if rare:
        found.append(f"echoes words of the dialogue: {', '.join(sorted(rare)[:4])}")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", required=True)
    ap.add_argument("--data-dir", default="other/data/data_test")
    ap.add_argument("--scenes", type=int, default=1)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--alm-repo", default=None)
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    from cares.alm import BACKENDS

    scenes_dir = Path(args.data_dir) / "audio_scenes"
    manifests = sorted(scenes_dir.glob("*.manifest.json"))
    if not manifests:
        print(f"No mixed scene in {scenes_dir}.")
        return 1

    kw = {}
    if args.model_id:
        kw["model_id"] = args.model_id
    if args.alm_repo:
        kw["repo"] = args.alm_repo
    try:
        chat = BACKENDS[args.backend](**kw)
    except TypeError:
        kw.pop("repo", None)
        chat = BACKENDS[args.backend](**kw)

    dialogues = {}
    f = Path(args.data_dir) / "filter_output" / "dialogues_filtered.json"
    if f.exists():
        d = json.loads(f.read_text())
        for x in (d if isinstance(d, list) else d.values()):
            dialogues[x["scenario_id"]] = " ".join(
                i.get("text", "") for i in x.get("timeline") or []
                if i.get("type") == "utterance")

    verdicts = []
    for path in manifests[:args.scenes]:
        m = json.loads(path.read_text())
        if not isinstance(m, dict) or not m.get("scenario_id"):
            continue
        sid = m["scenario_id"]
        wav = scenes_dir / f"{sid}.wav"
        print(f"\n{'=' * 72}\nSCENE {sid}")
        print(f"  place : {m.get('scene')}   duration {m.get('duration_s', 0):.0f} s")
        print(f"  sounds: {[e['event_id'] for e in m.get('events_placed') or []]}")
        for q in QUESTIONS:
            r = chat.ask(wav, q)
            text = (r.raw or "").strip()
            bad = degeneracy_reason(text)
            clues = listening_clues(text, m, dialogues.get(sid, ""))
            print(f"\n  Q: {q}")
            print(f"  A: {text[:400]}{'...' if len(text) > 400 else ''}")
            print(f"     {'DEGENERATE: ' + bad if bad else 'form: healthy'}")
            print(f"     listening clues: {'; '.join(clues) if clues else 'NONE'}")
            verdicts.append((bad is None, bool(clues)))

    healthy = sum(1 for ok, _ in verdicts if ok)
    heard = sum(1 for _, h in verdicts if h)
    print(f"\n{'=' * 72}")
    print(f"  {healthy}/{len(verdicts)} replies of healthy form")
    print(f"  {heard}/{len(verdicts)} carry a listening clue")
    if healthy < len(verdicts):
        print("\n  DO NOT LAUNCH A CAMPAIGN. A degenerate reply signals corrupted")
        print("  logits - most often a transformers version other than the")
        print("  `transformers_version` of the model's config.json.")
        return 2
    if not heard:
        print("\n  CAUTION: no reply shows that the model listened. Read them")
        print("  yourself: either the model is bad, or the audio does not reach it")
        print("  - and both produce scores at chance level.")
        return 3
    print("\n  The backend hears and answers. The campaign can start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
