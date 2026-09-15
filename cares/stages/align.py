from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..config import Paths
from ..dataset import strip_prosody_tags
from ..jsonio import load_json, save_json
from ..log import get_logger

log = get_logger(__name__)

WORD_RE = re.compile(r"[a-z']+")

MIN_COVERAGE = 0.90


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dialogues", type=Path, default=None,
                        help="Dialogues to align (default: dialogues of --data-dir).")
    parser.add_argument("--voices-dir", type=Path, default=None,
                        help="Voice tracks (default: out_voices of --data-dir).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Alignment directory (default: alignments of --data-dir).")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")


def turn_texts(record: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in record.get("timeline") or []:
        if item.get("type") == "utterance" and item.get("text"):
            out.append((str(item.get("speaker") or "?"), str(item["text"])))
    return out


def clean_turn_text(text: str) -> str:
    return strip_prosody_tags(text)


def words_with_turns(turns: list[tuple[str, str]]
                     ) -> tuple[list[str], list[int], list[int]]:
    words: list[str] = []
    owner: list[int] = []
    offsets: list[int] = []
    for i, (_, text) in enumerate(turns):
        for m in WORD_RE.finditer(clean_turn_text(text).lower()):
            words.append(m.group())
            owner.append(i)
            offsets.append(m.start())
    return words, owner, offsets


def align_one(audio, words: list[str], model, dictionary, sample_rate: int,
              device: str) -> list[tuple[float, float]] | None:
    import torch
    from torchaudio.functional import forced_align, merge_tokens

    tokenized = [[dictionary[c] for c in w if c in dictionary] for w in words]
    kept = [(i, t) for i, t in enumerate(tokenized) if t]
    if not kept:
        return None
    waveform = torch.from_numpy(audio).unsqueeze(0).to(device)
    with torch.inference_mode():
        emission, _ = model(waveform)
    targets = torch.tensor([c for _, t in kept for c in t],
                           dtype=torch.int32, device=device).unsqueeze(0)
    aligned, scores = forced_align(emission, targets, blank=0)
    spans = merge_tokens(aligned[0], scores[0])
    ratio = waveform.size(1) / emission.size(1) / sample_rate

    out: list[tuple[float, float]] = [(0.0, 0.0)] * len(words)
    cursor = 0
    for i, toks in kept:
        span = (spans[cursor].start * ratio, spans[cursor + len(toks) - 1].end * ratio)
        out[i] = span
        cursor += len(toks)
    for i in range(len(out)):
        if out[i] == (0.0, 0.0) and i > 0:
            out[i] = (out[i - 1][1], out[i - 1][1])
    return out


def turns_from_words(turns: list[tuple[str, str]], owner: list[int],
                     spans: list[tuple[float, float]], duration_s: float,
                     offsets: list[int] | None = None) -> list[dict]:
    bounds: list[dict] = []
    for i, (speaker, text) in enumerate(turns):
        idx = [k for k, o in enumerate(owner) if o == i]
        if not idx:
            continue
        entry = {"speaker": speaker, "text": text,
                 "t_start_s": float(spans[idx[0]][0]),
                 "t_end_s": float(spans[idx[-1]][1])}
        if offsets is not None:
            entry["text_clean"] = clean_turn_text(text)
            entry["words"] = [[round(spans[k][0], 4), round(spans[k][1], 4),
                               offsets[k]] for k in idx]
        bounds.append(entry)
    for a, b in zip(bounds, bounds[1:], strict=False):
        middle = (a["t_end_s"] + b["t_start_s"]) / 2.0
        a["t_end_s"] = b["t_start_s"] = middle
    if bounds:
        bounds[0]["t_start_s"] = 0.0
        bounds[-1]["t_end_s"] = duration_s
    return bounds


def run(args: argparse.Namespace) -> int:
    paths = Paths.resolve(args.data_dir)
    dialogues_path = args.dialogues or (paths.dialogues_dir("grounded") / "dialogues.json")
    voices_dir = args.voices_dir or (Path(args.data_dir or ".") / "out_voices")
    out_dir = args.output_dir or (Path(args.data_dir or ".") / "alignments")

    records = load_json(dialogues_path)
    if not records:
        log.error("Dialogues not found or empty: %s", dialogues_path)
        return 1
    if isinstance(records, dict):
        records = list(records.values())
    if not voices_dir.is_dir():
        log.error("Voice directory not found: %s", voices_dir)
        return 1

    todo = []
    for rec in records:
        sid = rec.get("scenario_id")
        if not sid:
            continue
        voice = next((voices_dir / f"{sid}{ext}" for ext in (".mp3", ".wav", ".flac")
                      if (voices_dir / f"{sid}{ext}").exists()), None)
        if voice is None:
            continue
        if not args.overwrite and (out_dir / f"{sid}.json").exists():
            continue
        todo.append((sid, voice, rec))
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        log.info("Nothing to align (already done? --overwrite to redo)")
        return 0

    import torch
    import torchaudio

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.MMS_FA
    log.info("Aligning %d voices on %s (MMS_FA model, %d Hz)",
             len(todo), device, bundle.sample_rate)
    model = bundle.get_model(with_star=False).to(device)
    dictionary = bundle.get_dict()

    from ..audio.io import load_audio_resampled

    out_dir.mkdir(parents=True, exist_ok=True)
    ok = weak = failed = 0
    for n, (sid, voice, rec) in enumerate(todo, 1):
        turns = turn_texts(rec)
        if not turns:
            failed += 1
            continue
        try:
            audio = load_audio_resampled(voice, target_sr=bundle.sample_rate)
            words, owner, offsets = words_with_turns(turns)
            spans = align_one(audio, words, model, dictionary, bundle.sample_rate, device)
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s: alignment impossible (%s)", sid, exc)
            failed += 1
            continue
        if spans is None:
            failed += 1
            continue
        duration = len(audio) / bundle.sample_rate
        coverage = spans[-1][1] / duration if duration > 0 else 0.0
        aligned = turns_from_words(turns, owner, spans, duration, offsets)
        if coverage < MIN_COVERAGE:
            weak += 1
            log.warning("  %s: the text only covers %.0f%% of the audio", sid, 100 * coverage)
        else:
            ok += 1
        save_json(out_dir / f"{sid}.json", {
            "scenario_id": sid,
            "aligner": "torchaudio.pipelines.MMS_FA",
            "audio_source": str(voice),
            "audio_duration_s": round(duration, 3),
            "coverage": round(coverage, 4),
            "n_words": len(words),
            "turns": aligned,
        })
        if n % 20 == 0:
            log.info("  %d/%d", n, len(todo))
    log.info("Done: %d alignments, %d weak, %d failures -> %s",
             ok, weak, failed, out_dir)
    return 0
