from __future__ import annotations

import argparse
import asyncio
import base64
import os
import random
import time
import traceback
import wave
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import VOICES_POOL, Paths
from ..dataset import load_scenarios, restrict_to_splits
from ..event_tags import event_to_tag
from ..jsonio import load_json, save_json
from ..log import get_logger
from ..runner import run_async, run_pool

log = get_logger(__name__)

PCM_SAMPWIDTH = 2
PCM_CHANNELS = 1

DEFAULT_MAX_CHARS = 2000


@dataclass
class Task:
    scenario_id: str
    inputs: list[dict]
    voices: dict
    turns_meta: list[dict]
    events_as_tags: list[dict]
    events_for_mixing: list[dict]
    script_text: str
    output_audio: Path
    output_meta: Path
    seed: int | None = None


@dataclass
class SynthConfig:
    model_id: str = "eleven_v3"
    output_format: str = "mp3_44100_128"
    language_code: str | None = None
    apply_text_normalization: str | None = None
    settings: dict | None = None
    seed: int | None = None
    timestamps: bool = True
    save_alignment: bool = False
    max_chars: int = DEFAULT_MAX_CHARS
    chunk_gap_ms: int = 0
    max_retries: int = 5


def build_dialogue_inputs(
    timeline: list[dict],
    voices: dict,
    *,
    include_sfx: bool = False,
    extra_tags: dict[str, str] | None = None,
    perform_events: bool = True,
) -> tuple[list[dict], list[dict], list[dict], list[dict], str]:
    utts: list[dict] = []
    for i, item in enumerate(timeline):
        if item.get("type") == "utterance":
            utts.append({
                "tl_pos": i,
                "speaker": item.get("speaker"),
                "text": (item.get("text") or "").strip(),
            })

    def prev_utt(pos: int, spk: str | None) -> dict | None:
        cand = [u for u in utts if u["tl_pos"] < pos
                and (spk is None or u["speaker"] == spk)]
        return cand[-1] if cand else None

    def next_utt(pos: int, spk: str | None) -> dict | None:
        cand = [u for u in utts if u["tl_pos"] > pos
                and (spk is None or u["speaker"] == spk)]
        return cand[0] if cand else None

    def n_utts_before(pos: int) -> int:
        return sum(1 for u in utts if u["tl_pos"] < pos)

    events_as_tags: list[dict] = []
    events_for_mixing: list[dict] = []

    for i, item in enumerate(timeline):
        if item.get("type") != "event":
            continue
        eid = item.get("event_id")
        tag = (event_to_tag(eid, include_sfx=include_sfx, extra=extra_tags)
               if perform_events else None)

        if tag is None:
            events_for_mixing.append({
                "event_id": eid,
                "after_turn_idx": n_utts_before(i) - 1,
            })
            continue

        spk = item.get("speaker")
        if spk not in ("A", "B"):
            p = prev_utt(i, None)
            n = next_utt(i, None)
            spk = (p or n or {}).get("speaker")

        anchor = prev_utt(i, spk)
        where = "append"
        if anchor is None:
            anchor = next_utt(i, spk)
            where = "prepend"
        if anchor is None:
            anchor = prev_utt(i, None) or next_utt(i, None)
            if anchor is not None:
                where = "append" if anchor["tl_pos"] < i else "prepend"
                spk = anchor["speaker"]
        if anchor is None:
            continue

        if where == "append":
            anchor["text"] = (anchor["text"] + " " + tag).strip()
        else:
            anchor["text"] = (tag + " " + anchor["text"]).strip()

        events_as_tags.append({
            "event_id": eid,
            "tag": tag,
            "speaker": spk,
            "_anchor_tl_pos": anchor["tl_pos"],
        })

    pos_to_idx = {u["tl_pos"]: k for k, u in enumerate(utts)}
    inputs: list[dict] = []
    turns_meta: list[dict] = []
    for idx, u in enumerate(utts):
        vid = voices.get(u["speaker"])
        inputs.append({"text": u["text"], "voice_id": vid})
        turns_meta.append({
            "turn_idx": idx,
            "speaker": u["speaker"],
            "voice_id": vid,
            "text": u["text"],
        })

    for e in events_as_tags:
        e["attached_to_input_index"] = pos_to_idx.get(e.pop("_anchor_tl_pos"))

    script_text = "\n".join(f'{u["speaker"]}: {u["text"]}' for u in utts)
    return inputs, turns_meta, events_as_tags, events_for_mixing, script_text


def plan_chunks(inputs: list[dict], max_chars: int) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    start = 0
    cur = 0
    for i, inp in enumerate(inputs):
        t = len(inp["text"])
        if cur > 0 and cur + t > max_chars:
            chunks.append((start, i))
            start = i
            cur = 0
        cur += t
    chunks.append((start, len(inputs)))
    return chunks


def load_voice_pool(path: Path | None = None) -> list[dict]:
    data = load_json(path or VOICES_POOL)
    pool = (data or {}).get("voices") if isinstance(data, dict) else data
    if not isinstance(pool, list) or not pool:
        raise ValueError(f"Empty or unreadable voice pool: {path or VOICES_POOL}")
    return [v for v in pool if v.get("voice_id")]


def make_voice_picker(
    voices: str | None,
    voice_a: str | None,
    voice_b: str | None,
    voices_file: Path | None = None,
):
    if voice_a and voice_b:
        return lambda sid, genders=None: {"A": voice_a, "B": voice_b}

    if voices:
        pool = [v.strip() for v in voices.split(",") if v.strip()]
        if len(pool) < 2:
            raise ValueError("--voices must list at least 2 voice_id.")

        def pick_flat(sid: str, genders: dict | None = None) -> dict:
            rng = random.Random(zlib.crc32(sid.encode()) & 0xFFFFFFFF)
            a, b = rng.sample(pool, 2)
            return {"A": a, "B": b}
        return pick_flat

    entries = load_voice_pool(voices_file)
    by_gender: dict[str, list[str]] = {}
    for v in entries:
        by_gender.setdefault(str(v.get("gender") or "?"), []).append(v["voice_id"])
    every = [v["voice_id"] for v in entries]
    if len(every) < 2:
        raise ValueError("The voice pool must contain at least 2 voices.")

    def pick(sid: str, genders: dict | None = None) -> dict:
        rng = random.Random(zlib.crc32(sid.encode()) & 0xFFFFFFFF)
        ga = (genders or {}).get("A")
        gb = (genders or {}).get("B")
        if ga is None or gb is None:
            a, b = rng.sample(every, 2)
            return {"A": a, "B": b}
        pa = by_gender.get(ga) or every
        pb = by_gender.get(gb) or every
        a = rng.choice(pa)
        rest = [v for v in pb if v != a] or [v for v in every if v != a]
        return {"A": a, "B": rng.choice(rest)}
    return pick


def _is_synthesized(meta_path: Path) -> bool:
    try:
        meta = load_json(meta_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("unreadable meta, the dialogue will be redone (%s): %s",
                    meta_path, exc)
        return False
    return isinstance(meta, dict) and not meta.get("dry_run")


def build_tasks(
    dialogues: list[dict],
    voices_dir: Path,
    cfg: SynthConfig,
    *,
    pick_voices,
    include_sfx: bool,
    extra_tags: dict[str, str] | None,
    seed_from_scenario: bool,
    overwrite: bool,
    perform_events: bool = True,
) -> list[Task]:
    ext = "mp3" if cfg.output_format.startswith("mp3_") else "wav"
    tasks: list[Task] = []
    skipped = 0

    for dlg in dialogues:
        sid = dlg["scenario_id"]
        audio_path = voices_dir / f"{sid}.{ext}"
        meta_path = voices_dir / f"{sid}.meta.json"

        if audio_path.exists() and _is_synthesized(meta_path) and not overwrite:
            skipped += 1
            continue

        timeline = dlg.get("timeline")
        if not timeline:
            log.warning("[%s] skipped: missing or empty timeline.", sid)
            continue

        voices = pick_voices(sid, {"A": dlg.get("gender_a"), "B": dlg.get("gender_b")})
        try:
            (inputs, turns_meta, events_as_tags,
             events_for_mixing, script_text) = build_dialogue_inputs(
                timeline, voices, include_sfx=include_sfx, extra_tags=extra_tags,
                perform_events=perform_events)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] skipped (build inputs): %s", sid, e)
            continue

        if not inputs:
            log.warning("[%s] skipped: no utterance.", sid)
            continue

        seed = cfg.seed
        if seed is None and seed_from_scenario:
            seed = zlib.crc32(("seed:" + sid).encode()) & 0xFFFFFFFF

        tasks.append(Task(
            scenario_id=sid,
            inputs=inputs,
            voices=voices,
            turns_meta=turns_meta,
            events_as_tags=events_as_tags,
            events_for_mixing=events_for_mixing,
            script_text=script_text,
            output_audio=audio_path,
            output_meta=meta_path,
            seed=seed,
        ))

    if skipped:
        log.info("Resume: %d dialogues already done skipped "
                 "(--overwrite to redo them).", skipped)
    return tasks


def _attr(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _to_bytes(audio: Any) -> bytes:
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    return b"".join(audio)


def _retry_after(exc: Exception, attempt: int) -> float:
    h = getattr(exc, "headers", None)
    try:
        if h:
            ra = h.get("retry-after") or h.get("Retry-After")
            if ra is not None:
                return float(ra)
    except Exception:  # noqa: BLE001
        pass
    return min(60.0, 2.0 ** attempt)


def _pcm_duration_s(raw: bytes, sr: int) -> float:
    return len(raw) / float(sr * PCM_CHANNELS * PCM_SAMPWIDTH)


def _seg_to_dict(seg: Any) -> dict:
    return {
        "voice_id": _attr(seg, "voice_id"),
        "start_time_seconds": _attr(seg, "start_time_seconds"),
        "end_time_seconds": _attr(seg, "end_time_seconds"),
        "character_start_index": _attr(seg, "character_start_index"),
        "character_end_index": _attr(seg, "character_end_index"),
        "dialogue_input_index": _attr(seg, "dialogue_input_index"),
    }


def _as_sdk_inputs(pairs: list[dict], dialogue_input_cls: Any) -> list:
    if dialogue_input_cls is not None:
        return [dialogue_input_cls(text=p["text"], voice_id=p["voice_id"])
                for p in pairs]
    return pairs


def _convert_chunk(
    client: Any,
    dialogue_input_cls: Any,
    seg_inputs: list[dict],
    cfg: SynthConfig,
    seed: int | None,
    sid: str,
) -> tuple[bytes, list[dict] | None, dict | None]:
    kwargs = dict(
        inputs=_as_sdk_inputs(seg_inputs, dialogue_input_cls),
        model_id=cfg.model_id,
        output_format=cfg.output_format,
    )
    if seed is not None:
        kwargs["seed"] = seed
    if cfg.settings is not None:
        kwargs["settings"] = cfg.settings
    if cfg.language_code:
        kwargs["language_code"] = cfg.language_code
    if cfg.apply_text_normalization:
        kwargs["apply_text_normalization"] = cfg.apply_text_normalization

    last = None
    for attempt in range(cfg.max_retries):
        audio = b""
        segs = None
        align = None
        try:
            if cfg.timestamps:
                resp = client.text_to_dialogue.convert_with_timestamps(**kwargs)
                b64 = _attr(resp, "audio_base_64") or _attr(resp, "audio_base64")
                audio = base64.b64decode(b64) if b64 else b""
                raw_segs = _attr(resp, "voice_segments") or []
                segs = [_seg_to_dict(s) for s in raw_segs]
                align = _attr(resp, "alignment") if cfg.save_alignment else None
                if align is not None and not isinstance(align, dict):
                    align = {
                        "characters": _attr(align, "characters"),
                        "character_start_times_seconds":
                            _attr(align, "character_start_times_seconds"),
                        "character_end_times_seconds":
                            _attr(align, "character_end_times_seconds"),
                    }
            else:
                audio = _to_bytes(client.text_to_dialogue.convert(**kwargs))
        except Exception as e:  # noqa: BLE001
            last = e
            status = getattr(e, "status_code", None)
            if status == 429 or (status is not None and 500 <= status < 600):
                wait = _retry_after(e, attempt)
                log.warning("[%s] HTTP %s -> sleeping %.1fs (attempt %d/%d)",
                            sid, status, wait, attempt + 1, cfg.max_retries)
                time.sleep(wait)
                continue
            if status == 422:
                raise
            if status is None:
                log.warning("[%s] network error -> retry (%d/%d): %s",
                            sid, attempt + 1, cfg.max_retries, str(e)[:160])
                time.sleep(min(30.0, 1.0 + attempt))
                continue
            raise

        if not audio:
            raise RuntimeError(
                f"[{sid}] ElevenLabs reply without audio "
                f"(empty audio_base_64 field). Check eleven_v3 / "
                f"Text-to-Dialogue access, the output format "
                f"({cfg.output_format}) and the voice_id."
            )
        return audio, segs, align
    raise last if last else RuntimeError("unknown failure")


def _write_wav(path: Path, pcm_bytes: bytes, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(PCM_CHANNELS)
        w.setsampwidth(PCM_SAMPWIDTH)
        w.setframerate(sr)
        w.writeframes(pcm_bytes)


def _mp3_bitrate(output_format: str) -> str | None:
    parts = output_format.split("_")
    if parts[0] != "mp3" or len(parts) < 3 or not parts[2].isdigit():
        return None
    return f"{parts[2]}k"


def _concat_mp3(path: Path, segments: list[bytes],
                bitrate: str | None = None) -> list[float]:
    try:
        import io

        from pydub import AudioSegment
    except ImportError as e:
        raise ImportError(
            "Dialogue longer than max-chars in MP3: stitching needs pydub + "
            "ffmpeg. Use --output-format pcm_44100 (dependency-free stitching) "
            "or install pydub."
        ) from e
    combined = None
    durations: list[float] = []
    for seg in segments:
        a = AudioSegment.from_file(io.BytesIO(seg), format="mp3")
        durations.append(len(a) / 1000.0)
        combined = a if combined is None else combined + a
    if combined is not None:
        kwargs = {"bitrate": bitrate} if bitrate else {}
        combined.export(str(path), format="mp3", **kwargs)
    return durations


def synthesize_task(
    task: Task,
    client: Any,
    dialogue_input_cls: Any,
    cfg: SynthConfig,
    sr: int,
) -> None:
    chunks = plan_chunks(task.inputs, cfg.max_chars)

    for inp in task.inputs:
        if len(inp["text"]) > cfg.max_chars:
            log.warning("[%s] a turn exceeds %d chars (%d).",
                        task.scenario_id, cfg.max_chars, len(inp["text"]))

    audio_parts: list[bytes] = []
    all_segments: list[dict] = []
    align_chars: list[str] = []
    align_starts: list[float] = []
    align_ends: list[float] = []

    pcm = cfg.output_format.startswith("pcm_")
    gap_bytes = (b"\x00" * (PCM_SAMPWIDTH * PCM_CHANNELS
                            * int(sr * cfg.chunk_gap_ms / 1000))
                 if (pcm and cfg.chunk_gap_ms) else b"")

    time_offset = 0.0
    input_offset = 0
    mp3_segments: list[bytes] = []

    for ci, (s, e) in enumerate(chunks):
        seg_inputs = task.inputs[s:e]
        audio, segs, align = _convert_chunk(
            client, dialogue_input_cls, seg_inputs, cfg, task.seed,
            task.scenario_id)
        audio_parts.append(audio)
        if not pcm:
            mp3_segments.append(audio)

        if cfg.timestamps and segs is not None:
            for seg in segs:
                if seg["start_time_seconds"] is not None:
                    seg["start_time_seconds"] += time_offset
                if seg["end_time_seconds"] is not None:
                    seg["end_time_seconds"] += time_offset
                if seg["dialogue_input_index"] is not None:
                    seg["dialogue_input_index"] += input_offset
                all_segments.append(seg)
            if cfg.save_alignment and align:
                chars = align.get("characters") or []
                st = align.get("character_start_times_seconds") or []
                en = align.get("character_end_times_seconds") or []
                align_chars.extend(chars)
                align_starts.extend([t + time_offset for t in st])
                align_ends.extend([t + time_offset for t in en])

        if pcm:
            time_offset += _pcm_duration_s(audio, sr)
            if ci < len(chunks) - 1 and gap_bytes:
                time_offset += cfg.chunk_gap_ms / 1000.0
        input_offset += (e - s)

    if pcm:
        _write_wav(task.output_audio, gap_bytes.join(audio_parts), sr)
    else:
        if len(mp3_segments) == 1:
            task.output_audio.write_bytes(mp3_segments[0])
        else:
            durations = _concat_mp3(task.output_audio, mp3_segments,
                                    _mp3_bitrate(cfg.output_format))
            if cfg.timestamps and all_segments:
                offs, acc = [], 0.0
                for d in durations:
                    offs.append(acc)
                    acc += d
                seg_i = 0
                for ci, (s, e) in enumerate(chunks):
                    n_in_chunk = e - s
                    for _ in range(n_in_chunk):
                        if seg_i < len(all_segments):
                            sg = all_segments[seg_i]
                            if sg["start_time_seconds"] is not None:
                                sg["start_time_seconds"] += offs[ci]
                            if sg["end_time_seconds"] is not None:
                                sg["end_time_seconds"] += offs[ci]
                            seg_i += 1

    written = task.output_audio.stat().st_size
    min_size = 45 if pcm else 1
    if written < min_size:
        raise RuntimeError(
            f"[{task.scenario_id}] audio file written empty or unreadable "
            f"({written} bytes): {task.output_audio}."
        )

    if cfg.timestamps and all_segments:
        end_by_input: dict[int, float] = {}
        start_by_input: dict[int, float] = {}
        for seg in all_segments:
            di = seg["dialogue_input_index"]
            if di is None:
                continue
            if seg["end_time_seconds"] is not None:
                end_by_input[di] = max(end_by_input.get(di, 0.0),
                                       seg["end_time_seconds"])
            if seg["start_time_seconds"] is not None:
                start_by_input[di] = min(start_by_input.get(di, float("inf")),
                                         seg["start_time_seconds"])
        for tm in task.turns_meta:
            di = tm["turn_idx"]
            tm["start_time_seconds"] = start_by_input.get(di)
            tm["end_time_seconds"] = end_by_input.get(di)
        for ev in task.events_for_mixing:
            t = ev["after_turn_idx"]
            ev["insert_at_seconds"] = 0.0 if t < 0 else end_by_input.get(t)

    meta = {
        "scenario_id": task.scenario_id,
        "engine": "elevenlabs",
        "model_id": cfg.model_id,
        "output_format": cfg.output_format,
        "language_code": cfg.language_code,
        "seed": task.seed,
        "settings": cfg.settings,
        "voices": task.voices,
        "n_chunks": len(chunks),
        "dialogue_inputs": task.inputs,
        "turns_meta": task.turns_meta,
        "events_rendered_as_tags": task.events_as_tags,
        "events_for_mixing": task.events_for_mixing,
        "script": task.script_text,
    }
    if cfg.timestamps:
        meta["voice_segments"] = all_segments
    if cfg.save_alignment and align_chars:
        meta["alignment"] = {
            "characters": align_chars,
            "character_start_times_seconds": align_starts,
            "character_end_times_seconds": align_ends,
        }
    save_json(task.output_meta, meta)


def write_dry_run_meta(task: Task, cfg: SynthConfig) -> None:
    chunks = plan_chunks(task.inputs, cfg.max_chars)
    total_chars = sum(len(i["text"]) for i in task.inputs)
    meta = {
        "scenario_id": task.scenario_id,
        "engine": "elevenlabs",
        "dry_run": True,
        "model_id": cfg.model_id,
        "output_format": cfg.output_format,
        "voices": task.voices,
        "seed": task.seed,
        "n_inputs": len(task.inputs),
        "total_chars": total_chars,
        "n_chunks": len(chunks),
        "chunk_ranges": [list(c) for c in chunks],
        "dialogue_inputs": task.inputs,
        "turns_meta": task.turns_meta,
        "events_rendered_as_tags": task.events_as_tags,
        "events_for_mixing": task.events_for_mixing,
        "script": task.script_text,
    }
    save_json(task.output_meta, meta)


def _parse_sample_rate(output_format: str) -> int:
    for part in output_format.split("_")[1:]:
        if part.isdigit():
            return int(part)
    raise ValueError(
        f"Cannot derive the sample rate from --output-format "
        f"'{output_format}'.")


def _make_client(api_key: str) -> tuple[Any, Any]:
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise ImportError(
            "This stage needs `pip install 'cares[tts]'` (elevenlabs SDK)."
        ) from e
    try:
        from elevenlabs import DialogueInput
    except Exception:  # noqa: BLE001
        DialogueInput = None
    return ElevenLabs(api_key=api_key), DialogueInput


def _synthesize_one(
    task: Task,
    client: Any,
    dialogue_input_cls: Any,
    cfg: SynthConfig,
    sr: int,
) -> bool:
    started = time.time()
    try:
        synthesize_task(task, client, dialogue_input_cls, cfg, sr)
    except Exception:  # noqa: BLE001
        log.error("[%s] FAILED\n%s", task.scenario_id, traceback.format_exc())
        return False
    log.info("[%s] OK in %.1fs", task.scenario_id, time.time() - started)
    return True


def _synthesize_all(
    tasks: list[Task],
    client: Any,
    dialogue_input_cls: Any,
    cfg: SynthConfig,
    sr: int,
    concurrency: int,
) -> tuple[int, int]:
    workers = max(1, concurrency)

    async def run_all() -> list:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="cares-tts") as pool:
            async def worker(task: Task) -> bool:
                return await loop.run_in_executor(
                    pool, _synthesize_one, task, client, dialogue_input_cls, cfg, sr)

            return await run_pool(tasks, worker, concurrency=workers, desc="Synthesis")

    results = run_async(run_all())
    n_ok = sum(1 for r in results if r)
    return n_ok, len(tasks) - n_ok


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=None,
                        help="dialogues.json (default: output of the dialogues stage)")
    parser.add_argument("--mode", choices=["grounded", "free"], default="grounded",
                        help="Input dialogue set when --input is absent.")
    parser.add_argument("--voices-dir", type=Path, default=None,
                        help="Output directory (default: <data-dir>/out_voices)")

    parser.add_argument("--voices-file", type=Path, default=None,
                        help="Voice pool with genders (default: "
                             "cares/resources/voices.json).")
    parser.add_argument("--voices", default=None,
                        help="Comma-separated voice_id pool. 2 distinct voices "
                             "are drawn per scenario, deterministically.")
    parser.add_argument("--voice-a", default=None,
                        help="voice_id of speaker A (fixed voice)")
    parser.add_argument("--voice-b", default=None,
                        help="voice_id of speaker B (fixed voice)")

    parser.add_argument("--model-id", default="eleven_v3")
    parser.add_argument("--output-format", default="mp3_44100_128",
                        help="pcm_44100 (-> .wav, dependency-free stitching) or "
                             "mp3_44100_128 (-> .mp3). PCM/WAV 44.1k requires the "
                             "Pro tier.")
    parser.add_argument("--language-code", default=None, help="ISO 639-1, e.g. 'en'")
    parser.add_argument("--apply-text-normalization", default=None,
                        choices=["auto", "on", "off"])
    parser.add_argument("--stability", type=float, default=None,
                        help="v3 setting (~0.0 Creative, 0.5 Natural, 1.0 Robust). "
                             "Not sent when absent.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global seed (0..4294967295) for reproducibility.")
    parser.add_argument("--seed-from-scenario", action="store_true",
                        help="Without --seed, derive one seed per scenario "
                             "(reproducible).")

    parser.add_argument("--timestamps", dest="timestamps", action="store_true",
                        default=True,
                        help="Use the with-timestamps endpoint (default) -> per-turn "
                             "timing + mixing instant of the ambient events.")
    parser.add_argument("--no-timestamps", dest="timestamps", action="store_false")
    parser.add_argument("--save-alignment", action="store_true",
                        help="Also save the character-by-character alignment "
                             "(bulky).")

    parser.add_argument("--perform-sfx", action="store_true",
                        help="Also render gunshot/explosion/applause as v3 tags "
                             "(these sounds are left to the mixing by default).")
    parser.add_argument("--no-perform-events", action="store_true",
                        help="Convert NO event into a tag: every event is left "
                             "to the mixing.")
    parser.add_argument("--event-tags", type=Path, default=None,
                        help="JSON {event_id: '[tag]'} extending or overriding the "
                             "cares.event_tags registry.")

    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                        help="Limit of cumulated characters per request.")
    parser.add_argument("--chunk-gap-ms", type=int, default=0,
                        help="Silence inserted between stitched chunks (PCM only).")

    parser.add_argument("--concurrency", type=int, default=3,
                        help="Parallel calls = concurrency limit of your "
                             "ElevenLabs tier.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--api-key", default=None,
                        help="Otherwise the ELEVENLABS_API_KEY env variable.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-split", default=None,
                        help="Synthesize these splits only, e.g. 'test'.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not call the API: only write the planned meta "
                             "(inputs, tags, chunking). 0 credit.")


def run(args: argparse.Namespace) -> int:
    paths = Paths.resolve(args.data_dir)
    input_path = args.input or paths.dialogues(args.mode)
    voices_dir = args.voices_dir or paths.voices_dir
    voices_dir.mkdir(parents=True, exist_ok=True)

    extra_tags = None
    if args.event_tags:
        extra_tags = load_json(args.event_tags)
        if not isinstance(extra_tags, dict):
            log.error("--event-tags unreadable or absent: %s", args.event_tags)
            return 2
        log.info("%d tag overrides loaded.", len(extra_tags))

    dialogues = load_scenarios(input_path)
    if dialogues is None:
        log.error("Dialogue file not found: %s", input_path)
        return 2
    if not dialogues:
        log.info("No dialogue in %s: nothing to do.", input_path)
        return 0
    if args.only_split:
        before = len(dialogues)
        dialogues = restrict_to_splits(dialogues, args.only_split)
        log.info("--only-split %s: %d dialogues out of %d", args.only_split,
                 len(dialogues), before)
        if not dialogues:
            log.error("No dialogue in that split or splits.")
            return 2
    if args.limit:
        dialogues = dialogues[: args.limit]
    log.info("%d dialogues loaded from %s.", len(dialogues), input_path)

    cfg = SynthConfig(
        model_id=args.model_id,
        output_format=args.output_format,
        language_code=args.language_code,
        apply_text_normalization=args.apply_text_normalization,
        settings=({"stability": args.stability}
                  if args.stability is not None else None),
        seed=args.seed,
        timestamps=args.timestamps,
        save_alignment=args.save_alignment,
        max_chars=args.max_chars,
        chunk_gap_ms=args.chunk_gap_ms,
        max_retries=args.max_retries,
    )

    try:
        pick_voices = make_voice_picker(args.voices, args.voice_a, args.voice_b,
                                        args.voices_file)
        sample_rate = _parse_sample_rate(cfg.output_format)
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    tasks = build_tasks(
        dialogues, voices_dir, cfg,
        pick_voices=pick_voices,
        include_sfx=args.perform_sfx,
        extra_tags=extra_tags,
        seed_from_scenario=args.seed_from_scenario,
        overwrite=args.overwrite,
        perform_events=not args.no_perform_events,
    )

    if not tasks:
        log.info("Nothing to do.")
        return 0

    n_tags = sum(len(t.events_as_tags) for t in tasks)
    n_mix = sum(len(t.events_for_mixing) for t in tasks)
    log.info("%d dialogues to synthesize. Events -> v3 tags: %d ; "
             "to be mixed later: %d.", len(tasks), n_tags, n_mix)

    if args.dry_run:
        for task in tasks:
            write_dry_run_meta(task, cfg)
        log.info("Dry-run done: %d meta written in %s (no API call).",
                 len(tasks), voices_dir)
        return 0

    api_key = args.api_key or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        log.error("Missing API key: --api-key or ELEVENLABS_API_KEY.")
        return 2
    try:
        client, dialogue_input_cls = _make_client(api_key)
    except ImportError as exc:
        log.error("%s", exc)
        return 2

    n_ok, n_fail = _synthesize_all(tasks, client, dialogue_input_cls, cfg,
                                   sample_rate, args.concurrency)
    log.info("Done. ok=%d fail=%d", n_ok, n_fail)
    return 0 if n_fail == 0 else 1
