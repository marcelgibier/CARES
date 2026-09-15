from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..jsonio import load_json
from ..log import get_logger
from .dsp import (
    BACKGROUND_EDGE_MARGIN_S,
    LUFS_VOICE,
    apply_fade,
    apply_release,
    measure_lufs,
    salient_offset,
    trim_silence,
)
from .io import (
    TARGET_SR,
    load_audio_resampled,
    require,
    stable_seed,
    takes_for_event,
)
from .reverb import (
    distance_profile,
    event_target_lufs,
    process_event,
    sample_event_distance,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

log = get_logger(__name__)

TURN_SPEAKER_KEYS = ("speaker", "speaker_id", "spk", "voice", "voice_id",
                     "speaker_label")
TURN_START_KEYS = ("t_start_s", "start_s", "start", "t_start",
                   "start_time_seconds")
TURN_END_KEYS = ("t_end_s", "end_s", "end", "t_end", "end_time_seconds")

EVENT_INSERT_MARGIN_S = 0.15
EVENT_CUT_BACK_S = 0.05
EVENT_CUT_FWD_S = 0.25
EVENT_CUT_ENV_MS = 20.0
EVENT_SIL_FACTOR = 0.15

EVENT_BEHAVIORAL_PRECUT_S = 1.5

BEHAVIORAL_PRECUT_JITTER_S = 0.30

BEHAVIORAL_DUCK_DB = -4.0
BEHAVIORAL_DUCK_RAMP_S = 0.08

MAX_EVENT_OVER_VOICE_DB = 6.0

BEHAVIORAL_OVERLAP_MAX_S = 2.0
BEHAVIORAL_OVERLAP_MAX_JITTER_S = 2.5
BEHAVIORAL_OVERLAP_FADE_S = 0.40

EVENT_ONSET_WINDOW_S = 0.35

TURN_GAP_MIN_S = EVENT_INSERT_MARGIN_S + EVENT_ONSET_WINDOW_S
MAX_INTRA_TURN_PAUSE_S = 0.45

TURN_CUT_SEARCH_S = 0.30
TURN_GAP_MAX_S = 0.85

BEHAVIORAL_NATURE_KEYS = ("reaction", "nature", "event_nature", "event_type",
                          "kind", "role", "category", "class", "label",
                          "sound_type", "behavior", "type")
BEHAVIORAL_NATURE_VALUES = {"behavioral", "behavioural"}
BEHAVIORAL_ID_KEYS = ("event_id", "id", "sound_id")

BEHAVIORAL_SUBSTRING_FALLBACK = True
BEHAVIORAL_PREFIXES = ("behavioral/",)
BEHAVIORAL_SUBSTRINGS = (
    "cough", "sneeze", "sniff", "throat", "sigh", "breath", "yawn",
    "laugh", "chuckle", "hiccup", "gulp", "swallow", "clear_throat",
    "paper_rustling",
)

DUCK_DEPTH_DB = -6.0
DUCK_DURATION_S = 1.5
DUCK_ATTACK_S = 0.05
DUCK_RELEASE_S = 0.4


@dataclass(frozen=True)
class MixSettings:
    event_insert_margin_s: float = EVENT_INSERT_MARGIN_S
    behavioral_cut: bool = True
    behavioral_precut_s: float = EVENT_BEHAVIORAL_PRECUT_S
    speaker_balance: bool = True
    background_leveling: bool = True
    background_edge_margin_s: float = BACKGROUND_EDGE_MARGIN_S
    distance_model: bool = True
    take_trim: bool = True
    take_release: bool = True
    turn_gaps: bool = True
    turn_gap_min_s: float = TURN_GAP_MIN_S
    turn_gap_max_s: float = TURN_GAP_MAX_S
    event_overlap: bool = True
    event_onset_window_s: float = EVENT_ONSET_WINDOW_S
    behavioral_duck_db: float = BEHAVIORAL_DUCK_DB
    behavioral_overlap_max_s: float = BEHAVIORAL_OVERLAP_MAX_S
    max_intra_turn_pause_s: float = 0.0
    rescale_turns: bool = True
    reverb_wet_scale: float = 1.0
    muted_event_ranks: frozenset[int] = frozenset()
    bg_lowpass_hz: float = 0.0
    master_lowpass_hz: float = 16000.0
    bg_lufs_jitter_db: float = 3.0
    file_ratios: dict | None = None


SETTINGS = MixSettings()


def configure(settings: MixSettings) -> None:
    global SETTINGS
    SETTINGS = settings


def _get_turn_field(turn: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in turn and turn[k] is not None:
            return turn[k]
    return None


def turns_candidates(sid: str, voices_dir: Path,
                     alignments_dir: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if alignments_dir is not None:
        candidates.append(alignments_dir / f"{sid}.json")
    candidates.append(voices_dir / f"{sid}.meta.json")
    candidates.append(voices_dir / f"{sid}.json")
    return candidates


def _warn_if_untimed(turns: list, path: Path) -> None:
    if any(isinstance(t, dict)
           and (_get_turn_field(t, TURN_END_KEYS) is not None
                or _get_turn_field(t, TURN_START_KEYS) is not None)
           for t in turns):
        return
    log.warning("  %s : %d turn(s) with no timing at all -> events will be piled "
                "at the end of the scene and speaker balancing will have no "
                "effect (voices synthesised with --no-timestamps ?)",
                path, len(turns))


def load_turns_for_scenario(
    sid: str, voices_dir: Path, alignments_dir: Path | None = None,
) -> tuple[list[dict] | None, Path | None]:
    for path in turns_candidates(sid, voices_dir, alignments_dir):
        if not path.exists():
            continue
        try:
            data = load_json(path)
        except Exception as exc:
            log.warning("  unreadable turns %s : %s", path, exc)
            continue
        for key in ("turns", "turns_meta"):
            seq = data.get(key) if isinstance(data, dict) else None
            if isinstance(seq, list) and seq:
                _warn_if_untimed(seq, path)
                return seq, path
        if isinstance(data, list) and data:
            _warn_if_untimed(data, path)
            return data, path
    return None, None


def balance_speakers(voice_audio: np.ndarray, turns_aligned: list[dict],
                     sr: int = TARGET_SR) -> tuple[np.ndarray, bool]:
    if len(voice_audio) == 0 or not turns_aligned:
        return voice_audio, False

    np = require("numpy")
    n = len(voice_audio)
    spk_segments: dict[str, list[tuple[int, int]]] = {}
    found_speaker_field = False
    for turn in turns_aligned:
        spk = _get_turn_field(turn, TURN_SPEAKER_KEYS)
        if spk is None:
            continue
        found_speaker_field = True
        t0 = _get_turn_field(turn, TURN_START_KEYS)
        t1 = _get_turn_field(turn, TURN_END_KEYS)
        if t0 is None or t1 is None:
            continue
        s0 = max(0, int(float(t0) * sr))
        s1 = min(n, int(float(t1) * sr))
        if s1 > s0:
            spk_segments.setdefault(str(spk), []).append((s0, s1))

    if not found_speaker_field:
        log.warning("  balance_speakers : no 'speaker' field in the turns "
                    "-> balancing skipped (check TURN_SPEAKER_KEYS)")
        return voice_audio, False
    if len(spk_segments) < 2:
        return voice_audio, False

    def concat(segs):
        return np.concatenate([voice_audio[a:b] for a, b in segs])

    levels_lufs, use_lufs = {}, True
    for spk, segs in spk_segments.items():
        lufs = measure_lufs(concat(segs), sr)
        if not np.isfinite(lufs) or lufs <= -60.0:
            use_lufs = False
        levels_lufs[spk] = lufs
    if use_lufs:
        levels = levels_lufs
    else:
        levels = {}
        for spk, segs in spk_segments.items():
            c = concat(segs).astype(np.float64)
            levels[spk] = 20.0 * np.log10(np.sqrt(np.mean(c ** 2) + 1e-12) + 1e-12)

    target = float(np.mean(list(levels.values())))

    out = voice_audio.copy()
    applied = False
    ramp_n = max(1, int(0.005 * sr))
    for spk, segs in spk_segments.items():
        gain_db = float(np.clip(target - levels[spk], -12.0, 12.0))
        if abs(gain_db) < 0.1:
            continue
        gain = 10.0 ** (gain_db / 20.0)
        for a, b in segs:
            seg_len = b - a
            r = min(ramp_n, seg_len // 2)
            env = np.full(seg_len, gain, dtype=np.float32)
            if r > 0:
                ramp = np.linspace(1.0, gain, r, dtype=np.float32)
                env[:r], env[-r:] = ramp, ramp[::-1]
            out[a:b] = (out[a:b] * env).astype(np.float32)
            applied = True
    return out, applied


def apply_rare_event_ducking(foreground: np.ndarray,
                             events_log: list[dict],
                             sr: int = TARGET_SR) -> np.ndarray:
    rare_events = [e for e in events_log if e.get("layer") == "rare"]
    if not rare_events:
        return foreground

    np = require("numpy")
    n = len(foreground)
    gain = np.ones(n, dtype=np.float32)
    duck_lin = 10.0 ** (DUCK_DEPTH_DB / 20.0)

    for evt in rare_events:
        t_start = evt["end_s"]
        attack_n = max(1, int(DUCK_ATTACK_S * sr))
        release_n = max(1, int(DUCK_RELEASE_S * sr))
        full_n = max(1, int(DUCK_DURATION_S * sr))
        sustain_n = max(0, full_n - attack_n - release_n)

        start_sample = int(t_start * sr)
        if start_sample >= n:
            continue

        attack_env = np.linspace(1.0, duck_lin, attack_n, dtype=np.float32)
        sustain_env = np.full(sustain_n, duck_lin, dtype=np.float32)
        release_env = np.linspace(duck_lin, 1.0, release_n, dtype=np.float32)
        local = np.concatenate([attack_env, sustain_env, release_env])

        end_sample = min(start_sample + len(local), n)
        local = local[: end_sample - start_sample]

        gain[start_sample:end_sample] = np.minimum(
            gain[start_sample:end_sample], local
        )

    return (foreground * gain).astype(np.float32)


@dataclass
class EventPlacement:
    event_id: str
    nature_layer: str
    after_turn_idx: int | None
    is_behavioral: bool = False
    reaction: str | None = None


def is_behavioral_event(event_id: str) -> bool:
    eid = event_id.lower()
    if any(eid.startswith(p) for p in BEHAVIORAL_PREFIXES):
        return True
    return any(s in eid for s in BEHAVIORAL_SUBSTRINGS)


def _event_obj_is_behavioral(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    for k in BEHAVIORAL_NATURE_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip().lower() in BEHAVIORAL_NATURE_VALUES:
            return True
    for k in ("is_behavioral", "behavioral"):
        if d.get(k) is True:
            return True
    return False


def _event_obj_id(d: Any) -> str | None:
    if not isinstance(d, dict):
        return None
    for k in BEHAVIORAL_ID_KEYS:
        v = d.get(k)
        if v:
            return str(v)
    return None


def collect_behavioral_event_ids(dialogue_record: dict) -> set[str]:
    ids: set[str] = set()
    lists = [v for k, v in dialogue_record.items()
             if k.endswith("elements") and isinstance(v, list)]
    if isinstance(dialogue_record.get("events"), list):
        lists.append(dialogue_record["events"])
    for lst in lists:
        for e in lst:
            if _event_obj_is_behavioral(e):
                eid = _event_obj_id(e)
                if eid:
                    ids.add(eid)
    for it in dialogue_record.get("timeline", []):
        if (isinstance(it, dict) and it.get("type") == "event"
                and _event_obj_is_behavioral(it)):
            eid = it.get("event_id") or _event_obj_id(it)
            if eid:
                ids.add(str(eid))
    return ids


def compute_event_placements(
    timeline: list[dict],
    core_event_ids_set: set[str],
    supp_event_ids_set: set[str],
    rare_event_ids_set: set[str],
    behavioral_ids: set[str] | None = None,
    reaction_by_id: dict[str, str] | None = None,
) -> list[EventPlacement]:
    behavioral_ids = behavioral_ids or set()
    reaction_by_id = reaction_by_id or {}
    placements: list[EventPlacement] = []
    last_utterance_idx: int | None = None
    turn_idx_counter = -1

    for item in timeline:
        if item.get("type") == "utterance":
            turn_idx_counter += 1
            last_utterance_idx = turn_idx_counter
        elif item.get("type") == "event":
            eid = item["event_id"]
            if eid in rare_event_ids_set or eid.startswith("rare_events/"):
                layer = "rare"
            elif eid in core_event_ids_set:
                layer = "core"
            elif eid in supp_event_ids_set:
                layer = "supporting"
            else:
                layer = "core"
            is_beh = (eid in behavioral_ids) or _event_obj_is_behavioral(item)
            if not is_beh and BEHAVIORAL_SUBSTRING_FALLBACK:
                is_beh = is_behavioral_event(eid)
            placements.append(EventPlacement(
                event_id=eid,
                nature_layer=layer,
                after_turn_idx=last_utterance_idx,
                is_behavioral=is_beh,
                reaction=reaction_by_id.get(eid),
            ))
    return placements


def _duck_envelope(n: int, duck_n: int, depth_db: float) -> np.ndarray:
    np = require("numpy")
    env = np.ones(n, dtype=np.float32)
    if depth_db >= 0.0 or duck_n <= 0:
        return env
    gain = float(10.0 ** (depth_db / 20.0))
    ramp = max(1, min(int(BEHAVIORAL_DUCK_RAMP_S * TARGET_SR), duck_n // 2))
    hold = max(0, duck_n - 2 * ramp)
    shape = np.concatenate([
        np.linspace(1.0, gain, ramp, dtype=np.float32),
        np.full(hold, gain, dtype=np.float32),
        np.linspace(gain, 1.0, ramp, dtype=np.float32),
    ])[:n]
    env[:len(shape)] = shape
    return env


def applied_duck_db(requested_db: float, event_lufs: float | None,
                    voice_lufs: float = LUFS_VOICE,
                    max_over_db: float = MAX_EVENT_OVER_VOICE_DB) -> float:
    if requested_db >= 0.0 or event_lufs is None:
        return min(requested_db, 0.0)
    headroom = max_over_db - (event_lufs - voice_lufs)
    return max(requested_db, -max(headroom, 0.0))


def limit_voice_overlap(evt_audio: np.ndarray, max_n: int,
                        fade_n: int) -> tuple[np.ndarray, float]:
    np = require("numpy")
    if max_n <= 0 or len(evt_audio) <= max_n:
        return evt_audio, 0.0
    fade = max(1, min(fade_n, max_n))
    out = evt_audio[:max_n].astype(np.float32).copy()
    out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out, float(fade / TARGET_SR)


BREAK_MARKERS = ("...", "\u2026")


def break_resumption_offset_s(turn: dict) -> float | None:
    text = turn.get("text_clean")
    words = turn.get("words")
    start = _get_turn_field(turn, TURN_START_KEYS)
    if not text or not words or start is None:
        return None
    hits = [text.find(m) for m in BREAK_MARKERS if m in text]
    if not hits:
        return None
    pos = min(hits)
    for w in words:
        if len(w) >= 3 and w[2] > pos:
            return max(0.0, float(w[0]) - float(start))
    return None


PAUSE_GAP_MIN_S = 1.2


_ELLIPSIS = ("...", "\u2026")
_SENTENCE_CLOSE = ".!?"


def _closes_a_sentence(span: str) -> bool:
    for e in _ELLIPSIS:
        span = span.replace(e, " ")
    return any(c in span for c in _SENTENCE_CLOSE)


def pause_resumption_offset_s(turn: dict,
                              min_gap_s: float = PAUSE_GAP_MIN_S) -> float | None:
    words = turn.get("words")
    text = turn.get("text_clean") or ""
    start = _get_turn_field(turn, TURN_START_KEYS)
    if not words or start is None:
        return None
    for a, b in zip(words, words[1:], strict=False):
        if len(a) < 2 or len(b) < 1:
            continue
        if float(b[0]) - float(a[1]) < min_gap_s:
            continue
        if len(a) >= 3 and len(b) >= 3 and _closes_a_sentence(text[a[2]:b[2]]):
            continue
        return max(0.0, float(b[0]) - float(start))
    return None


def resume_offset_s(turn: dict) -> tuple[float, str] | None:
    found = [(o, name) for o, name in (
        (break_resumption_offset_s(turn), "break"),
        (pause_resumption_offset_s(turn), "pause"),
    ) if o is not None]
    return min(found) if found else None


def _same_speaker_across(turns: list[dict], after_idx: int | None) -> bool:
    if after_idx is None or after_idx + 1 >= len(turns):
        return True
    return turns[after_idx].get("speaker") == turns[after_idx + 1].get("speaker")


def _next_turn_start_n(turns: list[dict], after_idx: int | None,
                       n_voice: int) -> int | None:
    if after_idx is None or after_idx + 1 >= len(turns):
        return None
    start = _get_turn_field(turns[after_idx + 1], TURN_START_KEYS)
    if start is None:
        return None
    return max(0, min(int(float(start) * TARGET_SR), n_voice))


def _found_key(turn: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in turn and turn[k] is not None:
            return k
    return None


def _min_turn_gap_s(settings: MixSettings) -> float:
    floor = settings.turn_gap_min_s
    if settings.event_overlap:
        floor = max(floor, settings.event_insert_margin_s + settings.event_onset_window_s)
    return floor


def space_turns(voice_audio: np.ndarray, turns_aligned: list[dict],
                sid: str = "") -> tuple[np.ndarray, list[dict], list[float]]:
    np = require("numpy")
    settings = SETTINGS
    n_voice = len(voice_audio)
    if not settings.turn_gaps or n_voice == 0 or len(turns_aligned) < 2:
        return voice_audio, turns_aligned, []

    bounds: list[int] = []
    for turn in turns_aligned[:-1]:
        t_end = _get_turn_field(turn, TURN_END_KEYS)
        if t_end is None:
            return voice_audio, turns_aligned, []
        bounds.append(max(0, min(n_voice, int(float(t_end) * TARGET_SR))))
    if any(b < a for a, b in pairwise(bounds)):
        log.warning("  non-monotonic turn timings : gaps not inserted")
        return voice_audio, turns_aligned, []

    cuts = _quiet_cuts(voice_audio, bounds, n_voice)

    gap_rng = random.Random(stable_seed(f"{sid}|turn-gaps"))
    lo = _min_turn_gap_s(settings)
    hi = max(lo, settings.turn_gap_max_s)
    gaps_n = [max(0, int(gap_rng.uniform(lo, hi) * TARGET_SR)) for _ in cuts]

    pieces: list[np.ndarray] = []
    prev = 0
    for c, g in zip(cuts, gaps_n, strict=True):
        pieces.append(voice_audio[prev:c])
        pieces.append(np.zeros(g, dtype=np.float32))
        prev = c
    pieces.append(voice_audio[prev:])
    spaced = np.concatenate(pieces).astype(np.float32)

    shifts = [0]
    for g in gaps_n:
        shifts.append(shifts[-1] + g)
    remapped: list[dict] = []
    for i, turn in enumerate(turns_aligned):
        out = dict(turn)
        start = cuts[i - 1] + gaps_n[i - 1] + shifts[i - 1] if i > 0 else 0
        end = (cuts[i] + shifts[i]) if i < len(cuts) else (n_voice + shifts[-1])
        ks = _found_key(turn, TURN_START_KEYS)
        ke = _found_key(turn, TURN_END_KEYS)
        if ks is not None:
            out[ks] = start / TARGET_SR
        if ke is not None:
            out[ke] = end / TARGET_SR
        remapped.append(out)
    return spaced, remapped, [g / TARGET_SR for g in gaps_n]


def compress_long_pauses(voice_audio: np.ndarray, turns_aligned: list[dict],
                         max_pause_s: float) -> tuple[np.ndarray, list[dict], float]:
    np = require("numpy")
    ndimage = require("scipy.ndimage")
    n_voice = len(voice_audio)
    if max_pause_s <= 0 or n_voice == 0 or not turns_aligned:
        return voice_audio, turns_aligned, 0.0

    ew = max(1, int(EVENT_CUT_ENV_MS * 1e-3 * TARGET_SR))
    env = np.sqrt(ndimage.uniform_filter1d(
        voice_audio.astype(np.float64) ** 2, size=ew, mode="nearest") + 1e-12)
    thr = (float(np.percentile(env, 75)) + 1e-9) * EVENT_SIL_FACTOR
    keep_n = int(max_pause_s * TARGET_SR)
    edge = int(0.10 * TARGET_SR)

    drops: list[tuple[int, int]] = []
    for turn in turns_aligned:
        t0 = _get_turn_field(turn, TURN_START_KEYS)
        t1 = _get_turn_field(turn, TURN_END_KEYS)
        if t0 is None or t1 is None:
            continue
        s = max(0, int(float(t0) * TARGET_SR) + edge)
        e = min(n_voice, int(float(t1) * TARGET_SR) - edge)
        if e - s <= keep_n:
            continue
        quiet = env[s:e] < thr
        diff = np.diff(quiet.astype(np.int8))
        starts = list(np.flatnonzero(diff == 1) + 1)
        ends = list(np.flatnonzero(diff == -1) + 1)
        if quiet[0]:
            starts.insert(0, 0)
        if quiet[-1]:
            ends.append(len(quiet))
        for a, b in zip(starts, ends, strict=True):
            if b - a > keep_n:
                drops.append((s + a + keep_n, (b - a) - keep_n))
    if not drops:
        return voice_audio, turns_aligned, 0.0

    drops.sort()
    pieces: list[np.ndarray] = []
    prev = 0
    for pos, n in drops:
        pieces.append(voice_audio[prev:pos])
        prev = pos + n
    pieces.append(voice_audio[prev:])
    out = np.concatenate(pieces).astype(np.float32)

    def shifted(t: float) -> float:
        sample = int(t * TARGET_SR)
        removed = sum(n for pos, n in drops if pos + n <= sample)
        return max(0.0, (sample - removed) / TARGET_SR)

    remapped: list[dict] = []
    for turn in turns_aligned:
        new = dict(turn)
        for keys in (TURN_START_KEYS, TURN_END_KEYS):
            key = _found_key(turn, keys)
            if key is not None:
                new[key] = shifted(float(turn[key]))
        remapped.append(new)
    return out, remapped, sum(n for _, n in drops) / TARGET_SR


TURN_RESCALE_LIMITS = (0.85, 1.25)


def rescale_turns_to_audio(turns_aligned: list[dict],
                           n_voice: int) -> tuple[list[dict], float]:
    if n_voice <= 0 or len(turns_aligned) < 2:
        return turns_aligned, 1.0
    last = _get_turn_field(turns_aligned[-1], TURN_END_KEYS)
    if last is None or float(last) <= 0:
        return turns_aligned, 1.0
    scale = (n_voice / TARGET_SR) / float(last)
    lo, hi = TURN_RESCALE_LIMITS
    if not (lo <= scale <= hi):
        log.warning("  turn timings inconsistent with the audio (factor %.3f) : "
                    "not rescaled", scale)
        return turns_aligned, 1.0
    if abs(scale - 1.0) < 1e-4:
        return turns_aligned, 1.0
    out: list[dict] = []
    for turn in turns_aligned:
        new = dict(turn)
        for keys in (TURN_START_KEYS, TURN_END_KEYS):
            key = _found_key(turn, keys)
            if key is not None:
                new[key] = float(turn[key]) * scale
        out.append(new)
    return out, scale


def _quiet_cuts(voice_audio: np.ndarray, bounds: list[int], n_voice: int) -> list[int]:
    np = require("numpy")
    ndimage = require("scipy.ndimage")
    ew = max(1, int(EVENT_CUT_ENV_MS * 1e-3 * TARGET_SR))
    env = np.sqrt(ndimage.uniform_filter1d(
        voice_audio.astype(np.float64) ** 2, size=ew, mode="nearest") + 1e-12)
    back = max(0, int(TURN_CUT_SEARCH_S * TARGET_SR))
    cuts: list[int] = []
    prev = 0
    for b in bounds:
        lo = max(prev, b - back)
        hi = min(n_voice, b + back)
        cuts.append(int(lo + np.argmin(env[lo:hi])) if hi > lo else max(b, prev))
        prev = cuts[-1]
    return cuts


def build_voice_with_events(
    voice_audio: np.ndarray,
    turns_aligned: list[dict],
    placements: list[EventPlacement],
    events_root: Path,
    scene: str,
    rng: random.Random,
    enable_event_eq: bool = True,
    enable_event_reverb: bool = True,
    split: str | None = None,
    floor_lufs: float | None = None,
    sid: str = "",
) -> tuple[np.ndarray, list[dict], dict]:
    np = require("numpy")
    ndimage = require("scipy.ndimage")
    uniform_filter1d = ndimage.uniform_filter1d

    settings = SETTINGS
    turn_scale = 1.0
    resume_offsets = [resume_offset_s(t) for t in turns_aligned]
    if settings.rescale_turns:
        turns_aligned, turn_scale = rescale_turns_to_audio(turns_aligned, len(voice_audio))
    voice_audio, turns_aligned, pause_removed_s = compress_long_pauses(
        voice_audio, turns_aligned, settings.max_intra_turn_pause_s)
    voice_audio, turns_aligned, turn_gaps_s = space_turns(voice_audio, turns_aligned, sid)
    n_voice = len(voice_audio)
    if pause_removed_s > 0.0:
        resume_offsets = [None] * len(resume_offsets)
    elif turn_scale != 1.0:
        resume_offsets = [None if r is None else (r[0] * turn_scale, r[1])
                          for r in resume_offsets]

    pre_n = max(0, int(settings.event_insert_margin_s * TARGET_SR))
    post_n = pre_n
    w_back = max(0, int(EVENT_CUT_BACK_S * TARGET_SR))
    w_fwd = max(1, int(EVENT_CUT_FWD_S * TARGET_SR))

    if n_voice > 0:
        ew = max(1, int(EVENT_CUT_ENV_MS * 1e-3 * TARGET_SR))
        env = np.sqrt(uniform_filter1d(voice_audio.astype(np.float64) ** 2,
                                       size=ew, mode="nearest") + 1e-12)
        sil_thr = (float(np.percentile(env, 75)) + 1e-9) * EVENT_SIL_FACTOR
    else:
        env = np.zeros(0, dtype=np.float64)
        sil_thr = 0.0

    items: list[tuple[int, bool, np.ndarray, EventPlacement, str, float | None]] = []
    for pl in placements:
        is_start = pl.after_turn_idx is None
        if is_start:
            b = 0
        elif pl.after_turn_idx >= len(turns_aligned):
            b = n_voice
        else:
            t_end = _get_turn_field(turns_aligned[pl.after_turn_idx],
                                    TURN_END_KEYS)
            b = int(float(t_end) * TARGET_SR) if t_end is not None else n_voice
        b = max(0, min(b, n_voice))

        candidates = takes_for_event(events_root, pl.event_id, split)
        if not candidates:
            log.warning("  event with no source file for split %s : %s",
                        split or "(none)", pl.event_id)
            continue
        src = rng.choice(candidates)
        try:
            evt_raw = load_audio_resampled(src)
        except Exception as exc:
            log.warning("  failed to load %s : %s", src, exc)
            continue
        if len(evt_raw) == 0:
            continue
        if settings.take_trim:
            evt_raw = trim_silence(evt_raw)
            if len(evt_raw) == 0:
                continue
        release_s = 0.0
        if settings.take_release:
            evt_raw, release_s = apply_release(evt_raw)
        distance_m = reference_m = None
        if settings.distance_model:
            distance_m = sample_event_distance(
                pl.event_id, pl.reaction, pl.is_behavioral, rng, scene=scene)
            reference_m = distance_profile(
                pl.event_id, scene, pl.is_behavioral).reference
        target_lufs = event_target_lufs(pl.nature_layer, distance_m, reference_m,
                                        floor_lufs)
        evt_audio = process_event(
            evt_raw, pl.nature_layer, scene, pl.event_id,
            enable_eq=enable_event_eq, enable_reverb=enable_event_reverb,
            split=split, distance_m=distance_m, reference_m=reference_m,
            target_lufs=target_lufs,
            wet_scale=settings.reverb_wet_scale,
        )
        evt_audio = apply_fade(evt_audio, fade_s=0.02)
        items.append((b, is_start, evt_audio, pl, str(src), distance_m, target_lufs,
                      release_s))

    items.sort(key=lambda x: x[0])

    chunks: list[np.ndarray] = []
    events_log: list[dict] = []
    overlays: list[tuple[int, np.ndarray]] = []
    inserts: list[tuple[int, int]] = []
    cursor = 0
    out_pos = 0
    behav_precut_n = max(1, int(settings.behavioral_precut_s * TARGET_SR))
    precut_rng = random.Random(stable_seed(f"{sid}|behavioral-precut"))
    jitter_n = max(0, int(BEHAVIORAL_PRECUT_JITTER_S * TARGET_SR))
    onset_n = max(0, int(settings.event_onset_window_s * TARGET_SR))
    overlap_max_n = max(0, int(settings.behavioral_overlap_max_s * TARGET_SR))
    overlap_jitter_n = max(0, int(BEHAVIORAL_OVERLAP_MAX_JITTER_S * TARGET_SR))
    overlap_fade_n = max(1, int(BEHAVIORAL_OVERLAP_FADE_S * TARGET_SR))
    max_gap_n = max(0, int(settings.turn_gap_max_s * TARGET_SR))

    for rank, (b, is_start, evt_audio, pl, src, distance_m, level_lufs,
               release_s) in enumerate(items):
        D = len(evt_audio)
        muted = rank in settings.muted_event_ranks

        behavioral = (settings.behavioral_cut and not is_start and n_voice > 0
                      and pl.is_behavioral
                      and (b - cursor) > 1)

        if behavioral:
            this_max_n = (overlap_max_n + precut_rng.randint(0, overlap_jitter_n)
                          if overlap_max_n and overlap_jitter_n else overlap_max_n)
            evt_audio, overlap_fade_s = limit_voice_overlap(
                evt_audio, this_max_n, overlap_fade_n)
            D = len(evt_audio)
            end_gap = precut_rng.randint(0, jitter_n) if jitter_n else 0
            precut_max = max(1, behav_precut_n
                             + precut_rng.randint(-jitter_n, jitter_n))
            nxt_start = _next_turn_start_n(turns_aligned, pl.after_turn_idx, n_voice)
            resume_n = None
            anchor_name = "turn_start"
            j = (pl.after_turn_idx + 1) if pl.after_turn_idx is not None else None
            if j is not None and j < len(turns_aligned) and j < len(resume_offsets) \
                    and resume_offsets[j] is not None:
                t0 = _get_turn_field(turns_aligned[j], TURN_START_KEYS)
                if t0 is not None:
                    offset, anchor_name = resume_offsets[j]
                    resume_n = int((float(t0) + offset) * TARGET_SR)
                    resume_n = max(0, min(resume_n, n_voice))
            if resume_n is None:
                resume_n = nxt_start
                anchor_name = "turn_start"
            on_break = resume_n is not None and (resume_n - end_gap - D) > cursor
            if on_break:
                a = resume_n - end_gap - D
                span = max(1, min(D, n_voice - a))
            else:
                anchor_name = "turn_end"
                span = min(D + end_gap, precut_max, b - cursor)
                a = b - span
            if a > cursor:
                chunks.append(voice_audio[cursor:a])
                out_pos += a - cursor
            duck_db = applied_duck_db(settings.behavioral_duck_db, level_lufs)
            tail = voice_audio[a:a + span].astype(np.float32).copy()
            duck_n = min(len(tail), D)
            if len(tail) > 1 and duck_n > 0:
                tail *= _duck_envelope(len(tail), duck_n, duck_db)
            precut = span
            cursor = a + len(tail)
            ev_start = out_pos
            chunks.append(tail)
            out_pos += len(tail)
            ev_end = ev_start + D
            clear_n = len(tail)
            if not muted:
                overlays.append((ev_start, evt_audio))
            events_log.append({
                "event_id": pl.event_id,
                "layer": pl.nature_layer,
                "reaction": pl.reaction,
                "distance_m": round(distance_m, 2) if distance_m is not None else None,
                "level_lufs": round(level_lufs, 2),
                "source_file": src,
                "start_s": float(ev_start / TARGET_SR),
                "end_s": float(ev_end / TARGET_SR),
                "duration_s": float(D / TARGET_SR),
                "release_fade_s": release_s,
                "overlap_fade_s": overlap_fade_s,
                "muted": muted,
                "overlapped_voice": True,
                "behavioral_cut": True,
                "anchor": anchor_name,
                "lead_s": float(((resume_n - a - D) if on_break
                                 else (b - a - D)) / TARGET_SR),
                "precut_s": float(precut / TARGET_SR),
                "voice_duck_db": float(duck_db),
                "clear_window_s": float(clear_n / TARGET_SR),
                "inserted_silence_s": 0.0,
                "reused_silence_s": 0.0,
            })
            continue

        needed = (pre_n + onset_n) if settings.event_overlap \
            else (pre_n + D + post_n)

        if is_start:
            cut = cursor
        elif b >= n_voice:
            cut = n_voice
        else:
            lo = max(cursor, b - w_back)
            hi = min(n_voice, b + w_fwd)
            cut = (lo + int(np.argmin(env[lo:hi]))) if hi > lo \
                else min(max(b, cursor), n_voice)

        if cut > cursor:
            chunks.append(voice_audio[cursor:cut])
            out_pos += cut - cursor
        cursor = cut

        gap_avail = 0
        jmax = min(n_voice, cut + needed + max_gap_n)
        j = cut
        while j < jmax and env[j] < sil_thr:
            gap_avail += 1
            j += 1
        inserted = max(0, needed - gap_avail)

        if inserted > 0:
            chunks.append(np.zeros(inserted, dtype=np.float32))
            out_pos += inserted
            inserts.append((cut, inserted))
        clear_n = max(0, inserted + gap_avail - pre_n)
        salient_n = (salient_offset(evt_audio, settings.event_onset_window_s)
                     if settings.event_overlap else 0)
        ev_start = max(0, out_pos + pre_n - salient_n)
        ev_end = ev_start + D
        if not muted:
            overlays.append((ev_start, evt_audio))

        events_log.append({
            "muted": muted,
            "event_id": pl.event_id,
            "layer": pl.nature_layer,
            "reaction": pl.reaction,
            "distance_m": round(distance_m, 2) if distance_m is not None else None,
            "level_lufs": round(level_lufs, 2),
            "source_file": src,
            "start_s": float(ev_start / TARGET_SR),
            "end_s": float(ev_end / TARGET_SR),
            "duration_s": float(D / TARGET_SR),
            "release_fade_s": release_s,
            "overlapped_voice": bool(D > clear_n),
            "behavioral_cut": False,
            "clear_window_s": float(clear_n / TARGET_SR),
            "salient_offset_s": float(salient_n / TARGET_SR),
            "inserted_silence_s": float(inserted / TARGET_SR),
            "reused_silence_s": float(min(gap_avail, needed) / TARGET_SR),
        })

    if cursor < n_voice:
        chunks.append(voice_audio[cursor:])

    foreground = (np.concatenate(chunks).astype(np.float32) if chunks
                  else np.zeros(0, dtype=np.float32))

    if overlays:
        end = max(start + len(evt) for start, evt in overlays)
        if end > len(foreground):
            foreground = np.concatenate(
                [foreground, np.zeros(end - len(foreground), dtype=np.float32)])
        for start, evt in overlays:
            foreground[start:start + len(evt)] += evt

    timeline = {
        "turn_time_scale": round(turn_scale, 5),
        "intra_turn_pause_removed_s": round(pause_removed_s, 3),
        "turn_gaps_s": [round(g, 3) for g in turn_gaps_s],
        "turns_placed": _placed_turns(turns_aligned, inserts),
    }
    return foreground, events_log, timeline


def _placed_turns(turns_aligned: list[dict], inserts: list[tuple[int, int]]) -> list[dict]:
    def shift_at(sample: int) -> int:
        return sum(n for pos, n in inserts if pos <= sample)

    out: list[dict] = []
    for turn in turns_aligned:
        start = _get_turn_field(turn, TURN_START_KEYS)
        end = _get_turn_field(turn, TURN_END_KEYS)
        if start is None or end is None:
            continue
        s0 = int(float(start) * TARGET_SR)
        e0 = int(float(end) * TARGET_SR)
        out.append({
            "speaker": _get_turn_field(turn, TURN_SPEAKER_KEYS),
            "start_s": round((s0 + shift_at(s0)) / TARGET_SR, 3),
            "end_s": round((e0 + shift_at(e0)) / TARGET_SR, 3),
        })
    return out
