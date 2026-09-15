from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from ..audio import placement
from ..audio.dsp import (
    BACKGROUND_EDGE_MARGIN_S,
    GLOBAL_FADE_S,
    LUFS_BACKGROUND,
    LUFS_EVENT_CORE,
    LUFS_EVENT_RARE,
    LUFS_EVENT_SUPP,
    LUFS_MASTER,
    LUFS_VOICE,
    MIN_EVENT_OVER_BACKGROUND_DB,
    apply_fade,
    extract_window_or_loop,
    level_background,
    limiter_gain,
    lowpass_filter,
    measure_lufs,
    normalize_lufs,
)
from ..audio.io import (
    TARGET_SR,
    ambiences_for_scene,
    find_voice_file,
    load_audio_resampled,
    pedalboard_available,
    pyroomacoustics_available,
    require,
    stable_seed,
)
from ..audio.placement import (
    BEHAVIORAL_DUCK_DB,
    EVENT_BEHAVIORAL_PRECUT_S,
    EVENT_INSERT_MARGIN_S,
    EVENT_ONSET_WINDOW_S,
    MixSettings,
    apply_rare_event_ducking,
    balance_speakers,
    build_voice_with_events,
    collect_behavioral_event_ids,
    compute_event_placements,
    load_turns_for_scenario,
    turns_candidates,
)
from ..audio.reverb import (
    DEFAULT_REVERB,
    DEFAULT_VOICE_WET,
    SCENE_REVERB_PARAMS,
    VOICE_WET_MIX_BY_SCENE,
    process_voice,
)
from ..config import Paths
from ..jsonio import load_json, save_json
from ..log import get_logger
from ..splits import SPLITS, parse_ratios, set_file_ratios

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

log = get_logger(__name__)


def background_target_lufs(sid: str) -> float:
    jitter_db = placement.SETTINGS.bg_lufs_jitter_db
    if not jitter_db:
        return LUFS_BACKGROUND
    jitter = random.Random(stable_seed(f"{sid}|bg-level"))
    return LUFS_BACKGROUND + jitter.uniform(-jitter_db, jitter_db)


def build_background_track(
    scene: str | None,
    target_len: int,
    backgrounds_root: Path,
    rng: random.Random,
    split: str | None = None,
    target_lufs: float = LUFS_BACKGROUND,
) -> tuple[np.ndarray | None, Path | None, float]:
    if not scene:
        return None, None, target_lufs
    candidates = ambiences_for_scene(backgrounds_root, scene, split)
    if not candidates:
        log.warning("  no ambience for scene '%s' (split %s)", scene, split or "-")
        return None, None, target_lufs
    src = rng.choice(candidates)
    try:
        bg = load_audio_resampled(src)
    except Exception as exc:
        log.warning("  failed to load background %s: %s", src, exc)
        return None, None, target_lufs
    if len(bg) == 0:
        return None, None, target_lufs
    settings = placement.SETTINGS
    bg = extract_window_or_loop(
        bg, target_len, rng,
        edge_margin_s=settings.background_edge_margin_s,
    )
    if settings.bg_lowpass_hz:
        bg = lowpass_filter(bg, settings.bg_lowpass_hz)
    if settings.background_leveling:
        bg = level_background(bg)
    bg = normalize_lufs(bg, target_lufs)
    return bg, src, target_lufs


SUPPORTING_LAYER_RATE = 0.35


def event_layers(dialogue_record: dict) -> tuple[set[str], set[str], set[str]]:
    core: set[str] = set()
    supp: set[str] = set()
    rare: set[str] = set()

    sid = str(dialogue_record.get("scenario_id") or "")
    layer_rng = random.Random(stable_seed(f"{sid}|event-layer"))
    for e in dialogue_record.get("events", []):
        eid = e.get("event_id")
        if not eid:
            continue
        if e.get("is_rare") or eid.startswith("rare_events/"):
            rare.add(eid)
        elif layer_rng.random() < SUPPORTING_LAYER_RATE:
            supp.add(eid)
        else:
            core.add(eid)

    known = core | supp | rare
    for e in dialogue_record.get("core_elements", []):
        eid = e.get("event_id") if e.get("nature") == "acoustic_event" else None
        if not eid or eid in known:
            continue
        (rare if eid.startswith("rare_events/") else core).add(eid)
    for e in dialogue_record.get("supporting_elements", []):
        eid = e.get("event_id") if e.get("nature") == "acoustic_event" else None
        if not eid or eid in known:
            continue
        supp.add(eid)
    return core, supp, rare


def events_rendered_by_voice(sid: str, voices_dir: Path) -> set[str]:
    path = voices_dir / f"{sid}.meta.json"
    try:
        meta = load_json(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("  [%s] unreadable tts sidecar (%s): %s", sid, path, exc)
        return set()
    rendered = meta.get("events_rendered_as_tags") if isinstance(meta, dict) else None
    if not isinstance(rendered, list):
        log.debug("  [%s] no list of voice-rendered events: "
                  "every timeline event will be placed", sid)
        return set()
    return {e["event_id"] for e in rendered
            if isinstance(e, dict) and e.get("event_id")}


def mix_one_scenario(
    dialogue_record: dict,
    voices_dir: Path,
    alignments_dir: Path | None,
    events_root: Path,
    backgrounds_root: Path,
    output_dir: Path,
    overwrite: bool,
    seed: int | None = None,
    enable_voice_processing: bool = True,
    enable_event_processing: bool = True,
    enable_ducking: bool = True,
    split: str | None = None,
    reference_master_lufs: float | None = None,
    reference_limiter_gain: float | None = None,
) -> dict:
    np = require("numpy")
    sid = dialogue_record["scenario_id"]
    out_wav = output_dir / f"{sid}.wav"
    out_manifest = output_dir / f"{sid}.manifest.json"

    if out_wav.exists() and out_manifest.exists() and not overwrite:
        return {"scenario_id": sid, "status": "skipped (exists)"}

    rng = random.Random(seed if seed is not None else stable_seed(sid))
    scene = dialogue_record.get("scene")
    if not scene:
        log.warning("  [%s] no scene: default reverb, no ambience", sid)

    turns_aligned, turns_src = load_turns_for_scenario(
        sid, voices_dir, alignments_dir
    )
    if not turns_aligned:
        candidates = turns_candidates(sid, voices_dir, alignments_dir)
        existing = [p for p in candidates if p.exists()]
        if not existing:
            looked = " | ".join(str(p) for p in candidates)
            status = f"error: turns JSON not found (looked in: {looked})"
        else:
            status = (f"error: turns JSON present but unusable "
                      f"('turns'/'turns_meta' list empty, unexpected key, or "
                      f"unreadable JSON - see warnings): {existing[0]}")
        return {"scenario_id": sid, "status": status}

    voice_path = find_voice_file(sid, voices_dir)
    if voice_path is None:
        return {"scenario_id": sid, "status": "error: voice missing"}
    voice_audio = load_audio_resampled(voice_path)
    if len(voice_audio) == 0:
        return {"scenario_id": sid, "status": "error: empty voice"}

    speaker_balanced = False
    if placement.SETTINGS.speaker_balance:
        voice_audio, speaker_balanced = balance_speakers(voice_audio, turns_aligned)

    if enable_voice_processing:
        voice_audio = process_voice(voice_audio, scene, split=split,
                                    wet_scale=placement.SETTINGS.reverb_wet_scale)
    else:
        voice_audio = normalize_lufs(voice_audio, LUFS_VOICE)

    core_eids, supp_eids, rare_eids = event_layers(dialogue_record)

    behavioral_ids = collect_behavioral_event_ids(dialogue_record)
    n_beh_meta = int(dialogue_record.get("metadata", {}).get("n_behavioral", 0) or 0)
    if n_beh_meta and not behavioral_ids:
        log.warning(
            "  %s: metadata n_behavioral=%d but no behavioral event recognised "
            "in the dialogue (unknown nature key?). Heuristic "
            "fallback=%s.",
            sid, n_beh_meta, placement.BEHAVIORAL_SUBSTRING_FALLBACK,
        )

    reaction_by_id = {e["event_id"]: e.get("reaction")
                      for e in dialogue_record.get("events", [])
                      if e.get("event_id")}
    timeline = dialogue_record["timeline"]
    voiced = events_rendered_by_voice(sid, voices_dir)
    if voiced:
        kept = [it for it in timeline
                if not (it.get("type") == "event" and it.get("event_id") in voiced)]
        log.info("  [%s] %d event(s) already played by the voice: not remixed",
                 sid, len(timeline) - len(kept))
        timeline = kept
    placements = compute_event_placements(
        timeline, core_eids, supp_eids, rare_eids,
        behavioral_ids, reaction_by_id=reaction_by_id,
    )

    bg_lufs = background_target_lufs(sid)
    event_floor_lufs = bg_lufs + MIN_EVENT_OVER_BACKGROUND_DB

    foreground, events_log, timeline = build_voice_with_events(
        voice_audio, turns_aligned, placements, events_root, scene, rng,
        split=split,
        enable_event_eq=enable_event_processing,
        enable_event_reverb=enable_event_processing,
        floor_lufs=event_floor_lufs,
        sid=sid,
    )
    if len(foreground) == 0:
        return {"scenario_id": sid, "status": "error: empty foreground"}

    duck_applied = False
    if enable_ducking and any(e.get("layer") == "rare" for e in events_log):
        foreground = apply_rare_event_ducking(foreground, events_log)
        duck_applied = True

    bg, bg_source, bg_lufs = build_background_track(
        scene, len(foreground), backgrounds_root, rng, split,
        target_lufs=bg_lufs)

    if bg is not None:
        if len(bg) > len(foreground):
            bg = bg[: len(foreground)]
        elif len(bg) < len(foreground):
            pad = np.zeros(len(foreground) - len(bg), dtype=np.float32)
            bg = np.concatenate([bg, pad])
        mix = foreground + bg
    else:
        mix = foreground.copy()

    if placement.SETTINGS.master_lowpass_hz:
        mix = lowpass_filter(mix, placement.SETTINGS.master_lowpass_hz)
    master_input_lufs = measure_lufs(mix)
    if placement.SETTINGS.muted_event_ranks and reference_master_lufs is not None:
        gain_db = LUFS_MASTER - reference_master_lufs
        mix = (mix * (10.0 ** (gain_db / 20.0))).astype(np.float32)
    else:
        mix = normalize_lufs(mix, LUFS_MASTER)
    mix = apply_fade(mix, fade_s=GLOBAL_FADE_S)
    gain = limiter_gain(mix, ceiling_db=-1.0)
    if placement.SETTINGS.muted_event_ranks and reference_limiter_gain is not None:
        gain = reference_limiter_gain
    mix = (mix * gain).astype(np.float32)

    sf = require("soundfile")
    output_dir.mkdir(parents=True, exist_ok=True)
    mix_int16 = np.clip(mix * 32767, -32768, 32767).astype(np.int16)
    sf.write(str(out_wav), mix_int16, TARGET_SR, subtype="PCM_16")

    settings = placement.SETTINGS
    manifest = {
        "scenario_id": sid,
        "scene": scene,
        "split": split,
        "audio_path": str(out_wav),
        "voice_source": str(voice_path),
        "turns_source": str(turns_src) if turns_src is not None else None,
        "duration_s": float(len(mix) / TARGET_SR),
        "sample_rate": TARGET_SR,
        "n_channels": 1,
        "bit_depth": 16,
        "events_placed": events_log,
        "turn_time_scale": timeline["turn_time_scale"],
        "intra_turn_pause_removed_s": timeline["intra_turn_pause_removed_s"],
        "turn_gaps_s": timeline["turn_gaps_s"],
        "turns_placed": timeline["turns_placed"],
        "has_background": bg is not None,
        "background_source": str(bg_source) if bg_source is not None else None,
        "levels_lufs": {
            "voice": LUFS_VOICE,
            "event_core": LUFS_EVENT_CORE,
            "event_supporting": LUFS_EVENT_SUPP,
            "event_rare": LUFS_EVENT_RARE,
            "background": LUFS_BACKGROUND,
            "background_applied": round(bg_lufs, 2) if bg is not None else None,
            "event_floor": round(event_floor_lufs, 2),
            "min_event_over_background_db": MIN_EVENT_OVER_BACKGROUND_DB,
            "master": LUFS_MASTER,
        },
        "processing": {
            "voice_processed": enable_voice_processing,
            "event_processed": enable_event_processing,
            "event_placement": "spacing",
            "voice_never_cut_midword": True,
            "events_overlap_voice": False,
            "event_insert_margin_s": settings.event_insert_margin_s,
            "behavioral_cut": settings.behavioral_cut,
            "behavioral_precut_s": (settings.behavioral_precut_s
                                    if settings.behavioral_cut else 0.0),
            "behavioral_event_ids": sorted(behavioral_ids),
            "ducking_applied": duck_applied,
            "speaker_balanced": speaker_balanced,
            "background_leveled": settings.background_leveling and bg is not None,
            "voice_wet_mix": (VOICE_WET_MIX_BY_SCENE.get(scene, DEFAULT_VOICE_WET)
                              * placement.SETTINGS.reverb_wet_scale
                              if enable_voice_processing else None),
            "master_input_lufs": (round(float(master_input_lufs), 3)
                                  if master_input_lufs is not None else None),
            "master_limiter_gain": round(float(gain), 6),
            "muted_event_ranks": sorted(placement.SETTINGS.muted_event_ranks) or None,
            "reverb_wet_scale": placement.SETTINGS.reverb_wet_scale,
            "distance_model": placement.SETTINGS.distance_model,
            "take_trim": placement.SETTINGS.take_trim,
            "take_release": placement.SETTINGS.take_release,
            "turn_gaps": placement.SETTINGS.turn_gaps,
            "event_overlap": placement.SETTINGS.event_overlap,
            "event_onset_window_s": (placement.SETTINGS.event_onset_window_s
                                     if placement.SETTINGS.event_overlap else None),
            "bg_lowpass_hz": placement.SETTINGS.bg_lowpass_hz or None,
            "master_lowpass_hz": placement.SETTINGS.master_lowpass_hz or None,
            "scene_reverb_rt60": SCENE_REVERB_PARAMS.get(
                scene, DEFAULT_REVERB)["rt60"],
            "pyroomacoustics": pyroomacoustics_available(),
            "pedalboard": pedalboard_available(),
        },
    }
    save_json(out_manifest, manifest)

    return {"scenario_id": sid, "status": "ok",
            "duration_s": manifest["duration_s"],
            "n_events": len(events_log)}


def _init_worker(settings: MixSettings) -> None:
    placement.configure(settings)
    if settings.file_ratios:
        set_file_ratios(settings.file_ratios)


def _check_pools(dialogues: list[dict], events_root: Path, backgrounds_root: Path,
                 ratios: dict[str, float] | None = None) -> int:
    from ..audio.io import list_files_for_background, list_files_for_event
    from ..splits import check_pool

    event_ids = sorted({e.get("event_id") for d in dialogues
                        for e in d.get("events", []) if e.get("event_id")})
    scenes = sorted({d.get("scene") for d in dialogues if d.get("scene")})

    problems: list[str] = []
    log.info("Checking pools: %d events, %d scenes",
             len(event_ids), len(scenes))

    for event_id in event_ids:
        ok, message = check_pool(list_files_for_event(events_root, event_id), event_id,
                                 ratios=ratios)
        (log.info if ok else log.error)("  %s", message)
        if not ok:
            problems.append(message)
    for scene in scenes:
        ok, message = check_pool(list_files_for_background(backgrounds_root, scene),
                                 f"ambience/{scene}", ratios=ratios)
        (log.info if ok else log.error)("  %s", message)
        if not ok:
            problems.append(message)

    if problems:
        log.error("%d pool(s) cannot be partitioned disjointly. Add takes, or "
                  "mix with --no-split-audio and accept the sharing.",
                  len(problems))
        return 1
    log.info("Every pool partitions without overlap.")
    return 0


def _audit_manifests(output_dir: Path) -> int:
    manifests = sorted(output_dir.glob("*.manifest.json"))
    if not manifests:
        log.error("No manifest in %s: nothing to audit.", output_dir)
        return 1

    users: dict[str, set[str]] = {}
    per_split: Counter = Counter()
    n_without_split = 0

    for path in manifests:
        manifest = load_json(path)
        if not isinstance(manifest, dict):
            continue
        split = manifest.get("split")
        if not split:
            n_without_split += 1
            continue
        per_split[split] += 1
        sources = [e.get("source_file") for e in manifest.get("events_placed", [])]
        sources.append(manifest.get("background_source"))
        for source in sources:
            if source:
                users.setdefault(source, set()).add(split)

    shared = {src: splits for src, splits in users.items() if len(splits) > 1}
    log.info("Audit of %d scenes: %s", len(manifests), dict(per_split))
    log.info("  %d distinct source files used", len(users))
    if n_without_split:
        log.warning("  %d scene(s) with no split in their manifest: not auditable.",
                    n_without_split)

    if shared:
        log.error("LEAK: %d source file(s) used by more than one split.", len(shared))
        for source, splits in sorted(shared.items())[:10]:
            log.error("  %s -> %s", source, ", ".join(sorted(splits)))
        return 1

    log.info("No take shared between splits: disjointness verified.")
    return 0 if not n_without_split else 1


def _reference_master(loudness_from: str | None,
                      sid: str | None) -> tuple[float | None, float | None]:
    if not loudness_from or not sid:
        return None, None
    manifest = load_json(Path(loudness_from) / f"{sid}.manifest.json")
    if not isinstance(manifest, dict):
        return None, None
    proc = manifest.get("processing") or {}
    def num(key):
        v = proc.get(key)
        return float(v) if isinstance(v, int | float) else None
    return num("master_input_lufs"), num("master_limiter_gain")


def _worker(args_tuple):
    (record, voices_dir, alignments_dir, events_root,
     backgrounds_root, output_dir, overwrite,
     enable_voice_processing, enable_event_processing, enable_ducking, split,
     loudness_from) = args_tuple
    try:
        return mix_one_scenario(
            record,
            Path(voices_dir),
            Path(alignments_dir) if alignments_dir else None,
            Path(events_root),
            Path(backgrounds_root),
            Path(output_dir),
            overwrite,
            enable_voice_processing=enable_voice_processing,
            enable_event_processing=enable_event_processing,
            enable_ducking=enable_ducking,
            split=split,
            **dict(zip(("reference_master_lufs", "reference_limiter_gain"),
                       _reference_master(loudness_from, record.get("scenario_id")),
                       strict=True)),
        )
    except Exception:
        return {"scenario_id": record.get("scenario_id", "?"),
                "status": "exception",
                "traceback": traceback.format_exc()}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dialogues", type=Path, default=None,
                        help="Dialogues to mix (default: dialogues of --data-dir).")
    parser.add_argument("--voices-dir", type=Path, default=None,
                        help="Voice tracks and timing sidecars "
                             "(default: out_voices of --data-dir).")
    parser.add_argument("--alignments-dir", type=Path, default=None,
                        help="Optional. Dia2 alignment directory ('turns' key, "
                             "t_end_s field). If absent, the JSON sidecar of "
                             "--voices-dir is read instead ('turns_meta' key).")
    parser.add_argument("--events-root", type=Path, default=None,
                        help="Root of the event recordings "
                             "(<root>/<event_id>/<take>).")
    parser.add_argument("--backgrounds-root", type=Path, default=None,
                        help="Root of the ambiences (<root>/<scene>/<take>).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory of the mixed scenes "
                             "(default: audio_scenes of --data-dir).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", default=None,
                        help="Force the split of EVERY dialogue processed. By default "
                             "each dialogue uses its own ('split' field), which allows "
                             "mixing the three splits in one pass.")
    parser.add_argument("--no-split-audio", action="store_true",
                        help="Do not partition takes, ambiences and RIRs by split. "
                             "Waveforms are then shared between train and test: "
                             "reserve this for monolithic datasets.")
    parser.add_argument("--audio-ratios", default=None,
                        help="Share of takes and ambiences per split, e.g. "
                             "'0.75/0.05/0.20'. Default: 0.8/0.1/0.1.")
    parser.add_argument("--audit-splits", action="store_true",
                        help="Re-read the manifests already produced, check that no take "
                             "was used in two splits, then stop.")
    parser.add_argument("--check-splits", action="store_true",
                        help="Check that every pool of takes and ambiences can be "
                             "partitioned without overlap, then stop.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-voice-processing", action="store_true",
                        help="Disable HPF/EQ/comp/reverb on the voice.")
    parser.add_argument("--no-event-processing", action="store_true",
                        help="Disable HPF/EQ/reverb on the events.")
    parser.add_argument("--no-ducking", action="store_true",
                        help="Disable ducking after rare events.")
    parser.add_argument("--event-margin-s", type=float,
                        default=EVENT_INSERT_MARGIN_S,
                        help="Silence added before AND after the event when the "
                             "voices are spread apart to make room for it.")
    parser.add_argument("--no-behavioral-cut", action="store_true",
                        help="Disable the cut: behavioral events (cough, sigh...) "
                             "are spaced cleanly like the others instead of "
                             "cutting the last word.")
    parser.add_argument("--behavioral-precut-s", type=float,
                        default=EVENT_BEHAVIORAL_PRECUT_S,
                        help="How long (s) before the end of the turn a behavioral "
                             "event starts (it overlaps/cuts the last word).")
    parser.add_argument("--no-background-leveling", action="store_true",
                        help="Disable the ambience leveler.")
    parser.add_argument("--no-speaker-balance", action="store_true",
                        help="Disable levelling the two speakers to each other.")
    parser.add_argument("--bg-edge-margin-s", type=float,
                        default=BACKGROUND_EDGE_MARGIN_S,
                        help="Margin (s) kept away from the edges of the ambience "
                             "file before extracting a random window.")
    parser.add_argument("--no-distance-model", action="store_true",
                        help="Historical fixed settings: every event at the same "
                             "level, same reverb, same spectrum, whatever its "
                             "assumed distance.")
    parser.add_argument("--no-take-trim", action="store_true",
                        help="Do not trim the head/tail silences of the takes "
                             "(reintroduces their noise pedestal).")
    parser.add_argument("--no-rescale-turns", action="store_true",
                        help="Do not rescale the turn timings onto the real audio "
                             "duration. The dialogue mode of the synthesiser "
                             "returns them on another time base.")
    parser.add_argument("--max-intra-turn-pause-s", type=float,
                        default=0.0,
                        help="Ceiling on silences INSIDE a turn (0 = leave them "
                             "alone). The synthesiser lays very long ones between "
                             "the sentences of one line. Default: 0 (disabled "
                             "while turn boundaries drift).")
    parser.add_argument("--mute-event-rank", type=int, default=None,
                        help="COUNTERFACTUAL: mix the scenes MUTING the event of this "
                             "chronological rank (0 = the first). Everything else in "
                             "the assembly is identical - placement, ducking, gaps, "
                             "levels - so only the presence of the sound tells the "
                             "pair apart. Write it to a separate --output-dir.")
    parser.add_argument("--loudness-from", default=None,
                        help="COUNTERFACTUAL: directory of the manifests of the "
                             "SOUNDED version. The master gain is reused as is instead "
                             "of being recomputed - without this, removing a sound "
                             "changes the loudness, hence the gain, hence the whole "
                             "file, and the pair is told apart at the global level.")
    parser.add_argument("--behavioral-overlap-max-s", type=float,
                        default=placement.BEHAVIORAL_OVERLAP_MAX_S,
                        help="Max duration (s) for which a behavioral event covers "
                             "speech; beyond it the event fades out. "
                             "0 = no bound. Default: "
                             f"{placement.BEHAVIORAL_OVERLAP_MAX_S}")
    parser.add_argument("--behavioral-duck-db", type=float,
                        default=BEHAVIORAL_DUCK_DB,
                        help="How far the voice dips under a behavioral event "
                             "(0 = it does not move). Default: "
                             f"{BEHAVIORAL_DUCK_DB}.")
    parser.add_argument("--reverb-wet-scale", type=float, default=1.0,
                        help="Factor on the reverb wet mixes, voice AND events. "
                             "0.5 dries everything one notch, 0 removes the reverb. "
                             "Default: 1.0 (the tables of cares.audio.reverb).")
    parser.add_argument("--no-turn-gaps", action="store_true",
                        help="Do not insert a gap between turns (the synthesiser "
                             "butts them together: the slightest silence in the "
                             "track would then point at an event).")
    parser.add_argument("--no-event-overlap", action="store_true",
                        help="Reserve a clear window of its whole duration for each "
                             "ordinary event instead of its onset alone. Old "
                             "behaviour: the silence then measures the event.")
    parser.add_argument("--event-onset-window-s", type=float,
                        default=EVENT_ONSET_WINDOW_S,
                        help="Clear time reserved for the onset of an ordinary "
                             f"event (default: {EVENT_ONSET_WINDOW_S}).")
    parser.add_argument("--no-take-release", action="store_true",
                        help="Do not add a release fade to the takes that stop at "
                             "full power (they then cut off dead, with only the "
                             "20 ms de-click).")
    parser.add_argument("--bg-lowpass-hz", type=float, default=0.0,
                        help="Lowpass on the ambience bed (0 = none; the old "
                             "behaviour was 8000).")
    parser.add_argument("--master-lowpass-hz", type=float, default=16000.0,
                        help="Common spectral ceiling of the final mix (0 = none).")
    parser.add_argument("--bg-lufs-jitter", type=float, default=3.0,
                        help="Per-scene +/- dB variation of the ambience level "
                             "(0 = constant SNR).")


def run(args: argparse.Namespace) -> int:
    paths = Paths.resolve(args.data_dir)
    dialogues_path = args.dialogues or paths.dialogues()
    voices_dir = args.voices_dir or paths.voices_dir
    output_dir = args.output_dir or paths.scenes_dir

    if args.audit_splits:
        return _audit_manifests(output_dir)

    missing = [name for name, value in (("--events-root", args.events_root),
                                        ("--backgrounds-root", args.backgrounds_root))
               if value is None]
    if missing:
        log.error("Options required to draw from the sound pools: %s.",
                  ", ".join(missing))
        return 2

    try:
        ratios = parse_ratios(args.audio_ratios)
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    settings = MixSettings(
        event_insert_margin_s=args.event_margin_s,
        behavioral_cut=not args.no_behavioral_cut,
        behavioral_precut_s=args.behavioral_precut_s,
        behavioral_overlap_max_s=args.behavioral_overlap_max_s,
        muted_event_ranks=(frozenset({args.mute_event_rank})
                           if args.mute_event_rank is not None else frozenset()),
        speaker_balance=not args.no_speaker_balance,
        background_leveling=not args.no_background_leveling,
        background_edge_margin_s=args.bg_edge_margin_s,
        distance_model=not args.no_distance_model,
        take_trim=not args.no_take_trim,
        take_release=not args.no_take_release,
        turn_gaps=not args.no_turn_gaps,
        event_overlap=not args.no_event_overlap,
        event_onset_window_s=args.event_onset_window_s,
        reverb_wet_scale=args.reverb_wet_scale,
        behavioral_duck_db=args.behavioral_duck_db,
        max_intra_turn_pause_s=args.max_intra_turn_pause_s,
        rescale_turns=not args.no_rescale_turns,
        bg_lowpass_hz=args.bg_lowpass_hz,
        master_lowpass_hz=args.master_lowpass_hz,
        bg_lufs_jitter_db=args.bg_lufs_jitter,
        file_ratios=ratios,
    )
    placement.configure(settings)
    if settings.file_ratios:
        set_file_ratios(settings.file_ratios)

    output_dir.mkdir(parents=True, exist_ok=True)

    dialogues = load_json(dialogues_path)
    if dialogues is None:
        log.error("Dialogues not found: %s", dialogues_path)
        return 1
    if isinstance(dialogues, dict):
        dialogues = list(dialogues.values())
    if args.limit:
        dialogues = dialogues[: args.limit]

    log.info("Dialogues to mix : %d", len(dialogues))
    log.info("Turns source     : %s",
             args.alignments_dir or "(JSON sidecar next to the voices)")
    log.info("Processing voice : %s", not args.no_voice_processing)
    log.info("Processing events: %s", not args.no_event_processing)
    log.info("Ducking rare evt : %s", not args.no_ducking)
    log.info("Speaker balance  : %s", settings.speaker_balance)
    log.info("BG leveling      : %s", settings.background_leveling)
    log.info("Event margin     : %ss before/after", settings.event_insert_margin_s)
    log.info("Behavioral cut   : %s (precut=%ss)",
             settings.behavioral_cut, settings.behavioral_precut_s)
    log.info("BG edge margin   : %ss", settings.background_edge_margin_s)
    log.info("Distance model   : %s | take trim: %s | take release: %s",
             settings.distance_model, settings.take_trim, settings.take_release)
    log.info("Turn gaps        : %s (%s-%s s) | event overlap: %s "
             "(onset %s s)",
             settings.turn_gaps, settings.turn_gap_min_s, settings.turn_gap_max_s,
             settings.event_overlap, settings.event_onset_window_s)
    log.info("Lowpass          : bed=%s Hz | master=%s Hz | bed jitter +/-%s dB",
             settings.bg_lowpass_hz or "none", settings.master_lowpass_hz or "none",
             settings.bg_lufs_jitter_db)
    log.info("pyroomacoustics  : %s", pyroomacoustics_available())
    log.info("pedalboard       : %s", pedalboard_available())

    def split_of(record: dict) -> str | None:
        if args.no_split_audio:
            return None
        return args.split or record.get("split")

    seen_splits = Counter(split_of(d) or "(none)" for d in dialogues)
    log.info("Audio splits     : %s", dict(seen_splits))
    log.info("Take ratios      : %s", "/".join(f"{ratios[s]:g}" for s in SPLITS))

    if args.no_split_audio:
        log.warning("Audio partition disabled: takes, ambiences and RIRs are "
                    "shared between train and test. Measurements on this dataset "
                    "no longer separate memorisation from generalisation.")
    else:
        orphans = [d["scenario_id"] for d in dialogues if not split_of(d)]
        if orphans:
            log.error("%d dialogue(s) with no split, including %s.",
                      len(orphans), orphans[:3])
            log.error("Without a split, mixing would draw from the WHOLE pool of "
                      "takes. Regenerate the scenarios with --split, force one with "
                      "--split train, or accept the sharing with --no-split-audio.")
            return 2

    if args.check_splits:
        return _check_pools(dialogues, args.events_root, args.backgrounds_root, ratios)

    work = [
        (d, str(voices_dir),
         str(args.alignments_dir) if args.alignments_dir is not None else "",
         str(args.events_root), str(args.backgrounds_root),
         str(output_dir), args.overwrite,
         not args.no_voice_processing,
         not args.no_event_processing,
         not args.no_ducking,
         split_of(d),
         args.loudness_from)
        for d in dialogues
    ]

    t_start = time.time()
    if args.workers <= 1:
        results = [_worker(w) for w in work]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers,
                      initializer=_init_worker,
                      initargs=(settings,)) as pool:
            results = []
            for i, res in enumerate(pool.imap_unordered(_worker, work)):
                results.append(res)
                done = i + 1
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(work) - done) / rate if rate > 0 else float("inf")
                if done % 10 == 0 or done == len(work):
                    log.info("Progress: %d/%d  rate=%.2f/s  eta=%.0fs",
                             done, len(work), rate, eta)

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_skipped = sum(1 for r in results if "skipped" in r.get("status", ""))
    n_err = sum(1 for r in results
                if r.get("status") not in ("ok",)
                and "skipped" not in r.get("status", ""))
    log.info("Done : ok=%d skipped=%d errors=%d", n_ok, n_skipped, n_err)

    summary_path = output_dir / "_mix_summary.json"
    save_json(summary_path, results)
    log.info("Summary: %s", summary_path)
    return 0
