from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..log import get_logger
from .dsp import (
    LUFS_EVENT_CORE,
    LUFS_EVENT_RARE,
    LUFS_EVENT_SUPP,
    LUFS_VOICE,
    TARGET_SR,
    compressor,
    highpass,
    lowpass_filter,
    normalize_lufs,
    peaking_eq,
)
from .io import pyroomacoustics_available, require, stable_seed

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

log = get_logger(__name__)

SCENE_REVERB_PARAMS: dict[str, dict] = {
    "cafe_restaurant":   {"rt60": 0.7,  "pre_delay": 0.015, "damping_hz": 5000,
                          "room_dim": [12, 8, 3.5], "absorption": 0.30},
    "office":            {"rt60": 0.4,  "pre_delay": 0.010, "damping_hz": 4500,
                          "room_dim": [6, 5, 2.8],  "absorption": 0.45},
    "shopping_mall":     {"rt60": 1.2,  "pre_delay": 0.020, "damping_hz": 5000,
                          "room_dim": [40, 20, 6],  "absorption": 0.25},

    "metro_station":     {"rt60": 2.2,  "pre_delay": 0.030, "damping_hz": 4500,
                          "room_dim": [50, 12, 5],  "absorption": 0.15},
    "train_station":     {"rt60": 2.5,  "pre_delay": 0.030, "damping_hz": 4500,
                          "room_dim": [80, 30, 12], "absorption": 0.15},
    "airport":           {"rt60": 1.8,  "pre_delay": 0.025, "damping_hz": 5000,
                          "room_dim": [60, 40, 10], "absorption": 0.20},
    "construction_site": {"rt60": 0.9,  "pre_delay": 0.015, "damping_hz": 5500,
                          "room_dim": [25, 25, 8],  "absorption": 0.25},

    "car":               {"rt60": 0.08, "pre_delay": 0.002, "damping_hz": 3500,
                          "room_dim": [2.5, 1.5, 1.3], "absorption": 0.70},

    "street_traffic":    {"rt60": 0.45, "pre_delay": 0.010, "damping_hz": 5500,
                          "room_dim": [30, 8, 8],   "absorption": 0.50},
    "public_park":       {"rt60": 0.20, "pre_delay": 0.005, "damping_hz": 6000,
                          "room_dim": [50, 50, 20], "absorption": 0.60},
    "beach":             {"rt60": 0.15, "pre_delay": 0.004, "damping_hz": 6000,
                          "room_dim": [80, 80, 30], "absorption": 0.70},
    "forest_path":       {"rt60": 0.15, "pre_delay": 0.005, "damping_hz": 4000,
                          "room_dim": [30, 30, 15], "absorption": 0.65},
    "nighttime_nature":  {"rt60": 0.20, "pre_delay": 0.005, "damping_hz": 5000,
                          "room_dim": [50, 50, 20], "absorption": 0.60},
}
DEFAULT_REVERB = {"rt60": 0.5, "pre_delay": 0.012, "damping_hz": 5000,
                  "room_dim": [10, 8, 3], "absorption": 0.35}

VOICE_WET_MIX_BY_SCENE: dict[str, float] = {
    "car": 0.02, "beach": 0.02, "forest_path": 0.02,
    "nighttime_nature": 0.03, "public_park": 0.03,
    "street_traffic": 0.05,
    "cafe_restaurant": 0.06, "airport": 0.06, "office": 0.06,
    "construction_site": 0.06,
    "shopping_mall": 0.08, "metro_station": 0.10, "train_station": 0.10,
}
DEFAULT_VOICE_WET = 0.06

EVENT_WET_MIX_BY_LAYER = {
    "core": 0.10,
    "supporting": 0.15,
    "rare": 0.06,
}

VOICE_HPF_HZ = 80
VOICE_EQ_MUD_HZ = 250
VOICE_EQ_MUD_GAIN_DB = -1.5
VOICE_EQ_MUD_Q = 0.7
VOICE_EQ_PRESENCE_HZ = 3000
VOICE_EQ_PRESENCE_GAIN_DB = 1.5
VOICE_EQ_PRESENCE_Q = 0.7
VOICE_COMP_THRESHOLD_DB = -18.0
VOICE_COMP_RATIO = 2.5
VOICE_COMP_ATTACK_MS = 10.0
VOICE_COMP_RELEASE_MS = 100.0

EVENT_HPF_HZ = 60
EVENT_EQ_MUD_HZ = 300
EVENT_EQ_MUD_GAIN_DB = -2.5
EVENT_EQ_MUD_Q = 0.7
EVENT_EQ_BRIGHT_HZ = 4000
EVENT_EQ_BRIGHT_GAIN_DB = 1.5
EVENT_EQ_BRIGHT_Q = 0.7

RARE_NO_HPF = {
    "rare_events/explosion",
    "rare_events/gunshot",
    "rare_events/thunder_clap",
}


@dataclass(frozen=True)
class DistanceProfile:
    near: float
    far: float
    reference: float

    def sample(self, rng: random.Random, bias: float = 1.0) -> float:
        span = (self.far - self.near) * max(0.0, min(1.0, bias))
        return rng.uniform(self.near, self.near + span)


DEFAULT_DISTANCE_PROFILE = DistanceProfile(1.0, 6.0, 2.0)

SPEAKER_DISTANCE_PROFILE = DistanceProfile(0.8, 1.5, 1.2)

RARE_DISTANCE_PROFILES: dict[str, DistanceProfile] = {
    "thunder_clap":    DistanceProfile(300.0, 3000.0, 800.0),
    "helicopter":      DistanceProfile(50.0, 400.0, 150.0),
    "explosion":       DistanceProfile(50.0, 600.0, 180.0),
    "firework_pop":    DistanceProfile(30.0, 400.0, 120.0),
    "siren_police":    DistanceProfile(30.0, 250.0, 90.0),
    "siren_ambulance": DistanceProfile(30.0, 250.0, 90.0),
    "gunshot":         DistanceProfile(20.0, 250.0, 70.0),
    "crash_vehicle":   DistanceProfile(15.0, 150.0, 45.0),
    "fire_alarm":      DistanceProfile(5.0, 40.0, 15.0),
    "scream":          DistanceProfile(3.0, 30.0, 9.0),
    "window_smashing": DistanceProfile(3.0, 25.0, 8.0),
    "animal_growling": DistanceProfile(2.0, 15.0, 5.0),
}

EVENT_DISTANCE_PROFILES: dict[str, DistanceProfile] = {
    "airport/aircraft_engine":        DistanceProfile(40.0, 400.0, 120.0),
    "beach/boat_motor_start":         DistanceProfile(20.0, 200.0, 60.0),
    "beach/drone_overhead":           DistanceProfile(10.0, 80.0, 25.0),
    "beach/seagull_cry":              DistanceProfile(5.0, 60.0, 15.0),
    "street_traffic/tram_bell":       DistanceProfile(5.0, 50.0, 15.0),
    "street_traffic/motorcycle_rev":  DistanceProfile(5.0, 60.0, 18.0),
    "street_traffic/tire_screech":    DistanceProfile(8.0, 80.0, 25.0),
    "street_traffic/car_horn":        DistanceProfile(5.0, 60.0, 18.0),
    "train_station/train_horn":       DistanceProfile(20.0, 250.0, 70.0),
    "train_station/brakes_squealing": DistanceProfile(10.0, 100.0, 30.0),
    "construction_site/jackhammer_burst": DistanceProfile(10.0, 100.0, 30.0),
    "rural/tractor_driving_by":       DistanceProfile(15.0, 150.0, 45.0),
    "rural/cow_moo":                  DistanceProfile(10.0, 120.0, 35.0),
    "rural/rooster_crow":             DistanceProfile(10.0, 120.0, 35.0),
    "nighttime_nature/wolf_howl":     DistanceProfile(50.0, 800.0, 200.0),
    "nighttime_nature/owl_hoot":      DistanceProfile(10.0, 150.0, 40.0),
    "public_park/child_shout":        DistanceProfile(5.0, 60.0, 18.0),
    "shared/dog_bark":                DistanceProfile(5.0, 80.0, 20.0),
}

EVENT_DISTANCE_PROFILES_BY_LEAF: dict[str, DistanceProfile] = {
    event_id.split("/", 1)[-1]: profile
    for event_id, profile in EVENT_DISTANCE_PROFILES.items()
}

OUTDOOR_SCENES = frozenset({
    "street_traffic", "public_park", "beach", "construction_site",
    "rural", "nighttime_nature", "forest_path",
})
OUTDOOR_DISTANCE_PROFILE = DistanceProfile(1.5, 20.0, 4.0)

DISTANCE_BIAS_BY_REACTION: dict[str, float] = {}

SPATIAL_REF_M = 2.0

DISTANCE_LEVEL_CLAMP_DB = (-9.0, 4.0)

DISTANCE_LOWPASS_FLOOR_HZ = 3500.0
DISTANCE_LOWPASS_EXPONENT = 0.6

DISTANCE_WET_CAP = 0.25


def distance_profile(event_id: str, scene: str | None = None,
                     is_behavioral: bool = False) -> DistanceProfile:
    if event_id in EVENT_DISTANCE_PROFILES:
        return EVENT_DISTANCE_PROFILES[event_id]
    namespace, _, name = event_id.partition("/")
    if namespace == "rare_events":
        return RARE_DISTANCE_PROFILES.get(name, DistanceProfile(5.0, 60.0, 15.0))
    leaf = name or namespace
    if leaf in EVENT_DISTANCE_PROFILES_BY_LEAF:
        return EVENT_DISTANCE_PROFILES_BY_LEAF[leaf]
    if (scene or namespace) in OUTDOOR_SCENES:
        return OUTDOOR_DISTANCE_PROFILE
    return DEFAULT_DISTANCE_PROFILE


def sample_event_distance(event_id: str, reaction: str | None, is_behavioral: bool,
                          rng: random.Random, scene: str | None = None) -> float:
    profile = distance_profile(event_id, scene, is_behavioral)
    bias = DISTANCE_BIAS_BY_REACTION.get(reaction or "", 1.0)
    return profile.sample(rng, bias)


def distance_level_offset_db(distance_m: float, reference_m: float = SPATIAL_REF_M) -> float:
    offset = -20.0 * math.log10(max(distance_m, 1e-3) / max(reference_m, 1e-3))
    lo, hi = DISTANCE_LEVEL_CLAMP_DB
    return min(hi, max(lo, offset))


def distance_wet_mix(base_wet: float, distance_m: float) -> float:
    if base_wet <= 0.0:
        return 0.0
    if base_wet >= 1.0:
        return DISTANCE_WET_CAP
    critical = SPATIAL_REF_M * math.sqrt(1.0 / base_wet - 1.0)
    wet = 1.0 / (1.0 + (critical / max(distance_m, 1e-3)) ** 2)
    return min(DISTANCE_WET_CAP, wet)


def distance_lowpass_hz(distance_m: float) -> float | None:
    if distance_m <= SPATIAL_REF_M:
        return None
    fc = 18000.0 * (SPATIAL_REF_M / distance_m) ** DISTANCE_LOWPASS_EXPONENT
    return max(DISTANCE_LOWPASS_FLOOR_HZ, fc)


_IR_CACHE: dict[str, np.ndarray] = {}


def _room_variation(scene: str, split: str | None) -> tuple[float, float, float]:
    if not split:
        return 1.0, 1.0, 1.0
    rng = random.Random(stable_seed(f"{scene}|{split}"))
    return (rng.uniform(0.90, 1.10), rng.uniform(0.90, 1.10), rng.uniform(0.75, 1.25))


def _generate_ir_pyroom(scene: str, sr: int = TARGET_SR,
                        split: str | None = None) -> np.ndarray:
    np = require("numpy")
    pra = require("pyroomacoustics")
    params = SCENE_REVERB_PARAMS.get(scene, DEFAULT_REVERB)
    fx, fy, f_mic = _room_variation(scene, split)
    dim = [params["room_dim"][0] * fx, params["room_dim"][1] * fy, params["room_dim"][2]]
    e_absorption = params["absorption"]

    try:
        room = pra.ShoeBox(
            dim, fs=sr,
            materials=pra.Material(e_absorption),
            max_order=12,
        )
        cx, cy, cz = dim[0] / 2, dim[1] / 2, min(1.6, dim[2] - 0.5)
        room.add_source([cx, cy, cz])
        mic_x = min(cx + 1.0 * f_mic, dim[0] - 0.2)
        room.add_microphone([mic_x, cy, cz])
        room.compute_rir()
        ir = room.rir[0][0].astype(np.float32)
        max_len = int(params["rt60"] * 1.5 * sr)
        if len(ir) > max_len:
            ir = ir[:max_len]
        peak = np.max(np.abs(ir))
        if peak > 0:
            ir = ir / peak
        return ir
    except Exception as exc:
        log.warning("pyroomacoustics failed on '%s': %s; falling back to synthetic",
                    scene, exc)
        return _generate_ir_synthetic(scene, sr, split)


def _generate_ir_synthetic(scene: str, sr: int = TARGET_SR,
                           split: str | None = None) -> np.ndarray:
    np = require("numpy")
    params = SCENE_REVERB_PARAMS.get(scene, DEFAULT_REVERB)
    rt60 = params["rt60"]
    pre_delay = params["pre_delay"]
    damping_hz = params["damping_hz"]

    n = max(1, int(rt60 * sr))
    rng = np.random.RandomState(stable_seed(f"{scene}|{split}" if split else scene))
    noise = rng.randn(n).astype(np.float32) * 0.3
    t = np.arange(n) / sr
    decay = np.exp(-6.91 * t / rt60).astype(np.float32)
    ir = noise * decay
    ir = lowpass_filter(ir, damping_hz, sr=sr)
    pre_n = max(0, int(pre_delay * sr))
    if pre_n > 0:
        ir = np.concatenate([np.zeros(pre_n, dtype=np.float32), ir])
    peak = np.max(np.abs(ir))
    if peak > 0:
        ir = ir / peak
    return ir.astype(np.float32)


def get_scene_ir(scene: str, sr: int = TARGET_SR, prefer_pyroom: bool = True,
                 split: str | None = None) -> np.ndarray:
    key = f"{scene}_{sr}_{split or ''}"
    if key in _IR_CACHE:
        return _IR_CACHE[key]
    if prefer_pyroom and pyroomacoustics_available():
        ir = _generate_ir_pyroom(scene, sr, split)
    else:
        ir = _generate_ir_synthetic(scene, sr, split)
    _IR_CACHE[key] = ir
    return ir


IR_ALIGN_MAX_DELAY_S = 0.05


def align_ir(ir: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    np = require("numpy")
    if ir is None or len(ir) == 0:
        return ir
    peak = int(np.argmax(np.abs(ir)))
    if peak <= 0 or peak > int(IR_ALIGN_MAX_DELAY_S * sr):
        return ir
    return ir[peak:]


def convolve_reverb(audio: np.ndarray, ir: np.ndarray,
                    wet_mix: float, allow_tail: bool = True,
                    sr: int = TARGET_SR) -> np.ndarray:
    if wet_mix <= 0.0 or ir is None or len(ir) == 0:
        return audio
    np = require("numpy")
    signal = require("scipy.signal")
    ir = align_ir(ir, sr=sr)
    wet = signal.oaconvolve(audio, ir, mode="full").astype(np.float32)
    wet_rms = np.sqrt(np.mean(wet ** 2) + 1e-12)
    dry_rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    if wet_rms > 0:
        wet = wet * (dry_rms / wet_rms)

    if not allow_tail:
        wet = wet[: len(audio)]
        return ((1.0 - wet_mix) * audio + wet_mix * wet).astype(np.float32)

    n_out = len(wet)
    dry_padded = np.zeros(n_out, dtype=np.float32)
    dry_padded[: len(audio)] = audio
    return ((1.0 - wet_mix) * dry_padded + wet_mix * wet).astype(np.float32)


def process_voice(voice: np.ndarray, scene: str,
                  enable_eq: bool = True,
                  enable_compressor: bool = True,
                  enable_reverb: bool = True,
                  sr: int = TARGET_SR,
                  split: str | None = None,
                  wet_scale: float = 1.0) -> np.ndarray:
    out = voice
    if enable_eq:
        out = highpass(out, VOICE_HPF_HZ, sr=sr)
        out = peaking_eq(out, VOICE_EQ_MUD_HZ,
                         VOICE_EQ_MUD_GAIN_DB, VOICE_EQ_MUD_Q, sr=sr)
        out = peaking_eq(out, VOICE_EQ_PRESENCE_HZ,
                         VOICE_EQ_PRESENCE_GAIN_DB, VOICE_EQ_PRESENCE_Q, sr=sr)
    if enable_compressor:
        out = compressor(out,
                         VOICE_COMP_THRESHOLD_DB, VOICE_COMP_RATIO,
                         VOICE_COMP_ATTACK_MS, VOICE_COMP_RELEASE_MS, sr=sr)
    if enable_reverb:
        ir = get_scene_ir(scene, sr=sr, split=split)
        wet = VOICE_WET_MIX_BY_SCENE.get(scene, DEFAULT_VOICE_WET)
        wet = max(0.0, min(1.0, wet * wet_scale))
        out = convolve_reverb(out, ir, wet_mix=wet, allow_tail=False)
    out = normalize_lufs(out, LUFS_VOICE, sr=sr)
    return out


def event_target_lufs(layer: str,
                      distance_m: float | None = None,
                      reference_m: float | None = None,
                      floor_lufs: float | None = None) -> float:
    target = {"rare": LUFS_EVENT_RARE,
              "core": LUFS_EVENT_CORE,
              "supporting": LUFS_EVENT_SUPP}.get(layer, LUFS_EVENT_CORE)
    if distance_m is not None:
        target += distance_level_offset_db(
            distance_m, reference_m if reference_m is not None else SPATIAL_REF_M)
    if floor_lufs is not None:
        target = max(target, floor_lufs)
    return target


def process_event(event_audio: np.ndarray, layer: str, scene: str,
                  event_id: str,
                  enable_eq: bool = True,
                  enable_reverb: bool = True,
                  sr: int = TARGET_SR,
                  split: str | None = None,
                  distance_m: float | None = None,
                  reference_m: float | None = None,
                  target_lufs: float | None = None,
                  wet_scale: float = 1.0) -> np.ndarray:
    out = event_audio
    if enable_eq and event_id not in RARE_NO_HPF:
        out = highpass(out, EVENT_HPF_HZ, sr=sr)
    if enable_eq:
        out = peaking_eq(out, EVENT_EQ_MUD_HZ,
                         EVENT_EQ_MUD_GAIN_DB, EVENT_EQ_MUD_Q, sr=sr)
        out = peaking_eq(out, EVENT_EQ_BRIGHT_HZ,
                         EVENT_EQ_BRIGHT_GAIN_DB, EVENT_EQ_BRIGHT_Q, sr=sr)
    if distance_m is not None:
        cutoff = distance_lowpass_hz(distance_m)
        if cutoff is not None:
            out = lowpass_filter(out, cutoff, sr=sr)
    if enable_reverb:
        ir = get_scene_ir(scene, sr=sr, split=split)
        base_wet = EVENT_WET_MIX_BY_LAYER.get(layer, 0.20)
        wet = (distance_wet_mix(base_wet, distance_m)
               if distance_m is not None else base_wet)
        wet = max(0.0, min(1.0, wet * wet_scale))
        out = convolve_reverb(out, ir, wet_mix=wet, allow_tail=True)
    if target_lufs is None:
        target_lufs = event_target_lufs(layer, distance_m, reference_m)
    out = normalize_lufs(out, target_lufs, sr=sr)
    return out
