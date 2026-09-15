from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .io import TARGET_SR, pedalboard_available, require

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

LUFS_VOICE = -20.0
LUFS_EVENT_CORE = -20.0
LUFS_EVENT_SUPP = -23.0
LUFS_EVENT_RARE = -17.0
LUFS_BACKGROUND = -32.0
LUFS_MASTER = -23.0

MIN_EVENT_OVER_BACKGROUND_DB = 6.0

LOOP_CROSSFADE_S = 0.2

GLOBAL_FADE_S = 0.5

RELEASE_PROBE_S = 0.050

RELEASE_THRESHOLD_DB = -12.0

RELEASE_FADE_S = 0.25

BACKGROUND_EDGE_MARGIN_S = 1.0

BACKGROUND_LEVEL_HEADROOM_DB = 6.0
BACKGROUND_LEVEL_RATIO = 6.0
BACKGROUND_LEVEL_WINDOW_MS = 80.0


def measure_lufs(audio: np.ndarray, sr: int = TARGET_SR) -> float:
    pyln = require("pyloudnorm")
    meter = pyln.Meter(sr)
    try:
        return meter.integrated_loudness(audio)
    except Exception:
        return -70.0


def normalize_lufs(audio: np.ndarray, target_lufs: float,
                   sr: int = TARGET_SR) -> np.ndarray:
    np = require("numpy")
    current = measure_lufs(audio, sr)
    if current <= -60.0 or not np.isfinite(current):
        return audio
    gain_db = target_lufs - current
    gain_lin = 10.0 ** (gain_db / 20.0)
    return (audio * gain_lin).astype(np.float32)


def apply_fade(audio: np.ndarray, fade_s: float = GLOBAL_FADE_S,
               sr: int = TARGET_SR) -> np.ndarray:
    np = require("numpy")
    n = int(fade_s * sr)
    if n <= 0 or 2 * n >= len(audio):
        return audio
    out = audio.copy()
    fade_curve = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    out[:n] *= fade_curve.astype(np.float32)
    out[-n:] *= fade_curve[::-1].astype(np.float32)
    return out


def salient_offset(audio: np.ndarray, window_s: float,
                   sr: int = TARGET_SR) -> int:
    np = require("numpy")
    n = int(window_s * sr)
    if n <= 0 or len(audio) <= n:
        return 0
    energy = np.cumsum(np.concatenate(
        [[0.0], audio.astype(np.float64) ** 2]))
    return int(np.argmax(energy[n:] - energy[:-n]))


def tail_level_db(audio: np.ndarray, sr: int = TARGET_SR,
                  probe_s: float = RELEASE_PROBE_S) -> float:
    np = require("numpy")
    ndimage = require("scipy.ndimage")
    n = int(probe_s * sr)
    if len(audio) < 2 * n or n <= 0:
        return float("-inf")
    window = max(1, int(0.010 * sr))
    envelope = np.sqrt(ndimage.uniform_filter1d(
        audio.astype(np.float64) ** 2, size=window, mode="nearest") + 1e-12)
    body = float(np.percentile(envelope, 90))
    tail = float(envelope[-n:].mean())
    if body <= 1e-5:
        return float("-inf")
    return 20.0 * float(np.log10(max(tail, 1e-12) / body))


def apply_release(audio: np.ndarray, sr: int = TARGET_SR,
                  threshold_db: float = RELEASE_THRESHOLD_DB,
                  fade_s: float = RELEASE_FADE_S) -> tuple[np.ndarray, float]:
    np = require("numpy")
    n = int(fade_s * sr)
    if n <= 0 or len(audio) < 2 * n:
        return audio, 0.0
    if tail_level_db(audio, sr=sr) < threshold_db:
        return audio, 0.0
    out = audio.copy()
    curve = 0.5 * (1 + np.cos(np.linspace(0, np.pi, n)))
    out[-n:] *= curve.astype(np.float32)
    return out, float(n / sr)


def trim_silence(audio: np.ndarray, sr: int = TARGET_SR,
                 threshold_db: float = -40.0, pad_ms: float = 30.0) -> np.ndarray:
    np = require("numpy")
    ndimage = require("scipy.ndimage")
    if len(audio) == 0:
        return audio
    window = max(1, int(0.010 * sr))
    envelope = np.sqrt(ndimage.uniform_filter1d(
        audio.astype(np.float64) ** 2, size=window, mode="nearest") + 1e-12)
    peak = float(envelope.max())
    if peak <= 0:
        return audio
    mask = envelope >= peak * (10.0 ** (threshold_db / 20.0))
    active = np.flatnonzero(mask)
    if len(active) == 0:
        return audio
    pad = int(pad_ms / 1000.0 * sr)
    start = max(0, int(active[0]) - pad)
    end = min(len(audio), int(active[-1]) + 1 + pad)
    return audio[start:end]


def loop_or_trim(audio: np.ndarray, target_len: int,
                 sr: int = TARGET_SR) -> np.ndarray:
    np = require("numpy")
    if len(audio) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(audio) >= target_len:
        return audio[:target_len].copy()

    n = len(audio)
    xfade_n = int(LOOP_CROSSFADE_S * sr)
    xfade_n = max(0, min(xfade_n, n // 2))

    out = np.zeros(target_len, dtype=np.float32)

    if xfade_n == 0:
        reps = target_len // n + 1
        tiled = np.tile(audio, reps)[:target_len]
        return tiled.astype(np.float32)

    fade_in = (0.5 * (1 - np.cos(np.linspace(0, np.pi, xfade_n)))).astype(np.float32)
    fade_out = fade_in[::-1]

    out[:n] = audio
    write_pos = n

    max_iters = target_len // max(1, (n - xfade_n)) + 4
    iters = 0
    while write_pos < target_len and iters < max_iters:
        iters += 1
        xs = write_pos - xfade_n
        if xs < 0:
            xs = 0
        xf_end = min(xs + xfade_n, target_len)
        xf_len = xf_end - xs
        if xf_len > 0:
            out[xs:xf_end] = (out[xs:xf_end] * fade_out[:xf_len]
                              + audio[:xf_len] * fade_in[:xf_len]).astype(np.float32)
        body_start = xf_end
        body_src_start = xf_len
        body_len = min(n - body_src_start, target_len - body_start)
        if body_len > 0:
            out[body_start:body_start + body_len] = audio[body_src_start:body_src_start + body_len]
            write_pos = body_start + body_len
        else:
            write_pos = xf_end
    return out


def extract_window_or_loop(audio: np.ndarray, target_len: int,
                           rng: random.Random,
                           edge_margin_s: float = BACKGROUND_EDGE_MARGIN_S,
                           sr: int = TARGET_SR) -> np.ndarray:
    if len(audio) <= target_len:
        return loop_or_trim(audio, target_len, sr=sr)

    margin = int(edge_margin_s * sr)
    earliest = margin
    latest = len(audio) - target_len - margin
    if latest <= earliest:
        latest_nomargin = len(audio) - target_len
        if latest_nomargin > 0:
            start = rng.randint(0, latest_nomargin)
        else:
            start = 0
    else:
        start = rng.randint(earliest, latest)
    return audio[start: start + target_len].copy()


def lowpass_filter(audio: np.ndarray, cutoff_hz: int,
                   sr: int = TARGET_SR) -> np.ndarray:
    np = require("numpy")
    signal = require("scipy.signal")
    nyq = sr / 2
    norm_cutoff = min(0.99, cutoff_hz / nyq)
    sos = signal.butter(4, norm_cutoff, btype="low", output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def limiter_gain(audio: np.ndarray, ceiling_db: float = -1.0) -> float:
    np = require("numpy")
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak == 0:
        return 1.0
    ceiling_lin = 10.0 ** (ceiling_db / 20.0)
    return ceiling_lin / peak if peak > ceiling_lin else 1.0


def hard_limiter(audio: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    return (audio * limiter_gain(audio, ceiling_db)).astype(audio.dtype)


def level_background(bg: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    if len(bg) == 0:
        return bg
    np = require("numpy")
    ndimage = require("scipy.ndimage")
    uniform_filter1d = ndimage.uniform_filter1d
    x = bg.astype(np.float64)
    win = max(1, int(BACKGROUND_LEVEL_WINDOW_MS * 1e-3 * sr))
    env = np.sqrt(uniform_filter1d(x ** 2, size=win, mode="nearest") + 1e-12)
    env_db = 20.0 * np.log10(env + 1e-12)
    ref_db = 20.0 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
    thresh_db = ref_db + BACKGROUND_LEVEL_HEADROOM_DB
    over = np.maximum(env_db - thresh_db, 0.0)
    gr_db = -over * (1.0 - 1.0 / BACKGROUND_LEVEL_RATIO)
    gr_db = uniform_filter1d(gr_db, size=max(1, int(0.030 * sr)), mode="nearest")
    gain = (10.0 ** (gr_db / 20.0)).astype(np.float32)
    return (bg * gain).astype(np.float32)


def _scipy_biquad(audio: np.ndarray, kind: str, freq: float,
                  gain_db: float = 0.0, q: float = 0.7,
                  sr: int = TARGET_SR) -> np.ndarray:
    np = require("numpy")
    signal = require("scipy.signal")
    nyq = sr / 2
    if kind == "highpass":
        order = 2
        norm = min(0.99, freq / nyq)
        sos = signal.butter(order, norm, btype="high", output="sos")
        return signal.sosfiltfilt(sos, audio).astype(np.float32)
    elif kind == "peak":
        norm = min(0.99, freq / nyq)
        b, a = signal.iirpeak(norm, q)
        sos = signal.tf2sos(b, a)
        peaked = signal.sosfiltfilt(sos, audio).astype(np.float32)
        gain = 10.0 ** (gain_db / 20.0) - 1.0
        return (audio + gain * peaked).astype(np.float32)
    else:
        raise ValueError(f"unknown kind: {kind}")


def highpass(audio: np.ndarray, freq: float, sr: int = TARGET_SR) -> np.ndarray:
    np = require("numpy")
    if pedalboard_available():
        from pedalboard import HighpassFilter, Pedalboard
        board = Pedalboard([HighpassFilter(cutoff_frequency_hz=freq)])
        return board(audio, sample_rate=sr).astype(np.float32)
    return _scipy_biquad(audio, "highpass", freq, sr=sr)


def peaking_eq(audio: np.ndarray, freq: float, gain_db: float, q: float = 0.7,
               sr: int = TARGET_SR) -> np.ndarray:
    np = require("numpy")
    if pedalboard_available():
        from pedalboard import PeakFilter, Pedalboard
        board = Pedalboard([PeakFilter(cutoff_frequency_hz=freq,
                                       gain_db=gain_db, q=q)])
        return board(audio, sample_rate=sr).astype(np.float32)
    return _scipy_biquad(audio, "peak", freq, gain_db, q, sr)


def compressor(audio: np.ndarray, threshold_db: float, ratio: float,
               attack_ms: float, release_ms: float,
               sr: int = TARGET_SR) -> np.ndarray:
    np = require("numpy")
    if pedalboard_available():
        from pedalboard import Compressor, Pedalboard
        board = Pedalboard([Compressor(threshold_db=threshold_db, ratio=ratio,
                                       attack_ms=attack_ms,
                                       release_ms=release_ms)])
        return board(audio, sample_rate=sr).astype(np.float32)

    eps = 1e-10
    abs_audio = np.abs(audio).astype(np.float64)
    alpha_a = float(np.exp(-1.0 / (attack_ms * 1e-3 * sr)))
    alpha_r = float(np.exp(-1.0 / (release_ms * 1e-3 * sr)))
    env = np.empty_like(abs_audio)
    e = 0.0
    for i in range(len(abs_audio)):
        if abs_audio[i] > e:
            e = alpha_a * e + (1 - alpha_a) * abs_audio[i]
        else:
            e = alpha_r * e + (1 - alpha_r) * abs_audio[i]
        env[i] = e
    env_db = 20.0 * np.log10(env + eps)
    over = np.maximum(env_db - threshold_db, 0)
    gr_db = -over * (1.0 - 1.0 / ratio)
    gr_lin = 10.0 ** (gr_db / 20.0)
    return (audio * gr_lin.astype(np.float32)).astype(np.float32)
