from __future__ import annotations

import importlib
import zlib
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

TARGET_SR = 44100

VOICE_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aif",
              ".aiff", ".opus")

_EXTRA_BY_MODULE = {
    "elevenlabs": "tts",
    "pyroomacoustics": "room",
}
_DEFAULT_EXTRA = "audio"


def install_hint(module_name: str) -> str:
    extra = _EXTRA_BY_MODULE.get(module_name.split(".", 1)[0], _DEFAULT_EXTRA)
    return f"This step requires `pip install 'cares[{extra}]'`"


def stable_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def require(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{install_hint(module_name)} (missing module: {module_name})") from exc


_PYROOM_AVAILABLE: bool | None = None
_PEDALBOARD_AVAILABLE: bool | None = None


def pyroomacoustics_available() -> bool:
    global _PYROOM_AVAILABLE
    if _PYROOM_AVAILABLE is None:
        try:
            import pyroomacoustics  # noqa: F401
            _PYROOM_AVAILABLE = True
        except Exception:
            _PYROOM_AVAILABLE = False
    return _PYROOM_AVAILABLE


def pedalboard_available() -> bool:
    global _PEDALBOARD_AVAILABLE
    if _PEDALBOARD_AVAILABLE is None:
        try:
            import pedalboard  # noqa: F401
            _PEDALBOARD_AVAILABLE = True
        except Exception:
            _PEDALBOARD_AVAILABLE = False
    return _PEDALBOARD_AVAILABLE


def load_audio_resampled(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    np = require("numpy")
    librosa = require("librosa")
    y, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return y.astype(np.float32)


def list_audio_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir()
                   if p.is_file() and not p.name.startswith(".")
                   and p.suffix.lower() in AUDIO_EXTS),
                  key=lambda p: p.name)


def list_files_for_event(events_root: Path, event_id: str) -> list[Path]:
    return list_audio_files(events_root / event_id)


def list_files_for_background(backgrounds_root: Path, scene: str) -> list[Path]:
    return list_audio_files(backgrounds_root / scene)


def takes_for_event(events_root: Path, event_id: str, split: str | None = None) -> list[Path]:
    files = list_files_for_event(events_root, event_id)
    if not split or not files:
        return files
    from ..splits import partition_files
    return partition_files(files, salt=event_id)[split]


def ambiences_for_scene(backgrounds_root: Path, scene: str,
                        split: str | None = None) -> list[Path]:
    files = list_files_for_background(backgrounds_root, scene)
    if not split or not files:
        return files
    from ..splits import partition_files
    return partition_files(files, salt=scene)[split]


def find_voice_file(sid: str, voices_dir: Path) -> Path | None:
    for ext in VOICE_EXTS:
        p = voices_dir / f"{sid}{ext}"
        if p.exists():
            return p
    if voices_dir.is_dir():
        for p in sorted(voices_dir.iterdir()):
            if p.is_file() and p.stem == sid and p.suffix.lower() != ".json":
                return p
    return None
