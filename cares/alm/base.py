from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from ..jsonparse import parse_json_loose
from ..log import get_logger

log = get_logger(__name__)

MAX_NEW_TOKENS = 512


class AudioReply(NamedTuple):
    value: Any | None
    raw: str
    status: str


class AudioChat(Protocol):
    name: str

    def ask(self, audio: Path, prompt: str) -> AudioReply: ...


_THINKING_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)

_THINKING_OPEN_RE = re.compile(r"^.*?<think(?:ing)?>", re.DOTALL | re.IGNORECASE)

LIBSTDCXX_ENV = "CARES_LIBSTDCXX"


def preload_libstdcxx(explicit: str | None = None) -> str | None:
    import ctypes
    import os
    import shutil
    import subprocess

    candidates = [explicit, os.environ.get(LIBSTDCXX_ENV)]
    gcc = shutil.which("gcc")
    if gcc:
        try:
            out = subprocess.run([gcc, "-print-file-name=libstdc++.so.6"],
                                 capture_output=True, text=True, timeout=10, check=False)
            candidates.append(out.stdout.strip() or None)
        except (OSError, subprocess.SubprocessError):
            pass
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        candidates.append(os.path.join(prefix, "lib", "libstdc++.so.6"))

    for path in candidates:
        if not path or not os.path.isabs(path) or not os.path.exists(path):
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        log.info("libstdc++ taken from %s", path)
        return path
    return None


REEXEC_GUARD = "CARES_LIBSTDCXX_REEXEC"


def find_libstdcxx() -> str | None:
    import os
    import shutil
    import subprocess

    candidates = [os.environ.get(LIBSTDCXX_ENV)]
    gcc = shutil.which("gcc")
    if gcc:
        try:
            out = subprocess.run([gcc, "-print-file-name=libstdc++.so.6"],
                                 capture_output=True, text=True, timeout=10, check=False)
            candidates.append(out.stdout.strip() or None)
        except (OSError, subprocess.SubprocessError):
            pass
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        candidates.append(os.path.join(prefix, "lib", "libstdc++.so.6"))
    for path in candidates:
        if path and os.path.isabs(path) and os.path.exists(path):
            return os.path.realpath(path)
    return None


def ensure_libstdcxx_on_path() -> None:
    import os
    import sys

    if os.environ.get(REEXEC_GUARD):
        return
    path = find_libstdcxx()
    if not path:
        return
    directory = os.path.dirname(path)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if directory in current.split(os.pathsep):
        return

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [directory, current]))
    env[REEXEC_GUARD] = "1"
    log.info("Re-exec with LD_LIBRARY_PATH=%s (system libstdc++ too old for the "
             "compiled extensions)", directory)
    os.execve(sys.executable, [sys.executable, *_reinvocation()], env)


def _reinvocation() -> list[str]:
    import os
    import sys

    if os.path.basename(sys.argv[0]) == "__main__.py":
        paquet = os.path.basename(os.path.dirname(os.path.abspath(sys.argv[0])))
        if paquet:
            return ["-m", paquet, *sys.argv[1:]]
    return list(sys.argv)


def strip_thinking(raw: str) -> str:
    out = _THINKING_RE.sub(" ", raw)
    if "</think" not in out.lower() and "<think" in out.lower():
        out = _THINKING_OPEN_RE.sub(" ", out)
    return out.strip()


def _python_literal(text: str) -> Any | None:
    start = text.find("{")
    if start < 0:
        return None
    span = text[start:]
    end = span.rfind("}")
    candidates = [span[: end + 1]] if end >= 0 else []
    candidates.append(span)
    for cand in candidates:
        for closer in ("", "']}", '"]}', "]}", "}"):
            try:
                value = ast.literal_eval(cand + closer)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                continue
            if isinstance(value, dict):
                return value
    return None


def read_reply(raw: str) -> AudioReply:
    if not raw or not raw.strip():
        return AudioReply(None, raw, "empty_response")
    cleaned = strip_thinking(raw)
    parsed = parse_json_loose(cleaned)
    if parsed is not None:
        return AudioReply(parsed, raw, "ok")
    parsed = _python_literal(cleaned)
    if parsed is not None:
        return AudioReply(parsed, raw, "ok_literal")
    return AudioReply(None, raw, "no_json")
