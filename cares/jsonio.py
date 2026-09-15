from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(f"{exc.msg} ({path})", exc.doc, exc.pos) from exc


def save_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)
