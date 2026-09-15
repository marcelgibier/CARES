from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING

    env_level = os.environ.get("CARES_LOG_LEVEL")
    unknown_env = ""
    if env_level and not (verbose or quiet):
        resolved = getattr(logging, env_level.upper(), None)
        if isinstance(resolved, int):
            level = resolved
        else:
            unknown_env = env_level

    logging.basicConfig(level=level, format=_FORMAT, force=True)
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
    if unknown_env:
        logging.getLogger(__name__).warning(
            "Unknown CARES_LOG_LEVEL=%r, keeping level %s.",
            unknown_env, logging.getLevelName(level))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
