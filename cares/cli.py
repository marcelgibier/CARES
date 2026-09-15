from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from . import __version__
from .log import setup_logging

COMMANDS: tuple[tuple[str, str, str], ...] = (
    ("templates", "cares.stages.templates",
     "Step 1: themes -> conversation templates"),
    ("scenarios", "cares.stages.scenarios",
     "Step 2: templates -> pre-allocated scenarios"),
    ("dialogues", "cares.stages.dialogues.grounded",
     "Step 3: scenarios -> dialogues (imposed reactions)"),
    ("filter", "cares.stages.filtering",
     "Step 4: filtering by reaction recovery"),
    ("tts", "cares.stages.tts",
     "Step 5: voice synthesis (ElevenLabs)"),
    ("align", "cares.stages.align",
     "Step 5 bis: forced alignment of the text on the voices"),
    ("mix", "cares.stages.mix",
     "Step 6: mixing of the audio scenes"),
    ("eval", "cares.stages.evaluate",
     "Step 7: evaluation of audio-language models on the mixed scenes"),
)


class _UnavailableStage:
    def __init__(self, module: str, error: Exception) -> None:
        self.module = module
        self.error = error

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(_unavailable=self)

    def run(self, args: argparse.Namespace) -> int:
        print(f"Command unavailable: {self.module} could not be imported.\n"
              f"  {type(self.error).__name__}: {self.error}\n"
              f"  Install the dependencies: pip install 'cares[all]'", file=sys.stderr)
        return 3


def _load(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        return _UnavailableStage(module_path, exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cares",
        description="Pipeline generating a dataset of conversational audio scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"cares {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default=None,
                        help="Root of the artifacts (default: ./data or $CARES_DATA_DIR)")
    verbosity = common.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="Debug logs")
    verbosity.add_argument("-q", "--quiet", action="store_true",
                           help="Warnings only")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for name, module_path, help_text in COMMANDS:
        stage = _load(module_path)
        subparser = subparsers.add_parser(name, parents=[common], help=help_text,
                                          description=help_text)
        stage.add_arguments(subparser)
        subparser.set_defaults(_run=stage.run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    if getattr(args, "_unavailable", None) is not None:
        return args._run(args)
    if unknown:
        parser.error(f"unknown argument(s): {' '.join(unknown)}")
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    return args._run(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
