from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

SCENARIO_BANKS = PACKAGE_DIR / "resources" / "scenario_banks_v3.json"

THEMES = PACKAGE_DIR / "resources" / "themes.json"
REACTION_REGISTERS = PACKAGE_DIR / "resources" / "reaction_registers.json"

VOICES_POOL = PACKAGE_DIR / "resources" / "voices.json"

DEFAULT_CLAUDE_MODEL = "claude-opus-5"

DEFAULT_DATA_DIR = Path("data")


@dataclass(frozen=True)
class Paths:
    root: Path = DEFAULT_DATA_DIR

    @classmethod
    def resolve(cls, data_dir: str | Path | None = None) -> Paths:
        if data_dir is None:
            data_dir = os.environ.get("CARES_DATA_DIR") or DEFAULT_DATA_DIR
        return cls(Path(data_dir))

    @property
    def templates_dir(self) -> Path:
        return self.root / "template_generation_output"

    @property
    def raw_templates(self) -> Path:
        return self.templates_dir / "raw_templates.json"

    @property
    def templates(self) -> Path:
        return self.templates_dir / "templates.json"

    @property
    def templates_flat(self) -> Path:
        return self.templates_dir / "templates_flat.json"

    @property
    def scenarios_dir(self) -> Path:
        return self.root / "scenario_generation_output"

    @property
    def raw_scenarios(self) -> Path:
        return self.scenarios_dir / "raw_scenarios.json"

    @property
    def scenarios(self) -> Path:
        return self.scenarios_dir / "scenarios.json"

    @property
    def distribution_log(self) -> Path:
        return self.scenarios_dir / "distribution_log.json"

    @property
    def scenarios_batches_state(self) -> Path:
        return self.scenarios_dir / "batches_state.json"

    def dialogues_dir(self, mode: str = "grounded") -> Path:
        suffix = "" if mode == "grounded" else "_generative"
        return self.root / f"dialogue_generation_output{suffix}"

    def raw_dialogues(self, mode: str = "grounded") -> Path:
        return self.dialogues_dir(mode) / "raw_dialogues.json"

    def dialogues(self, mode: str = "grounded") -> Path:
        return self.dialogues_dir(mode) / "dialogues.json"

    def batches_state(self, mode: str = "grounded") -> Path:
        return self.dialogues_dir(mode) / "batches_state.json"

    @property
    def filter_dir(self) -> Path:
        return self.root / "filter_output"

    @property
    def voices_dir(self) -> Path:
        return self.root / "out_voices"

    @property
    def scenes_dir(self) -> Path:
        return self.root / "audio_scenes"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval_output"

    @property
    def analysis_dir(self) -> Path:
        return self.root / "analysis_output"

    @property
    def pairs_dir(self) -> Path:
        return self.root / "pair_data"
