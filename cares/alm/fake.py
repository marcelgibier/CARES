from __future__ import annotations

import json
import random
import zlib
from pathlib import Path

from ..dataset import REACTIONS
from .base import AudioChat, AudioReply, read_reply


class FakeAudioChat(AudioChat):
    name = "fake"

    def __init__(self, vocabulary: list[str] | None = None, malformed_rate: float = 0.0):
        self.vocabulary = vocabulary or []
        self.malformed_rate = malformed_rate

    def ask(self, audio: Path, prompt: str) -> AudioReply:
        rng = random.Random(zlib.crc32(f"{audio}|{prompt[:64]}".encode()))
        if rng.random() < self.malformed_rate:
            return read_reply("Sure! I heard a few things but cannot say more.")
        if '"sound_present"' in prompt:
            return read_reply(json.dumps({"sound_present": rng.random() < 0.5}))
        if '"reaction"' in prompt:
            return read_reply(json.dumps({"reaction": rng.choice(REACTIONS)}))
        if "Summarise what happens" in prompt:
            import re

            m = re.search(r"in about (\d+) words", prompt)
            budget = int(m.group(1)) if m else 40
            mots = ("two people talk in a room while ordinary things happen "
                    "around them and the conversation carries on").split()
            return read_reply(" ".join(
                mots[i % len(mots)] for i in range(budget)) + ".")
        if '"answer"' in prompt:
            options = [line.strip()[2:].strip() for line in prompt.splitlines()
                       if line.strip().startswith("- ")]
            if options:
                return read_reply(json.dumps({"answer": rng.choice(options)}))
        pool = self.vocabulary or ["a door", "a phone", "footsteps"]
        n = rng.randint(0, min(3, len(pool)))
        return read_reply(json.dumps({"sounds": rng.sample(pool, n)}))
