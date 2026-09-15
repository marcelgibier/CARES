from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from ..log import get_logger
from .base import MAX_NEW_TOKENS, AudioChat, AudioReply, read_reply

log = get_logger(__name__)

DEFAULT_MODEL_ID = "nvidia/audio-flamingo-next-hf"

REPETITION_PENALTY = 1.2

AUDIO_WINDOW_S = 30.0

MIN_TAIL_S = 1.0


def tail_pad_s(duration_s: float) -> float:
    tail = duration_s % AUDIO_WINDOW_S
    if tail == 0.0 or tail >= MIN_TAIL_S:
        return 0.0
    return MIN_TAIL_S - tail


class FlamingoChat(AudioChat):
    name = "audio-flamingo"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device_map: str = "auto",
                 max_new_tokens: int = MAX_NEW_TOKENS):
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._padded: dict[str, Path] = {}

    def _prepared(self, audio: Path) -> Path:
        key = str(audio)
        if key in self._padded:
            return self._padded[key]
        import numpy as np
        import soundfile as sf

        info = sf.info(key)
        pad_s = tail_pad_s(info.frames / info.samplerate)
        if pad_s <= 0.0:
            self._padded[key] = audio
            return audio
        data, sr = sf.read(key)
        pad = np.zeros((int(pad_s * sr), *data.shape[1:]), dtype=data.dtype)
        tag = hashlib.sha1(str(audio.resolve()).encode()).hexdigest()[:12]
        out = Path(tempfile.gettempdir()) / f"cares_pad_{audio.stem}_{tag}.wav"
        sf.write(str(out), np.concatenate([data, pad]), sr)
        log.info("  fin de fenetre degeneree (%.3f s) : +%.3f s de silence -> %s",
                 info.frames / info.samplerate % AUDIO_WINDOW_S, pad_s, out.name)
        self._padded[key] = out
        return out

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import MusicFlamingoForConditionalGeneration
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "transformers does not know MusicFlamingoForConditionalGeneration "
                "(Audio Flamingo Next) : pip install --upgrade transformers."
            ) from exc

        if os.environ.get("HF_HUB_OFFLINE") == "1":
            log.info("HF_HUB_OFFLINE=1: weights are read from the local cache")
        log.info("Chargement de %s (bfloat16, device_map=%s)...",
                 self.model_id, self.device_map)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = MusicFlamingoForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=self.device_map,
        ).eval()
        log.info("Model loaded.")

    def ask(self, audio: Path, prompt: str) -> AudioReply:
        self._load()
        audio = self._prepared(audio)
        conversation = [[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": str(audio)},
            ],
        }]]
        batch = self._processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
        ).to(self._model.device)
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(self._model.dtype)

        log.debug("  prompt : %d jetons", batch["input_ids"].shape[1])
        generated = self._model.generate(
            **batch, max_new_tokens=self.max_new_tokens,
            repetition_penalty=REPETITION_PENALTY,
        )
        prompt_len = batch["input_ids"].shape[1]
        raw = self._processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return read_reply(raw)
