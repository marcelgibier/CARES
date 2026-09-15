from __future__ import annotations

from pathlib import Path

from ..log import get_logger
from .base import MAX_NEW_TOKENS, AudioChat, AudioReply, preload_libstdcxx, read_reply

log = get_logger(__name__)

DEFAULT_MODEL_ID = "moonshotai/Kimi-Audio-7B-Instruct"


FACADE_MODULE = "kimia_infer.api.kimia"


def _import_facade():
    import importlib
    import importlib.util

    installed = importlib.util.find_spec("kimia_infer") is not None
    try:
        return importlib.import_module(FACADE_MODULE).KimiAudio
    except ImportError as exc:
        if not installed:
            raise ImportError(
                "Kimi-Audio s'installe en paquet : pip install "
                "git+https://github.com/MoonshotAI/Kimi-Audio.git") from exc
        raise ImportError(
            f"kimia_infer is installed but {FACADE_MODULE} does not import: "
            f"{type(exc).__name__}: {exc}. One of ITS dependencies is missing "
            "or has drifted.") from exc


class KimiAudioChat(AudioChat):
    name = "kimi-audio"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        preload_libstdcxx()
        KimiAudio = _import_facade()
        log.info("Chargement de %s...", self.model_id)
        self._model = KimiAudio(model_path=self.model_id, load_detokenizer=False)
        log.info("Model loaded.")

    def ask(self, audio: Path, prompt: str) -> AudioReply:
        self._load()
        messages = [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": str(audio)},
        ]
        out = self._model.generate(
            messages, max_new_tokens=self.max_new_tokens, output_type="text")
        text = out[1] if isinstance(out, tuple) else out
        return read_reply(str(text))
