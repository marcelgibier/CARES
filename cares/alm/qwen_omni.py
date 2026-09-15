from __future__ import annotations

from pathlib import Path

from ..log import get_logger
from .base import MAX_NEW_TOKENS, AudioChat, AudioReply, read_reply

log = get_logger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


class QwenOmniChat(AudioChat):
    name = "qwen3-omni"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device_map: str = "auto",
                 max_new_tokens: int = MAX_NEW_TOKENS,
                 attn_implementation: str | None = None):
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self._model = None
        self._processor = None
        self._process_mm_info = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        try:
            from qwen_omni_utils import process_mm_info
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "Qwen3-Omni exige le paquet 'qwen-omni-utils' "
                "(pip install qwen-omni-utils -U)."
            ) from exc

        kwargs = {"dtype": "auto", "device_map": self.device_map}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        log.info("Chargement de %s...", self.model_id)
        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs)
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id)
        self._process_mm_info = process_mm_info
        if hasattr(self._model, "disable_talker"):
            self._model.disable_talker()
            log.info("  speech generation disabled")
        log.info("Model loaded.")

    def ask(self, audio: Path, prompt: str) -> AudioReply:
        self._load()
        conversation = [{
            "role": "user",
            "content": [{"type": "audio", "audio": str(audio)},
                        {"type": "text", "text": prompt}],
        }]
        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = self._process_mm_info(
            conversation, use_audio_in_video=False)
        inputs = self._processor(text=text, audio=audios, images=images, videos=videos,
                                 return_tensors="pt", padding=True,
                                 use_audio_in_video=False)
        inputs = inputs.to(self._model.device).to(self._model.dtype)

        try:
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       return_audio=False)
        except TypeError:
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text_ids = out[0] if isinstance(out, tuple) else out
        seq = getattr(text_ids, "sequences", text_ids)
        raw = self._processor.batch_decode(
            seq[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return read_reply(raw)
