from __future__ import annotations

import contextlib
from pathlib import Path

from ..log import get_logger
from .base import MAX_NEW_TOKENS, AudioChat, AudioReply, read_reply

log = get_logger(__name__)

DEFAULT_MODEL_ID = "mispeech/midashenglm-7b-0804-fp32"

SAMPLE_RATE = 16000

_MAPS_INFEREES = {"auto", "balanced", "balanced_low_0", "sequential"}


_CPU_FACTORIES = ("zeros", "ones", "full", "linspace",
                  "hann_window", "hamming_window", "blackman_window")


@contextlib.contextmanager
def _factories_on_cpu():
    saved = {}
    for name in _CPU_FACTORIES:
        original = getattr(torch_module(), name, None)
        if original is None:
            continue
        saved[name] = original

        def wrapper(f=original):
            def call(*a, **kw):
                kw.setdefault("device", "cpu")
                return f(*a, **kw)
            return call

        setattr(torch_module(), name, wrapper())
    try:
        yield
    finally:
        for name, f in saved.items():
            setattr(torch_module(), name, f)


def torch_module():
    import torch

    return torch


class MiDashengLMChat(AudioChat):
    name = "midasheng-lm"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device_map: str = "auto",
                 max_new_tokens: int = MAX_NEW_TOKENS):
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import importlib

        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        try:
            importlib.import_module(
                "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "MiDashengLM is built on Qwen2.5-Omni and needs a transformers "
                "that ships it (pip install -U transformers). The current one "
                "does not provide "
                "transformers.models.qwen2_5_omni."
            ) from exc

        log.info("Chargement de %s...", self.model_id)
        kwargs = {"trust_remote_code": True, "dtype": "auto"}
        infere = (self.device_map or "auto") in _MAPS_INFEREES
        if not infere:
            kwargs["device_map"] = self.device_map
        else:
            kwargs["low_cpu_mem_usage"] = False
        try:
            with _factories_on_cpu():
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            with _factories_on_cpu():
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, torch_dtype="auto", **kwargs)
        if infere:
            import torch

            cible = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("  device_map '%s' ignored (the model declares no "
                     "_no_split_modules): loading in one block on %s",
                     self.device_map, cible)
            self._model = self._model.to(cible)

        leftover = [n for n, t in self._model.named_parameters() if t.device.type == "meta"]
        leftover += [n for n, t in self._model.named_buffers() if t.device.type == "meta"]
        if leftover:
            raise RuntimeError(
                f"{len(leftover)} weights of {self.model_id} stayed on the 'meta' "
                f"device and were never materialised (e.g. "
                f"{', '.join(leftover[:3])}). The first item would fail. Known "
                "causes: a device_map the model does not support (it declares "
                "no _no_split_modules), or a checkpoint that does not cover "
                "every parameter. Try --device-map cuda:0, or the -bf16 "
                "variant if the card is too small.")

        self._model.requires_grad_(False)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True)
        log.info("Model loaded (%s).", getattr(self._model, "dtype", "unknown dtype"))

    def ask(self, audio: Path, prompt: str) -> AudioReply:
        from ..audio.io import load_audio_resampled

        self._load()
        vague = load_audio_resampled(Path(audio), SAMPLE_RATE)
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt},
                        {"type": "audio", "audio": vague}],
        }]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            add_special_tokens=True, return_dict=True,
        ).to(device=self._model.device, dtype=self._model.dtype)
        out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        raw = self._tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        return read_reply(_apres_le_prompt(raw, prompt))


_QUEUE = 60


def _apres_le_prompt(raw: str, prompt: str) -> str:
    queue = (prompt or "")[-_QUEUE:].strip()
    if queue and queue in raw:
        return raw.split(queue)[-1].lstrip()
    return raw
