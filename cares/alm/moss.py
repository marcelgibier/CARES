from __future__ import annotations

import importlib
import sys
from pathlib import Path

from ..log import get_logger
from .base import (
    MAX_NEW_TOKENS,
    AudioChat,
    AudioReply,
    preload_libstdcxx,
    read_reply,
)

log = get_logger(__name__)

DEFAULT_MODEL_ID = "OpenMOSS-Team/MOSS-Audio-8B-Thinking"

TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50

SAMPLING_SALT = ""


def sampling_seed(audio: Path, prompt: str, salt: str = "") -> int:
    import hashlib

    cle = f"{Path(audio).name}|{Path(audio).parent.name}|{prompt}|{salt}"
    return int(hashlib.sha1(cle.encode("utf-8")).hexdigest()[:8], 16)


REPO_MODULES = ("src/audio_io.py", "src/modeling_moss_audio.py",
                "src/processing_moss_audio.py")


def _import_classes(repo: str | None):
    preload_libstdcxx()
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    if repo:
        missing = [m for m in REPO_MODULES if not Path(repo).joinpath(m).exists()]
        if missing:
            raise ImportError(
                f"--alm-repo={repo} does not contain {', '.join(missing)}. "
                "Attendu : le clone de github.com/OpenMOSS/MOSS-Audio.")
    try:
        io_mod = importlib.import_module("src.audio_io")
        model_mod = importlib.import_module("src.modeling_moss_audio")
        proc_mod = importlib.import_module("src.processing_moss_audio")
    except ImportError as exc:
        if repo is None:
            raise ImportError(
                "MOSS-Audio exposes its classes in its repository, not in "
                "transformers : clonez github.com/OpenMOSS/MOSS-Audio et "
                "pass its path to --alm-repo.") from exc
        raise ImportError(
            f"The repository is in {repo} but fails to import: "
            f"{type(exc).__name__}: {exc}. A repository dependency is "
            "missing - install it (uv pip install <package>) WITHOUT touching "
            "torch, or the CUDA stack gets overwritten.") from exc
    return io_mod.load_audio, model_mod.MossAudioModel, proc_mod.MossAudioProcessor


class MossAudioChat(AudioChat):
    name = "moss-audio"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device_map: str = "auto",
                 max_new_tokens: int = MAX_NEW_TOKENS, repo: str | None = None):
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.repo = repo
        self._model = None
        self._processor = None
        self._load_audio = None
        self._fallback_logged = False

    def _load(self) -> None:
        if self._model is not None:
            return
        load_audio, MossAudioModel, MossAudioProcessor = _import_classes(self.repo)

        log.info("Chargement de %s...", self.model_id)
        self._model = MossAudioModel.from_pretrained(
            self.model_id, trust_remote_code=True, dtype="auto",
            device_map=self.device_map)
        self._processor = MossAudioProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, enable_time_marker=True)
        self._load_audio = load_audio
        log.info("Model loaded.")

    def _waveform(self, audio: Path, sample_rate: int):
        try:
            return self._load_audio(str(audio), sample_rate=sample_rate)
        except ImportError as exc:
            from ..audio.io import load_audio_resampled

            if not self._fallback_logged:
                log.warning("repository audio reader unavailable (%s): falling back to "
                            "cares.audio.io.load_audio_resampled", exc)
                self._fallback_logged = True
            return load_audio_resampled(audio, target_sr=sample_rate)

    def ask(self, audio: Path, prompt: str) -> AudioReply:
        self._load()
        raw_audio = self._waveform(audio, self._processor.config.mel_sr)
        inputs = self._processor(text=prompt, audios=[raw_audio], return_tensors="pt")
        inputs = inputs.to(self._model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(self._model.dtype)
        inputs["audio_input_mask"] = (
            inputs["input_ids"] == self._processor.audio_token_id)
        import torch

        torch.manual_seed(sampling_seed(audio, prompt, SAMPLING_SALT))
        generated = self._model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
            num_beams=1, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
            use_cache=True,
        )
        input_len = inputs["input_ids"].shape[1]
        raw = self._processor.decode(generated[0, input_len:], skip_special_tokens=True)
        return read_reply(raw)
