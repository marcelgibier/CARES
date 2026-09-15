from __future__ import annotations

import importlib
import sys
from pathlib import Path

from ..log import get_logger
from .base import AudioChat, AudioReply, preload_libstdcxx, read_reply

log = get_logger(__name__)

DEFAULT_MODEL_ID = "XiaomiMiMo/MiMo-Audio-7B-Instruct"
DEFAULT_TOKENIZER_ID = "XiaomiMiMo/MiMo-Audio-Tokenizer"


FACADE_PATH = ("src", "mimo_audio", "mimo_audio.py")
FACADE_MODULE = "src.mimo_audio.mimo_audio"


def _import_facade(repo: str | None):
    preload_libstdcxx()
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    facade = Path(repo).joinpath(*FACADE_PATH) if repo else None
    if facade is not None and not facade.exists():
        raise ImportError(
            f"--alm-repo={repo} does not contain {'/'.join(FACADE_PATH)}. "
            "Attendu : le clone de github.com/XiaomiMiMo/MiMo-Audio.")
    try:
        module = importlib.import_module(FACADE_MODULE)
    except ImportError as exc:
        if repo is None:
            raise ImportError(
                "MiMo-Audio is used from its repository: clone "
                "github.com/XiaomiMiMo/MiMo-Audio and pass its path to "
                "--alm-repo.") from exc
        raise ImportError(
            f"{FACADE_MODULE} is present in {repo} but fails to import: "
            f"{type(exc).__name__}: {exc}. A repository dependency is "
            "missing - install it (uv pip install <package>) WITHOUT touching "
            "torch, or the CUDA stack gets overwritten.") from exc
    _combler_post_init_manquant()
    return module.MimoAudio


_TRANSFORMERS_5_ATTRS = {"all_tied_weights_keys": {}}


def _combler_post_init_manquant() -> None:
    try:
        mod = importlib.import_module(
            "src.mimo_audio_tokenizer.modeling_audio_tokenizer")
    except ImportError:
        return
    cls = getattr(mod, "MiMoAudioTokenizer", None)
    if cls is None:
        return
    for name, default in _TRANSFORMERS_5_ATTRS.items():
        if not hasattr(cls, name):
            setattr(cls, name, dict(default) if isinstance(default, dict) else default)
            log.info("MiMoAudioTokenizer : '%s' absent, comble a %r "
                     "(repository written against transformers 4)", name, default)


class MimoAudioChat(AudioChat):
    name = "mimo-audio"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID,
                 tokenizer_id: str = DEFAULT_TOKENIZER_ID, repo: str | None = None):
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.repo = repo
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        MimoAudio = _import_facade(self.repo)

        log.info("Chargement de %s (tokenizer %s)...", self.model_id, self.tokenizer_id)
        self._model = MimoAudio(self.model_id, self.tokenizer_id)
        log.info("Model loaded.")

    def ask(self, audio: Path, prompt: str) -> AudioReply:
        self._load()
        _lire_sans_ffmpeg()
        raw = self._model.audio_understanding_sft(str(audio), prompt)
        return read_reply(str(raw))


_LECTEUR_REMPLACE = False


def _lire_sans_ffmpeg() -> None:
    global _LECTEUR_REMPLACE
    if _LECTEUR_REMPLACE:
        return
    import torchaudio

    original = torchaudio.load
    dit = []

    def _charge(path, *a, **kw):
        try:
            return original(path, *a, **kw)
        except (ImportError, OSError, RuntimeError) as exc:
            import soundfile as sf
            import torch

            donnees, taux = sf.read(str(path), dtype="float32", always_2d=True)
            if not dit:
                dit.append(True)
                log.warning("torchaudio.load indisponible (%s: %s) : lecture par "
                            "soundfile, taux natif conserve.",
                            type(exc).__name__, str(exc)[:90])
            return torch.from_numpy(donnees.T.copy()), taux

    torchaudio.load = _charge
    _LECTEUR_REMPLACE = True
