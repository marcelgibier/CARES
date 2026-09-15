from __future__ import annotations

from pathlib import Path

from ..log import get_logger
from .base import MAX_NEW_TOKENS, AudioChat, AudioReply, read_reply

log = get_logger(__name__)

DEFAULT_ASR_ID = "openai/whisper-large-v3"
DEFAULT_LLM_ID = "Qwen/Qwen3.6-35B-A3B"

CHUNK_LENGTH_S = 30

TRANSCRIPT_HEADER = """\
Below is an automatic transcription of the recording, with timestamps in \
seconds. Speaker identities are NOT available. Non-speech sounds are NOT \
transcribed.

{transcript}

"""


class CascadeChat(AudioChat):
    name = "whisper+llm"

    def __init__(self, asr_id: str = DEFAULT_ASR_ID, llm_id: str = DEFAULT_LLM_ID,
                 device_map: str = "auto", max_new_tokens: int = MAX_NEW_TOKENS,
                 transcripts_path: Path | None = None):
        self.asr_id = asr_id
        self.llm_id = llm_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.transcripts_path = transcripts_path
        self._asr = None
        self._llm = None
        self._processor = None
        self._transcripts: dict[str, str] = self._restore()

    def _restore(self) -> dict[str, str]:
        if self.transcripts_path is None:
            return {}
        from ..jsonio import load_json

        try:
            data = load_json(self.transcripts_path, default={})
        except Exception as exc:  # noqa: BLE001
            log.warning("Cache de transcriptions illisible (%s) : on repart a "
                        "empty, everything will be transcribed again.", exc)
            return {}
        if not isinstance(data, dict):
            log.warning("Cache de transcriptions mal forme (%s au lieu d'un "
                        "dict) : on repart a vide.", type(data).__name__)
            return {}
        log.info("Transcripts resumed: %d from %s",
                 len(data), self.transcripts_path)
        return data

    def _load(self) -> None:
        if self._llm is not None:
            return
        from transformers import AutoModelForMultimodalLM, AutoProcessor, pipeline

        log.info("Chargement de l'ASR %s...", self.asr_id)
        self._asr = pipeline("automatic-speech-recognition", model=self.asr_id,
                             chunk_length_s=CHUNK_LENGTH_S, return_timestamps=True,
                             device_map=self.device_map)
        log.info("Chargement du LLM %s...", self.llm_id)
        self._processor = AutoProcessor.from_pretrained(self.llm_id)
        self._llm = AutoModelForMultimodalLM.from_pretrained(
            self.llm_id, device_map=self.device_map)
        log.info("Cascade chargee.")

    def transcribe(self, audio: Path) -> str:
        key = str(audio)
        if key in self._transcripts:
            return self._transcripts[key]
        out = self._asr(key)
        chunks = out.get("chunks") if isinstance(out, dict) else None
        if chunks:
            lines = []
            for c in chunks:
                start, end = (c.get("timestamp") or (None, None))[:2]
                stamp = f"[{start:.1f}-{end:.1f}] " if start is not None and end is not None \
                    else ""
                lines.append(f"{stamp}{c.get('text', '').strip()}")
            text = "\n".join(lines)
        else:
            text = (out.get("text") if isinstance(out, dict) else str(out)).strip()
        self._transcripts[key] = text
        self._persist()
        return text

    def _persist(self) -> None:
        if self.transcripts_path is None:
            return
        from ..jsonio import save_json

        save_json(self.transcripts_path, self._transcripts)

    def ask(self, audio: Path, prompt: str) -> AudioReply:
        self._load()
        transcript = self.transcribe(audio)
        messages = [{"role": "user",
                     "content": TRANSCRIPT_HEADER.format(transcript=transcript) + prompt}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        ).to(self._llm.device)
        generated = self._llm.generate(**inputs, max_new_tokens=self.max_new_tokens)
        raw = self._processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return read_reply(raw)
