from __future__ import annotations

import json as _json
import queue
import re
from difflib import SequenceMatcher
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class WakeWordDetector(ABC):
    @abstractmethod
    def wait_for_wake_word(self) -> None:
        """Block until the wake word is detected."""
        ...


class AlwaysReadyDetector(WakeWordDetector):
    """Pass-through — used when wake word is disabled (keyboard/text mode)."""

    def wait_for_wake_word(self) -> None:
        return


@dataclass
class OpenWakeWordDetector(WakeWordDetector):
    """Detects a wake word using an openwakeword ONNX model (fully offline)."""

    model_path: str
    sample_rate: int = 16000
    chunk_size: int = 1280  # 80ms at 16 kHz — expected by openwakeword
    threshold: float = 0.5
    _model: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError(
                "Wake word detection requires openwakeword. "
                "Install with: pip install openwakeword"
            ) from exc
        self._model = Model(wakeword_models=[self.model_path], inference_framework="onnx")

    def wait_for_wake_word(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Wake word detection needs numpy and sounddevice installed."
            ) from exc

        audio_queue: queue.Queue[bytes] = queue.Queue()

        def callback(indata: bytes, frames: int, time_info: object, status: object) -> None:
            audio_queue.put(bytes(indata))

        print("[wake] Waiting for wake word...", flush=True)
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                audio_bytes = audio_queue.get()
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                prediction = self._model.predict(audio_array)
                for score in prediction.values():
                    if score >= self.threshold:
                        self._model.reset()
                        print("[wake] Wake word detected.", flush=True)
                        return


@dataclass
class VoskKeywordDetector(WakeWordDetector):
    """Detects a keyword ('adam') in a continuous audio stream using Vosk.

    Uses Vosk partial + final results to spot the keyword with very low
    latency (~100 ms chunks). No custom ONNX model required — works with
    the same Vosk model used for full STT.
    """

    model_path: str
    keyword: str = "adam"
    keyword_aliases: tuple[str, ...] = ("adam", "addam", "atom", "add em", "add him", "at him")
    sample_rate: int = 16000
    chunk_ms: int = 120
    audio_input_device: str | None = None
    _model: object = field(default=None, init=False, repr=False)
    _vosk: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import vosk
        except ImportError as exc:
            raise RuntimeError(
                "VoskKeywordDetector requires vosk installed."
            ) from exc
        self._vosk = vosk
        self._model = vosk.Model(self.model_path)

    def wait_for_wake_word(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("VoskKeywordDetector requires sounddevice installed.") from exc

        _push_event("state", state="waiting_wake_word", message=f"Say '{self.keyword}' to wake me up")

        chunk_frames = int(self.sample_rate * self.chunk_ms / 1000)
        rec = self._vosk.KaldiRecognizer(self._model, self.sample_rate)

        audio_q: queue.Queue[bytes] = queue.Queue()

        def callback(indata: bytes, frames: int, time_info: object, status: object) -> None:
            audio_q.put(bytes(indata))

        print(f"\n[wake] Listening for '{self.keyword}'...", flush=True)
        raw = self.audio_input_device
        device = int(raw) if raw and raw.isdigit() else (raw or None)
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=chunk_frames,
            dtype="int16",
            channels=1,
            device=device,
            callback=callback,
        ):
            while True:
                chunk = audio_q.get()
                if rec.AcceptWaveform(chunk):
                    result = _json.loads(rec.Result())
                    words = result.get("text", "").lower().split()
                else:
                    partial = _json.loads(rec.PartialResult())
                    words = partial.get("partial", "").lower().split()

                heard_text = " ".join(words)
                if _keyword_detected(heard_text, self.keyword, self.keyword_aliases):
                    rec.Reset()
                    print(f"[wake] '{self.keyword}' detected.", flush=True)
                    _push_event("wake", keyword=self.keyword, transcript=heard_text)
                    return


def _normalize_keyword_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _keyword_detected(text: str, keyword: str, aliases: tuple[str, ...] = ()) -> bool:
    normalized = _normalize_keyword_text(text)
    if not normalized:
        return False

    candidates = tuple(dict.fromkeys((keyword.lower().strip(), *aliases)))
    words = normalized.split()
    padded = f" {normalized} "
    compact = normalized.replace(" ", "")

    for candidate in candidates:
        candidate = _normalize_keyword_text(candidate)
        if not candidate:
            continue
        if f" {candidate} " in padded:
            return True
        if " " in candidate and candidate.replace(" ", "") in compact:
            return True

    base = _normalize_keyword_text(keyword).replace(" ", "")
    if len(base) < 4:
        return False
    for word in words:
        if len(word) < 3 or abs(len(word) - len(base)) > 1:
            continue
        if SequenceMatcher(None, word, base).ratio() >= 0.78:
            return True
    return False


def _push_event(event_type: str, **payload: object) -> None:
    try:
        from assistant_app import events
        events.push_event(event_type, **payload)
    except Exception:
        pass


def build_wake_word_detector(
    model_path: str | None,
    threshold: float = 0.5,
    keyword: str = "adam",
    use_keyword_detector: bool = False,
    audio_input_device: str | None = None,
    keyword_aliases: tuple[str, ...] = (),
) -> WakeWordDetector:
    if not model_path:
        return AlwaysReadyDetector()
    if use_keyword_detector:
        return VoskKeywordDetector(
            model_path=model_path,
            keyword=keyword,
            keyword_aliases=keyword_aliases or VoskKeywordDetector.keyword_aliases,
            audio_input_device=audio_input_device,
        )
    return OpenWakeWordDetector(model_path=model_path, threshold=threshold)
