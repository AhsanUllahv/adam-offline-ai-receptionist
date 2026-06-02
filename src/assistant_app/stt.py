from __future__ import annotations

import json
import queue
from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from assistant_app.audio_processing import AudioPreprocessor


class SpeechToText(ABC):
    @abstractmethod
    def listen(self, timeout_seconds: float | None = None) -> str: ...


class TextInputSTT(SpeechToText):
    """Keyboard input adapter for development and thesis demonstrations."""

    def listen(self, timeout_seconds: float | None = None) -> str:
        return input("\nYou: ")


@dataclass
class VoskSpeechToText(SpeechToText):
    model_path: str
    sample_rate: int = 16000
    frame_duration_ms: int = 30
    vad_aggressiveness: int = 2
    silence_limit_ms: int = 700
    max_recording_seconds: int = 20
    speech_start_timeout_seconds: float = 10.0
    pre_speech_ms: int = 300
    audio_preprocessor: AudioPreprocessor | None = None
    audio_input_device: str | None = None

    def __post_init__(self) -> None:
        try:
            import vosk
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError(
                "Voice mode needs sounddevice, vosk, and webrtcvad installed."
            ) from exc
        self._vosk = vosk
        self._model = vosk.Model(self.model_path)
        self._vad = webrtcvad.Vad(self.vad_aggressiveness)

    def listen(self, timeout_seconds: float | None = None) -> str:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Voice mode needs sounddevice installed."
            ) from exc

        recognizer = self._vosk.KaldiRecognizer(self._model, self.sample_rate)
        vad = self._vad
        audio_queue: queue.Queue[bytes] = queue.Queue()
        frame_size = int(self.sample_rate * self.frame_duration_ms / 1000) * 2
        silence_frames_needed = max(1, self.silence_limit_ms // self.frame_duration_ms)
        # max_speech_frames caps recording AFTER speech starts (prevents endless recording)
        max_speech_frames = max(1, self.max_recording_seconds * 1000 // self.frame_duration_ms)
        start_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.speech_start_timeout_seconds
        )
        # speech_start_frames_needed controls how long to wait for speech to BEGIN —
        # kept separate from max_speech_frames so follow-up timeouts (e.g. 45 s) are
        # honoured even though max_recording_seconds is only 20 s.
        speech_start_frames_needed = max(
            1, int(start_timeout * 1000 // self.frame_duration_ms)
        )
        pre_speech_frames = max(0, self.pre_speech_ms // self.frame_duration_ms)
        pre_roll: deque[bytes] = deque(maxlen=pre_speech_frames)

        def callback(indata: bytes, frames: int, time_info: object, status: object) -> None:
            if status:
                print(f"[audio] {status}")
            raw = bytes(indata)
            if self.audio_preprocessor:
                raw = self.audio_preprocessor.process(raw)
            audio_queue.put(raw)

        print("\nListening through microphone...")
        speech_started = False
        silence_frames = 0
        speech_frames = 0  # frames counted only after speech starts
        waiting_frames = 0

        raw = self.audio_input_device
        input_device = int(raw) if raw and raw.isdigit() else (raw or None)
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=frame_size // 2,
            dtype="int16",
            channels=1,
            device=input_device,
            callback=callback,
        ):
            while True:
                frame = audio_queue.get()
                if len(frame) != frame_size:
                    continue

                is_speech = vad.is_speech(frame, self.sample_rate)

                if is_speech:
                    if not speech_started:
                        for buffered_frame in pre_roll:
                            recognizer.AcceptWaveform(buffered_frame)
                        pre_roll.clear()
                    speech_started = True
                    silence_frames = 0
                    speech_frames += 1
                    recognizer.AcceptWaveform(frame)
                    if speech_frames >= max_speech_frames:
                        break
                    continue

                if not speech_started:
                    pre_roll.append(frame)
                    waiting_frames += 1
                    if waiting_frames >= speech_start_frames_needed:
                        return ""
                    continue

                # speech already started, silence frame
                silence_frames += 1
                speech_frames += 1
                recognizer.AcceptWaveform(frame)
                if silence_frames >= silence_frames_needed or speech_frames >= max_speech_frames:
                    break

        payload = json.loads(recognizer.FinalResult())
        return payload.get("text", "").strip()
