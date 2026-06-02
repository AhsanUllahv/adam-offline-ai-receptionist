from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


class TextToSpeech(ABC):
    @abstractmethod
    def say(self, text: str) -> None: ...

    def interrupt(self) -> None:
        """Stop any audio currently playing. No-op unless overridden."""


@dataclass
class ConsoleTTS(TextToSpeech):
    enabled: bool = True

    def say(self, text: str) -> None:
        if self.enabled and text:
            print(f"[voice] {text}")


@dataclass
class CommandTTS(TextToSpeech):
    command: list[str]
    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def say(self, text: str) -> None:
        if not text:
            return
        proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, text=True)
        with self._lock:
            self._proc = proc
        try:
            proc.communicate(input=text)
        finally:
            with self._lock:
                self._proc = None

    def interrupt(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()


@dataclass
class PiperTTS(TextToSpeech):
    model_path: str
    audio_device: str | None = None
    piper_binary: str = "piper"
    player_binary: str = "aplay"
    _play_proc: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _speak_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def say(self, text: str) -> None:
        if not text:
            return

        with self._speak_lock:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = Path(tmp.name)

            try:
                subprocess.run(
                    [
                        self._resolve_binary(self.piper_binary),
                        "--model",
                        self.model_path,
                        "--output_file",
                        str(output_path),
                    ],
                    input=text,
                    text=True,
                    check=False,
                )
                play_proc = subprocess.Popen(self._player_command(output_path))
                with self._lock:
                    self._play_proc = play_proc
                play_proc.wait()
                with self._lock:
                    self._play_proc = None
            finally:
                output_path.unlink(missing_ok=True)

    def interrupt(self) -> None:
        with self._lock:
            if self._play_proc and self._play_proc.poll() is None:
                self._play_proc.terminate()

    def _player_command(self, wav_path: Path) -> list[str]:
        command = [self._resolve_binary(self.player_binary)]
        if self.audio_device:
            command.extend(["-D", self.audio_device])
        command.append(str(wav_path))
        return command

    def _resolve_binary(self, name: str) -> str:
        found = shutil.which(name)
        if found:
            return found
        venv_candidate = Path(sys.executable).with_name(name)
        if venv_candidate.exists():
            return str(venv_candidate)
        return name


def build_answer_tts(
    command: str | None,
    piper_model_path: str | None = None,
    audio_device: str | None = None,
) -> TextToSpeech:
    if command and command.strip():
        return CommandTTS(command=shlex.split(command))
    if piper_model_path and piper_model_path.strip():
        return PiperTTS(model_path=piper_model_path, audio_device=audio_device or None)
    return ConsoleTTS(enabled=True)
