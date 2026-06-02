from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from assistant_app.tts import CommandTTS, ConsoleTTS, PiperTTS, build_answer_tts


class TTSFactoryTests(unittest.TestCase):
    def test_empty_command_and_no_piper_uses_console_tts(self) -> None:
        self.assertIsInstance(build_answer_tts(None), ConsoleTTS)
        self.assertIsInstance(build_answer_tts(""), ConsoleTTS)
        self.assertIsInstance(build_answer_tts("   "), ConsoleTTS)

    def test_command_is_parsed_with_shell_like_quotes(self) -> None:
        tts = build_answer_tts('piper --model "models/voice file.onnx" --output-raw')

        self.assertIsInstance(tts, CommandTTS)
        self.assertEqual(
            tts.command,
            ["piper", "--model", "models/voice file.onnx", "--output-raw"],
        )

    def test_explicit_command_takes_priority_over_piper_model(self) -> None:
        tts = build_answer_tts("custom-tts", piper_model_path="models/voice.onnx")

        self.assertIsInstance(tts, CommandTTS)

    def test_piper_model_uses_piper_tts(self) -> None:
        tts = build_answer_tts(None, piper_model_path="models/voice.onnx", audio_device="plughw:1,0")

        self.assertIsInstance(tts, PiperTTS)
        self.assertEqual(tts.model_path, "models/voice.onnx")
        self.assertEqual(tts.audio_device, "plughw:1,0")

    def test_piper_player_command_uses_audio_device_when_set(self) -> None:
        tts = PiperTTS(model_path="models/voice.onnx", audio_device="plughw:1,0")

        command = tts._player_command(Path("/tmp/answer.wav"))

        self.assertTrue(command[0].endswith("aplay"))
        self.assertEqual(command[1:], ["-D", "plughw:1,0", "/tmp/answer.wav"])

    def test_piper_say_generates_wav_then_plays_it(self) -> None:
        tts = PiperTTS(model_path="models/voice.onnx", audio_device="default")

        with patch("assistant_app.tts.subprocess.run") as run:
            tts.say("Hello")

        self.assertEqual(run.call_count, 2)
        piper_call = run.call_args_list[0]
        player_call = run.call_args_list[1]
        self.assertEqual(piper_call.kwargs["input"], "Hello")
        self.assertIn("--model", piper_call.args[0])
        self.assertIn("models/voice.onnx", piper_call.args[0])
        self.assertTrue(player_call.args[0][0].endswith("aplay"))
        self.assertEqual(player_call.args[0][1:3], ["-D", "default"])

    def test_piper_resolves_binary_from_python_environment(self) -> None:
        tts = PiperTTS(model_path="models/voice.onnx")

        with patch("assistant_app.tts.shutil.which", return_value=None), patch(
            "assistant_app.tts.Path.exists", return_value=True
        ), patch("assistant_app.tts.sys.executable", "/project/.venv/bin/python"):
            self.assertEqual(tts._resolve_binary("piper"), "/project/.venv/bin/piper")


if __name__ == "__main__":
    unittest.main()
