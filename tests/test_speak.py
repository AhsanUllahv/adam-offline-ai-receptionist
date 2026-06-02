from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


from assistant_app import speak


class SpeakCommandTests(unittest.TestCase):
    def test_speaks_text_arguments_with_console_fallback(self) -> None:
        output = io.StringIO()
        with patch("sys.argv", ["assistant-speak", "Hello", "there"]), redirect_stdout(output):
            speak.main()

        self.assertIn("[voice] Hello there", output.getvalue())

    def test_reads_text_from_stdin(self) -> None:
        output = io.StringIO()
        stdin = io.StringIO("Ready from stdin")
        stdin.isatty = lambda: False  # type: ignore[method-assign]

        with patch("sys.argv", ["assistant-speak"]), patch("sys.stdin", stdin), redirect_stdout(output):
            speak.main()

        self.assertIn("[voice] Ready from stdin", output.getvalue())

    def test_uses_piper_environment_for_speaker_test(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "ASSISTANT_PIPER_MODEL": "models/voice.onnx",
                "ASSISTANT_AUDIO_DEVICE": "default",
                "ASSISTANT_TTS_COMMAND": "",
            },
            clear=True,
        ), patch("assistant_app.speak.build_answer_tts") as build_tts, patch(
            "sys.argv", ["assistant-speak", "Hello"]
        ):
            tts = build_tts.return_value
            speak.main()

        build_tts.assert_called_once_with("", "models/voice.onnx", "default")
        tts.say.assert_called_once_with("Hello")

    def test_empty_text_exits_with_helpful_message(self) -> None:
        stdin = io.StringIO("")
        stdin.isatty = lambda: True  # type: ignore[method-assign]

        with patch("sys.argv", ["assistant-speak"]), patch("sys.stdin", stdin):
            with self.assertRaises(SystemExit) as context:
                speak.main()

        self.assertIn("No text provided", str(context.exception))


if __name__ == "__main__":
    unittest.main()
