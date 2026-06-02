from __future__ import annotations

import argparse
import sys

from assistant_app.config import AssistantConfig
from assistant_app.tts import build_answer_tts


def main() -> None:
    parser = argparse.ArgumentParser(description="Speak text through configured TTS")
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to speak. If omitted, text is read from standard input.",
    )
    args = parser.parse_args()

    text = " ".join(args.text).strip()
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()

    if not text:
        raise SystemExit("No text provided. Example: assistant-speak 'Hello, I am ready.'")

    config = AssistantConfig.from_env()
    tts = build_answer_tts(
        config.tts_command, config.piper_model_path, config.audio_device
    )
    tts.say(text)


if __name__ == "__main__":
    main()
