from __future__ import annotations

import argparse
import asyncio
import os
import threading
import time
from pathlib import Path

from assistant_app.audio_processing import AudioPreprocessor
from assistant_app.config import AssistantConfig
from assistant_app.eyes import build_eye_controller
from assistant_app.llm import OllamaLLM
from assistant_app.pipeline import AssistantPipeline
from assistant_app.query import QueryProcessor
from assistant_app.retrieval import ChromaRetriever
from assistant_app.state import ResponseStateManager
from assistant_app.stt import TextInputSTT, VoskSpeechToText
from assistant_app.tts import ConsoleTTS, build_answer_tts
from assistant_app.wakeword import build_wake_word_detector


def build_pipeline(config: AssistantConfig, voice_mode: bool = False, no_rag: bool = False) -> AssistantPipeline:
    state_manager = ResponseStateManager(
        tts=ConsoleTTS(enabled=config.enable_filler_voice),
        eyes=build_eye_controller(config.eye_serial_port, config.eye_serial_baudrate),
    )

    audio_preprocessor = AudioPreprocessor() if config.enable_audio_processing else None

    stt = TextInputSTT()
    if voice_mode:
        if not config.vosk_model_path:
            raise SystemExit(
                "Voice mode requires ASSISTANT_VOSK_MODEL pointing to a Vosk model."
            )
        stt = VoskSpeechToText(
            model_path=config.vosk_model_path,
            speech_start_timeout_seconds=config.stt_speech_start_timeout_seconds,
            pre_speech_ms=config.stt_pre_speech_ms,
            audio_preprocessor=audio_preprocessor,
            audio_input_device=config.audio_input_device,
        )

    wakeword_detector = build_wake_word_detector(
        model_path=config.wake_word_model_path or (config.vosk_model_path if voice_mode else None),
        threshold=config.wake_word_threshold,
        keyword=config.wake_word_keyword,
        use_keyword_detector=config.use_keyword_detector,
        audio_input_device=config.audio_input_device,
        keyword_aliases=config.wake_word_aliases,
    )

    return AssistantPipeline(
        state_manager=state_manager,
        stt=stt,
        retriever=ChromaRetriever.from_config(config),
        llm=OllamaLLM.from_config(config),
        answer_tts=build_answer_tts(
            config.tts_command, config.piper_model_path, config.audio_device
        ),
        query_processor=QueryProcessor(),
        wakeword_detector=wakeword_detector,
        enable_barge_in=config.enable_barge_in,
        sqlite_path=config.sqlite_path,
        no_rag=no_rag,
        continuous_conversation=config.continuous_conversation,
        followup_timeout_seconds=config.followup_timeout_seconds,
        wake_greeting_enabled=config.enable_wake_greeting,
        voice_answer_mode=config.voice_answer_mode,
        voice_retrieval_limit=config.voice_retrieval_limit,
        voice_context_chars=config.voice_context_chars,
    )


def start_dashboard(config: AssistantConfig) -> threading.Thread:
    """Start the FastAPI dashboard in a background thread with its own event loop."""
    import uvicorn
    from assistant_app import events
    from assistant_app.dashboard import app

    ready = threading.Event()

    ssl_kwargs: dict = {}
    cert = config.ssl_cert_path
    key = config.ssl_key_path
    if cert and key and Path(cert).exists() and Path(key).exists():
        ssl_kwargs = {"ssl_certfile": cert, "ssl_keyfile": key}
        scheme = "https"
    else:
        scheme = "http"

    host = config.dashboard_host
    port = config.dashboard_port
    print(f"[dashboard] {scheme}://{host}:{port}/monitor", flush=True)

    def run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        events.set_event_loop(loop)
        events.set_sqlite_path(config.sqlite_path)
        ready.set()

        uvicorn_config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
            **ssl_kwargs,
        )
        server = uvicorn.Server(uvicorn_config)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=run, name="dashboard", daemon=True)
    thread.start()
    ready.wait(timeout=5.0)
    return thread


def main() -> None:
    parser = argparse.ArgumentParser(description="Adam — Virtual Receptionist")
    parser.add_argument(
        "--no-filler-voice",
        action="store_true",
        help="Suppress filler voice lines.",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Use microphone input with Vosk STT and wake word detection.",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Run dashboard web server only, no voice pipeline.",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip document retrieval; answer every question directly from the model.",
    )
    args = parser.parse_args()

    config = AssistantConfig.from_env()
    config.enable_filler_voice = not args.no_filler_voice

    dashboard_thread = start_dashboard(config)

    if args.dashboard_only:
        print("[pipeline] Dashboard-only mode. Visit /monitor or /ask.", flush=True)
        try:
            dashboard_thread.join()
        except KeyboardInterrupt:
            print("\n[pipeline] Stopped.", flush=True)
        return

    if args.no_rag:
        os.environ["ASSISTANT_NO_RAG"] = "1"
        print("[pipeline] RAG disabled — all answers from model only.", flush=True)
    pipeline = build_pipeline(config, voice_mode=args.voice, no_rag=args.no_rag)

    # Pre-load the model in the background so the first question has no cold-start penalty.
    warmup_thread = threading.Thread(
        target=pipeline.llm.warmup, name="llm-warmup", daemon=True
    )
    warmup_thread.start()
    print("[pipeline] Warming up LLM in background…", flush=True)
    if args.voice and not args.no_rag:
        threading.Thread(
            target=lambda: pipeline.retriever.search("warmup", limit=1),
            name="retriever-warmup",
            daemon=True,
        ).start()
        print("[pipeline] Warming up retriever in background…", flush=True)

    try:
        pipeline.run_text_demo()
    except KeyboardInterrupt:
        print("\n[pipeline] Stopped.", flush=True)
