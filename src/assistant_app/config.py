from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class AssistantConfig:
    ollama_model: str = "llama3.2:latest"
    ollama_url: str = "http://127.0.0.1:11434"
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "documents"
    sqlite_path: str = "./data/assistant.db"
    embedding_model_path: str = "./models/embeddings/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    piper_model_path: str | None = None
    audio_device: str | None = None
    audio_input_device: str | None = None
    eye_serial_port: str | None = None
    eye_serial_baudrate: int = 115200
    tts_command: str | None = None
    vosk_model_path: str | None = None
    wake_word_model_path: str | None = None
    wake_word_threshold: float = 0.5
    wake_word_keyword: str = "adam"
    wake_word_aliases: tuple[str, ...] = ()
    use_keyword_detector: bool = True
    enable_barge_in: bool = True
    enable_audio_processing: bool = False
    enable_filler_voice: bool = True
    enable_wake_greeting: bool = False
    continuous_conversation: bool = True
    followup_timeout_seconds: float = 45.0
    stt_speech_start_timeout_seconds: float = 10.0
    stt_pre_speech_ms: int = 300
    voice_answer_mode: str = "fast"
    voice_retrieval_limit: int = 3
    voice_context_chars: int = 1800
    dashboard_answer_mode: str = "llm"
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8092
    ssl_cert_path: str = "./ssl/cert.pem"
    ssl_key_path: str = "./ssl/key.pem"
    ollama_num_predict: int = 400
    ollama_num_ctx: int = 3072
    ollama_temperature: float = 0.2
    dashboard_llm_context_chars: int = 2400
    dashboard_llm_retrieval_limit: int = 3
    no_rag: bool = False

    @classmethod
    def from_env(cls) -> "AssistantConfig":
        return cls(
            ollama_model=os.getenv("ASSISTANT_OLLAMA_MODEL", cls.ollama_model),
            ollama_url=os.getenv("ASSISTANT_OLLAMA_URL", cls.ollama_url),
            chroma_path=os.getenv("ASSISTANT_CHROMA_PATH", cls.chroma_path),
            chroma_collection=os.getenv(
                "ASSISTANT_CHROMA_COLLECTION", cls.chroma_collection
            ),
            sqlite_path=os.getenv("ASSISTANT_SQLITE_PATH", cls.sqlite_path),
            embedding_model_path=os.getenv(
                "ASSISTANT_EMBEDDING_MODEL", cls.embedding_model_path
            ),
            embedding_device=os.getenv("ASSISTANT_EMBEDDING_DEVICE", cls.embedding_device),
            piper_model_path=os.getenv("ASSISTANT_PIPER_MODEL"),
            audio_device=os.getenv("ASSISTANT_AUDIO_DEVICE"),
            audio_input_device=os.getenv("ASSISTANT_AUDIO_INPUT_DEVICE"),
            eye_serial_port=os.getenv("ASSISTANT_EYE_SERIAL_PORT"),
            eye_serial_baudrate=int(
                os.getenv("ASSISTANT_EYE_SERIAL_BAUDRATE", cls.eye_serial_baudrate)
            ),
            tts_command=os.getenv("ASSISTANT_TTS_COMMAND"),
            vosk_model_path=os.getenv("ASSISTANT_VOSK_MODEL"),
            wake_word_model_path=os.getenv("ASSISTANT_WAKE_WORD_MODEL"),
            wake_word_threshold=float(
                os.getenv("ASSISTANT_WAKE_WORD_THRESHOLD", cls.wake_word_threshold)
            ),
            wake_word_keyword=os.getenv("ASSISTANT_WAKE_WORD_KEYWORD", cls.wake_word_keyword).lower().strip(),
            wake_word_aliases=tuple(
                alias.strip().lower()
                for alias in os.getenv("ASSISTANT_WAKE_WORD_ALIASES", "").split(",")
                if alias.strip()
            ),
            use_keyword_detector=os.getenv("ASSISTANT_USE_KEYWORD_DETECTOR", "1").lower()
            in ("1", "true", "yes"),
            enable_barge_in=os.getenv("ASSISTANT_ENABLE_BARGE_IN", "1").lower()
            in ("1", "true", "yes"),
            enable_audio_processing=os.getenv(
                "ASSISTANT_ENABLE_AUDIO_PROCESSING", ""
            ).lower()
            in ("1", "true", "yes"),
            enable_wake_greeting=os.getenv(
                "ASSISTANT_ENABLE_WAKE_GREETING", ""
            ).lower()
            in ("1", "true", "yes"),
            continuous_conversation=os.getenv(
                "ASSISTANT_CONTINUOUS_CONVERSATION", "1"
            ).lower()
            in ("1", "true", "yes"),
            followup_timeout_seconds=float(
                os.getenv(
                    "ASSISTANT_FOLLOWUP_TIMEOUT_SECONDS",
                    cls.followup_timeout_seconds,
                )
            ),
            stt_speech_start_timeout_seconds=float(
                os.getenv(
                    "ASSISTANT_STT_SPEECH_START_TIMEOUT_SECONDS",
                    cls.stt_speech_start_timeout_seconds,
                )
            ),
            stt_pre_speech_ms=int(
                os.getenv("ASSISTANT_STT_PRE_SPEECH_MS", cls.stt_pre_speech_ms)
            ),
            voice_answer_mode=os.getenv(
                "ASSISTANT_VOICE_ANSWER_MODE", cls.voice_answer_mode
            ).strip().lower(),
            voice_retrieval_limit=int(
                os.getenv("ASSISTANT_VOICE_RETRIEVAL_LIMIT", cls.voice_retrieval_limit)
            ),
            voice_context_chars=int(
                os.getenv("ASSISTANT_VOICE_CONTEXT_CHARS", cls.voice_context_chars)
            ),
            dashboard_answer_mode=os.getenv(
                "ASSISTANT_DASHBOARD_ANSWER_MODE", cls.dashboard_answer_mode
            ).strip().lower(),
            ollama_num_predict=int(
                os.getenv("ASSISTANT_OLLAMA_NUM_PREDICT", cls.ollama_num_predict)
            ),
            ollama_num_ctx=int(os.getenv("ASSISTANT_OLLAMA_NUM_CTX", cls.ollama_num_ctx)),
            ollama_temperature=float(
                os.getenv("ASSISTANT_OLLAMA_TEMPERATURE", cls.ollama_temperature)
            ),
            dashboard_llm_context_chars=int(
                os.getenv(
                    "ASSISTANT_DASHBOARD_LLM_CONTEXT_CHARS",
                    cls.dashboard_llm_context_chars,
                )
            ),
            dashboard_llm_retrieval_limit=int(
                os.getenv(
                    "ASSISTANT_DASHBOARD_LLM_RETRIEVAL_LIMIT",
                    cls.dashboard_llm_retrieval_limit,
                )
            ),
            dashboard_host=os.getenv("ASSISTANT_DASHBOARD_HOST", cls.dashboard_host),
            dashboard_port=int(os.getenv("ASSISTANT_DASHBOARD_PORT", cls.dashboard_port)),
            ssl_cert_path=os.getenv("ASSISTANT_SSL_CERT", cls.ssl_cert_path),
            ssl_key_path=os.getenv("ASSISTANT_SSL_KEY", cls.ssl_key_path),
            no_rag=os.getenv("ASSISTANT_NO_RAG", "").lower() in ("1", "true", "yes"),
        )
