# Offline Humanized Voice Assistant

A local, privacy-focused assistant prototype with humanized voice/eye feedback and a low-latency response pipeline. The target deployment hardware is the NVIDIA Jetson Orin Nano Super Developer Kit running Ubuntu Server through JetPack / Jetson Linux.

## Project Structure

```text
.
├── README.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── docs/
│   ├── docker_deployment.md
│   ├── project_implementation_plan.md
│   └── progress_notes.md
├── src/
│   └── assistant_app/
│       ├── app.py
│       ├── audit.py
│       ├── config.py
│       ├── dashboard.py
│       ├── embeddings.py
│       ├── eyes.py
│       ├── ingest.py
│       ├── llm.py
│       ├── metadata.py
│       ├── pipeline.py
│       ├── query.py
│       ├── retrieval.py
│       ├── state.py
│       ├── stt.py
│       ├── tts.py
│       └── validation.py
├── documents/      # PDFs, text files, and markdown files to index
├── data/chroma/    # Generated ChromaDB vector store
├── models/         # Vosk/Piper/local model files
│   └── embeddings/  # Local sentence-transformers embedding model
├── scripts/        # Utility scripts for deployment later
└── tests/          # Tests
```

## Local Virtual Environment

This workspace has been tested with a project-local `.venv` for text/document mode:

```bash
.venv/bin/python -m assistant_app
.venv/bin/python -m assistant_app.ingest ./documents
.venv/bin/python -m assistant_app.speak "Hello, I am ready."
```

Voice mode needs system Python headers and PortAudio/ALSA libraries. These were installed on this machine with:

```bash
sudo apt install python3.12-dev portaudio19-dev libasound2-dev libportaudio2 alsa-utils
```

Then full dependencies were installed with:

```bash
.venv/bin/python -m pip install -e ".[all]"
```

## Quick Start

Run without installing the package:

```bash
PYTHONPATH=src python3 -m assistant_app
```

Or install it in editable mode:

```bash
pip install -e .
assistant
```

Type a question. Type `exit` to quit.

## Install Optional Dependencies

Minimal demo:

```bash
pip install -e .
```

Full voice and document mode:

```bash
pip install -e ".[all]"
```

The `requirements.txt` file is also kept for simple installation on devices that prefer it:

```bash
pip install -r requirements.txt
```

## Voice Mode

Use microphone input with Vosk and voice activity detection:

```bash
ASSISTANT_VOSK_MODEL=./models/vosk PYTHONPATH=src python3 -m assistant_app --voice
```

After editable install, you can use:

```bash
ASSISTANT_VOSK_MODEL=./models/vosk assistant --voice
```

Voice mode is designed to behave like a continuous local voice assistant rather than a one-question kiosk. Say the wake word once, wait for the short acknowledgment such as Yes? or I'm listening, then ask your question. After Adam answers, it keeps listening for follow-up questions for 45 seconds by default, so you do not need to repeat the wake word for every turn. Tune this with ASSISTANT_FOLLOWUP_TIMEOUT_SECONDS, or disable it with ASSISTANT_CONTINUOUS_CONVERSATION=0.

For noisy rooms, add comma-separated wake-word mishearings with ASSISTANT_WAKE_WORD_ALIASES, for example adam,atom,add him. STT keeps a small pre-speech buffer through ASSISTANT_STT_PRE_SPEECH_MS so the beginning of a question is less likely to be clipped.

Voice answers now use the same answer strategy as the web Ask page: relevant document questions use retrieved local knowledge, weak or irrelevant document matches fall back to general receptionist answers, and institution-specific questions without uploaded official knowledge ask for the missing brochure, policy, contact sheet, or source file instead of guessing. Short template answers handle greetings, thanks, and similar conversational turns instantly.

While a voice answer is generating, Adam may say brief thinking feedback such as One moment or Let me check that. As soon as the real answer is ready, filler audio is stopped and interrupted so it does not talk over the answer.

## Voice Configuration

Common voice settings are loaded from .env.example when running through adam.service:

    ASSISTANT_WAKE_WORD_KEYWORD=adam
    ASSISTANT_WAKE_WORD_ALIASES=adam,addam,atom,add em,add him,at him
    ASSISTANT_CONTINUOUS_CONVERSATION=1
    ASSISTANT_FOLLOWUP_TIMEOUT_SECONDS=45
    ASSISTANT_ENABLE_WAKE_GREETING=1
    ASSISTANT_STT_SPEECH_START_TIMEOUT_SECONDS=10
    ASSISTANT_STT_PRE_SPEECH_MS=300
    ASSISTANT_ENABLE_BARGE_IN=1
    ASSISTANT_VOICE_ANSWER_MODE=fast
    ASSISTANT_VOICE_RETRIEVAL_LIMIT=2
    ASSISTANT_VOICE_CONTEXT_CHARS=900

`ASSISTANT_VOICE_ANSWER_MODE=fast` is the low-latency voice path. When Adam finds a relevant knowledge chunk, it speaks a short direct answer from the top retrieved source without waiting for Ollama generation. Set it to `llm` when you prefer generated wording from compact retrieved context.

Restart the service after changing code or environment values:

    sudo systemctl restart adam.service
    systemctl status adam.service --no-pager
    journalctl -u adam.service --since '2 minutes ago' --no-pager

## Text-to-Speech

Final answers use console voice output by default. For Piper on Jetson, set a local Piper voice model and optional ALSA audio device:

```bash
ASSISTANT_PIPER_MODEL=./models/piper/voice.onnx
ASSISTANT_AUDIO_DEVICE=default
# This machine also played the Piper test WAV through HDMI with: ASSISTANT_AUDIO_DEVICE=plughw:0,3
```

The Piper adapter creates a temporary WAV file and plays it with `aplay`. Use `ASSISTANT_AUDIO_DEVICE=plughw:1,0` if your USB audio codec appears as that ALSA device. In voice mode, wake acknowledgments, thinking feedback, and final answers all use the configured answer TTS path.

For a fully custom command instead, set `ASSISTANT_TTS_COMMAND`. This command receives answer text through standard input and overrides the Piper settings.

Test the configured speaker/TTS path with:

```bash
PYTHONPATH=src python3 -m assistant_app.speak "Hello, I am ready."
```

After editable install:

```bash
assistant-speak "Hello, I am ready."
```

## Local Embedding Model

For fully offline document search, place a local sentence-transformers embedding model in:

```text
models/embeddings/all-MiniLM-L6-v2
```

Or point the app to another local model folder:

```bash
ASSISTANT_EMBEDDING_MODEL=./models/embeddings/your-model
ASSISTANT_EMBEDDING_DEVICE=cpu
```

The same embedding model must be used for ingestion and retrieval.

## Knowledge Dashboard

You can upload supported knowledge files from a local browser dashboard:

```bash
.venv/bin/python -m assistant_app.dashboard
```

Then open:

```text
http://127.0.0.1:8080
```

For Docker or LAN testing, set `ASSISTANT_DASHBOARD_HOST=0.0.0.0`. Keep the default `127.0.0.1` for local-only privacy.

For low-latency AI-generated dashboard answers, use the 1B Ollama model and capped generation settings:

```bash
ASSISTANT_DASHBOARD_ANSWER_MODE=llm \
ASSISTANT_OLLAMA_MODEL=llama3.2:1b \
ASSISTANT_OLLAMA_NUM_PREDICT=120 \
ASSISTANT_OLLAMA_NUM_CTX=1536 \
ASSISTANT_DASHBOARD_LLM_CONTEXT_CHARS=1600 \
ASSISTANT_DASHBOARD_LLM_RETRIEVAL_LIMIT=2 \
ASSISTANT_DASHBOARD_HOST=0.0.0.0 \
ASSISTANT_DASHBOARD_PORT=8092 \
.venv/bin/python -m assistant_app.dashboard
```

Set `ASSISTANT_DASHBOARD_ANSWER_MODE=extractive` only when you need sub-second direct source excerpts without AI generation.

The dashboard supports `.pdf`, `.txt`, and `.md` files. Uploaded files are saved under `documents/`, and the **Ingest Documents** button builds the SQLite + ChromaDB search index. Use the **Ask Documents** link, or open `/ask`, to type questions and receive answers with source labels from the indexed files.

The dashboard keeps persistent local records in SQLite:

- `/ask` is a modern virtual receptionist chat interface with New Chat, recent chat sessions, per-session history, streamed local model output, source chips, copy actions, saved user/assistant turns, document relevance checks, and general receptionist answers when no uploaded knowledge file matches.
- `/monitor` shows index health, chat records, voice events/logs, source labels, latency averages, Ollama/model status, document usage, error events, upload/ingest/delete events, system resources, and export links.
- `/monitor/interaction/{id}` shows the full saved question, answer, sources, model, and timing for one chat record.
- `/monitor/export/interactions.csv`, `/monitor/export/interactions.json`, `/monitor/export/events.csv`, and `/monitor/export/events.json` export local testing/evaluation records.
- `data/assistant.db` stores the dashboard history, event log, voice logs, feedback, chat sessions, and document metadata.

After editable install, you can also run:

```bash
assistant-dashboard
```

## Preload Documents

Process documents before runtime so questions search instantly. Ingestion now extracts PDF text with PyMuPDF, saves source/page/section metadata in SQLite, and stores searchable chunks in ChromaDB:

```bash
PYTHONPATH=src python3 -m assistant_app.ingest ./documents
```

After editable install:

```bash
assistant-ingest ./documents
```

Supported file types:

- `.pdf` with page numbers
- `.txt`
- `.md`

Generated local data:

- `data/assistant.db` stores document and chunk metadata in SQLite.
- `data/chroma/` stores vector search data in ChromaDB.

## ESP32-S3 Eye Feedback

By default, eye states are printed to the console. To send state commands to an ESP32-S3 over serial, set:

```bash
ASSISTANT_EYE_SERIAL_PORT=/dev/ttyUSB0
ASSISTANT_EYE_SERIAL_BAUDRATE=115200
```

The Python app sends simple newline-terminated commands:

```text
IDLE
LISTENING
THINKING
SEARCHING
READY
SPEAKING
ERROR
```

## Docker Deployment

Container deployment instructions are in [`docs/docker_deployment.md`](docs/docker_deployment.md). The Docker setup mounts `documents/`, `data/`, and `models/` as host folders so the same project can be moved to another system without baking private documents or large model files into the image.

Quick Docker commands:

```bash
docker compose build
docker compose --profile tools run --rm ingest
docker compose run --rm assistant
```

## Thesis Plan

The thesis-ready implementation plan is in [`docs/project_implementation_plan.md`](docs/project_implementation_plan.md). It covers the Ubuntu Server and Jetson Orin Nano Super hardware setup, audio devices, ESP32-S3 robotic eyes, privacy method, local processing pipeline, and evaluation criteria.

## Progress Notes

For new chat continuity, read [`docs/progress_notes.md`](docs/progress_notes.md). It records what has already been completed, verified commands, important files, and the next recommended work.

## Runtime Services

Keep Ollama warm before asking questions:

```bash
ollama serve
ollama run llama3.2:latest ""
```

Recommended environment variables:

```bash
ASSISTANT_OLLAMA_MODEL=llama3.2:latest
ASSISTANT_OLLAMA_URL=http://127.0.0.1:11434
ASSISTANT_CHROMA_PATH=./data/chroma
ASSISTANT_CHROMA_COLLECTION=documents
ASSISTANT_SQLITE_PATH=./data/assistant.db
ASSISTANT_EMBEDDING_MODEL=./models/embeddings/all-MiniLM-L6-v2
ASSISTANT_EMBEDDING_DEVICE=cpu
ASSISTANT_PIPER_MODEL=./models/piper/voice.onnx
ASSISTANT_AUDIO_DEVICE=default
# This machine also played the Piper test WAV through HDMI with: ASSISTANT_AUDIO_DEVICE=plughw:0,3
ASSISTANT_TTS_COMMAND=
ASSISTANT_EYE_SERIAL_PORT=/dev/ttyUSB0
ASSISTANT_EYE_SERIAL_BAUDRATE=115200
ASSISTANT_VOSK_MODEL=./models/vosk
ASSISTANT_DASHBOARD_HOST=127.0.0.1
ASSISTANT_DASHBOARD_PORT=8080
```

## Pipeline

```text
Wake Word Detection
↓
Humanized Listening Feedback
↓
Microphone Input
↓
Voice Activity Detection
↓
Streaming Speech-to-Text
↓
Query Processing
↓
Intent Routing and Query Reformulation
↓
Humanized Processing Feedback
↓
Document Retrieval from ChromaDB
↓
Metadata Lookup from SQLite
↓
Grounded Answer Validation
↓
Local LLM Answer Generation with Ollama
↓
Streaming Text-to-Speech
↓
Speaker Output
↓
Robotic Eye Feedback
```
