# Progress Notes for New Chat Context

This file records what has already been built in this project so future chats can continue without guessing.

## Current Project Goal

Build a fully offline, privacy-focused, voice-enabled PDF query assistant for the NVIDIA Jetson Orin Nano Super Developer Kit. The assistant should run on Ubuntu Server through JetPack / Jetson Linux, accept spoken questions, search local PDF documents, generate answers locally with Ollama, speak answers with Piper, and show assistant states through ESP32-S3 robotic eyes.

## Current Directory Structure

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
├── src/assistant_app/
├── documents/
├── data/chroma/
├── models/
│   └── embeddings/
├── scripts/
└── tests/
```

## Completed Work

1. Created a proper Python `src/` project layout.
2. Added installable package metadata in `pyproject.toml`.
3. Added command entry points:
   - `assistant`
   - `assistant-ingest`
   - `assistant-speak`
   - `assistant-dashboard`
4. Added a humanized response state manager with states:
   - idle
   - listening
   - processing
   - searching documents
   - answer ready
   - speaking
   - error
5. Added console eye feedback adapter as a placeholder for ESP32-S3 robotic eyes.
6. Added console TTS adapter as a placeholder/fallback for Piper.
7. Added text demo input mode for development without microphone hardware.
8. Added Vosk microphone STT adapter with voice activity detection logic.
9. Added Ollama streaming LLM adapter with fallback behavior when Ollama is unavailable.
10. Added ChromaDB retrieval adapter.
11. Added SQLite metadata store in `src/assistant_app/metadata.py`.
12. Rebuilt document ingestion so it now:
    - supports `.pdf`, `.txt`, and `.md`
    - extracts PDFs using PyMuPDF
    - preserves PDF page numbers
    - chunks text with overlap
    - stores document/chunk metadata in SQLite
    - stores searchable chunks in ChromaDB
13. Updated retrieval so retrieved chunks include source, page, and section labels.
14. Added thesis-ready implementation plan in `docs/project_implementation_plan.md`.
15. Added tests for chunk creation and SQLite metadata round-trip.
16. Added `.gitignore` for Python cache files, local data, documents, models, and environment files.
17. Added local sentence-transformers embedding configuration for ChromaDB ingestion and retrieval.
18. Added grounded-answer validation so the LLM is not called when no document context is retrieved.
19. Added Docker deployment files: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.env.example`, and `docs/docker_deployment.md`.
20. Wired `ASSISTANT_TTS_COMMAND` into final answer TTS with console fallback.
21. Added `assistant-speak` / `python -m assistant_app.speak` to test TTS and speaker output directly.
22. Added first-class Piper TTS playback using temporary WAV output and `aplay`, with optional `ASSISTANT_AUDIO_DEVICE` for Jetson/ALSA speaker selection.
23. Added explicit query processing: intent routing, simple conversation memory, and follow-up query reformulation.
24. Added optional ESP32-S3 serial eye controller with commands `IDLE`, `LISTENING`, `THINKING`, `SEARCHING`, `READY`, `SPEAKING`, and `ERROR`.
25. Created a local `.venv`, installed text/document dependencies, downloaded `all-MiniLM-L6-v2`, ingested a sample document, and verified retrieval plus Ollama `llama3.2:latest` answer generation.
26. Increased Ollama timeout to 120 seconds so local model warm-up does not fail during first response.
27. Installed system packages `python3.12-dev`, `portaudio19-dev`, `libasound2-dev`, `libportaudio2`, and `alsa-utils`; full `.venv/bin/python -m pip install -e ".[all]"` now succeeds, including Vosk, sounddevice, and webrtcvad.
28. Added `assistant-dashboard` / `python -m assistant_app.dashboard`, a local FastAPI dashboard for uploading `.pdf`, `.txt`, and `.md` files and triggering ingestion into SQLite + ChromaDB.
29. Fixed `assistant-speak` so it honors `ASSISTANT_PIPER_MODEL` and `ASSISTANT_AUDIO_DEVICE`, not only `ASSISTANT_TTS_COMMAND`.
30. Fixed Docker/package extras so `.[all]` includes dashboard dependencies and the image includes `alsa-utils` for Piper playback.
31. Added dashboard host/port environment variables for container use.
32. Added index cleanup so reingesting or deleting documents removes old SQLite/Chroma chunks.
33. Tightened the Ollama prompt so answers must stay grounded in retrieved document context.
34. Added a dashboard **Ask Documents** page at `/ask` for typing questions and getting source-labeled answers from indexed PDFs/text files.
35. Downloaded and unpacked `vosk-model-small-en-us-0.15` into `models/vosk`; verified the Vosk model loads successfully.
36. Downloaded Piper `en_US-amy-low` voice into `models/piper`; installed `piper-tts` in `.venv`; verified WAV generation.
37. Fixed Piper command resolution so `.venv/bin/python -m assistant_app.speak` can find the venv-installed `piper` binary.
38. Tested Piper playback successfully through HDMI ALSA device `plughw:0,3`; the USB audio codec is not visible yet in `aplay -l`.
39. Redesigned the `/ask` dashboard page into a modern chat-style interface with sidebar navigation, assistant/user message bubbles, source pills, and a fixed bottom composer.
40. Added the active Ollama answer model name to the `/ask` dashboard sidebar/status; current model is `llama3.2:latest`.
41. Restarted the updated dashboard on LAN port `8092` and verified the question `what is design of robo eyes` returns an answer from `For_designer_.pdf`.
42. Fixed dashboard ask-state handling so the global page state is only updated after the answer is ready, preventing half-updated pages with a question but default answer.
43. Added a browser-side loading indicator on `/ask` that immediately shows searching/generating feedback while local Ollama is working.
44. Added cached dashboard services so the retriever, embedding model, query processor, validator, and Ollama adapter stay warm between dashboard questions.
45. Reduced dashboard retrieval from 4 chunks to 3 chunks to shorten the prompt and lower generation time.
46. Added latency timing to dashboard answers, showing total, retrieval, and generation seconds.
47. Started a test dashboard on port `8094` using `llama3.2:1b`; measured first request 32.65s including embedding warm-up, then warm request 5.52s total with 0.25s retrieval and 5.27s generation.
48. Changed dashboard default strategy from LLM generation to extractive retrieval answers with `ASSISTANT_DASHBOARD_ANSWER_MODE=extractive`; Ollama generation is still available with `ASSISTANT_DASHBOARD_ANSWER_MODE=llm`.
49. Added cheap keyword reranking for retrieved chunks so extractive answers pick more specific chunks such as `For_designer_.pdf, page 2, section 3` for robo-eye questions.
50. Final low-latency benchmark on port `8092`: first request 10.78s including embedding warm-up, warm requests 0.28s and 0.25s with source `For_designer_.pdf, page 2, section 3`.
51. Removed dashboard chat-history persistence: `/ask` no longer stores the last question, answer, or sources in global state, and dashboard question processing no longer uses remembered conversation turns.
52. Restarted port `8092` and verified POST answers work while a fresh GET `/ask` does not contain the previous question or answer.
53. Reworked dashboard for low-latency AI generation: default answer mode is now `llm`, Ollama generation is capped with `ASSISTANT_OLLAMA_NUM_PREDICT`, `ASSISTANT_OLLAMA_NUM_CTX`, and low temperature, and dashboard context is compacted before calling Ollama.
54. Tested `llama3.2:1b` AI-fast mode on port `8092`: first request 26.14s including embedding warm-up, warm request 3.13s total with 0.25s retrieval and 2.86s generation.
55. Updated prompt wording so informal questions like `robo eyes` are matched to robot eye modules/eye displays in the retrieved context.
56. Added persistent dashboard audit/history storage in SQLite through `src/assistant_app/audit.py`.
57. Updated `/ask` so every submitted question, generated answer, source list, answer mode, AI model, and latency timing is saved and recent chat history is shown on the page.
58. Added `/monitor`, a tracking dashboard for chat history, knowledge file count, answer mode/model, latency, upload events, ingest events, rejected uploads, and delete events.
59. Added tests for audit storage, chat-history rendering, interaction logging, and monitor-page rendering.
60. Restarted the LAN dashboard on port `8092` with `llama3.2:1b`, verified `/ask` saves chat history, and verified `/monitor` shows records, events, file count, model, and latency.
61. Expanded `/monitor` into a fuller operations dashboard with index health, latency averages, model/Ollama status, document usage, error log, system resource readings, interaction detail pages, and CSV/JSON export routes.
62. Restarted port `8092` and verified the expanded monitor page, interaction detail page, chat CSV export, and events JSON export.
63. Reworked `/ask` into a ChatGPT-style chat page with continuous user/assistant turns, browser-side streaming from `/ask/stream`, live token display, source chips, and SQLite-backed history rendered as conversation turns.
64. Restarted port `8092` and verified `/ask/stream` returns newline-delimited streaming events while the dashboard serves the updated chat page.
65. Added real chat sessions: SQLite `chat_sessions`, `session_id` on interactions, `/ask/sessions`, `/ask/session/{session_id}`, per-session streaming, New Chat button, recent chat sidebar, delete current chat, copy chat/answer controls, and suggestion chips.
66. Restarted port `8092` and verified session listing, session-aware `/ask/stream`, and the modern New Chat UI.
67. Changed answer strategy to virtual receptionist mode: weak document matches are ignored, relevant document questions still use sources, general questions use a general receptionist prompt, and institution-specific questions without local documents return a safe upload/clarify response instead of hallucinated facts.
68. Restarted port `8092` and verified `i mean VBC college` no longer answers from the unrelated VBC connector PDF context.
69. Fixed the `/ask` ChatGPT-style page so initial session JSON no longer breaks JavaScript, Enter sends the question, Shift+Enter adds a new line, and the form has a POST fallback.
70. Restarted the LAN dashboard on port `8092` and verified `/ask/stream` returns status, token, and done events for a general receptionist question.
71. Fixed `/ask` layout so the sidebar stays fixed while only the chat transcript scrolls, and improved New Chat so it aborts any active stream, clears the composer, creates a visible fresh session, and opens an empty chat view.
72. Restarted the LAN dashboard on port `8092` and verified the served `/ask` HTML contains the fixed non-scrolling sidebar layout and improved New Chat JavaScript.
73. Fixed New Chat on LAN HTTP pages by replacing direct `crypto.randomUUID()` use with a safe fallback session ID generator for browsers where randomUUID is unavailable outside secure contexts.
74. Hardened New Chat further by turning it into a real `/ask?new=1` link with JavaScript enhancement, adding force-new-chat page-load handling, and keeping the instant client-side reset behavior.
75. Improved answer strategy for low-latency receptionist behavior: added instant template replies for greetings, stricter no-filler grounded prompts, lightweight per-session follow-up rewriting, and top-2 visible source limiting.
76. Added a Stop Generating control to the `/ask` page so long local Ollama responses can be aborted from the dashboard without waiting for completion.
77. Redesigned `/monitor` into a compact operations console with sticky sidebar/topbar, metric cards, scrollable chat history/events/error panels, searchable tables, compact badges, source pills, and runtime/model/source panels so history and events no longer stretch one long page.

78. Added Vosk wake-word alias matching for common mishearings such as atom, add him, and add em.
79. Changed voice mode to a continuous conversation flow: one wake word starts the session, then Adam listens for follow-up questions for 45 seconds before returning to wake-word mode.
80. Added short spoken wake acknowledgments such as Yes? and I'm listening so the user can tell Adam woke up.
81. Added STT speech-start timeout and pre-speech buffering so the first words of a question are less likely to be clipped.
82. Aligned voice answers with the web Ask strategy: relevant document questions use local knowledge, irrelevant document matches fall back to general receptionist answers, and institution-specific questions without uploaded official knowledge ask for the missing source file instead of guessing.
83. Added instant local voice replies for greetings, thanks, short acknowledgments, and similar conversational turns.
84. Tuned voice LLM generation to use shorter spoken answers and smaller voice generation caps so final speech starts sooner.
85. Added spoken thinking feedback during answer generation, then fixed it so filler audio is stopped and interrupted before final answer playback begins.
86. Added voice logging in SQLite through voice_logs, including wake time, transcript, answer, source type, filler count, barge-in, STT/retrieval/generation timing, resource readings, and errors.
87. Added focused voice-flow tests covering wake matching, continuous follow-up listening, and general-answer fallback when retrieved documents are irrelevant.
88. Restarted adam.service after the voice fixes and verified it runs under systemd with the updated environment.

## Important Files

- `README.md`: user-facing setup and run instructions.
- `Dockerfile`: container image definition for portable deployment.
- `docker-compose.yml`: container run configuration with mounted local data/model folders.
- `docs/docker_deployment.md`: Docker build, ingest, run, audio, and Jetson notes.
- `docs/project_implementation_plan.md`: thesis-ready hardware/software implementation plan.
- `docs/progress_notes.md`: this continuity note.
- `src/assistant_app/app.py`: CLI app entry point.
- `src/assistant_app/audit.py`: SQLite audit/history store for dashboard chat records, latency stats, source usage, exports, and system events.
- `src/assistant_app/dashboard.py`: local browser dashboard for uploading, listing, deleting, ingesting, asking questions, chat history, and monitoring.
- `src/assistant_app/embeddings.py`: local sentence-transformers embedding function for ChromaDB.
- `src/assistant_app/pipeline.py`: main interaction pipeline, including continuous voice conversation, voice answer routing, thinking feedback, and voice logging.
- `src/assistant_app/query.py`: intent router, conversation manager, and query reformulation.
- `src/assistant_app/state.py`: response state manager.
- `src/assistant_app/eyes.py`: console and optional ESP32-S3 serial eye controllers.
- `src/assistant_app/ingest.py`: document ingestion into SQLite and ChromaDB.
- `src/assistant_app/metadata.py`: SQLite metadata store plus index health/statistics helpers.
- `src/assistant_app/retrieval.py`: ChromaDB retrieval plus metadata formatting.
- `src/assistant_app/stt.py`: text input and Vosk/VAD STT adapter with speech-start timeout and pre-speech buffering.
- `src/assistant_app/speak.py`: standalone TTS/speaker test command.
- `src/assistant_app/llm.py`: Ollama streaming adapter.
- `src/assistant_app/tts.py`: TTS adapters, Piper WAV playback, ALSA device support, and `ASSISTANT_TTS_COMMAND` factory.
- `src/assistant_app/validation.py`: grounded-answer context validator.
- `src/assistant_app/wakeword.py`: Always-ready, OpenWakeWord, and Vosk keyword wake-word detectors with alias/fuzzy matching.
- `src/assistant_app/audio_processing.py`: optional NumPy-based audio preprocessing for noise suppression and automatic gain control.
- `tests/test_ingest_metadata.py`: ingestion and metadata tests.
- `tests/test_embeddings.py`: local embedding configuration tests.
- `tests/test_validation_pipeline.py`: grounded answer validation pipeline tests.
- `tests/test_tts.py`: TTS command parsing and console fallback tests.
- `tests/test_speak.py`: assistant-speak argument/stdin behavior tests.
- `tests/test_query.py`: query processor and follow-up reformulation tests.
- `tests/test_eyes.py`: ESP32 serial eye command mapping tests.
- `tests/test_dashboard.py`: dashboard file naming, delete/index cleanup, source extraction, web question-answer, history, monitor, and operations panel tests.
- `tests/test_audit.py`: persistent dashboard audit/history store tests.
- `tests/test_llm.py`: grounded Ollama prompt tests.
- `tests/test_voice_flow.py`: focused tests for wake matching, continuous conversation, and voice general-answer fallback.

## Verified Commands

These commands were run successfully:

```bash
PYTHONPATH=src python3 -m compileall src/assistant_app
PYTHONPATH=src python3 -m unittest discover -s tests
.venv/bin/python -m compileall src/assistant_app
.venv/bin/python -m unittest discover -s tests
python3 -m compileall src/assistant_app tests/test_voice_flow.py
PYTHONPATH=src python3 -m unittest tests.test_voice_flow
sudo systemctl restart adam.service
systemctl status adam.service --no-pager
.venv/bin/python - <<'PY'
import sounddevice, vosk, webrtcvad
print("voice imports ok")
PY
PYTHONPATH=src python3 -m assistant_app.ingest --help
.venv/bin/python -m assistant_app.ingest ./documents
.venv/bin/python -m assistant_app.dashboard
curl -fsS http://127.0.0.1:8080/
PYTHONPATH=src python3 -m assistant_app.speak "Hello, I am ready."
docker compose config
ASSISTANT_DASHBOARD_HOST=0.0.0.0 ASSISTANT_DASHBOARD_PORT=8092 .venv/bin/python -m assistant_app.dashboard
curl -fsS http://127.0.0.1:8092/ask
curl -fsS http://127.0.0.1:8092/monitor
.venv/bin/python - <<'PY'
import vosk
vosk.Model("models/vosk")
print("vosk model loaded ok")
PY
.venv/bin/piper --model models/piper/en_US-amy-low.onnx --output_file /tmp/assistant-piper-test.wav
aplay -D plughw:0,3 /tmp/assistant-piper-test.wav
printf 'what is indexed?\nexit\n' | PYTHONPATH=src python3 -m assistant_app
```

## Current Run Commands

Run text-mode demo:

```bash
PYTHONPATH=src python3 -m assistant_app
```

Ingest documents:

```bash
PYTHONPATH=src python3 -m assistant_app.ingest ./documents
```

Run voice mode after installing optional dependencies and adding a Vosk model:

```bash
ASSISTANT_VOSK_MODEL=./models/vosk PYTHONPATH=src python3 -m assistant_app --voice
```

Build and run with Docker:

```bash
docker compose build
docker compose --profile tools run --rm ingest
docker compose run --rm assistant
```

## Environment Variables

```bash
ASSISTANT_OLLAMA_MODEL=llama3.2:latest
ASSISTANT_OLLAMA_URL=http://127.0.0.1:11434
ASSISTANT_CHROMA_PATH=./data/chroma
ASSISTANT_CHROMA_COLLECTION=documents
ASSISTANT_SQLITE_PATH=./data/assistant.db
ASSISTANT_EMBEDDING_MODEL=./models/embeddings/all-MiniLM-L6-v2
ASSISTANT_EMBEDDING_DEVICE=cpu
ASSISTANT_PIPER_MODEL=./models/piper/en_US-amy-low.onnx
ASSISTANT_AUDIO_DEVICE=default
ASSISTANT_TTS_COMMAND=
ASSISTANT_EYE_SERIAL_PORT=/dev/ttyUSB0
ASSISTANT_EYE_SERIAL_BAUDRATE=115200
ASSISTANT_VOSK_MODEL=./models/vosk
ASSISTANT_DASHBOARD_HOST=127.0.0.1
ASSISTANT_DASHBOARD_PORT=8080
ASSISTANT_DASHBOARD_ANSWER_MODE=llm
ASSISTANT_OLLAMA_NUM_PREDICT=120
ASSISTANT_OLLAMA_NUM_CTX=1536
ASSISTANT_OLLAMA_TEMPERATURE=0.1
ASSISTANT_DASHBOARD_LLM_CONTEXT_CHARS=1600
ASSISTANT_DASHBOARD_LLM_RETRIEVAL_LIMIT=2
ASSISTANT_WAKE_WORD_KEYWORD=adam
ASSISTANT_WAKE_WORD_ALIASES=adam,addam,atom,add em,add him,at him
ASSISTANT_USE_KEYWORD_DETECTOR=1
ASSISTANT_CONTINUOUS_CONVERSATION=1
ASSISTANT_FOLLOWUP_TIMEOUT_SECONDS=45
ASSISTANT_ENABLE_WAKE_GREETING=1
ASSISTANT_STT_SPEECH_START_TIMEOUT_SECONDS=10
ASSISTANT_STT_PRE_SPEECH_MS=300
ASSISTANT_ENABLE_BARGE_IN=1
```

## Update Log

### 2026-06-01

- Improved voice mode to behave like a continuous local voice assistant: wake once, hear a short acknowledgment, ask follow-up questions for 45 seconds, then return to wake-word mode after silence.
- Added wake-word alias matching and STT pre-roll to reduce missed wakes and clipped first words.
- Aligned voice answer routing with the web Ask page so general questions can be answered by voice when document context is irrelevant.
- Added instant voice template replies for greetings and short social turns.
- Tuned voice LLM generation for shorter spoken answers and faster voice responses.
- Added spoken thinking feedback during generation and then fixed filler interruption so filler stops before final answer playback.
- Added SQLite voice logs and focused voice-flow regression tests.
- Restarted adam.service after the voice behavior changes and verified the service is active.

### 2026-05-27

- Redesigned `/monitor` into a compact one-page operations console with searchable scrollable Chat History and System Events panels, improved badges, source pills, quick exports, and runtime/model panels.
- Added a Stop Generating control to the chat UI and verified tests still pass.
- Improved answer quality and latency behavior: greetings now use instant local templates, document prompts use a strict Answer/Details/Source/Not found format, follow-up questions reuse the last session topic, and visible source chips are limited to the top two.
- Hardened New Chat with a real `/ask?new=1` fallback link plus JavaScript enhancement, so it can start a fresh chat even if a browser blocks part of the client script.
- Fixed New Chat on non-secure LAN browser sessions by adding a fallback session ID generator when `crypto.randomUUID()` is unavailable.
- Restarted the LAN dashboard on port `8092` and verified the served `/ask` page contains the fixed sidebar layout and New Chat behavior.
- Fixed `/ask` ChatGPT-style layout scrolling: sidebar now stays in place, the transcript scrolls inside the main panel, and New Chat reliably opens a fresh empty conversation.
- Restarted the LAN dashboard on port `8092` after the chat UI fix and verified `/ask/stream` still returns status, token, and done events.
- Fixed `/ask` browser chat behavior: repaired session JSON loading, added Enter-to-send with Shift+Enter for multiline input, and added a POST fallback so questions no longer become `/ask?question=...` URLs.
- Reviewed the source code end to end and fixed the remaining project-level gaps found during review.
- Wired `assistant-speak` to Piper/audio device environment settings.
- Corrected optional dependency groups and `requirements.txt` so dashboard installs work through both editable install and requirements install.
- Reformatted Dockerfile, added `alsa-utils`, and added dashboard host/port environment support for container port mapping.
- Added SQLite/Chroma cleanup for document reingestion and dashboard deletion to prevent stale knowledge results.
- Strengthened the Ollama prompt to avoid unsupported answers outside retrieved context.
- Added tests for dashboard delete cleanup, metadata deletion helpers, Piper speaker-test wiring, and grounded prompt wording.
- Verified `python3 -m compileall src/assistant_app` and `.venv/bin/python -m unittest discover -s tests` pass.
- Added the dashboard Ask Documents page, downloaded Vosk/Piper assets, verified Vosk load, verified Piper WAV generation, and verified ALSA playback through `plughw:0,3`.
- Modernized `/ask` into a chat-style dashboard, added visible answer mode/model name, restarted the updated LAN server on port `8092`, added an immediate loading indicator, cached services/timing, switched between extractive and low-latency AI modes, then added persistent SQLite chat history and the `/monitor` tracking dashboard.

### 2026-05-26

- Created the Python `src/` project layout and package metadata.
- Added the humanized response state manager and timed feedback pipeline.
- Added text demo mode, Vosk/VAD STT adapter, Ollama streaming adapter, ChromaDB retrieval adapter, and TTS placeholders.
- Added SQLite metadata storage and page-aware PyMuPDF document ingestion.
- Added thesis implementation plan for Jetson Orin Nano Super, Ubuntu Server through JetPack / Jetson Linux, USB audio, ESP32-S3 eyes, and offline privacy.
- Added tests for ingestion and SQLite metadata.
- Decided to keep this single progress notes file as both the new-chat handoff summary and running update log.
- Added Docker deployment support with mounted `documents/`, `data/`, and `models/` folders for portable deployment.
- Added explicit local embedding model support through `ASSISTANT_EMBEDDING_MODEL` and grounded no-context answer validation.
- Added `models/embeddings/.gitkeep` as the visible location for the local embedding model while keeping actual model files untracked.
- Wired final answer TTS to use `ASSISTANT_TTS_COMMAND` when set, with console TTS fallback when unset.
- Added standalone `assistant-speak` command for quick TTS/speaker testing.
- Added Piper playback support through `ASSISTANT_PIPER_MODEL` and `ASSISTANT_AUDIO_DEVICE`, while keeping `ASSISTANT_TTS_COMMAND` as an explicit override.
- Added query processor and ESP32-S3 serial eye controller to better match the architecture diagram.
- Installed/tested text, document, and voice dependency stacks in `.venv`; voice packages import successfully after installing Python headers and PortAudio/ALSA system libraries.
- Added a local-only upload dashboard so knowledge files can be managed without manually copying files into `documents/`.

## Next Recommended Work

1. Add dashboard controls for receptionist behavior: organization profile, opening hours, contact details, escalation message, answer style, knowledge-only/general/auto mode, and a stop-generating button.
2. Use `http://192.168.0.174:8092/ask` for low-latency AI-generated answers with `llama3.2:1b`, and use `http://192.168.0.174:8092/monitor` to track chat history, sources, latency, index health, Ollama status, document usage, errors, resources, files, and events.
2. Use `ASSISTANT_DASHBOARD_ANSWER_MODE=extractive` only for sub-second direct excerpts.
3. Connect the actual USB Audio Codec and confirm it appears in `aplay -l` / `sounddevice.query_devices()`.
4. Test live microphone voice mode with `ASSISTANT_VOSK_MODEL=./models/vosk .venv/bin/python -m assistant_app --voice`.
5. Test Piper playback on the Jetson USB audio codec using `ASSISTANT_AUDIO_DEVICE=plughw:X,Y ASSISTANT_PIPER_MODEL=./models/piper/en_US-amy-low.onnx .venv/bin/python -m assistant_app.speak "Hello"`.
6. Improve answer validation with source-aware checks after LLM generation.
7. Test ESP32-S3 serial output on real hardware.
8. Add Jetson setup scripts for Ubuntu Server / JetPack deployment.
8. Add systemd service files for Ollama warm-start and assistant startup.
9. Test Docker image build on the target Jetson device.
10. Add wake word detection and preload the Vosk model once instead of loading it on every voice turn.

## Rule for Future Work

Whenever new code, documentation, tests, setup scripts, or architecture changes are added, update this file before finishing the response. Add a short entry under `Update Log`, and keep the completed work and next recommended work accurate.
