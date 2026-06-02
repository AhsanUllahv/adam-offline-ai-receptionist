# Docker Deployment

This project can run inside Docker so it can be moved to another system with the same source code, documents, data folders, and model folders.

## What Goes Inside the Image

The Docker image contains:

- Python runtime
- assistant application code
- Python dependencies
- command line tools: `assistant` and `assistant-ingest`

The Docker image does not contain large local files such as PDFs, ChromaDB data, SQLite data, Vosk models, Piper voices, or Ollama models. Those stay on the host and are mounted into the container.

## Mounted Folders

```text
./documents -> /app/documents
./data      -> /app/data
./models    -> /app/models
```

This keeps the system portable and protects local project data from being baked into the image.

## Knowledge Dashboard

The dashboard can run inside the container and use the same mounted `documents/`, `data/`, and `models/` folders. Expose port `8080` when running it manually:

```bash
docker compose run --rm -p 8080:8080 -e ASSISTANT_DASHBOARD_HOST=0.0.0.0 assistant assistant-dashboard
```

Then open `http://127.0.0.1:8080`.

## Build the Container

```bash
docker compose build
```

## Ingest Documents

Put PDFs, text files, or markdown files into `documents/`, then run:

```bash
docker compose --profile tools run --rm ingest
```

This creates or updates:

- `data/assistant.db`
- `data/chroma/`

## Run the Assistant

```bash
docker compose run --rm assistant
```

For a long-running service:

```bash
docker compose up assistant
```

## Embedding Model

The assistant expects a local sentence-transformers model mounted under:

```text
./models/embeddings/all-MiniLM-L6-v2 -> /app/models/embeddings/all-MiniLM-L6-v2
```

You can use a different local folder by changing `ASSISTANT_EMBEDDING_MODEL` in `.env`.

The same model path must be available during both document ingestion and retrieval. This avoids cloud embedding APIs and keeps document search fully local.

## Ollama

Recommended Jetson setup: run Ollama on the host system and let the assistant container connect to it through:

```text
ASSISTANT_OLLAMA_URL=http://host.docker.internal:11434
```

On Linux, `docker-compose.yml` includes this mapping:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

This avoids putting large Ollama model files inside the assistant container.

## Audio Devices

For text mode, no audio device is needed.

To test TTS/speaker output through the container, first set these values in `.env`:

```bash
ASSISTANT_PIPER_MODEL=/app/models/piper/voice.onnx
ASSISTANT_AUDIO_DEVICE=default
```

Then run:

```bash
docker compose run --rm assistant assistant-speak "Hello, I am ready."
```

If the USB audio codec is not the default device, list host ALSA devices with `aplay -l` and set `ASSISTANT_AUDIO_DEVICE`, for example `plughw:1,0`.

For microphone and speaker access on Ubuntu Server / Jetson, the container may need access to `/dev/snd`. In `docker-compose.yml`, uncomment:

```yaml
devices:
  - /dev/snd:/dev/snd
group_add:
  - audio
```

Then run voice mode inside the container after adding a Vosk model under `models/vosk`:

```bash
docker compose run --rm assistant assistant --voice
```

## Environment

Copy `.env.example` to `.env` if you want local changes:

```bash
cp .env.example .env
```

Then edit `.env` for your model paths, Ollama URL, or TTS command.

## Jetson Notes

For NVIDIA Jetson deployment, use JetPack / Jetson Linux on the host. Docker is useful for the assistant application, but hardware drivers, audio devices, and Ollama/GPU support are usually easier to manage on the host first. Keep the assistant container focused on Python application logic and local mounted data.

## ESP32-S3 Eyes

For ESP32-S3 serial eye output, expose the serial device to the container. Example for `/dev/ttyUSB0`:

```yaml
devices:
  - /dev/ttyUSB0:/dev/ttyUSB0
```

Then set in `.env`:

```bash
ASSISTANT_EYE_SERIAL_PORT=/dev/ttyUSB0
ASSISTANT_EYE_SERIAL_BAUDRATE=115200
```

If no serial port is configured, the app uses console eye output.
