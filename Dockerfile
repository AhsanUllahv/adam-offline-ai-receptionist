FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        alsa-utils \
        build-essential \
        ffmpeg \
        libasound2 \
        libasound2-dev \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install -e ".[all]"

RUN mkdir -p /app/documents /app/data/chroma /app/models

ENV ASSISTANT_CHROMA_PATH=/app/data/chroma \
    ASSISTANT_SQLITE_PATH=/app/data/assistant.db \
    ASSISTANT_CHROMA_COLLECTION=documents \
    ASSISTANT_OLLAMA_URL=http://host.docker.internal:11434 \
    ASSISTANT_OLLAMA_MODEL=llama3.2:latest \
    ASSISTANT_EMBEDDING_MODEL=/app/models/embeddings/all-MiniLM-L6-v2 \
    ASSISTANT_EMBEDDING_DEVICE=cpu \
    ASSISTANT_PIPER_MODEL= \
    ASSISTANT_AUDIO_DEVICE=default \
    ASSISTANT_EYE_SERIAL_PORT= \
    ASSISTANT_EYE_SERIAL_BAUDRATE=115200 \
    ASSISTANT_DASHBOARD_HOST=127.0.0.1 \
    ASSISTANT_DASHBOARD_PORT=8092

CMD ["assistant"]
