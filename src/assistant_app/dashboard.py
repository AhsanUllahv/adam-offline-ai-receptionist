from __future__ import annotations

import base64
import csv
import html
import io
import json
import os
import secrets
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from assistant_app.audit import (
    AuditStore,
    ChatSessionRecord,
    EventRecord,
    InteractionRecord,
    LatencyStats,
    SourceUsageRecord,
    VoiceLogRecord,
)
from assistant_app.config import AssistantConfig
from assistant_app.ingest import (
    SUPPORTED_EXTENSIONS,
    ingest_path,
    iter_documents,
    remove_document_from_index,
)
from assistant_app.llm import OllamaLLM
from assistant_app.metadata import IndexedDocumentRecord, IndexStats, MetadataStore
from assistant_app.query import Intent, QueryProcessor
from assistant_app.retrieval import ChromaRetriever
from assistant_app.validation import GroundedAnswerValidator


DEFAULT_RETRIEVAL_LIMIT = 6
DISPLAY_SOURCE_LIMIT = 2
EXTRACTIVE_ANSWER_CHARS = 900
KEYWORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "design",
    "for",
    "from",
    "about",
    "do",
    "does",
    "how",
    "i",
    "is",
    "it",
    "of",
    "on",
    "or",
    "know",
    "mean",
    "the",
    "this",
    "to",
    "what",
    "with",
    "you",
    "your",
    "did",
    "not",
    "mention",
}


DOCUMENTS_DIR = Path("documents")


@dataclass(frozen=True)
class IndexHealth:
    uploaded_file_count: int
    indexed_document_count: int
    chunk_count: int
    last_indexed_at: str
    unindexed_files: tuple[str, ...]
    stale_files: tuple[str, ...]


@dataclass(frozen=True)
class ModelStatus:
    reachable: bool
    model: str
    url: str
    answer_mode: str
    message: str


@dataclass(frozen=True)
class SystemResources:
    load_1m: float
    memory_used_mb: int
    memory_total_mb: int
    memory_percent: float
    disk_used: str
    disk_total: str
    disk_percent: float
    temperature_c: float | None


@dataclass
class DashboardState:
    message: str = ""
    error: str = ""


state = DashboardState()
app = FastAPI(title="Offline Assistant Knowledge Dashboard")


class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        password = os.getenv("ASSISTANT_DASHBOARD_PASSWORD", "").strip()
        if not password:
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth[6:]).decode("utf-8")
                _, _, provided = decoded.partition(":")
                if secrets.compare_digest(provided.encode(), password.encode()):
                    return await call_next(request)
            except Exception:
                pass
        from starlette.responses import Response as SR
        return SR(
            "Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Offline Assistant"'},
        )


app.add_middleware(BasicAuthMiddleware)


@dataclass(frozen=True)
class DashboardAnswer:
    answer: str
    sources: tuple[str, ...]
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0


@dataclass
class DashboardServices:
    config: AssistantConfig
    retriever: ChromaRetriever = field(init=False)
    llm: OllamaLLM = field(init=False)
    validator: GroundedAnswerValidator = field(default_factory=GroundedAnswerValidator)

    def __post_init__(self) -> None:
        self.retriever = ChromaRetriever.from_config(self.config)
        self.llm = OllamaLLM.from_config(self.config)


_services: DashboardServices | None = None
_services_key: tuple[str, ...] | None = None


def get_dashboard_services(config: AssistantConfig) -> DashboardServices:
    global _services, _services_key
    key = (
        config.ollama_model,
        config.ollama_url,
        config.chroma_path,
        config.chroma_collection,
        config.sqlite_path,
        config.embedding_model_path,
        config.embedding_device,
        config.dashboard_answer_mode,
        str(config.dashboard_llm_context_chars),
        str(config.dashboard_llm_retrieval_limit),
        str(config.ollama_num_predict),
        str(config.ollama_num_ctx),
        str(config.ollama_temperature),
    )
    if _services is None or _services_key != key:
        _services = DashboardServices(config=config)
        _services_key = key
    return _services


def reset_dashboard_services() -> None:
    global _services, _services_key
    _services = None
    _services_key = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    files = iter_documents(DOCUMENTS_DIR)
    return render_page(files, state.message, state.error)


@app.get("/ask", response_class=HTMLResponse)
def ask_page() -> str:
    config = AssistantConfig.from_env()
    return render_ask_page("", "", (), "", [], recent_sessions(config))


@app.post("/ask", response_class=HTMLResponse)
def ask(question: str = Form(...)) -> str:
    submitted_question = question.strip()
    if not submitted_question:
        return render_ask_page(
            submitted_question, "", (), "Write a question first.", recent_interactions()
        )

    config = AssistantConfig.from_env()
    try:
        result = answer_dashboard_question(submitted_question, config)
    except Exception as exc:
        audit_store(config).log_event(
            "ask_error", str(exc), {"question": submitted_question}
        )
        return render_ask_page(submitted_question, "", (), str(exc), recent_interactions(config))

    answer_text = add_latency_summary(result)
    audit_store(config).log_interaction(
        question=submitted_question,
        answer=answer_text,
        sources=result.sources,
        answer_mode=config.dashboard_answer_mode,
        model=config.ollama_model,
        retrieval_seconds=result.retrieval_seconds,
        generation_seconds=result.generation_seconds,
    )
    history = recent_interactions(config)
    return render_ask_page(
        submitted_question,
        answer_text,
        result.sources,
        "",
        history,
    )




@app.get("/voice", response_class=HTMLResponse)
def voice_page() -> str:
    config = AssistantConfig.from_env()
    model_name = config.ollama_model
    return render_voice_page(model_name)


@app.post("/ask/stream")
def ask_stream(
    question: str = Form(...),
    session_id: str = Form("default"),
    voice: str = Form("0"),
) -> StreamingResponse:
    config = AssistantConfig.from_env()
    return StreamingResponse(
        stream_dashboard_answer(question.strip(), config, safe_session_id(session_id), voice_mode=voice == "1"),
        media_type="application/x-ndjson",
    )


@app.get("/ask/sessions")
def ask_sessions() -> Response:
    config = AssistantConfig.from_env()
    payload = [session_to_dict(item) for item in recent_sessions(config)]
    return Response(json.dumps(payload), media_type="application/json")


@app.get("/ask/session/{session_id}")
def ask_session(session_id: str) -> Response:
    config = AssistantConfig.from_env()
    items = audit_store(config).interactions_for_session(safe_session_id(session_id), 100)
    payload = [interaction_to_chat_dict(item) for item in items]
    return Response(json.dumps(payload), media_type="application/json")


@app.post("/ask/session/{session_id}/delete")
def delete_ask_session(session_id: str) -> Response:
    config = AssistantConfig.from_env()
    audit_store(config).delete_session(safe_session_id(session_id))
    return Response(json.dumps({"ok": True}), media_type="application/json")


@app.post("/ask/transcribe")
async def ask_transcribe(file: UploadFile = File(...)) -> Response:
    try:
        import json as _json
        import wave as _wave
        import io as _io
        import vosk as _vosk

        model_path = str(Path("models/vosk"))
        if not hasattr(ask_transcribe, "_vosk_model"):
            ask_transcribe._vosk_model = _vosk.Model(model_path)
        model = ask_transcribe._vosk_model
        data = await file.read()
        try:
            with _wave.open(_io.BytesIO(data)) as wf:
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
        except Exception:
            frames = data[44:]
            sample_rate = 16000
        rec = _vosk.KaldiRecognizer(model, sample_rate)
        rec.AcceptWaveform(frames)
        result = _json.loads(rec.FinalResult())
        text = result.get("text", "").strip()
        return Response(_json.dumps({"text": text}), media_type="application/json")
    except Exception as exc:
        return Response(
            json.dumps({"text": "", "error": str(exc)}),
            media_type="application/json",
            status_code=500,
        )


@app.post("/ask/feedback")
def ask_feedback(
    interaction_id: int = Form(...),
    rating: str = Form(...),
) -> Response:
    if rating not in {"up", "down"}:
        return Response(
            json.dumps({"ok": False, "error": "invalid rating"}),
            media_type="application/json",
            status_code=400,
        )
    config = AssistantConfig.from_env()
    audit_store(config).log_feedback(interaction_id, rating)
    return Response(json.dumps({"ok": True}), media_type="application/json")


@app.get("/documents/{filename}")
def serve_document(filename: str):
    target = DOCUMENTS_DIR / safe_filename(filename)
    if not target.exists() or not target.is_file() or target.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return Response("Not found", status_code=404)
    return FileResponse(target)


@app.get("/ask/suggestions")
def ask_suggestions(q: str = "") -> Response:
    q = q.strip()
    if len(q) < 3:
        return Response(json.dumps([]), media_type="application/json")
    config = AssistantConfig.from_env()
    suggestions = MetadataStore(config.sqlite_path).get_question_suggestions(q, limit=5)
    return Response(json.dumps(suggestions), media_type="application/json")


@app.post("/documents/{filename}/tags")
def set_document_tags_route(filename: str, tags: str = Form("")) -> RedirectResponse:
    config = AssistantConfig.from_env()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    MetadataStore(config.sqlite_path).set_document_tags(safe_filename(filename), tag_list)
    return RedirectResponse("/", status_code=303)


@app.post("/upload")
def upload(files: list[UploadFile] = File(...)) -> RedirectResponse:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    uploaded: list[str] = []
    rejected: list[str] = []

    for file in files:
        name = safe_filename(file.filename or "")
        suffix = Path(name).suffix.lower()
        if not name or suffix not in SUPPORTED_EXTENSIONS:
            rejected.append(file.filename or "unnamed")
            continue

        target = unique_path(DOCUMENTS_DIR / name)
        with target.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        uploaded.append(target.name)

    state.message = f"Uploaded: {', '.join(uploaded)}" if uploaded else ""
    if uploaded:
        audit_store().log_event("upload", state.message, {"files": uploaded})
    state.error = f"Rejected unsupported files: {', '.join(rejected)}" if rejected else ""
    if rejected:
        audit_store().log_event(
            "upload_rejected", state.error, {"files": rejected}
        )
    return RedirectResponse("/", status_code=303)


@app.post("/ingest")
def ingest(
    chunk_size: int = Form(900),
    overlap: int = Form(150),
) -> RedirectResponse:
    config = AssistantConfig.from_env()
    try:
        result = ingest_path(DOCUMENTS_DIR, config, chunk_size=chunk_size, overlap=overlap)
        reset_dashboard_services()
    except Exception as exc:
        state.message = ""
        state.error = str(exc)
        audit_store(config).log_event("ingest_error", state.error, {})
        return RedirectResponse("/", status_code=303)

    state.message = (
        f"Indexed {result.chunks_indexed} chunks from "
        f"{result.documents_indexed}/{result.documents_found} documents."
    )
    audit_store(config).log_event(
        "ingest",
        state.message,
        {
            "chunks_indexed": result.chunks_indexed,
            "documents_indexed": result.documents_indexed,
            "documents_found": result.documents_found,
            "skipped": list(result.skipped),
        },
    )
    state.error = "Skipped: " + ", ".join(result.skipped) if result.skipped else ""
    return RedirectResponse("/", status_code=303)


@app.post("/delete/{filename}")
def delete(filename: str) -> RedirectResponse:
    target = DOCUMENTS_DIR / safe_filename(filename)
    if target.exists() and target.is_file() and target.suffix.lower() in SUPPORTED_EXTENSIONS:
        config = AssistantConfig.from_env()
        remove_document_from_index(target, config)
        reset_dashboard_services()
        target.unlink()
        state.message = f"Deleted {target.name} and removed it from the knowledge index."
        audit_store(config).log_event("delete", state.message, {"file": target.name})
        state.error = ""
    else:
        state.message = ""
        state.error = "File not found or unsupported."
    return RedirectResponse("/", status_code=303)


@app.get("/monitor", response_class=HTMLResponse)
def monitor_page() -> str:
    config = AssistantConfig.from_env()
    store = audit_store(config)
    interactions = store.recent_interactions(50)
    audit_events = store.recent_events(50)
    return render_monitor_page(
        interactions=interactions,
        events=audit_events,
        error_events=store.error_events(25),
        latency_stats=store.latency_stats(),
        source_usage=store.source_usage(10),
        index_health=build_index_health(config),
        model_status=check_model_status(config),
        system_resources=read_system_resources(),
        interaction_count=store.interaction_count(),
        event_count=store.event_count(),
        config=config,
    )


@app.get("/monitor/live")
async def monitor_live_stream() -> StreamingResponse:
    """SSE endpoint — streams live voice pipeline events to the monitor dashboard."""
    from assistant_app import events as _live_events

    return StreamingResponse(
        _live_events.sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


_VOICE_EVENT_TYPES = {
    "state", "wake", "wake_response", "transcript", "sources",
    "filler", "barge_in", "error", "latency", "answer_complete",
}

@app.get("/monitor/voice-events")
def monitor_voice_events(limit: int = 200) -> Response:
    config = AssistantConfig.from_env()
    store = audit_store(config)
    rows = store.recent_events(limit)
    voice_rows = [event_to_dict(r) for r in rows if r.event_type in _VOICE_EVENT_TYPES]
    return Response(json.dumps(voice_rows), media_type="application/json")


@app.get("/monitor/voice-history")
def monitor_voice_history(limit: int = 60) -> Response:
    config = AssistantConfig.from_env()
    items = audit_store(config).recent_interactions(limit, session_id="voice")
    payload = [
        {
            "id": item.id,
            "created_at": item.created_at,
            "question": item.question,
            "answer": item.answer,
        }
        for item in reversed(items)
    ]
    return Response(json.dumps(payload), media_type="application/json")


@app.get("/monitor/voice-logs")
def monitor_voice_logs_api(limit: int = 100) -> Response:
    config = AssistantConfig.from_env()
    rows = audit_store(config).recent_voice_logs(limit)
    return Response(
        json.dumps([voice_log_to_dict(r) for r in rows]),
        media_type="application/json",
    )


def voice_log_to_dict(r: VoiceLogRecord) -> dict:
    return {
        "id": r.id,
        "created_at": r.created_at,
        "wake_detected_at": r.wake_detected_at,
        "transcript": r.transcript,
        "answer": r.answer,
        "sources": list(r.sources),
        "knowledge_source": r.knowledge_source,
        "filler_count": r.filler_count,
        "barge_in": r.barge_in,
        "answer_truncated": r.answer_truncated,
        "stt_seconds": r.stt_seconds,
        "retrieval_seconds": r.retrieval_seconds,
        "generation_seconds": r.generation_seconds,
        "total_seconds": r.total_seconds,
        "word_count": r.word_count,
        "source_count": r.source_count,
        "cpu_load": r.cpu_load,
        "memory_percent": r.memory_percent,
        "temperature_c": r.temperature_c,
        "error": r.error,
    }


@app.get("/monitor/interaction/{interaction_id}", response_class=HTMLResponse)
def interaction_detail_page(interaction_id: int) -> str:
    config = AssistantConfig.from_env()
    interaction = audit_store(config).get_interaction(interaction_id)
    return render_interaction_detail_page(interaction, config)


@app.get("/monitor/export/interactions.csv")
def export_interactions_csv() -> Response:
    rows = audit_store().all_interactions()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id",
        "created_at",
        "question",
        "answer",
        "sources",
        "answer_mode",
        "model",
        "retrieval_seconds",
        "generation_seconds",
        "total_seconds",
    ])
    for item in rows:
        writer.writerow([
            item.id,
            item.created_at,
            item.question,
            item.answer,
            " | ".join(item.sources),
            item.answer_mode,
            item.model,
            f"{item.retrieval_seconds:.4f}",
            f"{item.generation_seconds:.4f}",
            f"{item.total_seconds:.4f}",
        ])
    return Response(
        output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=interactions.csv"},
    )


@app.get("/monitor/export/events.csv")
def export_events_csv() -> Response:
    rows = audit_store().all_events()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "created_at", "event_type", "message", "details_json"])
    for item in rows:
        writer.writerow([
            item.id,
            item.created_at,
            item.event_type,
            item.message,
            json.dumps(item.details),
        ])
    return Response(
        output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=events.csv"},
    )


@app.get("/monitor/export/interactions.json")
def export_interactions_json() -> Response:
    payload = [interaction_to_dict(item) for item in audit_store().all_interactions()]
    return Response(json.dumps(payload, indent=2), media_type="application/json")


@app.get("/monitor/export/events.json")
def export_events_json() -> Response:
    payload = [event_to_dict(item) for item in audit_store().all_events()]
    return Response(json.dumps(payload, indent=2), media_type="application/json")


def audit_store(config: AssistantConfig | None = None) -> AuditStore:
    return AuditStore((config or AssistantConfig.from_env()).sqlite_path)


def recent_interactions(
    config: AssistantConfig | None = None, limit: int = 20
) -> list[InteractionRecord]:
    return audit_store(config).recent_interactions(limit)


def recent_sessions(
    config: AssistantConfig | None = None, limit: int = 20
) -> list[ChatSessionRecord]:
    return audit_store(config).recent_sessions(limit)


def latest_session_turn(
    config: AssistantConfig, session_id: str
) -> InteractionRecord | None:
    turns = audit_store(config).recent_interactions(limit=1, session_id=session_id)
    return turns[0] if turns else None


def simple_receptionist_answer(question: str) -> str | None:
    normalized = " ".join(question.lower().strip().split()).rstrip("?.!")
    if not normalized:
        return "Write a question first."
    words = set(normalized.split())
    greeting_words = {"hi", "hello", "hey", "salam", "assalamualaikum", "yo", "greetings", "howdy"}
    if words & greeting_words and len(normalized) <= 30:
        return "Hello! How can I help you today?"
    if normalized in {"how are you", "how r u", "how are you doing", "hows it going"}:
        return "I am ready to help. Ask me anything from your uploaded documents or a general question."
    if normalized in {"thanks", "thank you", "thank you very much", "thx", "ty"}:
        return "You are welcome! Let me know if you have more questions."
    if normalized in {"ok", "okay", "ok thanks", "okay thanks", "got it", "noted", "alright"}:
        return "Sure! Let me know if there is anything else I can help with."
    return None


FOLLOW_UP_TERMS = {
    "also",
    "board",
    "colour",
    "color",
    "explain",
    "it",
    "its",
    "more",
    "part",
    "previous",
    "spec",
    "specs",
    "specification",
    "specifications",
    "that",
    "them",
    "they",
    "those",
}


def rewrite_followup_question(question: str, previous: InteractionRecord | None) -> str:
    if previous is None:
        return question
    terms = set(keyword_terms(question))
    if len(terms) > 5 and not (terms & FOLLOW_UP_TERMS):
        return question
    if not terms or terms & FOLLOW_UP_TERMS or len(terms) <= 3:
        previous_hint = shorten_text(previous.question, 160)
        return f"{question} (follow-up about previous question: {previous_hint})"
    return question


def public_sources(contexts: list[str], limit: int = DISPLAY_SOURCE_LIMIT) -> tuple[str, ...]:
    return tuple(extract_source_labels(contexts)[:limit])


def public_source_chunks(contexts: list[str], limit: int = DISPLAY_SOURCE_LIMIT) -> list[dict]:
    chunks = []
    for ctx in contexts[:limit]:
        source, body = split_context(ctx)
        if source:
            chunks.append({"label": source, "text": shorten_text(body, 600)})
    return chunks


def knowledge_missing_answer() -> str:
    return (
        "Answer:\nI could not find that clearly in the uploaded knowledge.\n\n"
        "Details:\n- The current documents do not contain enough matching information for this question.\n"
        "- Try asking with a document keyword, page topic, or upload the missing official file.\n\n"
        "Source:\nNone\n\n"
        "Not found:\nThe requested detail is not available in the indexed knowledge."
    )


def answer_dashboard_question(question: str, config: AssistantConfig) -> DashboardAnswer:
    processed = QueryProcessor().process(question)
    if processed.intent is Intent.EMPTY:
        return DashboardAnswer(answer="Write a question first.", sources=())
    if processed.intent is Intent.EXIT:
        return DashboardAnswer(answer="Use the upload page or close the browser tab when finished.", sources=())

    simple_answer = simple_receptionist_answer(processed.text)
    if simple_answer:
        return DashboardAnswer(answer=simple_answer, sources=())

    services = get_dashboard_services(config)
    retrieval_started = time.monotonic()
    contexts = rerank_contexts(
        processed.text,
        services.retriever.search(processed.text, limit=DEFAULT_RETRIEVAL_LIMIT),
    )
    retrieval_seconds = time.monotonic() - retrieval_started

    relevant_context = has_relevant_document_context(processed.text, contexts)
    selected_contexts = contexts[: config.dashboard_llm_retrieval_limit] if relevant_context else []
    context = build_llm_context(selected_contexts, config.dashboard_llm_context_chars)

    sources = public_sources(contexts) if relevant_context else ()
    if relevant_context and config.dashboard_answer_mode == "extractive":
        answer = build_extractive_answer(contexts)
        return DashboardAnswer(
            answer=answer,
            sources=sources,
            retrieval_seconds=retrieval_seconds,
        )

    generation_started = time.monotonic()
    if relevant_context:
        answer = "".join(services.llm.stream_answer(processed.text, context)).strip()
    elif needs_official_knowledge_answer(processed.text):
        answer = official_knowledge_missing_answer(processed.text)
    else:
        answer = "".join(services.llm.stream_general_answer(processed.text)).strip()
    generation_seconds = time.monotonic() - generation_started
    return DashboardAnswer(
        answer=answer,
        sources=sources,
        retrieval_seconds=retrieval_seconds,
        generation_seconds=generation_seconds,
    )



def stream_dashboard_answer(question: str, config: AssistantConfig, session_id: str = "default", voice_mode: bool = False):
    def event(event_type: str, **payload: object) -> str:
        data = {"type": event_type, **payload}
        return json.dumps(data) + "\n"

    if not question:
        yield event("error", message="Write a question first.")
        return

    processed = QueryProcessor().process(question)
    if processed.intent is Intent.EMPTY:
        yield event("error", message="Write a question first.")
        return
    if processed.intent is Intent.EXIT:
        yield event("token", text="Use the upload page or close the browser tab when finished.")
        yield event("done", sources=[], retrieval_seconds=0.0, generation_seconds=0.0, total_seconds=0.0)
        return

    simple_answer = simple_receptionist_answer(processed.text)
    if simple_answer:
        for token in split_for_display_stream(simple_answer):
            yield event("token", text=token)
        interaction_id = audit_store(config).log_interaction(
            question=question,
            answer=simple_answer,
            sources=(),
            answer_mode="template",
            model="local-template",
            retrieval_seconds=0.0,
            generation_seconds=0.0,
            session_id=session_id,
        )
        yield event(
            "done",
            sources=[],
            chunks=[],
            retrieval_seconds=0.0,
            generation_seconds=0.0,
            total_seconds=0.0,
            model="local-template",
            answer_mode="template",
            knowledge_source="template",
            interaction_id=interaction_id,
        )
        return

    try:
        yield event("status", message="Preparing local services...")
        services = get_dashboard_services(config)
        previous_turn = latest_session_turn(config, session_id)
        search_text = rewrite_followup_question(processed.text, previous_turn)

        retrieval_seconds = 0.0
        sources: tuple[str, ...] = ()
        source_chunks: list[dict] = []
        knowledge_source = "general"
        relevant_context = False
        context = ""

        if not config.no_rag:
            yield event("status", message="Searching your documents...")
            retrieval_started = time.monotonic()
            contexts = rerank_contexts(
                search_text,
                services.retriever.search(search_text, limit=DEFAULT_RETRIEVAL_LIMIT),
            )
            retrieval_seconds = time.monotonic() - retrieval_started
            relevant_context = has_relevant_document_context(search_text, contexts)
            selected_contexts = contexts[: config.dashboard_llm_retrieval_limit] if relevant_context else []
            context = build_llm_context(selected_contexts, config.dashboard_llm_context_chars)
            knowledge_source = "documents" if relevant_context else "general"
            sources = public_sources(contexts) if relevant_context else ()
            source_chunks = public_source_chunks(contexts) if relevant_context else []
            if sources:
                yield event("sources", sources=list(sources), chunks=source_chunks)
        else:
            contexts = []

        yield event(
            "status",
            message="Generating from knowledge base..." if relevant_context else "Answering as virtual receptionist...",
        )
        generation_started = time.monotonic()
        answer_parts: list[str] = []
        if relevant_context and config.dashboard_answer_mode == "extractive":
            answer = build_extractive_answer(contexts)
            for token in split_for_display_stream(answer):
                answer_parts.append(token)
                yield event("token", text=token)
        elif relevant_context:
            stream_fn = services.llm.stream_voice_answer if voice_mode else services.llm.stream_answer
            for token in stream_fn(search_text, context):
                answer_parts.append(token)
                yield event("token", text=token)
        elif needs_official_knowledge_answer(processed.text) and not config.no_rag:
            for token in split_for_display_stream(official_knowledge_missing_answer(processed.text)):
                answer_parts.append(token)
                yield event("token", text=token)
        else:
            stream_fn = services.llm.stream_voice_general_answer if voice_mode else services.llm.stream_general_answer
            for token in stream_fn(processed.text):
                answer_parts.append(token)
                yield event("token", text=token)

        generation_seconds = time.monotonic() - generation_started
        raw_answer = "".join(answer_parts).strip()
        result = DashboardAnswer(
            answer=raw_answer,
            sources=sources,
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
        )
        saved_answer = add_latency_summary(result)
        interaction_id = audit_store(config).log_interaction(
            question=question,
            answer=saved_answer,
            sources=sources,
            answer_mode=config.dashboard_answer_mode,
            model=config.ollama_model,
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
            session_id=session_id,
        )
        total_seconds = retrieval_seconds + generation_seconds
        yield event(
            "done",
            sources=list(sources),
            chunks=source_chunks,
            retrieval_seconds=retrieval_seconds,
            generation_seconds=generation_seconds,
            total_seconds=total_seconds,
            model=config.ollama_model,
            answer_mode=config.dashboard_answer_mode,
            knowledge_source=knowledge_source,
            interaction_id=interaction_id,
        )
    except Exception as exc:
        audit_store(config).log_event("ask_error", str(exc), {"question": question})
        yield event("error", message=str(exc))


def split_for_display_stream(text: str):
    parts = text.split(" ")
    for index, part in enumerate(parts):
        if index:
            yield " "
        yield part


def rerank_contexts(query: str, contexts: list[str]) -> list[str]:
    query_terms = expand_query_terms(keyword_terms(query))
    if not query_terms or len(contexts) < 2:
        return contexts

    def score(item: tuple[int, str]) -> tuple[int, int, int]:
        index, context = item
        terms = keyword_terms(context)
        term_score = sum(terms.count(term) for term in query_terms)
        source, _ = split_context(context)
        source_score = sum(1 for term in query_terms if term in source.lower())
        return term_score, source_score, -index

    return [
        context
        for _, context in sorted(
            enumerate(contexts),
            key=score,
            reverse=True,
        )
    ]


def expand_query_terms(terms: list[str]) -> set[str]:
    expanded = set(terms)
    if "eyes" in expanded:
        expanded.add("eye")
    if "robo" in expanded:
        expanded.add("robot")
    if "display" in expanded:
        expanded.add("screen")
    return expanded


def keyword_terms(text: str) -> list[str]:
    import re

    return [
        term
        for term in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(term) > 2 and term not in KEYWORD_STOPWORDS
    ]


SPECIFIC_INFO_TERMS = {
    "admission",
    "address",
    "campus",
    "college",
    "contact",
    "course",
    "courses",
    "fee",
    "fees",
    "location",
    "phone",
    "policy",
    "principal",
    "school",
    "schedule",
    "timing",
    "university",
}


def needs_official_knowledge_answer(query: str) -> bool:
    terms = set(keyword_terms(query))
    if not terms & SPECIFIC_INFO_TERMS:
        return False
    return len(terms - SPECIFIC_INFO_TERMS) >= 1


def official_knowledge_missing_answer(query: str) -> str:
    return (
        "I do not have official local knowledge about that specific institution in the uploaded documents yet. "
        "Please upload a brochure, policy document, website copy, or contact sheet for it, and I can answer as the virtual receptionist from that knowledge base. "
        "I can still help with general questions, directions on what to upload, or questions from the documents already indexed."
    )


def has_relevant_document_context(query: str, contexts: list[str]) -> bool:
    if not contexts:
        return False
    query_terms = expand_query_terms(keyword_terms(query))
    if not query_terms:
        return False
    combined_terms: set[str] = set()
    for context in contexts[:3]:
        source, body = split_context(context)
        combined_terms.update(keyword_terms(source))
        combined_terms.update(keyword_terms(body))
    overlap = query_terms & combined_terms
    if len(query_terms) == 1:
        term = next(iter(query_terms))
        return len(term) >= 3 and term in combined_terms
    if len(overlap) >= 2:
        return True
    return len(overlap) / max(len(query_terms), 1) >= 0.6


def answer_kind_for_context(query: str, contexts: list[str]) -> str:
    return "knowledge" if has_relevant_document_context(query, contexts) else "general"


def build_llm_context(contexts: list[str], max_chars: int) -> str:
    compact: list[str] = []
    remaining = max(200, max_chars)
    for context in contexts:
        source, body = split_context(context)
        if not body:
            continue
        prefix = f"[{source}]\n" if source else ""
        allowance = max(120, remaining - len(prefix))
        excerpt = shorten_text(body, allowance)
        item = prefix + excerpt
        compact.append(item)
        remaining -= len(item)
        if remaining <= 120:
            break
    return "\n\n".join(compact)


def build_extractive_answer(contexts: list[str]) -> str:
    if not contexts:
        return "I could not find that clearly in your documents."

    source, body = split_context(contexts[0])
    excerpt = shorten_text(body, EXTRACTIVE_ANSWER_CHARS)
    if not excerpt:
        return "I found a relevant source, but it did not contain readable text."

    answer = "I found this in your documents:\n\n" + excerpt
    if source:
        answer += f"\n\nSource: {source}"
    return answer


def split_context(context: str) -> tuple[str, str]:
    lines = context.splitlines()
    if lines and lines[0].startswith("[") and lines[0].endswith("]"):
        return lines[0].strip("[]"), " ".join(lines[1:]).strip()
    return "", " ".join(lines).strip()


def shorten_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned

    window = cleaned[:max_chars].rstrip()
    sentence_end = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if sentence_end >= max_chars // 2:
        return window[: sentence_end + 1]
    return window + "..."


def add_latency_summary(result: DashboardAnswer) -> str:
    total_seconds = result.retrieval_seconds + result.generation_seconds
    if total_seconds <= 0:
        return result.answer
    if result.generation_seconds <= 0:
        return f"{result.answer}\n\nLatency: {total_seconds:.2f}s (retrieval only)."
    return (
        f"{result.answer}\n\n"
        f"Latency: {total_seconds:.2f}s "
        f"(retrieval {result.retrieval_seconds:.2f}s, "
        f"generation {result.generation_seconds:.2f}s)."
    )


def extract_source_labels(contexts: list[str]) -> list[str]:
    sources: list[str] = []
    for context in contexts:
        first_line = context.splitlines()[0].strip() if context.splitlines() else ""
        if first_line.startswith("[") and first_line.endswith("]"):
            sources.append(first_line.strip("[]"))
    return sources


def render_page(files: list[Path], message: str, error: str) -> str:
    config = AssistantConfig.from_env()
    all_tags = MetadataStore(config.sqlite_path).all_document_tags()
    rows = "".join(render_file_row(path, all_tags.get(path.name, [])) for path in files)
    sys_res = read_system_resources()
    model_st = check_model_status(config)
    temp_line = f"{sys_res.temperature_c:.1f} °C" if sys_res.temperature_c is not None else "—"
    model_color = "#4ade80" if model_st.reachable else "#f87171"
    model_label = "Online" if model_st.reachable else "Offline"
    if not rows:
        rows = '<tr><td colspan="5" class="muted">No knowledge files uploaded yet.</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Knowledge Files · Adam</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f4f6f8; color: #182230; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; }}
    .ops-shell {{ min-height: 100vh; display: grid; grid-template-columns: 248px minmax(0, 1fr); }}
    .ops-sidebar {{ position: sticky; top: 0; height: 100vh; background: #111827; color: #f9fafb; padding: 18px 14px; display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }}
    .brand {{ display: flex; gap: 10px; align-items: center; font-weight: 800; padding: 6px 4px; }}
    .brand-mark {{ width: 34px; height: 34px; border-radius: 9px; background: white; color: #111827; display: grid; place-items: center; font-weight: 900; font-size: 15px; }}
    .nav-stack {{ display: grid; gap: 6px; }}
    .nav-link {{ color: #d1d5db; text-decoration: none; padding: 10px 12px; border-radius: 8px; font-weight: 700; display: block; }}
    .nav-link:hover, .nav-link.active {{ background: #1f2937; color: white; }}
    .side-card {{ border: 1px solid #344054; border-radius: 10px; padding: 12px; color: #d0d5dd; font-size: 13px; line-height: 1.45; }}
    .ops-main {{ min-width: 0; padding: 18px; }}
    .topbar {{ position: sticky; top: 0; z-index: 3; background: rgba(244,246,248,.94); backdrop-filter: blur(12px); border-bottom: 1px solid #e4e7ec; margin: -18px -18px 18px; padding: 16px 18px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    h1 {{ margin: 0; font-size: 22px; line-height: 1.2; }}
    h2 {{ margin: 0 0 14px; font-size: 16px; font-weight: 800; }}
    .subtle {{ color: #667085; font-size: 13px; margin-top: 4px; }}
    .section-grid {{ display: grid; grid-template-columns: 1.5fr 1fr; gap: 14px; margin-bottom: 18px; align-items: start; }}
    .panel {{ background: white; border: 1px solid #e4e7ec; border-radius: 10px; padding: 20px; }}
    label {{ display: block; font-weight: 650; margin-bottom: 8px; font-size: 14px; }}
    input[type=file] {{ display: block; width: 100%; padding: 12px; border: 1px dashed #829ab1; border-radius: 6px; background: #f8fbff; }}
    .drop-zone {{ border: 2px dashed #829ab1; border-radius: 8px; background: #f8fbff; padding: 28px 16px; text-align: center; cursor: pointer; transition: border-color .2s, background .2s; }}
    .drop-zone.drag-over {{ border-color: #0b69a3; background: #e8f4ff; }}
    .drop-zone input[type=file] {{ display: none; }}
    .drop-zone-label {{ color: #486581; font-size: 14px; margin-top: 8px; }}
    .drop-zone-hint {{ color: #829ab1; font-size: 12px; margin-top: 4px; }}
    .tag-pill {{ display:inline-flex; align-items:center; background:#e0f2fe; color:#0369a1; border-radius:999px; padding:2px 8px; font-size:11px; font-weight:700; margin:2px 2px 0 0; }}
    .tag-input {{ width:130px; padding:4px 6px; border:1px solid #bcccdc; border-radius:4px; font:inherit; font-size:12px; }}
    .tag-save {{ font-size:11px; padding:4px 8px; background:#0b69a3; color:white; border:0; border-radius:4px; cursor:pointer; }}
    .field-row {{ display: flex; gap: 14px; align-items: end; flex-wrap: wrap; margin-bottom: 14px; }}
    input[type=number] {{ width: 110px; padding: 8px; border: 1px solid #bcccdc; border-radius: 5px; font: inherit; }}
    button {{ background: #111827; color: white; border: 0; border-radius: 6px; padding: 10px 16px; font-weight: 700; cursor: pointer; font: inherit; }}
    button.secondary {{ background: #486581; }}
    button.danger {{ background: #b42318; font: inherit; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    thead th {{ position: sticky; top: 0; background: #f8fafc; color: #667085; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .02em; }}
    th, td {{ border-bottom: 1px solid #eef2f6; padding: 10px; text-align: left; vertical-align: top; }}
    tr:hover td {{ background: #fafafa; }}
    .notice {{ padding: 12px 14px; border-radius: 8px; margin-bottom: 16px; font-size: 14px; border: 1px solid transparent; }}
    .ok {{ background: #e3f9e5; color: #14532d; border-color: #a7f3d0; }}
    .error {{ background: #ffe3e3; color: #7f1d1d; border-color: #fca5a5; }}
    .muted {{ color: #627d98; }}
    .actions {{ width: 1%; white-space: nowrap; }}
    .table-scroll {{ max-height: 480px; overflow: auto; }}
    .sys-card {{ border: 1px solid #334155; border-radius: 10px; padding: 14px; background: #0d1b2a; }}
    .sys-label {{ color: #94a3b8; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 10px; }}
    .sys-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 7px; font-size: 12px; color: #d1d5db; gap: 6px; }}
    .sys-row:last-child {{ margin-bottom: 0; }}
    .sys-key {{ color: #94a3b8; white-space: nowrap; }}
    .sys-val {{ display: flex; align-items: center; gap: 5px; }}
    .sys-bar-wrap {{ width: 52px; height: 4px; background: #1e293b; border-radius: 3px; overflow: hidden; }}
    .sys-bar {{ height: 100%; border-radius: 3px; }}
    .bar-ok {{ background: #10b981; }} .bar-warn {{ background: #f59e0b; }} .bar-bad {{ background: #f87171; }}
    .model-dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
    .bottom-cards {{ margin-top: auto; display: flex; flex-direction: column; gap: 10px; }}
    @media (max-width: 900px) {{ .ops-shell {{ display: block; }} .ops-sidebar {{ position: static; height: auto; }} .section-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="ops-shell">
    <aside class="ops-sidebar">
      <div class="brand"><div class="brand-mark">A</div><span>Adam · Receptionist</span></div>
      <nav class="nav-stack" aria-label="Dashboard navigation">
        <a class="nav-link active" href="/">Knowledge Files</a>
        <a class="nav-link" href="/ask">Ask Documents</a>
        <a class="nav-link" href="/voice">Voice</a>
        <a class="nav-link" href="/monitor">Monitor</a>
      </nav>
      <div class="bottom-cards">
        <div class="sys-card">
          <div class="sys-label">System</div>
          <div class="sys-row">
            <span class="sys-key">Memory</span>
            <span class="sys-val">
              {sys_res.memory_used_mb}/{sys_res.memory_total_mb} MB
              <span class="sys-bar-wrap"><span class="sys-bar {'bar-bad' if sys_res.memory_percent > 85 else 'bar-warn' if sys_res.memory_percent > 65 else 'bar-ok'}" style="width:{min(sys_res.memory_percent,100):.0f}%"></span></span>
            </span>
          </div>
          <div class="sys-row">
            <span class="sys-key">CPU load</span>
            <span class="sys-val">{sys_res.load_1m:.2f}</span>
          </div>
          <div class="sys-row">
            <span class="sys-key">Disk</span>
            <span class="sys-val">
              {sys_res.disk_used}/{sys_res.disk_total}
              <span class="sys-bar-wrap"><span class="sys-bar {'bar-bad' if sys_res.disk_percent > 90 else 'bar-warn' if sys_res.disk_percent > 70 else 'bar-ok'}" style="width:{min(sys_res.disk_percent,100):.0f}%"></span></span>
            </span>
          </div>
          <div class="sys-row">
            <span class="sys-key">Temperature</span>
            <span class="sys-val">{html.escape(temp_line)}</span>
          </div>
          <div class="sys-row">
            <span class="sys-key">Model</span>
            <span class="sys-val"><span class="model-dot" style="background:{model_color}"></span>{html.escape(model_label)}</span>
          </div>
        </div>
        <div class="side-card">Private local RAG — all files stay on this device.</div>
      </div>
    </aside>
    <main class="ops-main">
      <header class="topbar">
        <div><h1>Knowledge Files</h1><div class="subtle">Upload PDFs, text and Markdown for offline retrieval.</div></div>
      </header>

      {render_notice(message, 'ok')}
      {render_notice(error, 'error')}

      <div class="section-grid">
        <div class="panel">
          <h2>Upload Files</h2>
          <form method="post" action="/upload" enctype="multipart/form-data" id="upload-form">
            <div class="drop-zone" id="drop-zone" role="button" tabindex="0" aria-label="Drop files here or click to browse">
              <div style="font-size:32px">📂</div>
              <div class="drop-zone-label">Drag and drop files here, or click to browse</div>
              <div class="drop-zone-hint">Supported: .pdf, .txt, .md</div>
              <input id="files" name="files" type="file" multiple accept=".pdf,.txt,.md">
            </div>
            <p id="drop-file-names" style="color:#486581;font-size:13px;min-height:20px;margin:10px 0;"></p>
            <button type="submit">Upload Files</button>
          </form>
        </div>

        <div class="panel">
          <h2>Build Search Index</h2>
          <p style="color:#667085;font-size:13px;margin:0 0 16px;">Chunk and embed all uploaded files into the vector store.</p>
          <form method="post" action="/ingest">
            <div class="field-row">
              <div>
                <label for="chunk_size">Chunk size</label>
                <input id="chunk_size" name="chunk_size" type="number" min="200" max="3000" value="900">
              </div>
              <div>
                <label for="overlap">Overlap</label>
                <input id="overlap" name="overlap" type="number" min="0" max="1000" value="150">
              </div>
            </div>
            <button class="secondary" type="submit">Ingest Documents</button>
          </form>
        </div>
      </div>

      <div class="panel">
        <h2>Current Knowledge Files</h2>
        <div class="table-scroll">
          <table>
            <thead><tr><th>Name</th><th>Type</th><th>Size</th><th>Tags</th><th class="actions">Action</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
      </div>
    </main>
  </div>

  <script>
    (function() {{
      const zone = document.getElementById("drop-zone");
      const input = document.getElementById("files");
      const nameDisplay = document.getElementById("drop-file-names");
      function showNames(files) {{
        if (!files || !files.length) {{ nameDisplay.textContent = ""; return; }}
        nameDisplay.textContent = Array.from(files).map(f => f.name).join(", ");
      }}
      zone.addEventListener("click", () => input.click());
      zone.addEventListener("keydown", (e) => {{ if (e.key === "Enter" || e.key === " ") input.click(); }});
      input.addEventListener("change", () => showNames(input.files));
      zone.addEventListener("dragover", (e) => {{ e.preventDefault(); zone.classList.add("drag-over"); }});
      zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
      zone.addEventListener("drop", (e) => {{
        e.preventDefault(); zone.classList.remove("drag-over");
        const dt = e.dataTransfer;
        if (dt?.files?.length) {{
          input.files = dt.files;
          showNames(dt.files);
        }}
      }});
    }})();
  </script>
</body>
</html>"""


def render_ask_page(
    question: str,
    answer: str,
    sources: tuple[str, ...],
    error: str,
    history: list[InteractionRecord] | None = None,
    sessions: list[ChatSessionRecord] | None = None,
) -> str:
    config = AssistantConfig.from_env()
    answer_mode = html.escape(config.dashboard_answer_mode)
    model_name = html.escape(config.ollama_model)
    error_block = render_notice(error, "error")
    sessions_json = json.dumps([session_to_dict(item) for item in (sessions or [])]).replace(
        "</", "<\\/"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ask Documents</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f7f8; color: #111827; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; height: 100vh; overflow: hidden; background: #f7f7f8; color: #111827; }}
    button, textarea {{ font: inherit; }}
    .app-shell {{ height: 100vh; display: grid; grid-template-columns: 292px minmax(0, 1fr); overflow: hidden; }}
    .sidebar {{ height: 100vh; overflow: hidden; background: #0f172a; color: #f9fafb; padding: 18px 14px; display: flex; flex-direction: column; gap: 14px; }}
    .brand {{ display: flex; align-items: center; gap: 10px; font-weight: 760; padding: 6px 4px; }}
    .brand-mark {{ width: 34px; height: 34px; border-radius: 9px; display: grid; place-items: center; background: #f9fafb; color: #111827; font-weight: 800; }}
    .new-chat {{ width: 100%; min-height: 42px; border: 1px solid #334155; background: #111827; color: white; border-radius: 10px; cursor: pointer; font-weight: 760; text-align: left; padding: 0 12px; display: flex; align-items: center; text-decoration: none; }}
    .new-chat:hover {{ background: #1f2937; }}
    .nav-stack {{ display: grid; gap: 6px; }}
    .nav-link {{ color: #d1d5db; text-decoration: none; padding: 10px 12px; border-radius: 8px; font-weight: 650; }}
    .nav-link.active, .nav-link:hover {{ background: #1f2937; color: #ffffff; }}
    .sidebar-label {{ color: #94a3b8; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .04em; margin: 6px 6px 0; }}
    .sessions {{ display: grid; gap: 4px; overflow-y: auto; min-height: 0; padding-right: 2px; }}
    .session-button {{ border: 0; color: #d1d5db; background: transparent; text-align: left; border-radius: 8px; padding: 9px 10px; cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .session-button:hover, .session-button.active {{ background: #1f2937; color: white; }}
    .side-panel {{ margin-top: auto; border: 1px solid #334155; border-radius: 10px; padding: 12px; color: #cbd5e1; line-height: 1.45; font-size: 13px; }}
    .main {{ min-width: 0; height: 100vh; overflow: hidden; display: flex; flex-direction: column; }}
    .topbar {{ height: 68px; border-bottom: 1px solid #e5e7eb; background: rgba(255,255,255,.92); backdrop-filter: blur(12px); display: flex; align-items: center; justify-content: space-between; padding: 0 28px; position: sticky; top: 0; z-index: 2; }}
    .title-block h1 {{ margin: 0; font-size: 18px; line-height: 1.2; }}
    .title-block p {{ margin: 4px 0 0; color: #667085; font-size: 14px; }}
    .top-actions {{ display: flex; align-items: center; gap: 8px; }}
    .icon-button {{ min-width: 36px; height: 36px; border-radius: 999px; border: 1px solid #d0d5dd; background: white; cursor: pointer; }}
    .icon-button.danger {{ color: #b42318; border-color: #fecdca; }}
    .status-pill {{ border: 1px solid #d0d5dd; background: #ffffff; color: #344054; padding: 8px 10px; border-radius: 999px; font-size: 13px; font-weight: 650; white-space: nowrap; }}
    .chat-wrap {{ flex: 1; min-height: 0; width: 100%; overflow-y: auto; padding: 28px 22px 22px; }}
    .conversation {{ width: min(960px, 100%); margin: 0 auto; }}
    .notice {{ margin-bottom: 18px; padding: 12px 14px; border-radius: 8px; }}
    .error {{ background: #fef3f2; color: #912018; border: 1px solid #fecdca; }}
    .conversation {{ display: grid; gap: 22px; }}
    .turn {{ display: grid; grid-template-columns: 38px minmax(0, 1fr); gap: 12px; align-items: start; }}
    .turn.user-turn {{ grid-template-columns: minmax(0, 1fr) 38px; }}
    .turn.user-turn .avatar {{ order: 2; }}
    .turn.user-turn .message-body {{ order: 1; justify-self: end; }}
    .avatar {{ width: 34px; height: 34px; border-radius: 9px; display: grid; place-items: center; font-size: 12px; font-weight: 800; }}
    .assistant-avatar {{ background: #111827; color: #ffffff; }}
    .user-avatar {{ background: #e5e7eb; color: #111827; }}
    .message-body {{ max-width: min(760px, 100%); line-height: 1.7; white-space: pre-wrap; overflow-wrap: anywhere; }}
    .user-bubble {{ background: #111827; color: #ffffff; padding: 12px 14px; border-radius: 18px; border-top-right-radius: 6px; }}
    .assistant-text {{ padding-top: 4px; }}
    .message-meta {{ color: #667085; font-size: 12px; margin-top: 8px; }}
    .turn-actions {{ display: flex; gap: 6px; margin-top: 8px; }}
    .mini-button {{ border: 1px solid #d0d5dd; background: white; border-radius: 999px; padding: 4px 8px; color: #475467; font-size: 12px; cursor: pointer; }}
    .source-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .source-pill {{ display: inline-flex; align-items: center; min-height: 28px; border-radius: 999px; border: 1px solid #d0d5dd; background: #fff; color: #344054; padding: 5px 9px; font-size: 12px; }}
    .welcome {{ min-height: 54vh; display: grid; place-items: center; text-align: center; color: #667085; }}
    .welcome h2 {{ margin: 0; color: #111827; font-size: 28px; }}
    .suggestions {{ margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }}
    .suggestion {{ border: 1px solid #d0d5dd; background: white; border-radius: 999px; padding: 8px 12px; cursor: pointer; color: #344054; }}
    .composer-bar {{ flex: 0 0 auto; padding: 16px 22px 18px; background: linear-gradient(180deg, rgba(247,247,248,.72), #f7f7f8 28%); }}
    .composer {{ width: min(920px, 100%); margin: 0 auto; background: #fff; border: 1px solid #d0d5dd; border-radius: 18px; box-shadow: 0 12px 28px rgba(16,24,40,.10); display: grid; grid-template-columns: minmax(0,1fr) auto auto; gap: 10px; padding: 10px; align-items: end; }}
    .composer textarea {{ width: 100%; min-height: 52px; max-height: 180px; resize: none; border: 0; outline: 0; padding: 10px 12px; line-height: 1.45; color: #182230; }}
    .send-button {{ width: 44px; height: 44px; border-radius: 12px; border: 0; background: #111827; color: #fff; cursor: pointer; font-size: 18px; font-weight: 800; }}
    .send-button:disabled {{ opacity: .55; cursor: wait; }}
    .hint-row {{ width: min(920px,100%); margin: 8px auto 0; color: #667085; font-size: 12px; text-align: center; }}
    .thinking-dots {{ display: inline-flex; gap: 4px; margin-left: 6px; vertical-align: middle; }}
    .thinking-dots span {{ width: 6px; height: 6px; border-radius: 999px; background: #98a2b3; animation: pulse 1s infinite ease-in-out; }}
    .thinking-dots span:nth-child(2) {{ animation-delay: .15s; }}
    .thinking-dots span:nth-child(3) {{ animation-delay: .3s; }}
    @keyframes pulse {{ 0%,80%,100% {{ opacity: .35; transform: translateY(0); }} 40% {{ opacity: 1; transform: translateY(-3px); }} }}
    .hamburger {{ display:none; width:40px; height:40px; border:1px solid #d0d5dd; background:white; border-radius:9px; cursor:pointer; font-size:18px; flex-shrink:0; }}
    body.dark .hamburger {{ background:#1f2937; border-color:#374151; color:#f9fafb; }}
    .sidebar-backdrop {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.45); z-index:29; }}
    @media (max-width: 820px) {{
      body {{ overflow:auto; }}
      .app-shell {{ display:block; height:auto; overflow:visible; }}
      .sidebar {{ position:fixed; top:0; left:0; bottom:0; width:min(280px,85vw); height:100vh; z-index:30; transform:translateX(-100%); transition:transform .25s ease; overflow-y:auto; }}
      .sidebar.open {{ transform:translateX(0); }}
      .sidebar-backdrop.open {{ display:block; }}
      .hamburger {{ display:flex; align-items:center; justify-content:center; }}
      .side-panel {{ display:none; }}
      .main {{ height:auto; min-height:100vh; overflow:visible; }}
      .chat-wrap {{ overflow:visible; padding:16px 12px; }}
      .topbar {{ height:auto; padding:12px 14px; gap:8px; }}
      .title-block h1 {{ font-size:15px; }}
      .title-block p {{ display:none; }}
      .conversation {{ gap:16px; }}
      .turn,.turn.user-turn {{ grid-template-columns:30px minmax(0,1fr); gap:8px; }}
      .turn.user-turn .avatar,.turn.user-turn .message-body {{ order:0; justify-self:start; }}
      .avatar {{ width:30px; height:30px; font-size:10px; }}
      .composer-bar {{ padding:10px 10px 12px; }}
      .composer {{ border-radius:14px; gap:8px; padding:8px; }}
      .composer textarea {{ min-height:44px; font-size:15px; padding:8px 10px; }}
      .send-button,.mic-button {{ width:40px; height:40px; }}
      .hint-row {{ font-size:11px; }}
      .icon-button {{ min-width:32px; height:32px; font-size:14px; }}
      .status-pill {{ font-size:12px; padding:6px 8px; }}
      .source-pill {{ font-size:11px; padding:4px 8px; }}
      .chunk-panel {{ width:100vw; right:0; left:0; top:auto; bottom:0; border-radius:18px 18px 0 0; max-height:70vh; }}
      .welcome {{ min-height:40vh; }}
      .welcome h2 {{ font-size:20px; }}
      .suggestions {{ gap:6px; }}
      .suggestion {{ font-size:12px; padding:6px 10px; }}
    }}
    /* Dark mode */
    body.dark {{ background:#111827; color:#f9fafb; }}
    body.dark .topbar {{ background:rgba(17,24,39,.95); border-color:#374151; }}
    body.dark .chat-wrap {{ background:#111827; }}
    body.dark .source-pill {{ background:#1f2937; border-color:#374151; color:#d1d5db; }}
    body.dark .mini-button {{ background:#1f2937; border-color:#374151; color:#d1d5db; }}
    body.dark .icon-button {{ background:#1f2937; border-color:#374151; color:#f9fafb; }}
    body.dark .status-pill {{ background:#1f2937; border-color:#374151; color:#f9fafb; }}
    body.dark .composer {{ background:#1f2937; border-color:#374151; }}
    body.dark .composer textarea {{ background:#1f2937; color:#f9fafb; }}
    body.dark .suggestion {{ background:#1f2937; border-color:#374151; color:#f9fafb; }}
    body.dark .feedback-btn {{ background:#1f2937; border-color:#374151; color:#d1d5db; }}
    body.dark .message-meta {{ color:#9ca3af; }}
    /* Mic button */
    .mic-button {{ width:44px; height:44px; border-radius:12px; border:1px solid #d0d5dd; background:white; cursor:pointer; font-size:16px; flex-shrink:0; }}
    .mic-button.listening {{ background:#fee2e2; border-color:#fca5a5; animation:mic-pulse 1.2s infinite; }}
    @keyframes mic-pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:.6; }} }}
    body.dark .mic-button {{ background:#1f2937; border-color:#374151; color:#f9fafb; }}
    /* Confidence badges */
    .confidence-doc {{ background:#ecfdf3; color:#067647; border:1px solid #6ce9a6; border-radius:999px; padding:2px 7px; font-size:11px; font-weight:700; }}
    .confidence-gen {{ background:#eff6ff; color:#1d4ed8; border:1px solid #93c5fd; border-radius:999px; padding:2px 7px; font-size:11px; font-weight:700; }}
    body.dark .confidence-doc {{ background:#052e16; border-color:#166534; }}
    body.dark .confidence-gen {{ background:#0c1a3d; border-color:#1e3a8a; color:#93c5fd; }}
    /* Feedback buttons */
    .feedback-row {{ display:flex; gap:6px; margin-top:8px; }}
    .feedback-btn {{ border:1px solid #d0d5dd; background:white; border-radius:999px; padding:3px 10px; cursor:pointer; font-size:13px; }}
    .feedback-btn.voted-up {{ background:#ecfdf3; border-color:#6ce9a6; }}
    .feedback-btn.voted-down {{ background:#fef3f2; border-color:#fecdca; }}
    .feedback-btn:disabled {{ opacity:.6; cursor:default; }}
    .icon-button.vm-active {{ background:#dcfce7; border-color:#6ce9a6; color:#15803d; }}
    body.dark .icon-button.vm-active {{ background:#052e16; border-color:#166534; color:#4ade80; }}
    /* Source chunk preview overlay */
    .chunk-overlay {{ display:none; position:fixed; inset:0; z-index:50; background:rgba(0,0,0,.45); }}
    .chunk-overlay.open {{ display:block; }}
    .chunk-panel {{ position:absolute; right:0; top:0; bottom:0; width:min(460px,96vw); background:white; padding:24px; overflow-y:auto; box-shadow:-6px 0 32px rgba(0,0,0,.18); display:flex; flex-direction:column; gap:14px; }}
    body.dark .chunk-panel {{ background:#1f2937; color:#f9fafb; }}
    .chunk-panel-header {{ display:flex; align-items:center; justify-content:space-between; gap:12px; }}
    .chunk-close {{ border:0; background:transparent; font-size:22px; cursor:pointer; color:#6b7280; line-height:1; }}
    .chunk-label {{ font-weight:800; font-size:14px; color:#344054; flex:1; word-break:break-all; }}
    body.dark .chunk-label {{ color:#d1d5db; }}
    .chunk-body {{ white-space:pre-wrap; line-height:1.7; font-size:14px; color:#374151; background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:14px; }}
    body.dark .chunk-body {{ background:#111827; border-color:#374151; color:#d1d5db; }}
    .source-pill.has-chunk {{ cursor:pointer; }}
    .source-pill.has-chunk:hover {{ background:#eff6ff; border-color:#93c5fd; }}
    body.dark .source-pill.has-chunk:hover {{ background:#1e3a5f; border-color:#2563eb; }}
    /* Autocomplete */
    .composer-bar {{ position:relative; }}
    .ac-list {{ position:absolute; bottom:calc(100% + 6px); left:22px; right:22px; background:white; border:1px solid #d0d5dd; border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,.12); padding:4px; z-index:20; max-height:220px; overflow-y:auto; }}
    body.dark .ac-list {{ background:#1f2937; border-color:#374151; }}
    .ac-item {{ padding:9px 12px; border-radius:8px; cursor:pointer; font-size:13px; color:#344054; line-height:1.4; }}
    .ac-item:hover, .ac-item.ac-active {{ background:#eff6ff; color:#0369a1; }}
    body.dark .ac-item {{ color:#d1d5db; }}
    body.dark .ac-item:hover {{ background:#111827; color:#93c5fd; }}
  </style>
</head>
<body>
  <div class="sidebar-backdrop" id="sidebar-backdrop"></div>
  <div class="app-shell">
    <aside class="sidebar" id="sidebar">
      <div class="brand"><div class="brand-mark">OA</div><span>Offline Assistant</span></div>
      <a class="new-chat" id="new-chat-button" href="/ask?new=1" role="button">+ New chat</a>
      <nav class="nav-stack" aria-label="Dashboard navigation">
        <a class="nav-link" href="/">Knowledge Files</a>
        <a class="nav-link active" href="/ask">Ask Documents</a>
        <a class="nav-link" href="/voice">Voice</a>
        <a class="nav-link" href="/monitor">Monitor</a>
      </nav>
      <div class="sidebar-label">Recent chats</div>
      <div class="sessions" id="sessions"></div>
      <div class="side-panel"><strong>Answer mode:</strong> {answer_mode}<br><strong>AI model:</strong> {model_name}<br><br>Private local RAG on this device.</div>
    </aside>
    <main class="main">
      <header class="topbar">
        <button class="hamburger" id="hamburger" type="button" aria-label="Open menu">☰</button>
        <div class="title-block"><h1 id="chat-title">New chat</h1><p>Answers stream from your indexed local files.</p></div>
        <div class="top-actions"><button class="icon-button" id="voice-mode-toggle" title="Voice mode (speak → auto-send → robot replies)" type="button">🗣</button><button class="icon-button" id="dark-toggle" title="Dark mode" type="button">☾</button><button class="icon-button" id="export-chat" title="Export chat as .txt" type="button">⬇</button><button class="icon-button" id="copy-chat" title="Copy chat" type="button">⧉</button><button class="icon-button danger" id="stop-chat" title="Stop generating" type="button" hidden>■</button><button class="icon-button" id="delete-chat" title="Delete current chat" type="button">×</button><div class="status-pill" id="assistant-status">Ready · {answer_mode}</div></div>
      </header>
      <section class="chat-wrap" aria-label="Document chat">{error_block}<div class="conversation" id="conversation"></div></section>
      <div class="composer-bar"><form class="composer" id="ask-form" method="post" action="/ask"><textarea id="question" name="question" placeholder="Ask anything from your PDFs..." required rows="1" autocomplete="off" autocorrect="on" autocapitalize="sentences"></textarea><button class="mic-button" id="mic-button" type="button" title="Speak your question">🎤</button><button class="send-button" type="submit" id="send-button">↑</button></form><div class="hint-row" id="hint-row">Tap Enter to send · 🎤 for voice</div></div>
    </main>
  </div>
  <div class="chunk-overlay" id="chunk-overlay" role="dialog" aria-modal="true" aria-label="Source preview">
    <div class="chunk-panel">
      <div class="chunk-panel-header">
        <span class="chunk-label" id="chunk-preview-label"></span>
        <button class="chunk-close" id="chunk-close" type="button" aria-label="Close">✕</button>
      </div>
      <div id="chunk-pdf-link"></div>
      <div class="chunk-body" id="chunk-preview-body"></div>
    </div>
  </div>
  <script id="initial-sessions" type="application/json">{sessions_json}</script>
  <script>
    const form = document.getElementById("ask-form");
    const textarea = document.getElementById("question");
    const conversation = document.getElementById("conversation");
    const chatWrap = document.querySelector(".chat-wrap");
    const statusPill = document.getElementById("assistant-status");
    const button = document.getElementById("send-button");
    const sessionsNode = document.getElementById("sessions");
    const titleNode = document.getElementById("chat-title");
    const newChatButton = document.getElementById("new-chat-button");
    const deleteChatButton = document.getElementById("delete-chat");
    const stopChatButton = document.getElementById("stop-chat");
    const copyChatButton = document.getElementById("copy-chat");
    const micButton = document.getElementById("mic-button");
    const darkToggle = document.getElementById("dark-toggle");
    const exportButton = document.getElementById("export-chat");
    const chunkOverlay = document.getElementById("chunk-overlay");
    const hamburger = document.getElementById("hamburger");
    const sidebar = document.getElementById("sidebar");
    const sidebarBackdrop = document.getElementById("sidebar-backdrop");
    function openSidebar() {{ sidebar?.classList.add("open"); sidebarBackdrop?.classList.add("open"); }}
    function closeSidebar() {{ sidebar?.classList.remove("open"); sidebarBackdrop?.classList.remove("open"); }}
    hamburger?.addEventListener("click", () => {{ sidebar?.classList.contains("open") ? closeSidebar() : openSidebar(); }});
    sidebarBackdrop?.addEventListener("click", closeSidebar);
    function readInitialSessions() {{
      try {{
        return JSON.parse(document.getElementById("initial-sessions").textContent || "[]");
      }} catch (error) {{
        console.warn("Could not read initial sessions", error);
        return [];
      }}
    }}
    let sessions = readInitialSessions();
    const forceNewChat = new URLSearchParams(window.location.search).has("new");
    let activeSessionId = forceNewChat ? createSessionId() : (localStorage.getItem("assistant_active_session") || (sessions[0]?.session_id || createSessionId()));
    let activeController = null;
    let voiceMode = false;
    if (localStorage.getItem("oa_dark") === "1") {{ document.body.classList.add("dark"); if (darkToggle) darkToggle.title = "Light mode"; }}

    function createSessionId() {{
      if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {{
        return "chat-" + globalThis.crypto.randomUUID();
      }}
      const randomPart = Math.random().toString(36).slice(2, 10);
      return `chat-${{Date.now().toString(36)}}-${{randomPart}}`;
    }}
    function escapeHtml(value) {{ return String(value).replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}}[c])); }}
    function scrollToBottom() {{ chatWrap?.scrollTo({{ top: chatWrap.scrollHeight, behavior: "smooth" }}); }}
    function setActiveSession(id) {{ activeSessionId = id; localStorage.setItem("assistant_active_session", id); }}

    function renderSessions() {{
      sessionsNode.innerHTML = sessions.map(s => `<button class="session-button ${{s.session_id === activeSessionId ? "active" : ""}}" data-session="${{escapeHtml(s.session_id)}}">${{escapeHtml(s.title || "New chat")}}</button>`).join("");
      sessionsNode.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => loadSession(btn.dataset.session)));
    }}

    function welcome() {{
      conversation.innerHTML = `<div class="welcome"><div><h2>Ask your local documents</h2><p>Start a private chat from your indexed files.</p><div class="suggestions"><button class="suggestion">Summarize this PDF</button><button class="suggestion">What are the key design points?</button><button class="suggestion">Show sources for robot eyes</button></div></div></div>`;
      conversation.querySelectorAll(".suggestion").forEach(item => item.addEventListener("click", () => {{ textarea.value = item.textContent; textarea.focus(); }}));
    }}

    function appendUserTurn(text) {{
      conversation.querySelector(".welcome")?.remove();
      const turn = document.createElement("article");
      turn.className = "turn user-turn";
      turn.innerHTML = `<div class="avatar user-avatar">You</div><div class="message-body"><div class="user-bubble">${{escapeHtml(text)}}</div></div>`;
      conversation.appendChild(turn); scrollToBottom();
    }}
    function appendAssistantTurn() {{
      const turn = document.createElement("article");
      turn.className = "turn assistant-turn";
      turn.innerHTML = `<div class="avatar assistant-avatar">OA</div><div class="message-body"><div class="assistant-text"><span class="thinking-dots"><span></span><span></span><span></span></span></div><div class="source-row"></div><div class="turn-actions"><button class="mini-button copy-answer" type="button">Copy</button></div><div class="feedback-row"><button class="feedback-btn thumb-up" type="button" title="Good answer">👍</button><button class="feedback-btn thumb-down" type="button" title="Poor answer">👎</button></div><div class="message-meta"></div></div>`;
      conversation.appendChild(turn);
      turn.querySelector(".copy-answer").addEventListener("click", () => navigator.clipboard?.writeText(turn.querySelector(".assistant-text").textContent || ""));
      const thumbUp = turn.querySelector(".thumb-up");
      const thumbDown = turn.querySelector(".thumb-down");
      function doFeedback(rating) {{
        const iid = turn.dataset.interactionId;
        if (!iid) return;
        const fd = new FormData(); fd.append("interaction_id", iid); fd.append("rating", rating);
        fetch("/ask/feedback", {{method:"POST",body:fd}}).catch(()=>{{}});
        thumbUp.classList.toggle("voted-up", rating==="up"); thumbDown.classList.toggle("voted-down", rating==="down");
        thumbUp.disabled = true; thumbDown.disabled = true;
      }}
      thumbUp.addEventListener("click", () => doFeedback("up"));
      thumbDown.addEventListener("click", () => doFeedback("down"));
      scrollToBottom(); return turn;
    }}
    function showChunkPreview(label, text) {{
      document.getElementById("chunk-preview-label").textContent = label;
      document.getElementById("chunk-preview-body").textContent = text || "(no preview available)";
      const pdfLinkEl = document.getElementById("chunk-pdf-link");
      pdfLinkEl.innerHTML = "";
      const parts = label.split(","); const filename = parts[0].trim();
      const pageMatch = label.match(/,\\s*page\\s+(\\d+)/i);
      if (filename) {{
        const a = document.createElement("a");
        a.target = "_blank"; a.rel = "noopener noreferrer";
        a.style.cssText = "display:inline-block;color:#0369a1;font-size:13px;font-weight:700;text-decoration:none;border:1px solid #93c5fd;padding:4px 10px;border-radius:6px;margin-bottom:4px;";
        if (pageMatch && /\\.pdf$/i.test(filename)) {{
          a.href = "/documents/" + encodeURIComponent(filename) + "#page=" + pageMatch[1];
          a.textContent = "Open PDF · page " + pageMatch[1];
        }} else {{
          a.href = "/documents/" + encodeURIComponent(filename);
          a.textContent = "Open document";
        }}
        pdfLinkEl.appendChild(a);
      }}
      chunkOverlay?.classList.add("open");
    }}
    function renderSources(container, sources, chunks) {{ const row = container.querySelector(".source-row"); const chunkMap = {{}}; (chunks || []).forEach(c => {{ chunkMap[c.label] = c.text; }}); row.innerHTML = (sources || []).map(s => {{ const hc = !!chunkMap[s]; return `<span class="source-pill${{hc?" has-chunk":""}}" data-chunk-label="${{escapeHtml(s)}}">${{escapeHtml(s)}}</span>`; }}).join(""); row.querySelectorAll(".has-chunk").forEach(pill => pill.addEventListener("click", () => showChunkPreview(pill.dataset.chunkLabel, chunkMap[pill.dataset.chunkLabel]))); }}
    function renderTurns(turns) {{
      if (!turns.length) {{ welcome(); return; }}
      conversation.innerHTML = "";
      for (const item of turns) {{
        appendUserTurn(item.question);
        const turn = appendAssistantTurn();
        if (item.id) turn.dataset.interactionId = String(item.id);
        turn.querySelector(".assistant-text").textContent = item.answer || "";
        renderSources(turn, item.sources || []);
        turn.querySelector(".message-meta").textContent = `${{item.created_at}} · ${{item.answer_mode}} · ${{item.model}} · ${{Number(item.total_seconds || 0).toFixed(2)}}s`;
      }}
    }}
    async function refreshSessions() {{ const r = await fetch("/ask/sessions"); sessions = await r.json(); renderSessions(); }}
    async function loadSession(id) {{ setActiveSession(id); renderSessions(); const r = await fetch(`/ask/session/${{encodeURIComponent(id)}}`); const turns = await r.json(); titleNode.textContent = sessions.find(s => s.session_id === id)?.title || "New chat"; renderTurns(turns); }}
    function newChat() {{
      if (activeController) activeController.abort();
      button.disabled = false;
      button.textContent = "↑";
      textarea.value = "";
      textarea.style.height = "auto";
      const id = createSessionId();
      setActiveSession(id);
      if (!sessions.some(s => s.session_id === id)) {{
        sessions = [{{ session_id: id, title: "New chat", created_at: "", updated_at: "", message_count: 0 }}, ...sessions];
      }}
      titleNode.textContent = "New chat";
      statusPill.textContent = `Ready · {answer_mode}`;
      renderSessions();
      welcome();
      chatWrap?.scrollTo({{ top: 0 }});
      textarea.focus();
    }}

    textarea?.addEventListener("input", () => {{ textarea.style.height = "auto"; textarea.style.height = Math.min(textarea.scrollHeight, 180) + "px"; }});
    textarea?.addEventListener("keydown", (event) => {{
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {{
        event.preventDefault();
        form.requestSubmit();
      }}
    }});
    newChatButton.addEventListener("click", (event) => {{ event.preventDefault(); newChat(); }});
    deleteChatButton.addEventListener("click", async () => {{ await fetch(`/ask/session/${{encodeURIComponent(activeSessionId)}}/delete`, {{ method: "POST" }}); await refreshSessions(); newChat(); }});
    stopChatButton.addEventListener("click", () => {{ activeController?.abort(); window.speechSynthesis?.cancel(); }});
    copyChatButton.addEventListener("click", () => navigator.clipboard?.writeText(conversation.textContent || ""));
    darkToggle?.addEventListener("click", () => {{
      const isDark = document.body.classList.toggle("dark");
      localStorage.setItem("oa_dark", isDark ? "1" : "0");
      if (darkToggle) {{ darkToggle.title = isDark ? "Light mode" : "Dark mode"; darkToggle.textContent = isDark ? "☀" : "☾"; }}
    }});
    document.getElementById("voice-mode-toggle")?.addEventListener("click", () => {{
      voiceMode = !voiceMode;
      const vmBtn = document.getElementById("voice-mode-toggle");
      if (vmBtn) {{ vmBtn.classList.toggle("vm-active", voiceMode); vmBtn.title = voiceMode ? "Voice mode ON — tap mic, speak, robot replies (click to turn off)" : "Voice mode (speak → auto-send → robot replies)"; }}
      if (!voiceMode) window.speechSynthesis?.cancel();
    }});
    exportButton?.addEventListener("click", () => {{
      let text = "";
      conversation.querySelectorAll(".turn").forEach(turn => {{
        if (turn.classList.contains("user-turn")) {{
          text += "You: " + (turn.querySelector(".user-bubble")?.textContent || "") + "\\n\\n";
        }} else {{
          text += "Assistant: " + (turn.querySelector(".assistant-text")?.textContent || "") + "\\n\\n";
        }}
      }});
      if (!text.trim()) return;
      const blob = new Blob([text], {{type:"text/plain"}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url; a.download = "chat-export.txt"; a.click();
      URL.revokeObjectURL(url);
    }});
    chunkOverlay?.addEventListener("click", (e) => {{ if (e.target === chunkOverlay) chunkOverlay.classList.remove("open"); }});
    document.getElementById("chunk-close")?.addEventListener("click", () => chunkOverlay?.classList.remove("open"));
    (function setupMic() {{
      if (!micButton) return;

      function fillText(t) {{
        textarea.value = t;
        textarea.style.height = "auto";
        textarea.style.height = Math.min(textarea.scrollHeight, 180) + "px";
        textarea.focus();
      }}

      async function audioToWav(blob) {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)({{sampleRate: 16000}});
        const ab = await blob.arrayBuffer();
        let audioBuf;
        try {{ audioBuf = await ctx.decodeAudioData(ab); }} catch(e) {{ ctx.close(); throw e; }}
        ctx.close();
        const pcm = audioBuf.getChannelData(0);
        const i16 = new Int16Array(pcm.length);
        for (let i = 0; i < pcm.length; i++) i16[i] = Math.max(-32768, Math.min(32767, pcm[i] * 32768));
        const buf = new ArrayBuffer(44 + i16.byteLength);
        const v = new DataView(buf);
        const ws = (o, s) => {{ for (let i=0;i<s.length;i++) v.setUint8(o+i, s.charCodeAt(i)); }};
        ws(0,"RIFF"); v.setUint32(4,36+i16.byteLength,true); ws(8,"WAVE");
        ws(12,"fmt "); v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true);
        v.setUint32(24,16000,true); v.setUint32(28,32000,true); v.setUint16(32,2,true); v.setUint16(34,16,true);
        ws(36,"data"); v.setUint32(40,i16.byteLength,true);
        new Int16Array(buf,44).set(i16);
        return new Blob([buf], {{type:"audio/wav"}});
      }}

      async function sendForTranscription(audioBlob) {{
        micButton.disabled = true; micButton.textContent = "⏳";
        try {{
          const wav = await audioToWav(audioBlob);
          const fd = new FormData(); fd.append("file", wav, "rec.wav");
          const resp = await fetch("/ask/transcribe", {{method:"POST", body:fd}});
          const data = await resp.json();
          if (data.text) {{ fillText(data.text); if (voiceMode) setTimeout(() => form.requestSubmit(), 200); }}
        }} catch(e) {{ console.error("Transcription error", e); }}
        finally {{ micButton.disabled = false; micButton.textContent = "🎤"; }}
      }}

      /* --- Path 1: Web Speech API (HTTPS or localhost) --- */
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      const secureCtx = location.protocol === "https:" || location.hostname === "localhost" || location.hostname === "127.0.0.1";
      if (SR && secureCtx) {{
        const rec = new SR(); rec.lang = "en-US"; rec.interimResults = true; rec.continuous = false;
        rec.onresult = (e) => {{
          fillText(Array.from(e.results).map(r => r[0].transcript).join(""));
          if (e.results[e.results.length - 1].isFinal) {{ micButton.classList.remove("listening"); if (voiceMode && textarea.value.trim()) setTimeout(() => form.requestSubmit(), 200); }}
        }};
        rec.onerror = () => micButton.classList.remove("listening");
        rec.onend = () => micButton.classList.remove("listening");
        micButton.addEventListener("click", () => {{
          micButton.classList.contains("listening") ? rec.stop() : ((() => {{ try {{ rec.start(); micButton.classList.add("listening"); }} catch(_) {{}} }})());
        }});
        return;
      }}

      /* --- Path 2: MediaRecorder (getUserMedia available, e.g. some Android browsers on HTTP) --- */
      let mediaRec = null; let chunks = [];
      async function stopAndTranscribe() {{
        micButton.classList.remove("listening");
        const mimeType = mediaRec?.mimeType || "audio/webm";
        mediaRec?.stop();
        await new Promise(r => {{ if (mediaRec) mediaRec.onstop = r; else r(); }});
        const raw = new Blob(chunks, {{type: mimeType}});
        chunks = []; mediaRec = null;
        await sendForTranscription(raw);
      }}

      /* --- Path 3: File capture input (iOS Safari on HTTP - opens native voice recorder) --- */
      const fileCapture = document.createElement("input");
      fileCapture.type = "file"; fileCapture.accept = "audio/*";
      fileCapture.setAttribute("capture", "microphone");
      fileCapture.style.display = "none";
      document.body.appendChild(fileCapture);
      fileCapture.addEventListener("change", async () => {{
        const file = fileCapture.files[0];
        if (file) await sendForTranscription(file);
        fileCapture.value = "";
      }});

      micButton.addEventListener("click", async () => {{
        /* If already recording via MediaRecorder, stop it */
        if (mediaRec && mediaRec.state === "recording") {{ stopAndTranscribe(); return; }}

        /* Try getUserMedia first */
        if (navigator.mediaDevices?.getUserMedia) {{
          try {{
            const stream = await navigator.mediaDevices.getUserMedia({{audio:true}});
            chunks = [];
            const opts = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? {{mimeType:"audio/webm;codecs=opus"}} : {{}};
            mediaRec = new MediaRecorder(stream, opts);
            mediaRec.ondataavailable = e => {{ if (e.data.size > 0) chunks.push(e.data); }};
            mediaRec.start(250);
            micButton.classList.add("listening");
            setTimeout(() => {{ if (mediaRec?.state === "recording") stopAndTranscribe(); }}, 15000);
            return;
          }} catch(_) {{/* fall through to file capture */}}
        }}

        /* File capture fallback - works on iOS Safari over HTTP */
        fileCapture.click();
      }});
    }})();

    (function setupAutocomplete() {{
      let acTimer = null;
      let acEl = null;
      function getList() {{
        if (!acEl) {{
          acEl = document.createElement("div"); acEl.className = "ac-list"; acEl.style.display = "none";
          document.querySelector(".composer-bar")?.appendChild(acEl);
        }}
        return acEl;
      }}
      function clearList() {{ if (acEl) {{ acEl.style.display = "none"; acEl.innerHTML = ""; }} }}
      textarea?.addEventListener("input", () => {{
        clearTimeout(acTimer);
        const q = textarea.value.trim();
        if (q.length < 3 || button.disabled) {{ clearList(); return; }}
        acTimer = setTimeout(async () => {{
          try {{
            const r = await fetch("/ask/suggestions?q=" + encodeURIComponent(q));
            const items = await r.json();
            if (!items.length) {{ clearList(); return; }}
            const list = getList(); list.innerHTML = "";
            items.forEach(s => {{
              const el = document.createElement("div"); el.className = "ac-item"; el.textContent = s;
              el.addEventListener("mousedown", (e) => {{
                e.preventDefault();
                textarea.value = s.endsWith("?") ? s.slice(0,-1) : s;
                textarea.style.height = "auto"; textarea.style.height = Math.min(textarea.scrollHeight,180)+"px";
                clearList(); textarea.focus();
              }});
              list.appendChild(el);
            }});
            list.style.display = "block";
          }} catch(e) {{ clearList(); }}
        }}, 360);
      }});
      document.addEventListener("click", (e) => {{ if (!e.target.closest(".composer-bar")) clearList(); }});
    }})();

    form?.addEventListener("submit", async (event) => {{
      event.preventDefault();
      const question = textarea.value.trim();
      if (!question || button.disabled) return;
      document.querySelector(".ac-list")?.remove();
      appendUserTurn(question); textarea.value = ""; textarea.style.height = "auto";
      const assistantTurn = appendAssistantTurn();
      const answerNode = assistantTurn.querySelector(".assistant-text");
      const metaNode = assistantTurn.querySelector(".message-meta");
      const formData = new FormData(); formData.append("question", question); formData.append("session_id", activeSessionId);
      button.disabled = true; button.textContent = "…"; stopChatButton.hidden = false; statusPill.textContent = "Searching · {answer_mode}";
      let started = false; let buffer = ""; activeController = new AbortController();
      try {{
        const response = await fetch("/ask/stream", {{ method: "POST", body: formData, signal: activeController.signal }});
        const reader = response.body.getReader(); const decoder = new TextDecoder();
        while (true) {{
          const {{ value, done }} = await reader.read(); if (done) break;
          buffer += decoder.decode(value, {{ stream: true }});
          const lines = buffer.split("\\n"); buffer = lines.pop() || "";
          for (const line of lines) {{
            if (!line.trim()) continue; const data = JSON.parse(line);
            if (data.type === "status") statusPill.textContent = `${{data.message}} · {answer_mode}`;
            if (data.type === "sources") renderSources(assistantTurn, data.sources || [], data.chunks || []);
            if (data.type === "token") {{ if (!started) {{ answerNode.textContent = ""; started = true; }} answerNode.textContent += data.text || ""; scrollToBottom(); }}
            if (data.type === "done") {{ renderSources(assistantTurn, data.sources || [], data.chunks || []); if (data.interaction_id) assistantTurn.dataset.interactionId = String(data.interaction_id); const confBadge = data.knowledge_source === "documents" ? `<span class="confidence-doc">From documents</span> ` : data.knowledge_source === "general" ? `<span class="confidence-gen">General knowledge</span> ` : ""; metaNode.innerHTML = confBadge + escapeHtml(`${{data.answer_mode || "{answer_mode}"}} · ${{data.model || "{model_name}"}} · ${{Number(data.total_seconds || 0).toFixed(2)}}s (retrieval ${{Number(data.retrieval_seconds || 0).toFixed(2)}}s, generation ${{Number(data.generation_seconds || 0).toFixed(2)}}s)`); statusPill.textContent = `Ready · {answer_mode}`; await refreshSessions(); titleNode.textContent = sessions.find(s => s.session_id === activeSessionId)?.title || question; if (voiceMode && window.speechSynthesis) {{ const spokenText = answerNode.textContent.trim(); if (spokenText) {{ const utter = new SpeechSynthesisUtterance(spokenText); utter.rate = 1.05; utter.onend = () => {{ if (voiceMode) setTimeout(() => micButton.click(), 400); }}; window.speechSynthesis.cancel(); window.speechSynthesis.speak(utter); }} else if (voiceMode) {{ setTimeout(() => micButton.click(), 400); }} }} }}
            if (data.type === "error") {{ answerNode.textContent = data.message || "I could not answer that clearly."; statusPill.textContent = `Error · {answer_mode}`; }}
          }}
        }}
      }} catch (error) {{ answerNode.textContent = error.name === "AbortError" ? "Stopped." : "I could not complete the request. Check the monitor error log."; statusPill.textContent = error.name === "AbortError" ? `Stopped · {answer_mode}` : `Error · {answer_mode}`; }}
      finally {{ button.disabled = false; button.textContent = "↑"; stopChatButton.hidden = true; activeController = null; textarea.focus(); scrollToBottom(); }}
    }});

    renderSessions();
    if (forceNewChat) {{
      newChat();
      window.history.replaceState(null, "", "/ask");
    }} else {{
      loadSession(activeSessionId).catch(() => welcome());
    }}
  </script>
</body>
</html>"""


def render_history_items(history: list[InteractionRecord]) -> str:
    if not history:
        return """
        <div class="welcome">
          <div><h2>Ask your local documents</h2><p>Your private chat history will appear here.</p></div>
        </div>"""
    return "".join(render_history_item(item) for item in history)


def display_answer_text(answer: str) -> str:
    marker = "\n\nLatency:"
    if marker in answer:
        return answer.split(marker, 1)[0]
    return answer


def render_history_item(item: InteractionRecord) -> str:
    sources = "".join(
        f'<span class="source-pill">{html.escape(source)}</span>' for source in item.sources
    )
    source_row = f'<div class="source-row">{sources}</div>' if sources else ""
    meta = f"{html.escape(item.created_at)} · {html.escape(item.answer_mode)} · {html.escape(item.model)} · {item.total_seconds:.2f}s"
    return f"""
    <article class="turn user-turn">
      <div class="avatar user-avatar">You</div>
      <div class="message-body"><div class="user-bubble">{html.escape(item.question)}</div></div>
    </article>
    <article class="turn assistant-turn">
      <div class="avatar assistant-avatar">OA</div>
      <div class="message-body">
        <div class="assistant-text">{html.escape(display_answer_text(item.answer))}</div>
        {source_row}
        <div class="message-meta">{meta}</div>
      </div>
    </article>"""



def build_index_health(config: AssistantConfig) -> IndexHealth:
    files = iter_documents(DOCUMENTS_DIR)
    uploaded_by_name = {path.name: path for path in files}
    metadata = MetadataStore(config.sqlite_path)
    stats = metadata.index_stats()
    indexed = metadata.indexed_documents()
    indexed_by_name = {record.name: record for record in indexed}
    unindexed = tuple(sorted(name for name in uploaded_by_name if name not in indexed_by_name))
    stale: list[str] = []
    for name, path in uploaded_by_name.items():
        record = indexed_by_name.get(name)
        if not record:
            continue
        try:
            file_mtime = path.stat().st_mtime
            indexed_time = parse_sqlite_timestamp(record.indexed_at)
        except OSError:
            continue
        if indexed_time is not None and file_mtime > indexed_time:
            stale.append(name)
    return IndexHealth(
        uploaded_file_count=len(files),
        indexed_document_count=stats.document_count,
        chunk_count=stats.chunk_count,
        last_indexed_at=stats.last_indexed_at,
        unindexed_files=unindexed,
        stale_files=tuple(sorted(stale)),
    )


def parse_sqlite_timestamp(value: str) -> float | None:
    from datetime import datetime, timezone

    if not value or value == "-":
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return parsed.timestamp()


def check_model_status(config: AssistantConfig) -> ModelStatus:
    url = config.ollama_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            data = json.loads(response.read().decode("utf-8"))
        names = {str(model.get("name", "")) for model in data.get("models", [])}
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return ModelStatus(False, config.ollama_model, config.ollama_url, config.dashboard_answer_mode, str(exc))

    if config.ollama_model in names:
        message = "reachable and selected model is installed"
        reachable = True
    else:
        message = "reachable, but selected model was not listed by Ollama"
        reachable = False
    return ModelStatus(reachable, config.ollama_model, config.ollama_url, config.dashboard_answer_mode, message)


def read_system_resources() -> SystemResources:
    load_1m = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    memory = read_memory_info()
    disk = shutil.disk_usage(".")
    temp = read_temperature_c()
    memory_total = memory.get("MemTotal", 0)
    memory_available = memory.get("MemAvailable", 0)
    memory_used = max(0, memory_total - memory_available)
    memory_percent = (memory_used / memory_total * 100) if memory_total else 0.0
    disk_percent = (disk.used / disk.total * 100) if disk.total else 0.0
    return SystemResources(
        load_1m=load_1m,
        memory_used_mb=memory_used // 1024,
        memory_total_mb=memory_total // 1024,
        memory_percent=memory_percent,
        disk_used=format_size(disk.used),
        disk_total=format_size(disk.total),
        disk_percent=disk_percent,
        temperature_c=temp,
    )


def read_memory_info() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw_value = line.split(":", 1)
            values[key] = int(raw_value.strip().split()[0])
    except (OSError, ValueError):
        return {}
    return values


def read_temperature_c() -> float | None:
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            value = int(path.read_text().strip())
        except (OSError, ValueError):
            continue
        if value > 0:
            return value / 1000
    return None


def safe_session_id(session_id: str) -> str:
    cleaned = "".join(char for char in session_id if char.isalnum() or char in {"-", "_"})
    return cleaned[:80] or "default"


def session_to_dict(item: ChatSessionRecord) -> dict[str, object]:
    return {
        "session_id": item.session_id,
        "title": item.title,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "message_count": item.message_count,
    }


def interaction_to_chat_dict(item: InteractionRecord) -> dict[str, object]:
    return {
        "id": item.id,
        "session_id": item.session_id,
        "created_at": item.created_at,
        "question": item.question,
        "answer": display_answer_text(item.answer),
        "sources": list(item.sources),
        "answer_mode": item.answer_mode,
        "model": item.model,
        "retrieval_seconds": item.retrieval_seconds,
        "generation_seconds": item.generation_seconds,
        "total_seconds": item.total_seconds,
    }


def interaction_to_dict(item: InteractionRecord) -> dict[str, object]:
    return {
        "id": item.id,
        "session_id": item.session_id,
        "created_at": item.created_at,
        "question": item.question,
        "answer": item.answer,
        "sources": list(item.sources),
        "answer_mode": item.answer_mode,
        "model": item.model,
        "retrieval_seconds": item.retrieval_seconds,
        "generation_seconds": item.generation_seconds,
        "total_seconds": item.total_seconds,
    }


def event_to_dict(item: EventRecord) -> dict[str, object]:
    return {
        "id": item.id,
        "created_at": item.created_at,
        "event_type": item.event_type,
        "message": item.message,
        "details": item.details,
    }

def render_monitor_page(
    interactions: list[InteractionRecord],
    events: list[EventRecord],
    error_events: list[EventRecord],
    latency_stats: LatencyStats,
    source_usage: list[SourceUsageRecord],
    index_health: IndexHealth,
    model_status: ModelStatus,
    system_resources: SystemResources,
    interaction_count: int,
    event_count: int,
    config: AssistantConfig,
) -> str:
    interaction_rows = "".join(render_interaction_row(item) for item in interactions)
    event_rows = "".join(render_event_row(item) for item in events)
    error_rows = "".join(render_event_row(item) for item in error_events)
    source_rows = "".join(render_source_usage_row(item) for item in source_usage)
    if not interaction_rows:
        interaction_rows = '<tr><td colspan="7" class="muted">No chat interactions yet.</td></tr>'
    if not event_rows:
        event_rows = '<tr><td colspan="4" class="muted">No system events yet.</td></tr>'
    if not error_rows:
        error_rows = '<tr><td colspan="4" class="muted">No errors recorded.</td></tr>'
    if not source_rows:
        source_rows = '<tr><td colspan="2" class="muted">No source usage yet.</td></tr>'

    index_status = "Healthy"
    if index_health.unindexed_files or index_health.stale_files:
        index_status = "Needs ingest"
    model_badge = "Online" if model_status.reachable else "Check model"
    model_class = "status-ok" if model_status.reachable else "status-warn"
    index_class = "status-ok" if index_status == "Healthy" else "status-warn"
    temp_text = f"{system_resources.temperature_c:.1f} C" if system_resources.temperature_c is not None else "-"
    unanswered_count = sum(1 for item in interactions if "could not find" in item.answer.lower() or "not found" in item.answer.lower())
    slow_count = sum(1 for item in interactions if item.total_seconds >= 8.0)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adam · Live Monitor</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f4f6f8; color: #182230; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; background: #f4f6f8; color: #182230; }}
    a {{ color: inherit; }}
    .ops-shell {{ min-height: 100vh; display: grid; grid-template-columns: 248px minmax(0, 1fr); }}
    .ops-sidebar {{ position: sticky; top: 0; height: 100vh; background: #111827; color: #f9fafb; padding: 18px 14px; display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }}
    .brand {{ display: flex; gap: 10px; align-items: center; font-weight: 800; padding: 6px 4px; }}
    .brand-mark {{ width: 34px; height: 34px; border-radius: 9px; background: white; color: #111827; display: grid; place-items: center; font-weight: 900; font-size: 15px; }}
    .nav-stack {{ display: grid; gap: 6px; }}
    .nav-link {{ color: #d1d5db; text-decoration: none; padding: 10px 12px; border-radius: 8px; font-weight: 700; }}
    .nav-link:hover, .nav-link.active {{ background: #1f2937; color: white; }}
    .side-card {{ margin-top: auto; border: 1px solid #344054; border-radius: 10px; padding: 12px; color: #d0d5dd; font-size: 13px; line-height: 1.45; }}
    .live-sidebar {{ border: 1px solid #334155; border-radius: 10px; padding: 14px; background: #0d1b2a; }}
    .ls-label {{ color: #94a3b8; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 10px; }}
    .ls-orb-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }}
    .ls-orb {{ width: 18px; height: 18px; border-radius: 50%; background: #374151; transition: background .3s, box-shadow .3s; flex-shrink: 0; }}
    .ls-orb.idle {{ background: #374151; }}
    .ls-orb.waiting {{ background: #6366f1; box-shadow: 0 0 8px #6366f180; animation: orb-pulse 2s infinite; }}
    .ls-orb.listening {{ background: #2563eb; box-shadow: 0 0 14px #2563eb80; animation: orb-pulse 0.8s infinite; }}
    .ls-orb.processing, .ls-orb.searching {{ background: #d97706; box-shadow: 0 0 14px #d9770680; animation: orb-pulse 0.6s infinite; }}
    .ls-orb.speaking {{ background: #059669; box-shadow: 0 0 14px #05966980; animation: orb-pulse 0.5s infinite; }}
    .ls-orb.error {{ background: #dc2626; box-shadow: 0 0 14px #dc262680; }}
    @keyframes orb-pulse {{ 0%,100% {{ opacity:1; transform:scale(1); }} 50% {{ opacity:.65; transform:scale(1.18); }} }}
    .ls-state {{ color: #f9fafb; font-weight: 900; font-size: 13px; letter-spacing: .04em; }}
    .ls-you {{ font-size: 12px; color: #94a3b8; border-left: 2px solid #334155; padding-left: 8px; margin-bottom: 6px; line-height: 1.4; min-height: 16px; word-break: break-word; }}
    .ls-adam {{ font-size: 12px; color: #93c5fd; border-left: 2px solid #2563eb; padding-left: 8px; line-height: 1.4; min-height: 16px; word-break: break-word; max-height: 120px; overflow-y: auto; }}
    .ls-conn {{ font-size: 11px; margin-top: 10px; display: flex; align-items: center; gap: 6px; color: #6b7280; }}
    .conn-dot {{ width: 7px; height: 7px; border-radius: 50%; background: #374151; flex-shrink: 0; }}
    .conn-dot.live {{ background: #10b981; box-shadow: 0 0 5px #10b98160; }}
    .ops-main {{ min-width: 0; padding: 18px; }}
    .topbar {{ position: sticky; top: 0; z-index: 3; background: rgba(244,246,248,.94); backdrop-filter: blur(12px); border-bottom: 1px solid #e4e7ec; margin: -18px -18px 18px; padding: 16px 18px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    h1 {{ margin: 0; font-size: 22px; line-height: 1.2; }}
    h2 {{ margin: 0; font-size: 16px; }}
    .subtle {{ color: #667085; font-size: 13px; margin-top: 4px; }}
    .top-actions {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .button-link {{ display: inline-flex; align-items: center; justify-content: center; min-height: 34px; padding: 7px 10px; border: 1px solid #d0d5dd; background: white; color: #344054; border-radius: 8px; text-decoration: none; font-weight: 800; font-size: 13px; }}
    .button-link.primary {{ background: #111827; color: white; border-color: #111827; }}
    /* Live conversation card */
    .live-card {{ background: white; border-radius: 8px; padding: 20px 22px; margin-bottom: 18px; border: 1px solid #e4e7ec; }}
    .live-card-head {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; gap: 12px; flex-wrap: wrap; }}
    .live-title {{ display: flex; align-items: center; gap: 10px; }}
    .live-title-text {{ color: #111827; font-weight: 900; font-size: 17px; }}
    .live-subtitle {{ color: #667085; font-size: 13px; margin-top: 2px; }}
    .state-badge {{ display: inline-flex; align-items: center; gap: 7px; background: #f2f4f7; border: 1px solid #d0d5dd; border-radius: 999px; padding: 6px 12px; color: #344054; font-weight: 800; font-size: 13px; }}
    .state-dot {{ width: 10px; height: 10px; border-radius: 50%; background: #98a2b3; transition: background .3s, box-shadow .3s; }}
    .state-dot.idle {{ background: #98a2b3; }}
    .state-dot.waiting_wake_word {{ background: #6366f1; box-shadow: 0 0 6px #6366f180; }}
    .state-dot.listening {{ background: #2563eb; box-shadow: 0 0 8px #2563eb80; animation: orb-pulse 0.9s infinite; }}
    .state-dot.processing, .state-dot.searching_documents {{ background: #d97706; box-shadow: 0 0 8px #d9770680; animation: orb-pulse 0.6s infinite; }}
    .state-dot.answer_ready {{ background: #059669; box-shadow: 0 0 8px #05966980; }}
    .state-dot.speaking {{ background: #0ea5e9; box-shadow: 0 0 10px #0ea5e980; animation: orb-pulse 0.5s infinite; }}
    .state-dot.error {{ background: #dc2626; box-shadow: 0 0 8px #dc262680; }}
    .conn-badge {{ display: inline-flex; align-items: center; gap: 6px; background: #f2f4f7; border: 1px solid #d0d5dd; border-radius: 999px; padding: 5px 10px; font-size: 12px; color: #667085; }}
    .conn-badge .conn-dot {{ width: 8px; height: 8px; }}
    .live-convo {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .live-turn {{ background: #f8fafc; border: 1px solid #eef2f6; border-radius: 8px; padding: 14px; min-height: 80px; }}
    .live-turn-label {{ color: #667085; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px; }}
    .live-turn-text {{ color: #182230; font-size: 14px; line-height: 1.6; word-break: break-word; min-height: 20px; }}
    .live-turn-text.adam-text {{ color: #1d4ed8; }}
    .live-cursor {{ display: inline-block; width: 2px; height: 14px; background: #1d4ed8; border-radius: 1px; animation: blink 1s infinite; vertical-align: middle; margin-left: 1px; }}
    @keyframes blink {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0; }} }}
    .live-sources {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; min-height: 14px; }}
    .live-src-pill {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 999px; color: #1d4ed8; padding: 3px 9px; font-size: 11px; font-weight: 700; }}
    .live-filler {{ color: #b45309; font-size: 12px; margin-top: 8px; font-style: italic; min-height: 14px; }}
    /* Metrics grid */
    .metrics {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }}
    .metric {{ background: white; border: 1px solid #e4e7ec; border-radius: 8px; padding: 12px; min-height: 88px; }}
    .metric-label {{ color: #667085; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .03em; }}
    .metric-value {{ font-size: 21px; font-weight: 900; margin-top: 6px; overflow-wrap: anywhere; }}
    .metric-note {{ color: #667085; font-size: 12px; margin-top: 4px; line-height: 1.35; }}
    .dashboard-grid {{ display: grid; grid-template-columns: 1.1fr .9fr; gap: 14px; align-items: start; }}
    .panel {{ background: white; border: 1px solid #e4e7ec; border-radius: 8px; overflow: hidden; min-width: 0; }}
    .panel-head {{ min-height: 52px; padding: 12px 14px; border-bottom: 1px solid #eef2f6; display: flex; gap: 10px; align-items: center; justify-content: space-between; }}
    .panel-actions {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    .panel-body {{ padding: 14px; }}
    .panel.compact .panel-body {{ padding: 0; }}
    .table-scroll {{ max-height: 360px; overflow: auto; }}
    .table-scroll.tall {{ max-height: 520px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    thead th {{ position: sticky; top: 0; z-index: 1; background: #f8fafc; }}
    th, td {{ border-bottom: 1px solid #eef2f6; padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ color: #667085; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .02em; }}
    tr:hover td {{ background: #fafafa; }}
    .muted {{ color: #667085; }}
    .nowrap {{ white-space: nowrap; }}
    .answer-cell {{ max-width: 360px; line-height: 1.45; }}
    .question-cell {{ max-width: 260px; font-weight: 700; }}
    .detail-link {{ color: #175cd3; font-weight: 800; text-decoration: none; }}
    .badge {{ display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border-radius: 999px; background: #f2f4f7; color: #344054; font-weight: 800; font-size: 12px; }}
    .badge.good, .status-ok {{ color: #067647; background: #ecfdf3; }}
    .badge.warn, .status-warn {{ color: #b54708; background: #fffaeb; }}
    .badge.bad {{ color: #b42318; background: #fef3f2; }}
    .source-list {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .source-pill {{ display: inline-flex; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding: 4px 8px; border: 1px solid #d0d5dd; border-radius: 999px; color: #344054; background: white; font-size: 12px; }}
    .search-input {{ width: min(260px, 100%); min-height: 34px; border: 1px solid #d0d5dd; border-radius: 8px; padding: 7px 10px; font: inherit; font-size: 13px; }}
    .health-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .kv {{ border: 1px solid #eef2f6; border-radius: 8px; padding: 10px; }}
    .kv-label {{ color: #667085; font-size: 12px; font-weight: 800; }}
    .kv-value {{ margin-top: 4px; font-weight: 800; overflow-wrap: anywhere; }}
    .split-stack {{ display: grid; gap: 14px; }}
    .event-type {{ font-weight: 900; }}
    .details-cell {{ max-width: 320px; color: #667085; }}
    .empty-filter {{ display: none; padding: 14px; color: #667085; }}
    .live-log {{ max-height: 280px; overflow-y: auto; background: #0f172a; border-radius: 8px; padding: 10px 12px; margin-top: 12px; }}
    .log-entry {{ color: #9ca3af; font-size: 12px; font-family: ui-monospace, "Cascadia Mono", monospace; padding: 2px 0; border-bottom: 1px solid #1e293b; }}
    .log-entry:last-child {{ border-bottom: none; }}
    .log-ts {{ color: #4b5563; margin-right: 8px; }}
    .log-type {{ font-weight: 800; margin-right: 6px; }}
    .log-type.state {{ color: #6366f1; }} .log-type.wake {{ color: #f59e0b; }} .log-type.transcript {{ color: #34d399; }}
    .log-type.token {{ color: #60a5fa; }} .log-type.barge_in {{ color: #f87171; }} .log-type.filler {{ color: #fbbf24; }}
    /* ── Voice Session Log ── */
    .vl-table th, .vl-table td {{ padding:9px 10px; font-size:12px; border-bottom:1px solid #eef2f6; text-align:left; vertical-align:top; }}
    .vl-table thead th {{ background:#f8fafc; color:#667085; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:.03em; white-space:nowrap; position:sticky; top:0; z-index:1; }}
    .vl-table tr:hover td {{ background:#f8fafc; cursor:pointer; }}
    .vl-table tr.vl-expanded td {{ background:#eef2ff; }}
    .vl-detail-row td {{ background:#f0f4ff !important; padding:0; }}
    .vl-detail-inner {{ display:grid; grid-template-columns:minmax(0,1fr) 260px; gap:0; }}
    .vl-detail-left {{ padding:14px 16px; border-right:1px solid #e0e7ff; }}
    .vl-detail-right {{ padding:14px 16px; display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .vl-bubble-you {{ background:#1e2939; color:#f1f5f9; border-radius:10px; padding:8px 12px; font-size:13px; line-height:1.5; margin-bottom:8px; overflow-wrap:anywhere; }}
    .vl-bubble-adam {{ background:white; border:1px solid #e0e7ff; border-radius:10px; padding:8px 12px; font-size:13px; line-height:1.6; color:#182230; white-space:pre-wrap; overflow-wrap:anywhere; max-height:160px; overflow-y:auto; }}
    .vl-stat-box {{ background:white; border:1px solid #e0e7ff; border-radius:8px; padding:10px 12px; }}
    .vl-stat-title {{ font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.04em; color:#667085; margin-bottom:6px; }}
    .vl-stat-row {{ display:flex; justify-content:space-between; font-size:12px; margin-bottom:3px; }}
    .vl-stat-key {{ color:#667085; }}
    .vl-stat-val {{ font-weight:700; color:#111827; }}
    .vl-stat-val.fast {{ color:#067647; }} .vl-stat-val.slow {{ color:#b42318; }}
    .vl-src-pill {{ display:inline-flex; background:#eff6ff; border:1px solid #bfdbfe; border-radius:999px; color:#1d4ed8; padding:2px 8px; font-size:11px; font-weight:700; margin:2px 3px 0 0; }}
    .vl-badge-ok {{ background:#ecfdf3; color:#067647; border:1px solid #6ce9a6; border-radius:999px; padding:1px 7px; font-size:11px; font-weight:700; }}
    .vl-badge-warn {{ background:#fef3f2; color:#b42318; border:1px solid #fecdca; border-radius:999px; padding:1px 7px; font-size:11px; font-weight:700; }}
    .vl-badge-gray {{ background:#f2f4f7; color:#344054; border:1px solid #d0d5dd; border-radius:999px; padding:1px 7px; font-size:11px; font-weight:700; }}
    .vl-badge-blue {{ background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; border-radius:999px; padding:1px 7px; font-size:11px; font-weight:700; }}
    .log-type.error {{ color: #f87171; }} .log-type.sources {{ color: #a78bfa; }} .log-type.latency {{ color: #6b7280; }}
    /* Raw log toggle section */
    .vpl-log-toggle-head {{ padding:7px 14px; background:#f0f4ff; color:#6366f1; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:.04em; cursor:pointer; display:flex; align-items:center; justify-content:space-between; border-top:1px solid #e0e7ff; }}
    .vpl-log-toggle-head:hover {{ background:#e0e7ff; }}
    .vpl-log-raw {{ background:#0f172a; padding:6px 10px; }}
    /* VPL log rows (micro-log style) */
    .vpl-card {{ display:flex; align-items:baseline; gap:0; padding:3px 0; border-bottom:1px solid #1e293b; font-family:ui-monospace,"Cascadia Mono",monospace; }}
    .vpl-card:last-child {{ border-bottom:none; }}
    .vpl-row-ts {{ color:#6b7280; font-size:12px; margin-right:8px; white-space:nowrap; flex-shrink:0; }}
    .vpl-row-type {{ font-size:12px; font-weight:800; margin-right:8px; white-space:nowrap; flex-shrink:0; min-width:100px; }}
    .vpl-row-msg {{ font-size:12px; color:#9ca3af; word-break:break-word; flex:1; min-width:0; }}
    .vpl-expand {{ font-size:11px; color:#60a5fa; background:none; border:none; cursor:pointer; padding:0 4px; font-weight:700; font-family:inherit; flex-shrink:0; }}
    .vpl-pending {{ background:#eef2ff !important; border-left-color:#6366f1 !important; }}
    /* VPL live dot */
    .vpl-live-dot {{ width:8px; height:8px; border-radius:50%; background:#d0d5dd; display:inline-block; vertical-align:middle; margin-left:4px; transition:background .1s; }}
    .vpl-live-dot.on {{ background:#10b981; animation:vpl-flash .8s ease-out; }}
    @keyframes vpl-flash {{ 0% {{ box-shadow:0 0 0 0 #10b98160; }} 100% {{ box-shadow:0 0 0 8px transparent; }} }}
    /* Voice chat history bubbles */
    .vc-turn {{ display: flex; flex-direction: column; gap: 6px; }}
    .vc-you {{ align-self: flex-end; max-width: 72%; background: #111827; color: #fff; border-radius: 14px 14px 4px 14px; padding: 10px 13px; font-size: 13px; line-height: 1.5; word-break: break-word; }}
    .vc-adam {{ align-self: flex-start; max-width: 80%; background: white; border: 1px solid #e4e7ec; color: #182230; border-radius: 14px 14px 14px 4px; padding: 10px 13px; font-size: 13px; line-height: 1.6; word-break: break-word; white-space: pre-wrap; }}
    .vc-adam.pending {{ color: #98a2b3; font-style: italic; }}
    .vc-meta {{ font-size: 11px; color: #98a2b3; text-align: right; }}
    @media (max-width: 1180px) {{ .metrics {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} .dashboard-grid {{ grid-template-columns: 1fr; }} .live-convo {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 820px) {{ .ops-shell {{ display: block; }} .ops-sidebar {{ position: static; height: auto; }} .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .topbar {{ position: static; margin: -18px -18px 14px; align-items: flex-start; flex-direction: column; }} }}
    @media (max-width: 560px) {{ .metrics, .health-grid {{ grid-template-columns: 1fr; }} .ops-main {{ padding: 12px; }} }}
    /* ── Conversation session cards ── */
    /* ── Conversation session cards (two-column) ── */
    .sess-card {{ display:grid; grid-template-columns:minmax(0,1fr) 110px; gap:10px; border:1px solid #e4e7ec; border-radius:10px; padding:14px 16px; cursor:pointer; transition:background .15s,border-color .15s; align-items:start; }}
    .sess-card:hover {{ background:#f0f4ff; border-color:#bfdbfe; }}
    .sess-card-title {{ font-weight:800; font-size:14px; color:#182230; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-bottom:5px; }}
    .sess-card-meta {{ color:#667085; font-size:12px; display:flex; align-items:center; gap:7px; flex-wrap:wrap; }}
    .sess-card-right {{ display:flex; flex-direction:column; align-items:flex-end; gap:4px; }}
    .sess-card-count {{ font-size:13px; font-weight:800; color:#344054; }}
    .sess-card-date {{ font-size:11px; color:#98a2b3; white-space:nowrap; text-align:right; }}
    .sess-card-arrow {{ color:#c7d2fe; font-size:20px; line-height:1; margin-top:2px; }}
    .sess-tag {{ display:inline-flex; align-items:center; background:#eff6ff; color:#1d4ed8; border-radius:999px; padding:1px 7px; font-size:11px; font-weight:700; }}
    .sess-tag.voice {{ background:#fef9c3; color:#854d0e; }}
    .sess-tag.default {{ background:#f0fdf4; color:#166534; }}
    /* ── Session detail overlay ── */
    .sess-overlay {{ display:none; position:fixed; inset:0; z-index:70; background:rgba(0,0,0,.5); }}
    .sess-overlay.open {{ display:block; }}
    .sess-detail-panel {{ position:absolute; right:0; top:0; bottom:0; width:min(800px,98vw); background:#f8fafc; display:flex; flex-direction:column; box-shadow:-8px 0 40px rgba(0,0,0,.2); overflow:hidden; }}
    .sess-panel-head {{ background:white; padding:20px 22px 16px; border-bottom:1px solid #e4e7ec; display:flex; align-items:flex-start; justify-content:space-between; gap:12px; flex-shrink:0; }}
    .sess-panel-title {{ font-weight:900; font-size:17px; color:#111827; overflow-wrap:anywhere; line-height:1.3; }}
    .sess-panel-meta {{ color:#667085; font-size:13px; margin-top:5px; }}
    .sess-close-btn {{ border:0; background:transparent; font-size:24px; cursor:pointer; color:#6b7280; line-height:1; padding:2px 6px; flex-shrink:0; }}
    .sess-panel-body {{ flex:1; min-height:0; overflow-y:auto; padding:18px 20px; display:flex; flex-direction:column; gap:14px; }}
    /* ── Turn detail cards (two-column) ── */
    .td-wrap {{ display:grid; grid-template-columns:minmax(0,1fr) 180px; border:1px solid #e4e7ec; border-radius:10px; overflow:hidden; background:white; }}
    .td-left {{ display:flex; flex-direction:column; min-width:0; }}
    .td-you {{ background:#1e2939; padding:12px 16px; }}
    .td-you-label {{ font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.04em; color:#94a3b8; margin-bottom:5px; }}
    .td-you-q {{ color:#f1f5f9; font-size:14px; line-height:1.55; font-weight:600; overflow-wrap:anywhere; }}
    .td-adam {{ padding:14px 16px; flex:1; }}
    .td-adam-label {{ font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.04em; color:#1d4ed8; margin-bottom:8px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
    .td-adam-a {{ color:#182230; font-size:14px; line-height:1.65; white-space:pre-wrap; overflow-wrap:anywhere; }}
    .td-src-row {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }}
    .td-src-pill {{ display:inline-flex; align-items:center; background:#eff6ff; border:1px solid #bfdbfe; border-radius:999px; color:#1d4ed8; padding:4px 10px; font-size:12px; font-weight:700; }}
    /* ── Right stats column ── */
    .td-right {{ background:#f8fafc; border-left:1px solid #e4e7ec; padding:14px 12px; display:flex; flex-direction:column; gap:10px; }}
    .td-stat-label {{ font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.04em; color:#667085; margin-bottom:6px; }}
    .td-latency {{ display:flex; flex-direction:column; gap:5px; }}
    .td-lat-row {{ display:flex; justify-content:space-between; align-items:center; font-size:12px; gap:6px; }}
    .td-lat-key {{ color:#667085; white-space:nowrap; }}
    .td-lat-val {{ font-weight:800; color:#111827; }}
    .td-lat-val.fast {{ color:#067647; }} .td-lat-val.slow {{ color:#b42318; }}
    .td-divider {{ height:1px; background:#e4e7ec; }}
    .td-meta-stack {{ display:flex; flex-direction:column; gap:5px; }}
    .td-meta-item {{ font-size:11px; color:#667085; display:flex; align-items:center; gap:5px; }}
    .td-num {{ font-size:11px; color:#94a3b8; font-weight:700; }}
  </style>
</head>
<body>
  <div class="ops-shell">
    <aside class="ops-sidebar">
      <div class="brand"><div class="brand-mark">A</div><span>Adam · Receptionist</span></div>
      <nav class="nav-stack" aria-label="Dashboard navigation">
        <a class="nav-link" href="/">Knowledge Files</a>
        <a class="nav-link" href="/ask">Ask Documents</a>
        <a class="nav-link" href="/voice">Voice</a>
        <a class="nav-link active" href="/monitor">Monitor</a>
      </nav>
      <!-- Mini live view in sidebar -->
      <div class="live-sidebar">
        <div class="ls-label">Live Voice</div>
        <div class="ls-orb-row">
          <div class="ls-orb idle" id="sb-orb"></div>
          <span class="ls-state" id="sb-state">IDLE</span>
        </div>
        <div class="ls-label" style="margin-bottom:4px">You said</div>
        <div class="ls-you" id="sb-you">—</div>
        <div class="ls-label" style="margin-top:8px;margin-bottom:4px">Adam replied</div>
        <div class="ls-adam" id="sb-adam">—</div>
        <div class="ls-conn"><span class="conn-dot" id="sb-dot"></span><span id="sb-conn-text">Connecting...</span></div>
      </div>
      <div class="side-card">
        <strong>{html.escape(config.ollama_model)}</strong><br>
        Mode: {html.escape(config.dashboard_answer_mode)}<br><br>
        Private local RAG on this device.
      </div>
    </aside>
    <main class="ops-main">
      <header class="topbar">
        <div><h1>Live Monitor</h1><div class="subtle">Real-time view of voice pipeline · interaction history · system health</div></div>
        <div class="top-actions">
          <a class="button-link primary" href="/ask">Open Chat</a>
          <a class="button-link" href="/monitor/export/interactions.csv">Chat CSV</a>
          <a class="button-link" href="/monitor/export/events.csv">Events CSV</a>
        </div>
      </header>

      <!-- ===== METRICS ===== -->
      <section class="metrics" style="margin-bottom:18px;" aria-label="System summary">
        <div class="metric"><div class="metric-label">Index</div><div class="metric-value"><span class="badge {'good' if index_status == 'Healthy' else 'warn'}">{html.escape(index_status)}</span></div><div class="metric-note">{index_health.indexed_document_count}/{index_health.uploaded_file_count} files · {index_health.chunk_count} chunks</div></div>
        <div class="metric"><div class="metric-label">Chat Records</div><div class="metric-value">{interaction_count}</div><div class="metric-note">Last answer {latency_stats.last_total:.2f}s</div></div>
        <div class="metric"><div class="metric-label">Average Latency</div><div class="metric-value">{latency_stats.average_total:.2f}s</div><div class="metric-note">retrieval {latency_stats.average_retrieval:.2f}s · generation {latency_stats.average_generation:.2f}s</div></div>
        <div class="metric"><div class="metric-label">Model</div><div class="metric-value"><span class="badge {'good' if model_status.reachable else 'warn'}">{html.escape(model_badge)}</span></div><div class="metric-note">{html.escape(config.ollama_model)}</div></div>
        <div class="metric"><div class="metric-label">Needs Review</div><div class="metric-value">{unanswered_count}</div><div class="metric-note">answers marked not found / unclear</div></div>
        <div class="metric"><div class="metric-label">Slow Turns</div><div class="metric-value">{slow_count}</div><div class="metric-note">recent answers over 8 seconds</div></div>
      </section>

      <!-- ===== INDEX AND RUNTIME / MODEL AND SOURCES ===== -->
      <section class="dashboard-grid" style="margin-bottom:18px;">
        <div class="panel">
          <div class="panel-head"><div><h2>Index And Runtime</h2><div class="subtle">Knowledge readiness and Jetson resource snapshot.</div></div></div>
          <div class="panel-body health-grid">
            <div class="kv"><div class="kv-label">Uploaded files</div><div class="kv-value">{index_health.uploaded_file_count}</div></div>
            <div class="kv"><div class="kv-label">Indexed documents</div><div class="kv-value">{index_health.indexed_document_count}</div></div>
            <div class="kv"><div class="kv-label">Chunks</div><div class="kv-value">{index_health.chunk_count}</div></div>
            <div class="kv"><div class="kv-label">Last ingest</div><div class="kv-value">{html.escape(index_health.last_indexed_at)}</div></div>
            <div class="kv"><div class="kv-label">Need ingest</div><div class="kv-value">{render_inline_list(index_health.unindexed_files)}</div></div>
            <div class="kv"><div class="kv-label">Stale files</div><div class="kv-value">{render_inline_list(index_health.stale_files)}</div></div>
            <div class="kv"><div class="kv-label">CPU load 1m</div><div class="kv-value">{system_resources.load_1m:.2f}</div></div>
            <div class="kv"><div class="kv-label">Memory</div><div class="kv-value">{system_resources.memory_used_mb}/{system_resources.memory_total_mb} MB ({system_resources.memory_percent:.1f}%)</div></div>
            <div class="kv"><div class="kv-label">Disk</div><div class="kv-value">{system_resources.disk_used}/{system_resources.disk_total} ({system_resources.disk_percent:.1f}%)</div></div>
            <div class="kv"><div class="kv-label">Temperature</div><div class="kv-value">{html.escape(temp_text)}</div></div>
          </div>
        </div>

        <div class="panel compact">
          <div class="panel-head"><div><h2>Model And Sources</h2><div class="subtle">Current Ollama endpoint and most-used documents.</div></div></div>
          <div class="panel-body health-grid">
            <div class="kv"><div class="kv-label">Ollama</div><div class="kv-value"><span class="badge {'good' if model_status.reachable else 'warn'}">{html.escape(model_badge)}</span></div></div>
            <div class="kv"><div class="kv-label">Answer mode</div><div class="kv-value">{html.escape(model_status.answer_mode)}</div></div>
            <div class="kv"><div class="kv-label">Model</div><div class="kv-value">{html.escape(model_status.model)}</div></div>
            <div class="kv"><div class="kv-label">Endpoint</div><div class="kv-value">{html.escape(model_status.url)}</div></div>
            <div class="kv" style="grid-column:1/-1;"><div class="kv-label">Status message</div><div class="kv-value">{html.escape(model_status.message)}</div></div>
          </div>
          <div class="table-scroll">
            <table>
              <thead><tr><th>Source</th><th>Answers</th></tr></thead>
              <tbody>{source_rows}</tbody>
            </table>
          </div>
        </div>
      </section>

      <!-- ===== LIVE VOICE PANEL ===== -->
      <div class="live-card" aria-label="Live voice pipeline">
        <div class="live-card-head">
          <div class="live-title">
            <div>
              <div class="live-title-text">🤖 Adam · Virtual Receptionist</div>
              <div class="live-subtitle">Live voice pipeline — say "Adam" to wake</div>
            </div>
          </div>
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            <div class="state-badge"><div class="state-dot idle" id="state-dot"></div><span id="state-text">IDLE</span></div>
            <div class="conn-badge"><span class="conn-dot" id="conn-dot"></span><span id="conn-text">Connecting…</span></div>
          </div>
        </div>
        <div class="live-convo">
          <div class="live-turn">
            <div class="live-turn-label">You said</div>
            <div class="live-turn-text" id="lv-you">Waiting for wake word "Adam"…</div>
          </div>
          <div class="live-turn">
            <div class="live-turn-label">Adam replied</div>
            <div class="live-turn-text adam-text" id="lv-adam">—</div>
            <div class="live-sources" id="lv-sources"></div>
            <div class="live-filler" id="lv-filler"></div>
          </div>
        </div>
        <!-- Live event micro-log -->
        <div class="live-log" id="live-log" aria-label="Live event log"></div>
      </div>

      <!-- ===== VOICE CHAT HISTORY ===== -->
      <section class="panel" style="margin-bottom:18px;" aria-label="Voice chat history">
        <div class="panel-head">
          <div><h2>Voice Chat History</h2><div class="subtle">Every voice conversation with Adam — newest at the bottom</div></div>
          <button class="button-link" id="clear-voice-history" style="font-size:12px;min-height:28px;padding:4px 10px;">Clear</button>
        </div>
        <div id="voice-history" style="max-height:420px;overflow-y:auto;padding:14px 16px;display:flex;flex-direction:column;gap:14px;background:#f8fafc;">
          <div id="voice-history-empty" style="color:#667085;font-size:13px;text-align:center;padding:24px 0;">No voice conversations yet — say "Adam" to start</div>
        </div>
      </section>

      <!-- ===== VOICE SESSION LOG ===== -->
      <section class="panel" style="margin-bottom:18px;" aria-label="Voice session log">
        <div class="panel-head">
          <div><h2>Voice Session Log</h2><div class="subtle">Full detail per wake-word cycle — latency breakdown, system health, knowledge source</div></div>
          <div class="panel-actions">
            <button class="button-link" id="vl-refresh-btn" style="font-size:12px;min-height:28px;padding:4px 10px;">↻ Refresh</button>
          </div>
        </div>
        <div class="table-scroll tall">
          <table class="vl-table" id="vl-table" style="width:100%;border-collapse:collapse;">
            <thead>
              <tr>
                <th>Time</th>
                <th>Transcript</th>
                <th>Source</th>
                <th title="Speech-to-text time">STT</th>
                <th title="RAG retrieval time">Retrieval</th>
                <th title="LLM generation time">Generation</th>
                <th title="Total end-to-end time">Total</th>
                <th title="Filler phrases played">Fillers</th>
                <th title="User interrupted">Barge-in</th>
                <th title="CPU 1-min load">CPU</th>
                <th title="Jetson temperature">°C</th>
                <th title="RAM usage">RAM%</th>
                <th>Words</th>
              </tr>
            </thead>
            <tbody id="vl-tbody">
              <tr><td colspan="13" style="text-align:center;color:#667085;padding:28px;">Loading voice session log…</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <!-- ===== CONVERSATIONS ===== -->
      <section class="panel" style="margin-bottom:18px;" aria-label="Conversation history">
        <div class="panel-head">
          <div><h2>Conversations</h2><div class="subtle">Every session — click to view full Q&amp;A with sources and latency</div></div>
          <div class="panel-actions"><input id="sess-search" class="search-input" placeholder="Search conversations…"></div>
        </div>
        <div id="sess-list" style="max-height:520px;overflow-y:auto;padding:10px 14px;display:flex;flex-direction:column;gap:8px;">
          <div id="sess-empty" style="padding:28px 0;text-align:center;color:#667085;font-size:13px;">Loading conversations…</div>
        </div>
      </section>

      <!-- System events + error log -->
      <section style="display:grid;grid-template-columns:1.1fr .9fr;gap:14px;margin-bottom:18px;align-items:start;">
        <div class="panel compact">
          <div class="panel-head">
            <div><h2>System Events</h2><div class="subtle">Uploads, ingests, deletes, and errors.</div></div>
            <div class="panel-actions"><input id="event-filter" class="search-input" placeholder="Search events..."></div>
          </div>
          <div class="table-scroll">
            <table id="event-table">
              <thead><tr><th>Time</th><th>Type</th><th>Message</th><th>Details</th></tr></thead>
              <tbody>{event_rows}</tbody>
            </table>
            <div class="empty-filter" id="event-empty">No event rows match this search.</div>
          </div>
        </div>

        <div class="panel compact">
          <div class="panel-head"><div><h2>Error Log</h2><div class="subtle">Recent errors only.</div></div><span class="badge {'bad' if error_events else 'good'}">{len(error_events)} errors</span></div>
          <div class="table-scroll">
            <table>
              <thead><tr><th>Time</th><th>Type</th><th>Message</th><th>Details</th></tr></thead>
              <tbody>{error_rows}</tbody>
            </table>
          </div>
        </div>
      </section>

    </main>
  </div>

  <!-- ===== SESSION DETAIL OVERLAY ===== -->
  <div id="sess-overlay" class="sess-overlay" role="dialog" aria-modal="true" aria-label="Conversation detail">
    <div class="sess-detail-panel">
      <div class="sess-panel-head">
        <div style="flex:1;min-width:0;">
          <div class="sess-panel-title" id="sess-panel-title">Session</div>
          <div class="sess-panel-meta" id="sess-panel-meta"></div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-shrink:0;">
          <button class="button-link" id="sess-export-btn" style="font-size:12px;min-height:28px;padding:4px 10px;">⬇ Export</button>
          <button class="sess-close-btn" id="sess-close-btn" aria-label="Close">✕</button>
        </div>
      </div>
      <div class="sess-panel-body" id="sess-panel-body"></div>
    </div>
  </div>

  <script>
    function attachFilter(inputId, tableId, emptyId) {{
      const input = document.getElementById(inputId);
      const table = document.getElementById(tableId);
      const empty = document.getElementById(emptyId);
      if (!input || !table || !empty) return;
      input.addEventListener('input', () => {{
        const needle = input.value.trim().toLowerCase();
        let visible = 0;
        table.querySelectorAll('tbody tr').forEach(row => {{
          const haystack = (row.dataset.search || row.textContent || '').toLowerCase();
          const show = !needle || haystack.includes(needle);
          row.hidden = !show;
          if (show) visible += 1;
        }});
        empty.style.display = visible ? 'none' : 'block';
      }});
    }}
    attachFilter('event-filter', 'event-table', 'event-empty');

    // ===== LIVE SSE PANEL =====
    (function setupLivePanel() {{
      const stateDot   = document.getElementById("state-dot");
      const stateText  = document.getElementById("state-text");
      const connDot    = document.getElementById("conn-dot");
      const connText   = document.getElementById("conn-text");
      const lvYou      = document.getElementById("lv-you");
      const lvAdam     = document.getElementById("lv-adam");
      const lvSources  = document.getElementById("lv-sources");
      const lvFiller   = document.getElementById("lv-filler");
      const liveLog    = document.getElementById("live-log");
      // sidebar elements
      const sbOrb      = document.getElementById("sb-orb");
      const sbState    = document.getElementById("sb-state");
      const sbYou      = document.getElementById("sb-you");
      const sbAdam     = document.getElementById("sb-adam");
      const sbDot      = document.getElementById("sb-dot");
      const sbConnText = document.getElementById("sb-conn-text");

      let adamBuffer = "";
      let cursorEl = null;

      // ── Voice history ──────────────────────────────────────────────
      const voiceHistoryEl = document.getElementById("voice-history");
      const voiceHistoryEmpty = document.getElementById("voice-history-empty");
      let pendingTurnEl = null;  // the .vc-turn div waiting for Adam's answer

      function escVH(s) {{ return String(s||"").replace(/[&<>"']/g, c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}}[c])); }}

      function renderVoiceTurn(question, answer, ts) {{
        const div = document.createElement("div");
        div.className = "vc-turn";
        const time = ts ? new Date(ts).toLocaleTimeString("en-GB",{{hour12:false,hour:"2-digit",minute:"2-digit"}}) : new Date().toLocaleTimeString("en-GB",{{hour12:false,hour:"2-digit",minute:"2-digit"}});
        div.innerHTML = `<div class="vc-meta">${{escVH(time)}}</div>
          <div class="vc-you">${{escVH(question)}}</div>
          <div class="vc-adam ${{answer ? "" : "pending"}}">${{escVH(answer || "Adam is thinking…")}}</div>`;
        return div;
      }}

      function addVoiceTurn(question) {{
        if (voiceHistoryEmpty) voiceHistoryEmpty.style.display = "none";
        pendingTurnEl = renderVoiceTurn(question, "");
        voiceHistoryEl.appendChild(pendingTurnEl);
        voiceHistoryEl.scrollTop = voiceHistoryEl.scrollHeight;
      }}

      function completeVoiceTurn(answerText) {{
        if (!pendingTurnEl) return;
        const adamDiv = pendingTurnEl.querySelector(".vc-adam");
        if (adamDiv) {{
          adamDiv.textContent = answerText || "(no answer)";
          adamDiv.classList.remove("pending");
        }}
        pendingTurnEl = null;
        voiceHistoryEl.scrollTop = voiceHistoryEl.scrollHeight;
      }}

      // Load past turns on page open
      fetch("/monitor/voice-history")
        .then(r => r.json())
        .then(items => {{
          if (!items.length) return;
          if (voiceHistoryEmpty) voiceHistoryEmpty.style.display = "none";
          items.forEach(item => voiceHistoryEl.appendChild(renderVoiceTurn(item.question, item.answer, item.created_at)));
          voiceHistoryEl.scrollTop = voiceHistoryEl.scrollHeight;
        }}).catch(() => {{}});

      document.getElementById("clear-voice-history")?.addEventListener("click", () => {{
        voiceHistoryEl.querySelectorAll(".vc-turn").forEach(el => el.remove());
        if (voiceHistoryEmpty) voiceHistoryEmpty.style.display = "";
        pendingTurnEl = null;
      }});
      // ─────────────────────────────────────────────────────────────

      const STATE_LABELS = {{
        idle: "IDLE",
        waiting_wake_word: "WAITING FOR 'ADAM'",
        listening: "LISTENING",
        processing: "THINKING",
        searching_documents: "SEARCHING DOCS",
        answer_ready: "ANSWER READY",
        speaking: "SPEAKING",
        error: "ERROR",
      }};

      function setConnected(ok) {{
        connDot.className = "conn-dot" + (ok ? " live" : "");
        sbDot.className   = "conn-dot" + (ok ? " live" : "");
        connText.textContent = ok ? "Connected · live" : "Reconnecting…";
        sbConnText.textContent = ok ? "Live" : "Reconnecting…";
      }}

      function applyState(state) {{
        const label = STATE_LABELS[state] || state.toUpperCase().replace(/_/g," ");
        // main panel
        stateDot.className = "state-dot " + state;
        stateText.textContent = label;
        // sidebar orb
        const orbClass = state === "waiting_wake_word" ? "waiting"
                       : state === "searching_documents" ? "searching"
                       : state === "answer_ready" ? "speaking"
                       : state;
        sbOrb.className = "ls-orb " + orbClass;
        sbState.textContent = label;
      }}

      function addCursor() {{
        if (cursorEl) return;
        cursorEl = document.createElement("span");
        cursorEl.className = "live-cursor";
        lvAdam.appendChild(cursorEl);
      }}

      function removeCursor() {{
        if (cursorEl) {{ cursorEl.remove(); cursorEl = null; }}
      }}

      function appendToken(text) {{
        removeCursor();
        adamBuffer += text;
        // display up to last 600 chars so the card doesn't overflow
        const display = adamBuffer.length > 600 ? "…" + adamBuffer.slice(-580) : adamBuffer;
        lvAdam.textContent = display;
        sbAdam.textContent = display.length > 120 ? display.slice(-120) : display;
        addCursor();
      }}

      function logEvent(type, summary) {{
        if (!liveLog) return;
        const now = new Date().toLocaleTimeString("en-GB", {{hour12:false}});
        const entry = document.createElement("div");
        entry.className = "log-entry";
        entry.innerHTML = `<span class="log-ts">${{now}}</span><span class="log-type ${{type}}">${{type}}</span>${{summary}}`;
        liveLog.insertBefore(entry, liveLog.firstChild);
        while (liveLog.children.length > 200) liveLog.removeChild(liveLog.lastChild);
      }}

      function handleEvent(ev) {{
        switch (ev.type) {{
          case "state":
            applyState(ev.state);
            if (ev.state === "listening") {{
              lvYou.textContent = "…";
              sbYou.textContent = "…";
              adamBuffer = "";
              lvAdam.textContent = "—";
              sbAdam.textContent = "—";
              lvSources.innerHTML = "";
              lvFiller.textContent = "";
              removeCursor();
            }}
            if (ev.state === "idle" || ev.state === "waiting_wake_word") {{
              removeCursor();
              lvFiller.textContent = "";
            }}
            logEvent("state", ev.state);
            break;

          case "wake":
            applyState("listening");
            lvYou.textContent = `Wake word detected`;
            sbYou.textContent = `Wake word`;
            adamBuffer = "";
            lvAdam.textContent = "—";
            lvSources.innerHTML = "";
            lvFiller.textContent = "";
            removeCursor();
            logEvent("wake", `'${{ev.keyword || "adam"}}'`);
            break;

          case "wake_response":
            adamBuffer = ev.text || "";
            lvAdam.textContent = adamBuffer;
            sbAdam.textContent = adamBuffer;
            logEvent("wake", `Adam: "${{ev.text}}"`);
            break;

          case "transcript":
            lvYou.textContent = ev.text || "(empty)";
            sbYou.textContent = (ev.text || "(empty)").slice(0, 80);
            adamBuffer = "";
            lvAdam.textContent = "—";
            lvSources.innerHTML = "";
            lvFiller.textContent = "";
            removeCursor();
            logEvent("transcript", ev.text || "");
            if (ev.text && ev.text.trim()) addVoiceTurn(ev.text.trim());
            break;

          case "token":
            appendToken(ev.text || "");
            break;

          case "sources":
            lvSources.innerHTML = (ev.sources || []).map(s =>
              `<span class="live-src-pill">${{s}}</span>`).join("");
            logEvent("sources", (ev.sources || []).join(", "));
            break;

          case "filler":
            lvFiller.textContent = ev.text || "";
            logEvent("filler", ev.text || "");
            break;

          case "barge_in":
            removeCursor();
            lvFiller.textContent = "";
            const bargeNote = document.createElement("span");
            bargeNote.style.cssText = "color:#f87171;font-size:12px;margin-left:6px";
            bargeNote.textContent = " [interrupted]";
            lvAdam.appendChild(bargeNote);
            logEvent("barge_in", "user interrupted");
            break;

          case "answer_complete":
            removeCursor();
            lvFiller.textContent = "";
            logEvent("state", "answer_complete");
            completeVoiceTurn(ev.text || adamBuffer);
            break;

          case "latency":
            logEvent("latency", `${{ev.seconds}}s`);
            break;

          case "error":
            applyState("error");
            lvAdam.textContent = ev.message || "Error";
            sbAdam.textContent = (ev.message || "Error").slice(0, 80);
            removeCursor();
            logEvent("error", ev.message || "");
            break;
        }}
      }}

      let es = null;
      let retryDelay = 2000;

      function connect() {{
        if (es) {{ es.close(); es = null; }}
        es = new EventSource("/monitor/live");
        es.onopen = () => {{ setConnected(true); retryDelay = 2000; }};
        es.onerror = () => {{
          setConnected(false);
          es.close(); es = null;
          setTimeout(connect, retryDelay);
          retryDelay = Math.min(retryDelay * 1.5, 15000);
        }};
        es.onmessage = (e) => {{
          if (!e.data || e.data.startsWith(":")) return;
          try {{ handleEvent(JSON.parse(e.data)); }} catch(_) {{}}
        }};
      }}

      connect();
    }})();

    // ===== TABLE FILTERS =====
    attachFilter('event-filter', 'event-table', 'event-empty');

    // ===== CONVERSATIONS =====
    (function setupConversations() {{
      const listEl    = document.getElementById("sess-list");
      const emptyEl   = document.getElementById("sess-empty");
      const searchEl  = document.getElementById("sess-search");
      const overlay   = document.getElementById("sess-overlay");
      const panelTitle = document.getElementById("sess-panel-title");
      const panelMeta  = document.getElementById("sess-panel-meta");
      const panelBody  = document.getElementById("sess-panel-body");
      const closeBtn   = document.getElementById("sess-close-btn");
      const exportBtn  = document.getElementById("sess-export-btn");

      let allSessions  = [];
      let currentSession = null;
      let currentTurns   = [];

      function escH(s) {{
        return String(s || "").replace(/[&<>"']/g, c => ({{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"}}[c]));
      }}

      function sessionTag(sid) {{
        if (sid && sid.startsWith("voice")) return `<span class="sess-tag voice">🎤 Voice</span>`;
        if (sid === "default")              return `<span class="sess-tag default">💬 Default</span>`;
        return `<span class="sess-tag">💬 Chat</span>`;
      }}

      function renderList(sessions) {{
        const q = searchEl ? searchEl.value.trim().toLowerCase() : "";
        const filtered = q ? sessions.filter(s =>
          (s.title || "").toLowerCase().includes(q) || (s.session_id || "").toLowerCase().includes(q)
        ) : sessions;
        if (!filtered.length) {{
          listEl.innerHTML = `<div style="padding:28px 0;text-align:center;color:#667085;font-size:13px;">${{q ? "No matching conversations." : "No conversations yet — start a chat or say 'Adam' to wake the voice assistant."}}</div>`;
          return;
        }}
        listEl.innerHTML = filtered.map(s => {{
          const mc      = Number(s.message_count || 0);
          const dateStr = (s.updated_at || s.created_at || "").slice(0, 16).replace("T", " ");
          return '<div class="sess-card" data-id="' + escH(s.session_id) + '">' +
            '<div>' +
              '<div class="sess-card-title">' + escH(s.title || "(untitled)") + '</div>' +
              '<div class="sess-card-meta">' +
                sessionTag(s.session_id) +
                '<span>' + escH(dateStr) + '</span>' +
              '</div>' +
            '</div>' +
            '<div class="sess-card-right">' +
              '<span class="sess-card-count">' + escH(mc + " turn" + (mc === 1 ? "" : "s")) + '</span>' +
              '<span class="sess-card-arrow">›</span>' +
            '</div>' +
          '</div>';
        }}).join("");
        listEl.querySelectorAll(".sess-card").forEach(card => {{
          card.addEventListener("click", () => openSession(card.dataset.id));
        }});
      }}

      function latClass(v) {{ return v < 3 ? "fast" : v > 8 ? "slow" : ""; }}

      function renderTurn(turn, index) {{
        const srcs = (turn.sources || []).map(function(s) {{ return '<span class="td-src-pill">' + escH(s) + '</span>'; }}).join("");
        const srcRow = srcs ? '<div class="td-src-row">' + srcs + '</div>' : "";
        const hasDocs = (turn.sources || []).length > 0;
        const confBadge = hasDocs
          ? '<span class="confidence-doc">📄 From documents</span>'
          : '<span class="confidence-gen">💡 General</span>';
        const ret = Number(turn.retrieval_seconds || 0);
        const gen = Number(turn.generation_seconds || 0);
        const tot = Number(turn.total_seconds || 0);
        return (
          '<div class="td-wrap">' +
            '<div class="td-left">' +
              '<div class="td-you">' +
                '<div class="td-you-label"><span class="td-num">#' + (index + 1) + '</span> · You asked · ' + escH(turn.created_at || "") + '</div>' +
                '<div class="td-you-q">' + escH(turn.question || "") + '</div>' +
              '</div>' +
              '<div class="td-adam">' +
                '<div class="td-adam-label">🤖 Adam replied ' + confBadge + '</div>' +
                '<div class="td-adam-a">' + escH(turn.answer || "") + '</div>' +
                srcRow +
              '</div>' +
            '</div>' +
            '<div class="td-right">' +
              '<div>' +
                '<div class="td-stat-label">Latency</div>' +
                '<div class="td-latency">' +
                  '<div class="td-lat-row"><span class="td-lat-key">Retrieval</span><span class="td-lat-val ' + latClass(ret) + '">' + ret.toFixed(2) + 's</span></div>' +
                  '<div class="td-lat-row"><span class="td-lat-key">Generation</span><span class="td-lat-val ' + latClass(gen) + '">' + gen.toFixed(2) + 's</span></div>' +
                  '<div class="td-divider" style="margin:4px 0;"></div>' +
                  '<div class="td-lat-row"><span class="td-lat-key">Total</span><span class="td-lat-val ' + latClass(tot) + '">' + tot.toFixed(2) + 's</span></div>' +
                '</div>' +
              '</div>' +
              '<div class="td-divider"></div>' +
              '<div class="td-meta-stack">' +
                '<div class="td-meta-item">' + confBadge + '</div>' +
                (turn.answer_mode ? '<div class="td-meta-item"><span class="badge">' + escH(turn.answer_mode) + '</span></div>' : '') +
                (turn.model ? '<div class="td-meta-item" style="color:#98a2b3;">' + escH(turn.model) + '</div>' : '') +
              '</div>' +
            '</div>' +
          '</div>'
        );
      }}

      async function openSession(sid) {{
        const session = allSessions.find(s => s.session_id === sid);
        currentSession = session || null;
        panelTitle.textContent = session ? (session.title || "(untitled)") : sid;
        panelMeta.textContent  = "Loading…";
        panelBody.innerHTML    = `<div style="padding:32px;text-align:center;color:#667085;">Loading turns…</div>`;
        overlay.classList.add("open");
        try {{
          const r = await fetch(`/ask/session/${{encodeURIComponent(sid)}}`);
          currentTurns = await r.json();
        }} catch(e) {{
          currentTurns = [];
        }}
        const mc = currentTurns.length;
        panelMeta.textContent = mc + " turn" + (mc === 1 ? "" : "s") +
          (session ? " · Updated " + (session.updated_at || "") : "");
        panelBody.innerHTML = currentTurns.length
          ? currentTurns.map((t, i) => renderTurn(t, i)).join("")
          : `<div style="padding:32px;text-align:center;color:#667085;">No turns found in this session.</div>`;
      }}

      function closeOverlay() {{ overlay.classList.remove("open"); }}
      closeBtn?.addEventListener("click", closeOverlay);
      overlay?.addEventListener("click", e => {{ if (e.target === overlay) closeOverlay(); }});
      document.addEventListener("keydown", e => {{ if (e.key === "Escape") closeOverlay(); }});

      exportBtn?.addEventListener("click", () => {{
        if (!currentTurns.length) return;
        let text = "";
        currentTurns.forEach((t, i) => {{
          text += `=== Turn ${{i + 1}} · ${{t.created_at || ""}} ===\n`;
          text += `You: ${{t.question || ""}}\n\n`;
          text += `Adam: ${{t.answer || ""}}\n`;
          if ((t.sources || []).length) text += `Sources: ${{t.sources.join(", ")}}\n`;
          text += `Latency: ${{Number(t.total_seconds || 0).toFixed(2)}}s (retrieval ${{Number(t.retrieval_seconds || 0).toFixed(2)}}s, generation ${{Number(t.generation_seconds || 0).toFixed(2)}}s)\n`;
          text += `Mode: ${{t.answer_mode || ""}} · Model: ${{t.model || ""}}\n\n`;
        }});
        const blob = new Blob([text], {{ type: "text/plain" }});
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "session-" + (currentSession?.session_id || "export") + ".txt";
        a.click();
        URL.revokeObjectURL(a.href);
      }});

      searchEl?.addEventListener("input", () => renderList(allSessions));

      fetch("/ask/sessions")
        .then(r => r.json())
        .then(data => {{
          allSessions = data;
          if (emptyEl) emptyEl.remove();
          renderList(data);
        }})
        .catch(() => {{
          if (emptyEl) emptyEl.textContent = "Could not load conversations. Server may be offline.";
        }});
    }})();

    // ===== VOICE SESSION LOG =====
    (function setupVoiceSessionLog() {{
      const tbody  = document.getElementById("vl-tbody");
      const refBtn = document.getElementById("vl-refresh-btn");
      let expandedRow = null;
      let expandedDetailRow = null;
      let allLogs = [];

      function escH(s) {{
        return String(s || "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}}[c]));
      }}

      function latCls(v) {{ return v > 8 ? "slow" : v < 3 ? "fast" : ""; }}

      function srcBadge(ks) {{
        if (ks === "documents") return '<span class="vl-badge-blue">📄 Docs</span>';
        if (ks === "quick")     return '<span class="vl-badge-gray">⚡ Quick</span>';
        if (ks === "missing")   return '<span class="vl-badge-warn">⚠ Missing</span>';
        return '<span class="vl-badge-gray">💡 General</span>';
      }}

      function renderRow(log) {{
        const time = log.created_at ? log.created_at.slice(11, 19) : "";
        const trans = escH((log.transcript || "").slice(0, 45)) + ((log.transcript || "").length > 45 ? "…" : "");
        const tot   = Number(log.total_seconds || 0);
        const stt   = Number(log.stt_seconds || 0);
        const ret   = Number(log.retrieval_seconds || 0);
        const gen   = Number(log.generation_seconds || 0);
        const cpu   = Number(log.cpu_load || 0);
        const temp  = log.temperature_c != null ? Number(log.temperature_c).toFixed(1) : "—";
        const mem   = Number(log.memory_percent || 0);
        const barge = log.barge_in ? '<span class="vl-badge-warn">⚡</span>' : '<span style="color:#d0d5dd">—</span>';
        const err   = log.error ? '<span class="vl-badge-warn" title="' + escH(log.error) + '">ERR</span>' : "";
        return '<tr data-id="' + log.id + '">' +
          '<td>' + escH(time) + '</td>' +
          '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + trans + (err ? " " + err : "") + '</td>' +
          '<td>' + srcBadge(log.knowledge_source) + '</td>' +
          '<td class="' + latCls(stt) + '">' + stt.toFixed(2) + 's</td>' +
          '<td class="' + latCls(ret) + '">' + ret.toFixed(2) + 's</td>' +
          '<td class="' + latCls(gen) + '">' + gen.toFixed(2) + 's</td>' +
          '<td><strong class="' + latCls(tot) + '">' + tot.toFixed(2) + 's</strong></td>' +
          '<td>' + (log.filler_count || 0) + '</td>' +
          '<td>' + barge + '</td>' +
          '<td>' + cpu.toFixed(2) + '</td>' +
          '<td>' + temp + '</td>' +
          '<td>' + mem.toFixed(1) + '%</td>' +
          '<td>' + (log.word_count || 0) + '</td>' +
        '</tr>';
      }}

      function renderDetail(log) {{
        const srcs = (log.sources || []).map(function(s) {{ return '<span class="vl-src-pill">' + escH(s) + '</span>'; }}).join("");
        const stt = Number(log.stt_seconds || 0);
        const ret = Number(log.retrieval_seconds || 0);
        const gen = Number(log.generation_seconds || 0);
        const tot = Number(log.total_seconds || 0);
        const temp = log.temperature_c != null ? Number(log.temperature_c).toFixed(1) + " °C" : "—";
        return '<tr class="vl-detail-row"><td colspan="13"><div class="vl-detail-inner">' +
          '<div class="vl-detail-left">' +
            '<div class="vl-bubble-you">' + escH(log.transcript || "") + '</div>' +
            '<div class="vl-bubble-adam">' + escH(log.answer || "(no answer recorded)") + '</div>' +
            (srcs ? '<div style="margin-top:8px;">' + srcs + '</div>' : '') +
            (log.error ? '<div style="margin-top:8px;color:#b42318;font-size:12px;">Error: ' + escH(log.error) + '</div>' : '') +
          '</div>' +
          '<div class="vl-detail-right">' +
            '<div class="vl-stat-box">' +
              '<div class="vl-stat-title">Latency</div>' +
              '<div class="vl-stat-row"><span class="vl-stat-key">STT</span><span class="vl-stat-val ' + latCls(stt) + '">' + stt.toFixed(3) + 's</span></div>' +
              '<div class="vl-stat-row"><span class="vl-stat-key">Retrieval</span><span class="vl-stat-val ' + latCls(ret) + '">' + ret.toFixed(3) + 's</span></div>' +
              '<div class="vl-stat-row"><span class="vl-stat-key">Generation</span><span class="vl-stat-val ' + latCls(gen) + '">' + gen.toFixed(3) + 's</span></div>' +
              '<div style="border-top:1px solid #e0e7ff;margin:5px 0;"></div>' +
              '<div class="vl-stat-row"><span class="vl-stat-key">Total</span><span class="vl-stat-val ' + latCls(tot) + '"><strong>' + tot.toFixed(3) + 's</strong></span></div>' +
            '</div>' +
            '<div class="vl-stat-box">' +
              '<div class="vl-stat-title">System</div>' +
              '<div class="vl-stat-row"><span class="vl-stat-key">CPU load</span><span class="vl-stat-val">' + Number(log.cpu_load||0).toFixed(2) + '</span></div>' +
              '<div class="vl-stat-row"><span class="vl-stat-key">RAM</span><span class="vl-stat-val">' + Number(log.memory_percent||0).toFixed(1) + '%</span></div>' +
              '<div class="vl-stat-row"><span class="vl-stat-key">Temp</span><span class="vl-stat-val">' + temp + '</span></div>' +
            '</div>' +
            '<div class="vl-stat-box">' +
              '<div class="vl-stat-title">Pipeline</div>' +
              '<div class="vl-stat-row"><span class="vl-stat-key">Source</span><span class="vl-stat-val">' + srcBadge(log.knowledge_source) + '</span></div>' +
              '<div class="vl-stat-row"><span class="vl-stat-key">Fillers</span><span class="vl-stat-val">' + (log.filler_count||0) + '</span></div>' +
              '<div class="vl-stat-row"><span class="vl-stat-key">Barge-in</span><span class="vl-stat-val">' + (log.barge_in ? "Yes" : "No") + '</span></div>' +
              '<div class="vl-stat-row"><span class="vl-stat-key">Words</span><span class="vl-stat-val">' + (log.word_count||0) + '</span></div>' +
              '<div class="vl-stat-row"><span class="vl-stat-key">Docs used</span><span class="vl-stat-val">' + (log.source_count||0) + '</span></div>' +
            '</div>' +
            '<div class="vl-stat-box">' +
              '<div class="vl-stat-title">Time</div>' +
              '<div style="font-size:11px;color:#667085;line-height:1.6;">' +
                '<div>Wake: ' + escH(log.wake_detected_at || "—") + '</div>' +
                '<div>Logged: ' + escH(log.created_at || "—") + '</div>' +
              '</div>' +
            '</div>' +
          '</div>' +
        '</div></td></tr>';
      }}

      function load() {{
        fetch("/monitor/voice-logs?limit=100")
          .then(function(r) {{ return r.json(); }})
          .then(function(data) {{
            allLogs = data;
            expandedRow = null;
            expandedDetailRow = null;
            if (!data.length) {{
              tbody.innerHTML = '<tr><td colspan="13" style="text-align:center;color:#667085;padding:28px;">No voice sessions recorded yet — say "Adam" to start.</td></tr>';
              return;
            }}
            tbody.innerHTML = data.map(renderRow).join("");
            tbody.querySelectorAll("tr[data-id]").forEach(function(tr) {{
              tr.addEventListener("click", function() {{
                const id = Number(tr.dataset.id);
                const log = allLogs.find(function(l) {{ return l.id === id; }});
                if (!log) return;
                // Collapse if same row
                if (expandedRow === tr) {{
                  expandedDetailRow && expandedDetailRow.remove();
                  tr.classList.remove("vl-expanded");
                  expandedRow = null; expandedDetailRow = null;
                  return;
                }}
                // Collapse previous
                if (expandedRow) {{
                  expandedDetailRow && expandedDetailRow.remove();
                  expandedRow.classList.remove("vl-expanded");
                }}
                // Expand this row
                tr.classList.add("vl-expanded");
                const detail = document.createElement("tbody");
                detail.innerHTML = renderDetail(log);
                const detailTr = detail.firstElementChild;
                tr.after(detailTr);
                expandedRow = tr;
                expandedDetailRow = detailTr;
              }});
            }});
          }}).catch(function() {{
            tbody.innerHTML = '<tr><td colspan="13" style="text-align:center;color:#b42318;padding:28px;">Could not load voice session log.</td></tr>';
          }});
      }}

      refBtn?.addEventListener("click", load);
      load();
    }})();

    // Auto-refresh every 60 seconds (SSE keeps live view current, page refresh just updates the tables)
    (function setupAutoRefresh() {{
      const INTERVAL = 60;
      let remaining = INTERVAL;
      let paused = false;
      const btn = document.createElement("button");
      btn.className = "button-link"; btn.style.cssText = "font-size:12px;min-height:30px;padding:5px 8px;";
      btn.textContent = "Refresh tables: " + INTERVAL + "s";
      document.querySelector(".top-actions")?.appendChild(btn);
      const timer = setInterval(() => {{
        if (paused) return;
        remaining -= 1;
        btn.textContent = "Refresh tables: " + remaining + "s";
        if (remaining <= 0) {{ window.location.reload(); }}
      }}, 1000);
      btn.addEventListener("click", () => {{
        paused = !paused;
        if (paused) {{ btn.textContent = "Resume refresh"; btn.style.borderColor = "#f59e0b"; }}
        else {{ remaining = INTERVAL; btn.textContent = "Refresh tables: " + remaining + "s"; btn.style.borderColor = ""; }}
      }});
    }})();
  </script>
</body>
</html>"""

def render_inline_list(items: tuple[str, ...]) -> str:
    if not items:
        return '<span class="muted">-</span>'
    return html.escape(", ".join(items))


def render_source_usage_row(item: SourceUsageRecord) -> str:
    return f"""
    <tr>
      <td>{html.escape(item.source)}</td>
      <td>{item.count}</td>
    </tr>"""


def render_interaction_row(item: InteractionRecord) -> str:
    source_html = (
        "".join(
            f'<span class="source-pill">{html.escape(source)}</span>'
            for source in item.sources[:2]
        )
        or '<span class="muted">-</span>'
    )
    badge_class = "good" if item.total_seconds < 4 else "warn" if item.total_seconds < 8 else "bad"
    search = " ".join([
        item.created_at,
        item.question,
        display_answer_text(item.answer),
        " ".join(item.sources),
        item.answer_mode,
        item.model,
    ])
    return f"""
    <tr data-search="{html.escape(search)}">
      <td class="nowrap">{html.escape(item.created_at)}</td>
      <td class="question-cell">{html.escape(shorten_text(item.question, 140))}</td>
      <td class="answer-cell">{html.escape(shorten_text(display_answer_text(item.answer), 320))}</td>
      <td><div class="source-list">{source_html}</div></td>
      <td><span class="badge">{html.escape(item.answer_mode)}</span><div class="muted">{html.escape(item.model)}</div></td>
      <td class="nowrap"><span class="badge {badge_class}">{item.total_seconds:.2f}s</span><div class="muted">R {item.retrieval_seconds:.2f}s · G {item.generation_seconds:.2f}s</div></td>
      <td><a class="detail-link" href="/monitor/interaction/{item.id}">Open</a></td>
    </tr>"""


def render_event_row(item: EventRecord) -> str:
    details = ", ".join(f"{key}: {value}" for key, value in item.details.items())
    search = " ".join([item.created_at, item.event_type, item.message, details])
    event_class = "bad" if "error" in item.event_type.lower() else "warn" if "reject" in item.event_type.lower() else "good"
    return f"""
    <tr data-search="{html.escape(search)}">
      <td class="nowrap">{html.escape(item.created_at)}</td>
      <td><span class="badge {event_class}">{html.escape(item.event_type)}</span></td>
      <td>{html.escape(shorten_text(item.message, 180))}</td>
      <td class="details-cell">{html.escape(shorten_text(details or '-', 220))}</td>
    </tr>"""



def render_interaction_detail_page(
    item: InteractionRecord | None, config: AssistantConfig
) -> str:
    if item is None:
        body = "<section><h2>Interaction Not Found</h2><p>No saved chat record matched that id.</p></section>"
    else:
        sources = "".join(f"<li>{html.escape(source)}</li>" for source in item.sources) or "<li>-</li>"
        body = f"""
        <section>
          <h2>Question</h2>
          <p>{html.escape(item.question)}</p>
        </section>
        <section>
          <h2>Answer</h2>
          <pre>{html.escape(item.answer)}</pre>
        </section>
        <section>
          <h2>Sources</h2>
          <ul>{sources}</ul>
        </section>
        <section>
          <h2>Run Details</h2>
          <table><tbody>
            <tr><th>Time</th><td>{html.escape(item.created_at)}</td></tr>
            <tr><th>Mode</th><td>{html.escape(item.answer_mode)}</td></tr>
            <tr><th>Model</th><td>{html.escape(item.model)}</td></tr>
            <tr><th>Retrieval</th><td>{item.retrieval_seconds:.2f}s</td></tr>
            <tr><th>Generation</th><td>{item.generation_seconds:.2f}s</td></tr>
            <tr><th>Total</th><td>{item.total_seconds:.2f}s</td></tr>
          </tbody></table>
        </section>"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interaction Detail</title>
  <style>
    :root {{ font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; color: #182230; background: #f6f7f9; }}
    body {{ margin: 0; }}
    header {{ background: #111827; color: white; padding: 22px 32px; }}
    main {{ max-width: 980px; margin: 28px auto; padding: 0 20px; }}
    section {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 20px; margin-bottom: 18px; }}
    h1 {{ margin: 0; font-size: 24px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    a {{ color: white; font-weight: 700; text-decoration: none; }}
    pre {{ white-space: pre-wrap; font: inherit; line-height: 1.55; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e6edf5; padding: 10px; text-align: left; }}
  </style>
</head>
<body>
  <header><h1>Interaction Detail</h1><nav><a href="/monitor">Back to Monitor</a> · <a href="/ask">Ask Documents</a></nav></header>
  <main>{body}</main>
</body>
</html>"""


def render_notice(text: str, kind: str) -> str:
    if not text:
        return ""
    return f'<div class="notice {kind}">{html.escape(text)}</div>'


def render_file_row(path: Path, tags: list[str] | None = None) -> str:
    tag_pills = " ".join(f'<span class="tag-pill">{html.escape(t)}</span>' for t in (tags or []))
    tag_display = tag_pills or '<span class="muted">—</span>'
    current_tags = html.escape(", ".join(tags or []))
    return f"""
    <tr>
      <td>{html.escape(path.name)}</td>
      <td>{html.escape(path.suffix.lower())}</td>
      <td>{format_size(path.stat().st_size)}</td>
      <td>{tag_display}<br>
        <form style="display:inline-flex;gap:4px;margin-top:4px;" method="post" action="/documents/{html.escape(path.name)}/tags">
          <input class="tag-input" name="tags" type="text" placeholder="e.g. fees, policy" value="{current_tags}">
          <button class="tag-save" type="submit">Save</button>
        </form>
      </td>
      <td class="actions">
        <form method="post" action="/delete/{html.escape(path.name)}">
          <button class="danger" type="submit">Delete</button>
        </form>
      </td>
    </tr>"""


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace(" ", "_")
    return "".join(char for char in name if char.isalnum() or char in {"-", "_", "."})


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not create a unique file name.")


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def render_voice_page(model_name: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>Voice Assistant</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ height: 100%; width: 100%; overflow: hidden; }}
    body {{
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      background: #0a0a0f;
      color: #f0f0f5;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: space-between;
      min-height: 100dvh;
      padding: env(safe-area-inset-top, 0) env(safe-area-inset-right, 0) env(safe-area-inset-bottom, 0) env(safe-area-inset-left, 0);
    }}

    /* ── top bar ── */
    .topbar {{
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px 24px 0;
    }}
    .back-link {{
      color: #9ca3af;
      text-decoration: none;
      font-size: 14px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .back-link:hover {{ color: #f0f0f5; }}
    .model-chip {{
      font-size: 12px;
      color: #6b7280;
      border: 1px solid #1f2937;
      border-radius: 999px;
      padding: 4px 10px;
    }}

    /* ── orb ── */
    .orb-wrap {{
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 32px;
    }}
    .orb-ring {{
      position: relative;
      width: 220px;
      height: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    /* outer glow ring */
    .orb-ring::before {{
      content: "";
      position: absolute;
      inset: -18px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(99,179,237,0.18) 0%, transparent 70%);
      animation: ring-idle 3s ease-in-out infinite;
      opacity: 0;
      transition: opacity 0.5s;
    }}
    .orb-ring.listening::before {{ opacity: 1; animation: ring-listen 1s ease-in-out infinite; }}
    .orb-ring.speaking::before {{ opacity: 1; animation: ring-speak 0.6s ease-in-out infinite; }}

    .orb {{
      width: 200px;
      height: 200px;
      border-radius: 50%;
      position: relative;
      overflow: hidden;
      cursor: pointer;
      transition: transform 0.3s ease;
      box-shadow: 0 0 60px rgba(99,179,237,0.15), inset 0 0 40px rgba(255,255,255,0.05);
    }}
    .orb:hover {{ transform: scale(1.04); }}
    .orb:active {{ transform: scale(0.97); }}

    /* gradient layers that animate */
    .orb-layer {{
      position: absolute;
      inset: 0;
      border-radius: 50%;
    }}
    .orb-base {{
      background: radial-gradient(ellipse at 40% 35%, #2563eb 0%, #1e3a8a 45%, #0f172a 100%);
      animation: orb-idle-base 6s ease-in-out infinite;
    }}
    .orb-mist {{
      background: radial-gradient(ellipse at 60% 65%, rgba(147,197,253,0.55) 0%, rgba(59,130,246,0.2) 40%, transparent 70%);
      animation: orb-idle-mist 7s ease-in-out infinite alternate;
    }}
    .orb-shine {{
      background: radial-gradient(ellipse at 30% 25%, rgba(255,255,255,0.22) 0%, transparent 55%);
    }}

    /* idle animations */
    @keyframes orb-idle-base {{
      0%,100% {{ filter: brightness(1); }}
      50% {{ filter: brightness(1.12); }}
    }}
    @keyframes orb-idle-mist {{
      0% {{ transform: translate(0,0) scale(1); opacity: 0.7; }}
      100% {{ transform: translate(12px,-8px) scale(1.06); opacity: 1; }}
    }}
    @keyframes ring-idle {{
      0%,100% {{ transform: scale(1); opacity: 0.4; }}
      50% {{ transform: scale(1.06); opacity: 0.8; }}
    }}

    /* listening state */
    .orb-ring.listening .orb-base {{
      background: radial-gradient(ellipse at 40% 35%, #1d4ed8 0%, #1e3a8a 45%, #0f172a 100%);
      animation: orb-listen-pulse 0.9s ease-in-out infinite;
    }}
    .orb-ring.listening .orb-mist {{
      background: radial-gradient(ellipse at 55% 60%, rgba(165,243,252,0.65) 0%, rgba(56,189,248,0.3) 40%, transparent 70%);
      animation: orb-listen-mist 0.6s ease-in-out infinite alternate;
    }}
    @keyframes orb-listen-pulse {{
      0%,100% {{ filter: brightness(1); transform: scale(1); }}
      50% {{ filter: brightness(1.25); transform: scale(1.03); }}
    }}
    @keyframes orb-listen-mist {{
      0% {{ transform: translate(0,0) scale(1); opacity: 0.8; }}
      100% {{ transform: translate(-8px,6px) scale(1.1); opacity: 1; }}
    }}
    @keyframes ring-listen {{
      0%,100% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(99,179,237,0.4); }}
      50% {{ transform: scale(1.04); box-shadow: 0 0 0 12px rgba(99,179,237,0); }}
    }}

    /* speaking state */
    .orb-ring.speaking .orb-base {{
      background: radial-gradient(ellipse at 40% 35%, #7c3aed 0%, #4c1d95 45%, #0f172a 100%);
      animation: orb-speak-wave 0.45s ease-in-out infinite alternate;
    }}
    .orb-ring.speaking .orb-mist {{
      background: radial-gradient(ellipse at 60% 55%, rgba(216,180,254,0.6) 0%, rgba(167,139,250,0.3) 40%, transparent 70%);
      animation: orb-speak-mist 0.5s ease-in-out infinite alternate;
    }}
    @keyframes orb-speak-wave {{
      0% {{ filter: brightness(1); transform: scale(1) scaleX(1); }}
      100% {{ filter: brightness(1.3); transform: scale(1.04) scaleX(0.97); }}
    }}
    @keyframes orb-speak-mist {{
      0% {{ transform: translate(0,0) scale(1); opacity: 0.7; }}
      100% {{ transform: translate(10px,-5px) scale(1.12); opacity: 1; }}
    }}
    @keyframes ring-speak {{
      0%,100% {{ transform: scale(1); opacity: 0.6; }}
      50% {{ transform: scale(1.08); opacity: 1; }}
    }}

    /* thinking state */
    .orb-ring.thinking .orb-base {{
      background: radial-gradient(ellipse at 40% 35%, #0f766e 0%, #134e4a 45%, #0f172a 100%);
      animation: orb-think 2s ease-in-out infinite;
    }}
    .orb-ring.thinking .orb-mist {{
      background: radial-gradient(ellipse at 55% 60%, rgba(110,231,183,0.55) 0%, rgba(52,211,153,0.25) 40%, transparent 70%);
      animation: orb-think-mist 1.5s ease-in-out infinite alternate;
    }}
    @keyframes orb-think {{
      0%,100% {{ filter: brightness(1); }}
      50% {{ filter: brightness(1.18); }}
    }}
    @keyframes orb-think-mist {{
      0% {{ transform: scale(1) rotate(0deg); opacity: 0.7; }}
      100% {{ transform: scale(1.08) rotate(12deg); opacity: 0.95; }}
    }}

    /* ── status text ── */
    .status-text {{
      font-size: 16px;
      font-weight: 500;
      color: #9ca3af;
      letter-spacing: 0.02em;
      min-height: 24px;
      text-align: center;
      transition: color 0.3s;
    }}
    .orb-ring.listening ~ .status-text {{ color: #93c5fd; }}
    .orb-ring.speaking ~ .status-text {{ color: #c4b5fd; }}
    .orb-ring.thinking ~ .status-text {{ color: #6ee7b7; }}

    /* ── transcript / response text ── */
    .transcript-wrap {{
      width: min(480px, 90vw);
      min-height: 80px;
      max-height: 180px;
      overflow-y: auto;
      text-align: center;
      padding: 0 8px;
    }}
    .transcript-you {{
      font-size: 17px;
      color: #e2e8f0;
      font-weight: 500;
      line-height: 1.5;
      margin-bottom: 8px;
    }}
    .transcript-ai {{
      font-size: 15px;
      color: #94a3b8;
      line-height: 1.6;
    }}

    /* ── bottom controls ── */
    .controls {{
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 28px;
      padding: 24px 24px 36px;
    }}
    .ctrl-btn {{
      width: 56px;
      height: 56px;
      border-radius: 50%;
      border: 1px solid #1f2937;
      background: #111827;
      color: #9ca3af;
      font-size: 20px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.2s, color 0.2s, transform 0.15s;
    }}
    .ctrl-btn:hover {{ background: #1f2937; color: #f0f0f5; }}
    .ctrl-btn:active {{ transform: scale(0.93); }}
    .ctrl-btn.danger:hover {{ background: #450a0a; color: #fca5a5; border-color: #7f1d1d; }}

    .mic-main {{
      width: 72px;
      height: 72px;
      font-size: 26px;
      background: #1e3a8a;
      border-color: #3b82f6;
      color: #93c5fd;
    }}
    .mic-main:hover {{ background: #1d4ed8; color: #bfdbfe; }}
    .mic-main.active {{ background: #1d4ed8; border-color: #60a5fa; color: #fff; animation: mic-pulse 1s ease-in-out infinite; }}
    @keyframes mic-pulse {{ 0%,100% {{ box-shadow: 0 0 0 0 rgba(59,130,246,0.5); }} 50% {{ box-shadow: 0 0 0 10px rgba(59,130,246,0); }} }}

    /* ── error toast ── */
    .toast {{
      position: fixed;
      bottom: 110px;
      left: 50%;
      transform: translateX(-50%);
      background: #450a0a;
      color: #fca5a5;
      border: 1px solid #7f1d1d;
      border-radius: 10px;
      padding: 10px 18px;
      font-size: 13px;
      display: none;
      z-index: 99;
    }}
    .toast.show {{ display: block; }}
  </style>
</head>
<body>
  <header class="topbar">
    <a class="back-link" href="/ask">← Back to chat</a>
    <span class="model-chip">{html.escape(model_name)}</span>
  </header>

  <div class="orb-wrap">
    <div class="orb-ring" id="orb-ring">
      <div class="orb" id="orb" title="Tap to speak">
        <div class="orb-layer orb-base"></div>
        <div class="orb-layer orb-mist"></div>
        <div class="orb-layer orb-shine"></div>
      </div>
    </div>
    <div class="status-text" id="status-text">Tap to start</div>
    <div class="transcript-wrap">
      <div class="transcript-you" id="transcript-you"></div>
      <div class="transcript-ai" id="transcript-ai"></div>
    </div>
  </div>

  <div class="controls">
    <button class="ctrl-btn danger" id="btn-stop" title="Stop">■</button>
    <button class="ctrl-btn mic-main" id="btn-mic" title="Speak">🎤</button>
    <button class="ctrl-btn" id="btn-mute" title="Mute TTS">🔊</button>
  </div>

  <div class="toast" id="toast"></div>

  <script>
    const orbRing = document.getElementById("orb-ring");
    const orb = document.getElementById("orb");
    const statusText = document.getElementById("status-text");
    const transcriptYou = document.getElementById("transcript-you");
    const transcriptAi = document.getElementById("transcript-ai");
    const btnMic = document.getElementById("btn-mic");
    const btnStop = document.getElementById("btn-stop");
    const btnMute = document.getElementById("btn-mute");
    const toastEl = document.getElementById("toast");

    let ttsEnabled = true;
    let activeController = null;
    let mediaRec = null;
    let autoLoop = false;

    function setState(s) {{
      orbRing.className = "orb-ring " + s;
      const labels = {{ idle:"Tap to speak", listening:"Listening…", thinking:"Thinking…", speaking:"Speaking…" }};
      statusText.textContent = labels[s] || s;
    }}
    setState("idle");

    function showToast(msg, ms=3500) {{
      toastEl.textContent = msg; toastEl.classList.add("show");
      setTimeout(() => toastEl.classList.remove("show"), ms);
    }}

    /* ── TTS toggle ── */
    btnMute.addEventListener("click", () => {{
      ttsEnabled = !ttsEnabled;
      btnMute.textContent = ttsEnabled ? "🔊" : "🔇";
      btnMute.title = ttsEnabled ? "Mute TTS" : "Unmute TTS";
      if (!ttsEnabled) window.speechSynthesis?.cancel();
    }});

    /* ── Stop ── */
    function stopAll() {{
      autoLoop = false;
      streamFinished = true; pendingUtterances = 0;
      if (resumeTimer) {{ clearInterval(resumeTimer); resumeTimer = null; }}
      if (activeController) {{ activeController.abort(); activeController = null; }}
      window.speechSynthesis?.cancel();
      stopMic(); stopViz();
      setState("idle");
      btnMic.classList.remove("active");
    }}
    btnStop.addEventListener("click", stopAll);

    /* ── Natural voice selection ── */
    let _selectedVoice = null;
    function getVoice() {{
      if (_selectedVoice) return _selectedVoice;
      const voices = window.speechSynthesis?.getVoices() || [];
      const preferred = ["Samantha","Karen","Moira","Google UK English Female","Microsoft Zira Desktop","Google US English","Daniel","Fiona","Veena"];
      for (const name of preferred) {{
        const v = voices.find(v => v.name.includes(name));
        if (v) {{ _selectedVoice = v; return v; }}
      }}
      _selectedVoice = voices.find(v => v.lang?.startsWith("en")) || null;
      return _selectedVoice;
    }}
    if (window.speechSynthesis) window.speechSynthesis.onvoiceschanged = () => {{ _selectedVoice = null; }};

    /* ── Filler phrases spoken instantly while LLM is thinking ── */
    const FILLERS = [
      "Hmm, let me think about that.",
      "Sure, one moment.",
      "Good question, let me check.",
      "Right, give me a second.",
      "Okay, let me look into that.",
      "Interesting, let me think.",
    ];

    /* ── Queue a sentence chunk to speak immediately as tokens arrive ── */
    let pendingUtterances = 0;
    let streamFinished = false;
    let resumeTimer = null;

    function keepSpeechAlive() {{
      if (resumeTimer) clearInterval(resumeTimer);
      resumeTimer = setInterval(() => {{
        if (!window.speechSynthesis?.speaking) {{ clearInterval(resumeTimer); resumeTimer = null; return; }}
        window.speechSynthesis.pause();
        window.speechSynthesis.resume();
      }}, 10000);
    }}

    function speakChunk(text, isFiller) {{
      if (!ttsEnabled || !window.speechSynthesis || !text.trim()) return;
      const u = new SpeechSynthesisUtterance(text.trim());
      u.rate = 1.08; u.lang = "en-US"; u.pitch = 1.05;
      const v = getVoice(); if (v) u.voice = v;
      if (!isFiller) {{
        pendingUtterances++;
        setState("speaking");
      }}
      const onDone = () => {{
        if (!isFiller) {{
          pendingUtterances--;
          if (streamFinished && pendingUtterances === 0) {{
            if (resumeTimer) {{ clearInterval(resumeTimer); resumeTimer = null; }}
            setState("idle");
            if (autoLoop) setTimeout(startListening, 500);
          }}
        }}
      }};
      u.onend = onDone; u.onerror = onDone;
      window.speechSynthesis.speak(u);
      keepSpeechAlive();
    }}

    function sentenceReady(text) {{
      return text.length > 80 || /[.!?][\\s"']/.test(text) || text.endsWith("?") || text.endsWith("!") || text.endsWith(".");
    }}

    /* ── Start answering: speak filler immediately, then stream LLM response ── */
    function startAnswering(question) {{
      transcriptYou.textContent = question;
      transcriptAi.textContent = "";
      pendingUtterances = 0; streamFinished = false;
      window.speechSynthesis?.cancel();
      if (ttsEnabled) speakChunk(FILLERS[Math.floor(Math.random() * FILLERS.length)], true);
      streamAnswer(question);
    }}

    /* ── Stream answer — speak each sentence the moment it is complete ── */
    async function streamAnswer(question) {{
      setState("thinking");
      const fd = new FormData();
      fd.append("question", question);
      fd.append("session_id", "voice-" + (localStorage.getItem("voice_session") || "default"));
      fd.append("voice", "1");
      activeController = new AbortController();
      let buffer = "";
      let allParts = [];
      let sentenceBuf = "";
      try {{
        const resp = await fetch("/ask/stream", {{ method: "POST", body: fd, signal: activeController.signal }});
        const reader = resp.body.getReader();
        const dec = new TextDecoder();
        while (true) {{
          const {{ value, done }} = await reader.read(); if (done) break;
          buffer += dec.decode(value, {{ stream: true }});
          const lines = buffer.split("\\n"); buffer = lines.pop() || "";
          for (const line of lines) {{
            if (!line.trim()) continue;
            const data = JSON.parse(line);
            if (data.type === "token") {{
              const tok = data.text || "";
              allParts.push(tok); sentenceBuf += tok;
              transcriptAi.textContent = allParts.join("");
              if (sentenceReady(sentenceBuf)) {{
                speakChunk(sentenceBuf, false);
                sentenceBuf = "";
              }}
            }}
            if (data.type === "done") {{
              if (sentenceBuf.trim()) speakChunk(sentenceBuf, false);
              activeController = null;
              streamFinished = true;
              if (!ttsEnabled || pendingUtterances === 0) {{
                setState("idle");
                if (autoLoop) setTimeout(startListening, 500);
              }}
              return;
            }}
            if (data.type === "error") {{
              showToast(data.message || "Error from assistant.");
              setState("idle");
              if (autoLoop) setTimeout(startListening, 600);
              return;
            }}
          }}
        }}
      }} catch (err) {{
        if (err.name !== "AbortError") showToast("Connection error — is the server running?");
        setState("idle"); btnMic.classList.remove("active");
      }}
    }}

    /* ── Audio → WAV ── */
    async function audioToWav(blob) {{
      const ctx = new (window.AudioContext || window.webkitAudioContext)({{ sampleRate: 16000 }});
      const ab = await blob.arrayBuffer();
      let buf;
      try {{ buf = await ctx.decodeAudioData(ab); }} catch(e) {{ ctx.close(); throw e; }}
      ctx.close();
      const pcm = buf.getChannelData(0);
      const i16 = new Int16Array(pcm.length);
      for (let i = 0; i < pcm.length; i++) i16[i] = Math.max(-32768, Math.min(32767, pcm[i] * 32768));
      const out = new ArrayBuffer(44 + i16.byteLength);
      const v = new DataView(out);
      const ws = (o, s) => {{ for (let i=0;i<s.length;i++) v.setUint8(o+i, s.charCodeAt(i)); }};
      ws(0,"RIFF"); v.setUint32(4,36+i16.byteLength,true); ws(8,"WAVE");
      ws(12,"fmt "); v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true);
      v.setUint32(24,16000,true); v.setUint32(28,32000,true); v.setUint16(32,2,true); v.setUint16(34,16,true);
      ws(36,"data"); v.setUint32(40,i16.byteLength,true);
      new Int16Array(out,44).set(i16);
      return new Blob([out], {{ type:"audio/wav" }});
    }}

    /* ── Transcribe audio ── */
    async function transcribeAudio(audioBlob) {{
      setState("thinking");
      btnMic.classList.remove("active");
      try {{
        const wav = await audioToWav(audioBlob);
        const fd = new FormData(); fd.append("file", wav, "rec.wav");
        const resp = await fetch("/ask/transcribe", {{ method: "POST", body: fd }});
        const data = await resp.json();
        const text = (data.text || "").trim();
        if (!text) {{
          showToast("Couldn't hear that — please try again.");
          setState("idle");
          if (autoLoop) setTimeout(startListening, 600);
          return;
        }}
        startAnswering(text);
      }} catch(e) {{
        showToast("Transcription failed. Is the Vosk model loaded?");
        setState("idle");
        if (autoLoop) setTimeout(startListening, 800);
      }}
    }}

    /* ── MediaRecorder mic recording ── */
    let chunks = [];
    let vizAudioCtx = null;
    function stopMic() {{
      if (mediaRec && mediaRec.state === "recording") mediaRec.stop();
    }}
    function stopViz() {{
      if (vizAudioCtx) {{ try {{ vizAudioCtx.close(); }} catch(_) {{}} vizAudioCtx = null; }}
      orb.style.transform = "";
    }}

    function startVizAndVAD(stream) {{
      try {{
        vizAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const src = vizAudioCtx.createMediaStreamSource(stream);
        const analyser = vizAudioCtx.createAnalyser();
        analyser.fftSize = 64;
        src.connect(analyser);
        const buf = new Uint8Array(analyser.frequencyBinCount);

        const SPEECH_THRESHOLD = 12;   /* volume level = speech */
        const SILENCE_MS = 700;        /* stop after 700 ms quiet */
        const MIN_SPEECH_MS = 400;     /* ignore taps shorter than this */
        let speechStart = null;
        let silenceStart = null;

        function tick() {{
          if (!orbRing.classList.contains("listening")) {{ stopViz(); return; }}
          analyser.getByteFrequencyData(buf);
          const vol = buf.reduce((a,b)=>a+b,0)/buf.length;

          /* orb visualization */
          orb.style.transform = `scale(${{1 + Math.min(vol/255,1)*0.22}})`;

          /* VAD */
          if (vol > SPEECH_THRESHOLD) {{
            if (!speechStart) speechStart = Date.now();
            silenceStart = null;
          }} else if (speechStart) {{
            if (!silenceStart) silenceStart = Date.now();
            const spokenMs = silenceStart - speechStart;
            if (spokenMs >= MIN_SPEECH_MS && Date.now() - silenceStart >= SILENCE_MS) {{
              stopMic();   /* auto-stop — user finished speaking */
              return;
            }}
          }}
          requestAnimationFrame(tick);
        }}
        tick();
      }} catch(_) {{}}
    }}

    async function startMediaRecorder() {{
      const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
      chunks = [];
      const opts = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? {{ mimeType:"audio/webm;codecs=opus" }} : {{}};
      mediaRec = new MediaRecorder(stream, opts);
      mediaRec.ondataavailable = e => {{ if (e.data.size > 0) chunks.push(e.data); }};
      mediaRec.onstop = async () => {{
        stopViz();
        stream.getTracks().forEach(t => t.stop());
        const raw = new Blob(chunks, {{ type: mediaRec.mimeType || "audio/webm" }});
        chunks = []; mediaRec = null;
        await transcribeAudio(raw);
      }};
      mediaRec.start(250);
      setState("listening");
      btnMic.classList.add("active");
      startVizAndVAD(stream);
      setTimeout(() => {{ if (mediaRec?.state === "recording") stopMic(); }}, 15000);
    }}

    /* ── Web Speech API path (fast, works on HTTPS / localhost) ── */
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const secureCtx = location.protocol === "https:" || location.hostname === "localhost" || location.hostname === "127.0.0.1";
    let srRec = null;

    async function startListening() {{
      if (orbRing.classList.contains("listening") || orbRing.classList.contains("thinking") || orbRing.classList.contains("speaking")) return;
      transcriptYou.textContent = "";
      transcriptAi.textContent = "";

      if (SR && secureCtx) {{
        srRec = new SR();
        srRec.lang = "en-US"; srRec.interimResults = true; srRec.continuous = false;
        srRec.onresult = (e) => {{
          const transcript = Array.from(e.results).map(r => r[0].transcript).join("");
          transcriptYou.textContent = transcript;
          if (e.results[e.results.length-1].isFinal) {{
            srRec = null; btnMic.classList.remove("active");
            if (transcript.trim()) startAnswering(transcript.trim());
            else {{ setState("idle"); if (autoLoop) setTimeout(startListening, 500); }}
          }}
        }};
        srRec.onerror = (e) => {{
          btnMic.classList.remove("active"); setState("idle");
          if (e.error !== "no-speech") showToast("Mic error: " + e.error);
          if (autoLoop) setTimeout(startListening, 600);
        }};
        srRec.onend = () => {{ btnMic.classList.remove("active"); }};
        try {{ srRec.start(); setState("listening"); btnMic.classList.add("active"); return; }} catch(_) {{}}
      }}

      /* fallback: MediaRecorder */
      if (navigator.mediaDevices?.getUserMedia) {{
        try {{ await startMediaRecorder(); return; }} catch(e) {{ showToast("Microphone access denied."); return; }}
      }}
      showToast("Microphone not available on this browser/connection.");
    }}

    /* ── iOS audio unlock — must call from a user gesture before any auto-speak ── */
    let audioUnlocked = false;
    function unlockAudio() {{
      if (audioUnlocked) return;
      audioUnlocked = true;
      if (window.speechSynthesis) {{
        const u = new SpeechSynthesisUtterance("");
        u.volume = 0;
        window.speechSynthesis.speak(u);
      }}
      /* also resume any suspended AudioContext */
      try {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        if (ctx.state === "suspended") ctx.resume();
        ctx.close();
      }} catch(_) {{}}
    }}

    /* ── tap orb or mic button ── */
    function handleMicTap() {{
      unlockAudio();
      if (orbRing.classList.contains("listening")) {{
        /* stop early */
        if (srRec) {{ srRec.stop(); srRec = null; }}
        stopMic();
        return;
      }}
      if (orbRing.classList.contains("speaking") || orbRing.classList.contains("thinking")) {{
        stopAll(); return;
      }}
      autoLoop = true;
      startListening();
    }}

    orb.addEventListener("click", handleMicTap);
    btnMic.addEventListener("click", handleMicTap);

    /* session id per page load */
    if (!localStorage.getItem("voice_session")) {{
      localStorage.setItem("voice_session", Date.now().toString(36));
    }}
  </script>
</body>
</html>"""


def main() -> None:
    import uvicorn

    host = os.getenv("ASSISTANT_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.getenv("ASSISTANT_DASHBOARD_PORT", "8080"))
    ssl_cert = os.getenv("ASSISTANT_SSL_CERT")
    ssl_key = os.getenv("ASSISTANT_SSL_KEY")
    uvicorn.run(
        "assistant_app.dashboard:app",
        host=host,
        port=port,
        ssl_certfile=ssl_cert or None,
        ssl_keyfile=ssl_key or None,
    )


if __name__ == "__main__":
    main()
