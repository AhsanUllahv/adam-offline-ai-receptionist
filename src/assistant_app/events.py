from __future__ import annotations

import asyncio
import json
import threading
from typing import AsyncGenerator

_lock = threading.Lock()
_clients: list[asyncio.Queue] = []
_loop: asyncio.AbstractEventLoop | None = None
_sqlite_path: str | None = None

# Event types that are too frequent or not worth persisting
_NO_PERSIST = {"token"}


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


def set_sqlite_path(path: str) -> None:
    global _sqlite_path
    _sqlite_path = path


def push_event(event_type: str, **payload: object) -> None:
    """Push an event from any thread to all connected SSE clients and persist to DB."""
    loop = _loop
    if loop is None or loop.is_closed():
        return
    data = json.dumps({"type": event_type, **payload})
    try:
        loop.call_soon_threadsafe(_broadcast, data)
    except RuntimeError:
        pass

    if event_type not in _NO_PERSIST and _sqlite_path:
        _persist_event(event_type, payload)


def _persist_event(event_type: str, payload: dict) -> None:
    try:
        from assistant_app.audit import AuditStore
        message = _event_message(event_type, payload)
        AuditStore(_sqlite_path).log_event(event_type, message, dict(payload))
    except Exception:
        pass


def _event_message(event_type: str, payload: dict) -> str:
    if event_type == "state":
        return f"state → {payload.get('state', '')}"
    if event_type == "wake":
        return f"Wake word '{payload.get('keyword', 'adam')}' detected"
    if event_type == "wake_response":
        return f"Adam: {payload.get('text', '')}"
    if event_type == "transcript":
        return f"You said: {payload.get('text', '')}"
    if event_type == "sources":
        return "Sources: " + ", ".join(payload.get("sources", []))
    if event_type == "filler":
        return payload.get("text", "")
    if event_type == "barge_in":
        return "User interrupted (barge-in)"
    if event_type == "error":
        return payload.get("message", "Unknown error")
    if event_type == "latency":
        return f"{payload.get('seconds', 0)}s total latency"
    if event_type == "answer_complete":
        text = payload.get("text", "")
        preview = text[:120] + "…" if len(text) > 120 else text
        return f"Answer: {preview}"
    return json.dumps(payload)


def _broadcast(data: str) -> None:
    with _lock:
        snapshot = list(_clients)
    for q in snapshot:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


async def sse_stream() -> AsyncGenerator[str, None]:
    """Async generator that yields SSE-formatted event data for one client."""
    q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=300)
    with _lock:
        _clients.append(q)
    try:
        while True:
            try:
                data = await asyncio.wait_for(q.get(), timeout=20.0)
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue
            if data is None:
                break
            yield f"data: {data}\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        with _lock:
            try:
                _clients.remove(q)
            except ValueError:
                pass
