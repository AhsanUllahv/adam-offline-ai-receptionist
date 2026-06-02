from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VoiceLogRecord:
    id: int
    created_at: str
    wake_detected_at: str
    transcript: str
    answer: str
    sources: tuple[str, ...]
    knowledge_source: str
    filler_count: int
    barge_in: bool
    answer_truncated: bool
    stt_seconds: float
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    word_count: int
    source_count: int
    cpu_load: float
    memory_percent: float
    temperature_c: float | None
    error: str


@dataclass(frozen=True)
class LatencyStats:
    total_count: int
    average_total: float
    average_retrieval: float
    average_generation: float
    slowest_question: str
    slowest_total: float
    last_total: float


@dataclass(frozen=True)
class SourceUsageRecord:
    source: str
    count: int


@dataclass(frozen=True)
class ChatSessionRecord:
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


@dataclass(frozen=True)
class InteractionRecord:
    id: int
    created_at: str
    question: str
    answer: str
    sources: tuple[str, ...]
    answer_mode: str
    model: str
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    session_id: str = "default"


@dataclass(frozen=True)
class EventRecord:
    id: int
    created_at: str
    event_type: str
    message: str
    details: dict[str, Any]


class AuditStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS dashboard_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT 'default',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    answer_mode TEXT NOT NULL,
                    model TEXT NOT NULL,
                    retrieval_seconds REAL NOT NULL,
                    generation_seconds REAL NOT NULL,
                    total_seconds REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_dashboard_interactions_created_at
                ON dashboard_interactions(created_at);

                CREATE TABLE IF NOT EXISTS dashboard_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_dashboard_events_created_at
                ON dashboard_events(created_at);

                CREATE TABLE IF NOT EXISTS interaction_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id INTEGER NOT NULL,
                    rating TEXT NOT NULL CHECK(rating IN ('up', 'down')),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS voice_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    wake_detected_at TEXT NOT NULL DEFAULT '',
                    transcript TEXT NOT NULL DEFAULT '',
                    answer TEXT NOT NULL DEFAULT '',
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    knowledge_source TEXT NOT NULL DEFAULT 'general',
                    filler_count INTEGER NOT NULL DEFAULT 0,
                    barge_in INTEGER NOT NULL DEFAULT 0,
                    answer_truncated INTEGER NOT NULL DEFAULT 0,
                    stt_seconds REAL NOT NULL DEFAULT 0.0,
                    retrieval_seconds REAL NOT NULL DEFAULT 0.0,
                    generation_seconds REAL NOT NULL DEFAULT 0.0,
                    total_seconds REAL NOT NULL DEFAULT 0.0,
                    word_count INTEGER NOT NULL DEFAULT 0,
                    source_count INTEGER NOT NULL DEFAULT 0,
                    cpu_load REAL NOT NULL DEFAULT 0.0,
                    memory_percent REAL NOT NULL DEFAULT 0.0,
                    temperature_c REAL,
                    error TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_voice_logs_created_at
                ON voice_logs(created_at);
                """
            )
            self._migrate_schema(connection)

    def _migrate_schema(self, connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(dashboard_interactions)").fetchall()
        }
        if "session_id" not in columns:
            connection.execute(
                "ALTER TABLE dashboard_interactions ADD COLUMN session_id TEXT NOT NULL DEFAULT 'default'"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_dashboard_interactions_session_id ON dashboard_interactions(session_id)"
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO chat_sessions(session_id, title, created_at, updated_at)
            SELECT
                session_id,
                COALESCE((SELECT question FROM dashboard_interactions AS first_turn WHERE first_turn.session_id = dashboard_interactions.session_id ORDER BY first_turn.id ASC LIMIT 1), 'Previous Chat'),
                MIN(created_at),
                MAX(created_at)
            FROM dashboard_interactions
            GROUP BY session_id
            """
        )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def log_interaction(
        self,
        question: str,
        answer: str,
        sources: tuple[str, ...],
        answer_mode: str,
        model: str,
        retrieval_seconds: float,
        generation_seconds: float,
        session_id: str = "default",
    ) -> int:
        total_seconds = retrieval_seconds + generation_seconds
        title = question[:72] or "New Chat"
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions(session_id, title)
                VALUES (?, ?)
                ON CONFLICT(session_id) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, title),
            )
            existing_count = connection.execute(
                "SELECT COUNT(*) AS total FROM dashboard_interactions WHERE session_id = ?",
                (session_id,),
            ).fetchone()["total"]
            if int(existing_count) == 0:
                connection.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                    (title, session_id),
                )
            cursor = connection.execute(
                """
                INSERT INTO dashboard_interactions(
                    session_id, question, answer, sources_json, answer_mode, model,
                    retrieval_seconds, generation_seconds, total_seconds
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    question,
                    answer,
                    json.dumps(list(sources)),
                    answer_mode,
                    model,
                    retrieval_seconds,
                    generation_seconds,
                    total_seconds,
                ),
            )
            connection.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,),
            )
            return int(cursor.lastrowid)

    def log_feedback(self, interaction_id: int, rating: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO interaction_feedback (interaction_id, rating) VALUES (?, ?)",
                (interaction_id, rating),
            )

    def log_event(
        self, event_type: str, message: str, details: dict[str, Any] | None = None
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO dashboard_events(event_type, message, details_json)
                VALUES (?, ?, ?)
                """,
                (event_type, message, json.dumps(details or {})),
            )
            return int(cursor.lastrowid)

    def recent_interactions(
        self, limit: int = 25, session_id: str | None = None
    ) -> list[InteractionRecord]:
        with self.connect() as connection:
            if session_id:
                rows = connection.execute(
                    """
                    SELECT * FROM dashboard_interactions
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM dashboard_interactions
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [self._interaction_from_row(row) for row in rows]

    def recent_sessions(self, limit: int = 20) -> list[ChatSessionRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    chat_sessions.session_id,
                    chat_sessions.title,
                    chat_sessions.created_at,
                    chat_sessions.updated_at,
                    COUNT(dashboard_interactions.id) AS message_count
                FROM chat_sessions
                LEFT JOIN dashboard_interactions
                    ON dashboard_interactions.session_id = chat_sessions.session_id
                GROUP BY chat_sessions.session_id
                ORDER BY chat_sessions.updated_at DESC, chat_sessions.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._session_from_row(row) for row in rows]

    def interactions_for_session(
        self, session_id: str, limit: int = 100
    ) -> list[InteractionRecord]:
        return list(reversed(self.recent_interactions(limit=limit, session_id=session_id)))

    def delete_session(self, session_id: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM dashboard_interactions WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))

    def recent_events(self, limit: int = 25) -> list[EventRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM dashboard_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def interaction_count(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM dashboard_interactions"
            ).fetchone()
        return int(row["total"])

    def event_count(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM dashboard_events").fetchone()
        return int(row["total"])


    def get_interaction(self, interaction_id: int) -> InteractionRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM dashboard_interactions WHERE id = ?", (interaction_id,)
            ).fetchone()
        return self._interaction_from_row(row) if row else None

    def latency_stats(self) -> LatencyStats:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_count,
                    COALESCE(AVG(total_seconds), 0) AS average_total,
                    COALESCE(AVG(retrieval_seconds), 0) AS average_retrieval,
                    COALESCE(AVG(generation_seconds), 0) AS average_generation
                FROM dashboard_interactions
                """
            ).fetchone()
            slowest = connection.execute(
                """
                SELECT question, total_seconds FROM dashboard_interactions
                ORDER BY total_seconds DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
            last = connection.execute(
                """
                SELECT total_seconds FROM dashboard_interactions
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()

        return LatencyStats(
            total_count=int(row["total_count"]),
            average_total=float(row["average_total"]),
            average_retrieval=float(row["average_retrieval"]),
            average_generation=float(row["average_generation"]),
            slowest_question=str(slowest["question"]) if slowest else "-",
            slowest_total=float(slowest["total_seconds"]) if slowest else 0.0,
            last_total=float(last["total_seconds"]) if last else 0.0,
        )

    def source_usage(self, limit: int = 10) -> list[SourceUsageRecord]:
        with self.connect() as connection:
            try:
                rows = connection.execute(
                    """
                    SELECT
                        SUBSTR(value, 1,
                            CASE
                                WHEN INSTR(value, ', page') > 0 THEN INSTR(value, ', page') - 1
                                WHEN INSTR(value, ', section') > 0 THEN INSTR(value, ', section') - 1
                                ELSE LENGTH(value)
                            END
                        ) AS name,
                        COUNT(*) AS cnt
                    FROM dashboard_interactions,
                         json_each(dashboard_interactions.sources_json)
                    GROUP BY name
                    ORDER BY cnt DESC, name ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [SourceUsageRecord(source=str(row["name"]), count=int(row["cnt"])) for row in rows]
            except Exception:
                raw = connection.execute("SELECT sources_json FROM dashboard_interactions").fetchall()

        counts: dict[str, int] = {}
        for row in raw:
            for source in json.loads(row["sources_json"]):
                name = str(source).split(", page", 1)[0].split(", section", 1)[0]
                counts[name] = counts.get(name, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [SourceUsageRecord(source=source, count=count) for source, count in ranked]

    def error_events(self, limit: int = 25) -> list[EventRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM dashboard_events
                WHERE event_type LIKE '%error%' OR event_type LIKE '%rejected%'
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def all_interactions(self) -> list[InteractionRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM dashboard_interactions ORDER BY id DESC"
            ).fetchall()
        return [self._interaction_from_row(row) for row in rows]

    def all_events(self) -> list[EventRecord]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM dashboard_events ORDER BY id DESC").fetchall()
        return [self._event_from_row(row) for row in rows]

    def events_in_window(
        self,
        start_ts: str,
        end_ts: str,
        event_types: tuple[str, ...] | None = None,
    ) -> list[EventRecord]:
        with self.connect() as connection:
            if event_types:
                placeholders = ",".join("?" * len(event_types))
                rows = connection.execute(
                    f"""
                    SELECT * FROM dashboard_events
                    WHERE created_at > ? AND created_at <= ?
                      AND event_type IN ({placeholders})
                    ORDER BY id ASC
                    """,
                    (start_ts, end_ts, *event_types),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM dashboard_events
                    WHERE created_at > ? AND created_at <= ?
                    ORDER BY id ASC
                    """,
                    (start_ts, end_ts),
                ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def previous_voice_interaction_time(self, interaction_id: int) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT created_at FROM dashboard_interactions
                WHERE id < ? AND session_id = 'voice'
                ORDER BY id DESC
                LIMIT 1
                """,
                (interaction_id,),
            ).fetchone()
        return str(row["created_at"]) if row else None

    def save_voice_log(
        self,
        *,
        wake_detected_at: str = "",
        transcript: str = "",
        answer: str = "",
        sources: tuple[str, ...] = (),
        knowledge_source: str = "general",
        filler_count: int = 0,
        barge_in: bool = False,
        answer_truncated: bool = False,
        stt_seconds: float = 0.0,
        retrieval_seconds: float = 0.0,
        generation_seconds: float = 0.0,
        total_seconds: float = 0.0,
        word_count: int = 0,
        source_count: int = 0,
        cpu_load: float = 0.0,
        memory_percent: float = 0.0,
        temperature_c: float | None = None,
        error: str = "",
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO voice_logs (
                    wake_detected_at, transcript, answer, sources_json,
                    knowledge_source, filler_count, barge_in, answer_truncated,
                    stt_seconds, retrieval_seconds, generation_seconds, total_seconds,
                    word_count, source_count, cpu_load, memory_percent, temperature_c, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    wake_detected_at, transcript, answer,
                    json.dumps(list(sources)), knowledge_source,
                    filler_count, int(barge_in), int(answer_truncated),
                    stt_seconds, retrieval_seconds, generation_seconds, total_seconds,
                    word_count, source_count, cpu_load, memory_percent, temperature_c, error,
                ),
            )
            return int(cursor.lastrowid)

    def recent_voice_logs(self, limit: int = 50) -> list[VoiceLogRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM voice_logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._voice_log_from_row(row) for row in rows]

    def _voice_log_from_row(self, row: sqlite3.Row) -> VoiceLogRecord:
        return VoiceLogRecord(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            wake_detected_at=str(row["wake_detected_at"] or ""),
            transcript=str(row["transcript"] or ""),
            answer=str(row["answer"] or ""),
            sources=tuple(json.loads(row["sources_json"] or "[]")),
            knowledge_source=str(row["knowledge_source"] or "general"),
            filler_count=int(row["filler_count"] or 0),
            barge_in=bool(row["barge_in"]),
            answer_truncated=bool(row["answer_truncated"]),
            stt_seconds=float(row["stt_seconds"] or 0),
            retrieval_seconds=float(row["retrieval_seconds"] or 0),
            generation_seconds=float(row["generation_seconds"] or 0),
            total_seconds=float(row["total_seconds"] or 0),
            word_count=int(row["word_count"] or 0),
            source_count=int(row["source_count"] or 0),
            cpu_load=float(row["cpu_load"] or 0),
            memory_percent=float(row["memory_percent"] or 0),
            temperature_c=float(row["temperature_c"]) if row["temperature_c"] is not None else None,
            error=str(row["error"] or ""),
        )

    def _session_from_row(self, row: sqlite3.Row) -> ChatSessionRecord:
        return ChatSessionRecord(
            session_id=str(row["session_id"]),
            title=str(row["title"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            message_count=int(row["message_count"]),
        )

    def _interaction_from_row(self, row: sqlite3.Row) -> InteractionRecord:
        return InteractionRecord(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            sources=tuple(json.loads(row["sources_json"])),
            answer_mode=str(row["answer_mode"]),
            model=str(row["model"]),
            retrieval_seconds=float(row["retrieval_seconds"]),
            generation_seconds=float(row["generation_seconds"]),
            total_seconds=float(row["total_seconds"]),
            session_id=str(row["session_id"]),
        )

    def _event_from_row(self, row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            event_type=str(row["event_type"]),
            message=str(row["message"]),
            details=json.loads(row["details_json"]),
        )
