from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path




@dataclass(frozen=True)
class IndexStats:
    document_count: int
    chunk_count: int
    last_indexed_at: str


@dataclass(frozen=True)
class IndexedDocumentRecord:
    path: str
    name: str
    file_type: str
    indexed_at: str
    chunk_count: int


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_path: str
    document_name: str
    page_number: int | None
    section_number: int
    text: str


class MetadataStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id INTEGER NOT NULL,
                    page_number INTEGER,
                    section_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks(document_id);

                CREATE TABLE IF NOT EXISTS document_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_name TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    UNIQUE(document_name, tag)
                );
                """
            )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def upsert_document(self, path: Path) -> int:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO documents(path, name, file_type)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    name = excluded.name,
                    file_type = excluded.file_type,
                    indexed_at = CURRENT_TIMESTAMP
                """,
                (str(path), path.name, path.suffix.lower().lstrip(".")),
            )
            row = connection.execute(
                "SELECT id FROM documents WHERE path = ?", (str(path),)
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Could not create metadata row for {path}")
            return int(row["id"])

    def delete_chunks_for_document(self, document_id: int) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    def get_document_id(self, path: Path) -> int | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM documents WHERE path = ?", (str(path),)
            ).fetchone()
        return int(row["id"]) if row else None

    def get_chunk_ids_for_document(self, document_id: int) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def delete_document(self, document_id: int) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def insert_chunks(self, document_id: int, records: list[ChunkRecord]) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO chunks(
                    id, document_id, page_number, section_number, text
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.chunk_id,
                        document_id,
                        record.page_number,
                        record.section_number,
                        record.text,
                    )
                    for record in records
                ],
            )


    def set_document_tags(self, document_name: str, tags: list[str]) -> None:
        cleaned = [t.strip().lower() for t in tags if t.strip()]
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM document_tags WHERE document_name = ?", (document_name,)
            )
            connection.executemany(
                "INSERT OR IGNORE INTO document_tags (document_name, tag) VALUES (?, ?)",
                [(document_name, tag) for tag in cleaned],
            )

    def get_document_tags(self, document_name: str) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT tag FROM document_tags WHERE document_name = ? ORDER BY tag",
                (document_name,),
            ).fetchall()
        return [str(row["tag"]) for row in rows]

    def all_document_tags(self) -> dict[str, list[str]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT document_name, tag FROM document_tags ORDER BY document_name, tag"
            ).fetchall()
        result: dict[str, list[str]] = {}
        for row in rows:
            result.setdefault(str(row["document_name"]), []).append(str(row["tag"]))
        return result

    def get_question_suggestions(self, query: str, limit: int = 5) -> list[str]:
        q = query.strip().lower()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT chunks.text FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE LOWER(chunks.text) LIKE ?
                LIMIT 40
                """,
                (f"%{q}%",),
            ).fetchall()
        suggestions: list[str] = []
        seen: set[str] = set()
        for row in rows:
            sentences = re.split(r"(?<=[.!?])\s+", str(row["text"]).strip())
            for sentence in sentences[:4]:
                sentence = sentence.strip()
                if q in sentence.lower() and 12 <= len(sentence) <= 110:
                    key = sentence[:50].lower()
                    if key not in seen:
                        seen.add(key)
                        text = sentence.rstrip(".!?,;")
                        suggestions.append(text + "?")
                        if len(suggestions) >= limit:
                            return suggestions
        return suggestions

    def index_stats(self) -> IndexStats:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM documents) AS document_count,
                    (SELECT COUNT(*) FROM chunks) AS chunk_count,
                    COALESCE(MAX(indexed_at), '-') AS last_indexed_at
                FROM documents
                """
            ).fetchone()
        return IndexStats(
            document_count=int(row["document_count"]),
            chunk_count=int(row["chunk_count"]),
            last_indexed_at=str(row["last_indexed_at"]),
        )

    def indexed_documents(self) -> list[IndexedDocumentRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    documents.path,
                    documents.name,
                    documents.file_type,
                    documents.indexed_at,
                    COUNT(chunks.id) AS chunk_count
                FROM documents
                LEFT JOIN chunks ON chunks.document_id = documents.id
                GROUP BY documents.id
                ORDER BY documents.indexed_at DESC, documents.name
                """
            ).fetchall()
        return [
            IndexedDocumentRecord(
                path=str(row["path"]),
                name=str(row["name"]),
                file_type=str(row["file_type"]),
                indexed_at=str(row["indexed_at"]),
                chunk_count=int(row["chunk_count"]),
            )
            for row in rows
        ]

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, ChunkRecord]:
        if not chunk_ids:
            return {}

        placeholders = ",".join("?" for _ in chunk_ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    chunks.id,
                    documents.path,
                    documents.name,
                    chunks.page_number,
                    chunks.section_number,
                    chunks.text
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()

        return {
            row["id"]: ChunkRecord(
                chunk_id=row["id"],
                document_path=row["path"],
                document_name=row["name"],
                page_number=row["page_number"],
                section_number=row["section_number"],
                text=row["text"],
            )
            for row in rows
        }
