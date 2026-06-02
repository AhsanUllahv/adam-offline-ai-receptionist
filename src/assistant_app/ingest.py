from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from assistant_app.config import AssistantConfig
from assistant_app.embeddings import build_embedding_function
from assistant_app.metadata import ChunkRecord, MetadataStore


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class TextBlock:
    text: str
    page_number: int | None
    section_number: int


@dataclass(frozen=True)
class IngestResult:
    documents_found: int
    documents_indexed: int
    chunks_indexed: int
    skipped: tuple[str, ...]
    sqlite_path: str
    chroma_path: str
    collection_name: str


def ingest_path(
    path: Path,
    config: AssistantConfig,
    chunk_size: int = 900,
    overlap: int = 150,
) -> IngestResult:
    documents = iter_documents(path)
    if not documents:
        raise ValueError("No supported documents found.")

    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "ChromaDB is required for ingestion. Install with: pip install -e '.[documents]'"
        ) from exc

    metadata_store = MetadataStore(config.sqlite_path)
    metadata_store.initialize()

    embedding_function = build_embedding_function(
        config.embedding_model_path, config.embedding_device
    )
    client = chromadb.PersistentClient(path=config.chroma_path)
    collection = client.get_or_create_collection(
        config.chroma_collection, embedding_function=embedding_function
    )

    total_chunks = 0
    indexed_documents = 0
    skipped: list[str] = []
    for document_path in documents:
        records = build_chunk_records(document_path, chunk_size, overlap)
        if not records:
            skipped.append(str(document_path))
            continue

        document_id = metadata_store.upsert_document(document_path)
        old_chunk_ids = metadata_store.get_chunk_ids_for_document(document_id)
        delete_chroma_ids(collection, old_chunk_ids)
        metadata_store.delete_chunks_for_document(document_id)
        metadata_store.insert_chunks(document_id, records)

        collection.upsert(
            ids=[record.chunk_id for record in records],
            documents=[record.text for record in records],
            metadatas=[
                {
                    "source": record.document_path,
                    "document_name": record.document_name,
                    "page_number": record.page_number or 0,
                    "section_number": record.section_number,
                }
                for record in records
            ],
        )
        total_chunks += len(records)
        indexed_documents += 1

    if total_chunks == 0:
        raise ValueError("Documents were found, but no text could be extracted.")

    return IngestResult(
        documents_found=len(documents),
        documents_indexed=indexed_documents,
        chunks_indexed=total_chunks,
        skipped=tuple(skipped),
        sqlite_path=config.sqlite_path,
        chroma_path=config.chroma_path,
        collection_name=config.chroma_collection,
    )


def remove_document_from_index(path: Path, config: AssistantConfig) -> int:
    metadata_store = MetadataStore(config.sqlite_path)
    metadata_store.initialize()
    document_id = metadata_store.get_document_id(path)
    if document_id is None:
        return 0

    old_chunk_ids = metadata_store.get_chunk_ids_for_document(document_id)
    if old_chunk_ids:
        try:
            import chromadb
        except ImportError:
            pass
        else:
            client = chromadb.PersistentClient(path=config.chroma_path)
            try:
                collection = client.get_collection(config.chroma_collection)
            except Exception:
                collection = None
            if collection is not None:
                delete_chroma_ids(collection, old_chunk_ids)

    metadata_store.delete_document(document_id)
    return len(old_chunk_ids)


def delete_chroma_ids(collection: object, chunk_ids: list[str]) -> None:
    if not chunk_ids:
        return
    try:
        collection.delete(ids=chunk_ids)  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preload documents into SQLite metadata and ChromaDB"
    )
    parser.add_argument("path", help="Document file or directory")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    config = AssistantConfig.from_env()
    try:
        result = ingest_path(Path(args.path), config, args.chunk_size, args.overlap)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for skipped in result.skipped:
        print(f"Skipped {skipped}: no extractable text")
    print(
        f"Indexed {result.chunks_indexed} chunks from "
        f"{result.documents_indexed}/{result.documents_found} documents."
    )
    print(f"SQLite metadata: {result.sqlite_path}")
    print(f"Chroma path: {result.chroma_path}")
    print(f"Collection: {result.collection_name}")


def iter_documents(path: Path) -> list[Path]:
    path = path.expanduser()
    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(
            child
            for child in path.rglob("*")
            if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return []


def build_chunk_records(path: Path, chunk_size: int, overlap: int) -> list[ChunkRecord]:
    blocks = extract_text_blocks(path)
    records: list[ChunkRecord] = []
    section_number = 0

    for block in blocks:
        for text in chunk_text(block.text, chunk_size, overlap):
            section_number += 1
            chunk_id = stable_id(path, block.page_number, section_number, text)
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_path=str(path),
                    document_name=path.name,
                    page_number=block.page_number,
                    section_number=section_number,
                    text=text,
                )
            )

    return records


def extract_text_blocks(path: Path) -> list[TextBlock]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_blocks(path)
    return [TextBlock(text=path.read_text(encoding="utf-8", errors="ignore"), page_number=None, section_number=1)]


def extract_pdf_blocks(path: Path) -> list[TextBlock]:
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is required for PDF ingestion. Install with: pip install -e '.[documents]'"
        ) from exc

    blocks: list[TextBlock] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text")
            if text.strip():
                blocks.append(
                    TextBlock(
                        text=text,
                        page_number=page_index,
                        section_number=page_index,
                    )
                )
    return blocks


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(cleaned):
        chunk = cleaned[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def stable_id(path: Path, page_number: int | None, section_number: int, chunk: str) -> str:
    payload = f"{path.resolve()}:{page_number}:{section_number}:{chunk}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
