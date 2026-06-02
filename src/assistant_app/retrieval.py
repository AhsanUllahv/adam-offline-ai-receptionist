from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from assistant_app.config import AssistantConfig
from assistant_app.embeddings import build_embedding_function
from assistant_app.metadata import MetadataStore


class DocumentRetriever(ABC):
    @abstractmethod
    def search(self, query: str, limit: int = 4) -> list[str]: ...


@dataclass
class ChromaRetriever(DocumentRetriever):
    path: str
    collection_name: str
    sqlite_path: str
    embedding_model_path: str
    embedding_device: str = "cpu"
    _embedding_function: object = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: AssistantConfig) -> "ChromaRetriever":
        return cls(
            path=config.chroma_path,
            collection_name=config.chroma_collection,
            sqlite_path=config.sqlite_path,
            embedding_model_path=config.embedding_model_path,
            embedding_device=config.embedding_device,
        )

    def _get_embedding_function(self) -> object:
        if self._embedding_function is None:
            self._embedding_function = build_embedding_function(
                self.embedding_model_path, self.embedding_device
            )
        return self._embedding_function

    def search(self, query: str, limit: int = 4) -> list[str]:
        try:
            import chromadb
        except ImportError:
            return []

        try:
            embedding_function = self._get_embedding_function()
        except (FileNotFoundError, RuntimeError):
            return []

        client = chromadb.PersistentClient(path=self.path)
        try:
            collection = client.get_collection(
                self.collection_name, embedding_function=embedding_function
            )
        except Exception:
            return []

        results = collection.query(query_texts=[query], n_results=limit)
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        store = MetadataStore(self.sqlite_path)
        records = store.get_chunks(ids)

        contexts: list[str] = []
        for index, chunk_id in enumerate(ids):
            record = records.get(chunk_id)
            if record:
                source = format_source(
                    record.document_name, record.page_number, record.section_number
                )
                contexts.append(f"[{source}]\n{record.text}")
                continue

            document = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            source = format_source(
                str(metadata.get("document_name") or metadata.get("source") or "document"),
                metadata.get("page_number"),
                metadata.get("section_number", index + 1),
            )
            if document:
                contexts.append(f"[{source}]\n{document}")

        return contexts


def format_source(
    document_name: str,
    page_number: int | None,
    section_number: int | None,
) -> str:
    parts = [document_name]
    if page_number:
        parts.append(f"page {page_number}")
    if section_number:
        parts.append(f"section {section_number}")
    return ", ".join(parts)
