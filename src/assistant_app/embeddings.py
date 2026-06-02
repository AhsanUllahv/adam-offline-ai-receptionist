from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class LocalSentenceTransformerEmbeddingFunction:
    """Chroma-compatible embedding function backed by a local model folder."""

    model_path: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        path = Path(self.model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Embedding model not found at {path}. "
                "Place a local sentence-transformers model there or set "
                "ASSISTANT_EMBEDDING_MODEL to the correct local path."
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install -e '.[documents]'"
            ) from exc

        try:
            self._model = SentenceTransformer(
                str(path), device=self.device, local_files_only=True
            )
        except TypeError:
            self._model = SentenceTransformer(str(path), device=self.device)

    def name(self) -> str:
        return "local-sentence-transformer"

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(list(input))

    def embed_documents(
        self,
        documents: Sequence[str] | None = None,
        input: Sequence[str] | None = None,
    ) -> list[list[float]]:
        texts = documents if documents is not None else input
        if texts is None:
            texts = []
        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(
        self,
        query: str | Sequence[str] | None = None,
        input: str | Sequence[str] | None = None,
    ) -> list[float] | list[list[float]]:
        text = query if query is not None else input
        if text is None:
            text = ""
        if isinstance(text, str):
            return self.embed_documents([text])[0]
        return self.embed_documents(text)


def build_embedding_function(model_path: str, device: str = "cpu"):
    return LocalSentenceTransformerEmbeddingFunction(model_path=model_path, device=device)
