from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant_app.embeddings import build_embedding_function


class EmbeddingConfigTests(unittest.TestCase):
    def test_missing_local_embedding_model_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-model"
            with self.assertRaises(FileNotFoundError) as context:
                build_embedding_function(str(missing))

        self.assertIn("Embedding model not found", str(context.exception))


if __name__ == "__main__":
    unittest.main()
