from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant_app.ingest import build_chunk_records
from assistant_app.metadata import MetadataStore


class IngestMetadataTests(unittest.TestCase):
    def test_text_document_chunks_include_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            document = Path(tmp) / "policy.txt"
            document.write_text("alpha beta gamma " * 80, encoding="utf-8")

            records = build_chunk_records(document, chunk_size=120, overlap=20)

            self.assertGreater(len(records), 1)
            self.assertEqual(records[0].document_name, "policy.txt")
            self.assertIsNone(records[0].page_number)
            self.assertEqual(records[0].section_number, 1)
            self.assertTrue(records[0].text)

    def test_sqlite_store_round_trips_chunk_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            document = Path(tmp) / "manual.txt"
            document.write_text("offline assistant manual", encoding="utf-8")
            records = build_chunk_records(document, chunk_size=200, overlap=20)

            store = MetadataStore(str(Path(tmp) / "assistant.db"))
            store.initialize()
            document_id = store.upsert_document(document)
            store.insert_chunks(document_id, records)

            loaded = store.get_chunks([records[0].chunk_id])

            self.assertIn(records[0].chunk_id, loaded)
            self.assertEqual(loaded[records[0].chunk_id].document_name, "manual.txt")
            self.assertEqual(loaded[records[0].chunk_id].text, records[0].text)

    def test_store_returns_chunk_ids_and_deletes_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            document = Path(tmp) / "manual.txt"
            document.write_text("offline assistant manual", encoding="utf-8")
            records = build_chunk_records(document, chunk_size=200, overlap=20)

            store = MetadataStore(str(Path(tmp) / "assistant.db"))
            store.initialize()
            document_id = store.upsert_document(document)
            store.insert_chunks(document_id, records)

            self.assertEqual(store.get_document_id(document), document_id)
            self.assertEqual(store.get_chunk_ids_for_document(document_id), [records[0].chunk_id])

            store.delete_document(document_id)

            self.assertIsNone(store.get_document_id(document))
            self.assertEqual(store.get_chunks([records[0].chunk_id]), {})


if __name__ == "__main__":
    unittest.main()
