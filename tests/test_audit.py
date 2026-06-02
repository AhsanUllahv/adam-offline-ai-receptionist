from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from assistant_app.audit import AuditStore


class AuditStoreTests(unittest.TestCase):
    def test_logs_interactions_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AuditStore(str(Path(tmp) / "assistant.db"))

            interaction_id = store.log_interaction(
                question="What is indexed?",
                answer="A manual.",
                sources=("manual.txt, section 1",),
                answer_mode="llm",
                model="llama3.2:1b",
                retrieval_seconds=0.2,
                generation_seconds=1.1,
            )
            event_id = store.log_event("upload", "Uploaded manual.txt", {"file": "manual.txt"})

            interactions = store.recent_interactions()
            events = store.recent_events()

            self.assertEqual(interaction_id, 1)
            self.assertEqual(event_id, 1)
            self.assertEqual(store.interaction_count(), 1)
            self.assertEqual(store.event_count(), 1)
            self.assertEqual(interactions[0].question, "What is indexed?")
            self.assertEqual(interactions[0].sources, ("manual.txt, section 1",))
            self.assertEqual(events[0].details["file"], "manual.txt")


if __name__ == "__main__":
    unittest.main()
