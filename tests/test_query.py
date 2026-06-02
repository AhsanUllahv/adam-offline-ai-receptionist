from __future__ import annotations

import unittest

from assistant_app.query import Intent, QueryProcessor


class QueryProcessorTests(unittest.TestCase):
    def test_empty_and_exit_intents(self) -> None:
        processor = QueryProcessor()

        self.assertEqual(processor.process("   ").intent, Intent.EMPTY)
        self.assertEqual(processor.process("exit").intent, Intent.EXIT)

    def test_document_question_is_normalized(self) -> None:
        processed = QueryProcessor().process("  What   is   the policy? ")

        self.assertEqual(processed.intent, Intent.DOCUMENT_QUERY)
        self.assertEqual(processed.text, "What is the policy?")

    def test_followup_question_uses_last_question(self) -> None:
        processor = QueryProcessor()
        processor.remember("What is the admission policy?", "It is in the document.")

        processed = processor.process("what about fees?")

        self.assertIn("Previous question: What is the admission policy?", processed.text)
        self.assertIn("Follow-up question: what about fees?", processed.text)


if __name__ == "__main__":
    unittest.main()
