from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

from assistant_app.audit import EventRecord, InteractionRecord, LatencyStats, SourceUsageRecord
from assistant_app.dashboard import (
    DashboardAnswer,
    IndexHealth,
    ModelStatus,
    SystemResources,
    add_latency_summary,
    answer_dashboard_question,
    build_extractive_answer,
    build_llm_context,
    extract_source_labels,
    get_dashboard_services,
    render_ask_page,
    render_history_items,
    render_monitor_page,
    has_relevant_document_context,
    public_sources,
    rewrite_followup_question,
    simple_receptionist_answer,
    needs_official_knowledge_answer,
    official_knowledge_missing_answer,
    rerank_contexts,
    reset_dashboard_services,
    safe_filename,
    unique_path,
)


class DashboardTests(unittest.TestCase):
    def test_safe_filename_removes_paths_and_unsafe_chars(self) -> None:
        self.assertEqual(safe_filename("../My Policy!.pdf"), "My_Policy.pdf")

    def test_unique_path_adds_suffix_when_file_exists(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manual.txt"
            path.write_text("one", encoding="utf-8")

            self.assertEqual(unique_path(path), Path(tmp) / "manual_1.txt")

    def test_delete_removes_file_and_index_entry(self) -> None:
        import tempfile
        from assistant_app import dashboard

        with tempfile.TemporaryDirectory() as tmp:
            documents_dir = Path(tmp) / "documents"
            documents_dir.mkdir()
            target = documents_dir / "manual.txt"
            target.write_text("hello", encoding="utf-8")

            with patch.object(dashboard, "DOCUMENTS_DIR", documents_dir), patch(
                "assistant_app.dashboard.remove_document_from_index"
            ) as remove_from_index:
                response = dashboard.delete("manual.txt")

            self.assertEqual(response.status_code, 303)
            self.assertFalse(target.exists())
            remove_from_index.assert_called_once()

    def test_extract_source_labels_from_contexts(self) -> None:
        contexts = [
            "[manual.txt, section 1]\nHow does it run? The system runs offline.",
            "[policy.pdf, page 2, section 5]\nPrivacy text.",
        ]

        self.assertEqual(
            extract_source_labels(contexts),
            ["manual.txt, section 1", "policy.pdf, page 2, section 5"],
        )

    def test_dashboard_question_uses_llm_answer_by_default(self) -> None:
        from assistant_app.config import AssistantConfig

        class FakeRetriever:
            def search(self, query, limit=4):
                return ["[manual.txt, section 1]\nHow does it run? The system runs offline."]

        class FakeLLM:
            def stream_answer(self, question, context):
                yield "The system runs offline."

        class FakeServices:
            retriever = FakeRetriever()
            llm = FakeLLM()

            def __init__(self) -> None:
                from assistant_app.validation import GroundedAnswerValidator

                self.validator = GroundedAnswerValidator()

        with patch("assistant_app.dashboard.get_dashboard_services", return_value=FakeServices()):
            result = answer_dashboard_question("How does it run?", AssistantConfig())

        self.assertEqual(result.answer, "The system runs offline.")
        self.assertEqual(result.sources, ("manual.txt, section 1",))

    def test_dashboard_question_uses_extractive_answer_when_enabled(self) -> None:
        from assistant_app.config import AssistantConfig

        class FakeRetriever:
            def search(self, query, limit=4):
                return ["[manual.txt, section 1]\nHow does it run? The system runs offline."]

        class FakeLLM:
            def stream_answer(self, question, context):
                raise AssertionError("LLM should not be called in extractive mode")

        class FakeServices:
            retriever = FakeRetriever()
            llm = FakeLLM()

            def __init__(self) -> None:
                from assistant_app.validation import GroundedAnswerValidator

                self.validator = GroundedAnswerValidator()

        config = AssistantConfig(dashboard_answer_mode="extractive")
        with patch("assistant_app.dashboard.get_dashboard_services", return_value=FakeServices()):
            result = answer_dashboard_question("How does it run?", config)

        self.assertIn("I found this in your documents", result.answer)
        self.assertIn("The system runs offline.", result.answer)

    def test_dashboard_question_can_use_llm_mode_when_enabled(self) -> None:
        from assistant_app.config import AssistantConfig

        class FakeRetriever:
            def search(self, query, limit=4):
                return ["[manual.txt, section 1]\nHow does it run? The system runs offline."]

        class FakeLLM:
            def stream_answer(self, question, context):
                yield "LLM answer."

        class FakeServices:
            retriever = FakeRetriever()
            llm = FakeLLM()

            def __init__(self) -> None:
                from assistant_app.validation import GroundedAnswerValidator

                self.validator = GroundedAnswerValidator()

        config = AssistantConfig(dashboard_answer_mode="llm", dashboard_llm_retrieval_limit=1)
        with patch("assistant_app.dashboard.get_dashboard_services", return_value=FakeServices()):
            result = answer_dashboard_question("How does it run?", config)

        self.assertEqual(result.answer, "LLM answer.")



    def test_specific_institution_question_uses_safe_receptionist_guard(self) -> None:
        self.assertTrue(needs_official_knowledge_answer("i mean VBC college"))
        answer = official_knowledge_missing_answer("i mean VBC college")
        self.assertIn("do not have official local knowledge", answer)
        self.assertIn("uploaded documents", answer)

    def test_simple_receptionist_answer_avoids_model_for_greetings(self) -> None:
        self.assertEqual(simple_receptionist_answer("hi"), "Hello. How can I help you today?")
        self.assertIn("ready to help", simple_receptionist_answer("how are you") or "")
        self.assertIsNone(simple_receptionist_answer("what is robo eyes"))

    def test_rewrite_followup_question_adds_previous_topic(self) -> None:
        previous = InteractionRecord(
            id=1,
            created_at="2026-05-28 10:00:00",
            question="Which board are we using for robo eyes?",
            answer="ESP32-S3",
            sources=("manual.txt, section 1",),
            answer_mode="llm",
            model="llama3.2:1b",
            retrieval_seconds=0.1,
            generation_seconds=0.2,
            total_seconds=0.3,
        )

        rewritten = rewrite_followup_question("what are the specs", previous)

        self.assertIn("what are the specs", rewritten)
        self.assertIn("Which board are we using for robo eyes?", rewritten)

    def test_public_sources_limits_visible_source_chips(self) -> None:
        contexts = [
            "[one.pdf, page 1]\nA",
            "[two.pdf, page 2]\nB",
            "[three.pdf, page 3]\nC",
        ]

        self.assertEqual(public_sources(contexts), ("one.pdf, page 1", "two.pdf, page 2"))

    def test_relevance_gate_rejects_weak_acronym_only_match(self) -> None:
        contexts = [
            "[manual.pdf, page 5]\nVBC means visual board connector in this wiring diagram."
        ]

        self.assertFalse(has_relevant_document_context("VBC college", contexts))

    def test_relevance_gate_accepts_specific_document_match(self) -> None:
        contexts = [
            "[eyes.pdf, page 2]\nThe robo eyes use circular IPS displays and ESP32-S3."
        ]

        self.assertTrue(has_relevant_document_context("what is design of robo eyes", contexts))

    def test_rerank_contexts_prefers_keyword_overlap(self) -> None:
        contexts = [
            "[general.pdf, page 1]\nRobot enclosure and general design notes.",
            "[eyes.pdf, page 2]\nThe robo eyes use circular IPS displays and ESP32-S3 behind the visor.",
        ]

        ranked = rerank_contexts("what is design of robo eyes", contexts)

        self.assertTrue(ranked[0].startswith("[eyes.pdf"))

    def test_build_llm_context_limits_context_size(self) -> None:
        contexts = [
            "[manual.txt, section 1]\n" + "alpha " * 200,
            "[manual.txt, section 2]\n" + "beta " * 200,
        ]

        compact = build_llm_context(contexts, max_chars=260)

        self.assertIn("[manual.txt, section 1]", compact)
        self.assertLessEqual(len(compact), 340)

    def test_build_extractive_answer_uses_first_source_excerpt(self) -> None:
        answer = build_extractive_answer(["[manual.txt, section 1]\nAlpha beta gamma."])

        self.assertIn("Alpha beta gamma.", answer)
        self.assertIn("Source: manual.txt, section 1", answer)

    def test_ask_page_logs_and_renders_history(self) -> None:
        from assistant_app import dashboard

        answer = DashboardAnswer(
            answer="Private answer",
            sources=("manual.txt, section 1",),
            retrieval_seconds=0.1,
        )

        class FakeStore:
            def log_interaction(self, **kwargs):
                self.kwargs = kwargs
                return 1

            def recent_interactions(self, limit=20):
                return [
                    InteractionRecord(
                        id=1,
                        created_at="2026-05-28 10:00:00",
                        question="What is private?",
                        answer="Private answer",
                        sources=("manual.txt, section 1",),
                        answer_mode="llm",
                        model="llama3.2:1b",
                        retrieval_seconds=0.1,
                        generation_seconds=0.2,
                        total_seconds=0.3,
                    )
                ]

        store = FakeStore()
        with patch("assistant_app.dashboard.answer_dashboard_question", return_value=answer), patch(
            "assistant_app.dashboard.audit_store", return_value=store
        ):
            posted = dashboard.ask("What is private?")

        self.assertIn("New chat", posted)
        self.assertIn("/ask/stream", posted)
        self.assertEqual(store.kwargs["question"], "What is private?")

    def test_render_history_items_outputs_records(self) -> None:
        html = render_history_items([
            InteractionRecord(
                id=1,
                created_at="2026-05-28 10:00:00",
                question="Question?",
                answer="Answer.",
                sources=("manual.txt, section 1",),
                answer_mode="llm",
                model="llama3.2:1b",
                retrieval_seconds=0.1,
                generation_seconds=0.2,
                total_seconds=0.3,
            )
        ])

        self.assertIn("Question?", html)
        self.assertIn("Answer.", html)
        self.assertIn("llama3.2:1b", html)

    def test_render_monitor_page_outputs_metrics(self) -> None:
        from assistant_app.config import AssistantConfig

        html = render_monitor_page(
            interactions=[
                InteractionRecord(
                    id=1,
                    created_at="2026-05-28 10:00:00",
                    question="Question?",
                    answer="Answer.",
                    sources=("manual.txt, section 1",),
                    answer_mode="llm",
                    model="llama3.2:1b",
                    retrieval_seconds=0.1,
                    generation_seconds=0.2,
                    total_seconds=0.3,
                )
            ],
            events=[
                EventRecord(
                    id=1,
                    created_at="2026-05-28 10:01:00",
                    event_type="upload",
                    message="Uploaded manual.txt",
                    details={"file": "manual.txt"},
                )
            ],
            error_events=[],
            latency_stats=LatencyStats(
                total_count=1,
                average_total=0.3,
                average_retrieval=0.1,
                average_generation=0.2,
                slowest_question="Question?",
                slowest_total=0.3,
                last_total=0.3,
            ),
            source_usage=[SourceUsageRecord("manual.txt", 1)],
            index_health=IndexHealth(
                uploaded_file_count=1,
                indexed_document_count=1,
                chunk_count=4,
                last_indexed_at="2026-05-28 10:00:00",
                unindexed_files=(),
                stale_files=(),
            ),
            model_status=ModelStatus(
                reachable=True,
                model="llama3.2:1b",
                url="http://127.0.0.1:11434",
                answer_mode="llm",
                message="reachable",
            ),
            system_resources=SystemResources(
                load_1m=0.2,
                memory_used_mb=100,
                memory_total_mb=1000,
                memory_percent=10.0,
                disk_used="1 GB",
                disk_total="10 GB",
                disk_percent=10.0,
                temperature_c=45.0,
            ),
            interaction_count=1,
            event_count=1,
            config=AssistantConfig(ollama_model="llama3.2:1b"),
        )

        self.assertIn("Assistant Monitor", html)
        self.assertIn('class="ops-shell"', html)
        self.assertIn("One-page operations view", html)
        self.assertIn("Index And Runtime", html)
        self.assertIn("Model And Sources", html)
        self.assertIn("Error Log", html)
        self.assertIn("Chat History", html)
        self.assertIn("System Events", html)
        self.assertIn('id="chat-filter"', html)
        self.assertIn('id="event-filter"', html)
        self.assertIn('class="table-scroll tall"', html)
        self.assertIn("Uploaded manual.txt", html)

    def test_render_ask_page_uses_chat_layout(self) -> None:
        html = render_ask_page(
            "What is local search?",
            "It searches local documents.",
            ("manual.txt, section 1",),
            "",
        )

        self.assertIn('class="app-shell"', html)
        self.assertIn('class="composer"', html)
        self.assertIn('id="new-chat-button"', html)
        self.assertIn('href="/ask?new=1"', html)
        self.assertIn('id="stop-chat"', html)
        self.assertIn('id="sessions"', html)
        self.assertIn('/ask/stream', html)
        self.assertIn("body { margin: 0; height: 100vh; overflow: hidden;", html)
        self.assertIn(".sidebar { height: 100vh; overflow: hidden;", html)
        self.assertIn("const chatWrap = document.querySelector", html)
        self.assertIn("if (activeController) activeController.abort();", html)
        self.assertIn("typeof globalThis.crypto.randomUUID", html)
        self.assertIn("Date.now().toString(36)", html)
        self.assertIn('const forceNewChat = new URLSearchParams', html)
        self.assertIn('window.history.replaceState(null, "", "/ask")', html)
        self.assertIn("stopChatButton.addEventListener", html)
        self.assertIn("activeController?.abort()", html)
        self.assertIn("Press Enter to send", html)
        self.assertIn("Answer mode:", html)
        self.assertIn("llm", html)

    def test_dashboard_services_are_cached_by_config(self) -> None:
        from assistant_app.config import AssistantConfig

        class FakeRetriever:
            @classmethod
            def from_config(cls, config):
                return cls()

        class FakeLLM:
            @classmethod
            def from_config(cls, config):
                return cls()

        reset_dashboard_services()
        with patch("assistant_app.dashboard.ChromaRetriever", FakeRetriever), patch(
            "assistant_app.dashboard.OllamaLLM", FakeLLM
        ):
            first = get_dashboard_services(AssistantConfig(ollama_model="one"))
            second = get_dashboard_services(AssistantConfig(ollama_model="one"))
            third = get_dashboard_services(AssistantConfig(ollama_model="two"))

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        reset_dashboard_services()

    def test_latency_summary_is_added_when_timing_exists(self) -> None:
        result = DashboardAnswer(
            answer="Done.",
            sources=(),
            retrieval_seconds=0.25,
            generation_seconds=1.50,
        )

        summary = add_latency_summary(result)

        self.assertIn("Done.", summary)
        self.assertIn("Latency: 1.75s", summary)
        self.assertIn("retrieval 0.25s", summary)
        self.assertIn("generation 1.50s", summary)

    def test_latency_summary_for_extractive_mode(self) -> None:
        result = DashboardAnswer(
            answer="Done.",
            sources=(),
            retrieval_seconds=0.25,
            generation_seconds=0.0,
        )

        summary = add_latency_summary(result)

        self.assertIn("Latency: 0.25s", summary)
        self.assertIn("retrieval only", summary)


if __name__ == "__main__":
    unittest.main()
