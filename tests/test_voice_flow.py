from __future__ import annotations

import unittest

from assistant_app.eyes import EyeController
from assistant_app.llm import LocalLLM
from assistant_app.pipeline import AssistantPipeline, _has_relevant_document_context, _normalize_voice_query_text
from assistant_app.retrieval import DocumentRetriever
from assistant_app.state import ResponseStateManager
from assistant_app.stt import SpeechToText
from assistant_app.tts import TextToSpeech
from assistant_app.wakeword import WakeWordDetector, _keyword_detected


class SilentEyes(EyeController):
    def set_expression(self, expression: str) -> None:
        pass


class CapturingTTS(TextToSpeech):
    def __init__(self) -> None:
        self.messages: list[str] = []

    def say(self, text: str) -> None:
        self.messages.append(text)


class ScriptedVoiceSTT(SpeechToText):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.timeouts: list[float | None] = []

    def listen(self, timeout_seconds: float | None = None) -> str:
        self.timeouts.append(timeout_seconds)
        return self.responses.pop(0)


class CountingWakeDetector(WakeWordDetector):
    def __init__(self) -> None:
        self.calls = 0

    def wait_for_wake_word(self) -> None:
        self.calls += 1


class EmptyRetriever(DocumentRetriever):
    def search(self, query: str, limit: int = 4) -> list[str]:
        return []


class IrrelevantRetriever(DocumentRetriever):
    def search(self, query: str, limit: int = 4) -> list[str]:
        return ["[robot_manual.pdf]\nServo calibration and display wiring notes."]


class RelevantRetriever(DocumentRetriever):
    def __init__(self) -> None:
        self.limits: list[int] = []

    def search(self, query: str, limit: int = 4) -> list[str]:
        self.limits.append(limit)
        return [
            "[robot.pdf, page 1]\n"
            "Design intent: The robot should look friendly. "
            "The head and body are white, the face is a smooth black visor with two blue eyes, "
            "and the lower base is black with a blue accent groove."
        ]


class ExplainRetriever(DocumentRetriever):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, limit: int = 4) -> list[str]:
        self.queries.append(query)
        return [
            "[Thesis.pdf, page 15, section 41]\n"
            "Voice-to-text Converting voice into words is the main pillar of this system. "
            "The human voice is converted into digital form, sound features are separated, and these features are matched with existing patterns so the machine can understand spoken questions."
        ]


class TinyLLM(LocalLLM):
    def __init__(self) -> None:
        self.document_calls = 0
        self.general_calls = 0

    def stream_answer(self, question: str, context: str):
        self.document_calls += 1
        yield "Document answer."

    def stream_general_answer(self, question: str):
        self.general_calls += 1
        yield "General answer."

    def stream_voice_answer(self, question: str, context: str):
        self.document_calls += 1
        yield "Voice document answer."

    def stream_voice_general_answer(self, question: str):
        self.general_calls += 1
        yield "Voice general answer."


class VoiceFlowTests(unittest.TestCase):
    def test_keyword_matching_accepts_common_vosk_variants(self) -> None:
        self.assertTrue(_keyword_detected("adam", "adam"))
        self.assertTrue(_keyword_detected("hey atom", "adam", ("atom",)))
        self.assertTrue(_keyword_detected("please add him", "adam", ("add him",)))
        self.assertFalse(_keyword_detected("hello assistant", "adam"))

    def test_voice_loop_keeps_followup_window_without_second_wake_word(self) -> None:
        stt = ScriptedVoiceSTT(["first question", "exit"])
        wake = CountingWakeDetector()
        tts = CapturingTTS()
        pipeline = AssistantPipeline(
            state_manager=ResponseStateManager(tts=tts, eyes=SilentEyes()),
            stt=stt,
            retriever=EmptyRetriever(),
            llm=TinyLLM(),
            answer_tts=tts,
            wakeword_detector=wake,
            no_rag=True,
            continuous_conversation=True,
            followup_timeout_seconds=3.5,
        )

        pipeline.run_text_demo()

        self.assertEqual(wake.calls, 1)
        self.assertEqual(stt.timeouts, [None, 3.5])
        self.assertIn("Voice general answer.", tts.messages)

    def test_voice_uses_general_answer_when_documents_are_irrelevant(self) -> None:
        tts = CapturingTTS()
        llm = TinyLLM()
        pipeline = AssistantPipeline(
            state_manager=ResponseStateManager(tts=tts, eyes=SilentEyes()),
            stt=ScriptedVoiceSTT([]),
            retriever=IrrelevantRetriever(),
            llm=llm,
            answer_tts=tts,
        )
        pipeline._is_voice = True

        answer = pipeline.answer("who is the founder of usa")

        self.assertEqual(answer, "Voice general answer.")
        self.assertEqual(llm.general_calls, 1)
        self.assertEqual(llm.document_calls, 0)

    def test_voice_fast_mode_uses_high_confidence_fact_without_llm(self) -> None:
        tts = CapturingTTS()
        llm = TinyLLM()
        retriever = RelevantRetriever()
        pipeline = AssistantPipeline(
            state_manager=ResponseStateManager(tts=tts, eyes=SilentEyes()),
            stt=ScriptedVoiceSTT([]),
            retriever=retriever,
            llm=llm,
            answer_tts=tts,
            voice_answer_mode="fast",
            voice_retrieval_limit=1,
        )
        pipeline._is_voice = True

        answer = pipeline.answer("what is the color of this robot")

        self.assertIn("white", answer.lower())
        self.assertIn("black", answer.lower())
        self.assertNotIn("From robot.pdf", answer)
        self.assertEqual(retriever.limits, [1])
        self.assertEqual(llm.document_calls, 0)
        self.assertEqual(llm.general_calls, 0)

    def test_voice_fast_mode_extracts_head_section_fact(self) -> None:
        tts = CapturingTTS()
        llm = TinyLLM()
        pipeline = AssistantPipeline(
            state_manager=ResponseStateManager(tts=tts, eyes=SilentEyes()),
            stt=ScriptedVoiceSTT([]),
            retriever=EmptyRetriever(),
            llm=llm,
            answer_tts=tts,
            voice_answer_mode="fast",
        )
        answer = pipeline._build_fast_voice_answer(
            "which board am i using in the head of robot",
            [
                "[For_designer_.pdf, page 5, section 11]\n"
                "Robot 3D Design Brief - Sectioned for Designer Page 5 5. "
                "Inner-design area What it tells the designer "
                "Top section / head Only the two eye displays and the centered ESP32-S3 controller should stay in the head area. No battery should be added there. "
                "Middle section Two side speakers should stay left and right, while the audio board sits centrally above the Jetson section."
            ],
        )

        self.assertIn("ESP32-S3", answer)
        self.assertNotIn("From For_designer_", answer)
        self.assertNotIn("Robot 3D Design Brief", answer)
        self.assertEqual(llm.document_calls, 0)

    def test_voice_fast_mode_extracts_robot_color_fact(self) -> None:
        tts = CapturingTTS()
        llm = TinyLLM()
        pipeline = AssistantPipeline(
            state_manager=ResponseStateManager(tts=tts, eyes=SilentEyes()),
            stt=ScriptedVoiceSTT([]),
            retriever=EmptyRetriever(),
            llm=llm,
            answer_tts=tts,
            voice_answer_mode="fast",
        )
        contexts = [
            "[For_designer_.pdf, page 1, section 1]\n"
            "Design intent: The robot should look like a small, friendly, modern receptionist robot. "
            "Keep the form soft and rounded. The head and body are white, the face is a smooth black visor with two blue eyes, "
            "and the lower base is black with a blue accent groove. Keep the front body area clean for branding."
        ]

        self.assertTrue(_has_relevant_document_context("what is the color of this robot", contexts))
        answer = pipeline._build_fast_voice_answer("what is the color of this robot", contexts)

        self.assertIn("white", answer.lower())
        self.assertIn("black", answer.lower())
        self.assertIn("blue", answer.lower())
        self.assertNotIn("From For_designer_", answer)

    def test_color_question_about_flag_does_not_use_robot_color_context(self) -> None:
        contexts = [
            "[For_designer_.pdf, page 1, section 1]\n"
            "The head and body are white, the face is a smooth black visor with two blue eyes, "
            "and the lower base is black with a blue accent groove."
        ]

        self.assertFalse(
            _has_relevant_document_context("what does that color of pakistani flag", contexts)
        )


    def test_voice_fast_mode_falls_back_to_llm_for_weak_extract(self) -> None:
        tts = CapturingTTS()
        llm = TinyLLM()
        pipeline = AssistantPipeline(
            state_manager=ResponseStateManager(tts=tts, eyes=SilentEyes()),
            stt=ScriptedVoiceSTT([]),
            retriever=ExplainRetriever(),
            llm=llm,
            answer_tts=tts,
            voice_answer_mode="fast",
        )
        pipeline._is_voice = True

        answer = pipeline.answer("what is the voice to text part in the thesis")

        self.assertEqual(answer, "Voice document answer.")
        self.assertEqual(llm.document_calls, 1)


    def test_voice_explain_question_uses_compact_llm_not_fast_extract(self) -> None:
        tts = CapturingTTS()
        llm = TinyLLM()
        retriever = ExplainRetriever()
        pipeline = AssistantPipeline(
            state_manager=ResponseStateManager(tts=tts, eyes=SilentEyes()),
            stt=ScriptedVoiceSTT([]),
            retriever=retriever,
            llm=llm,
            answer_tts=tts,
            voice_answer_mode="fast",
        )
        pipeline._is_voice = True

        answer = pipeline.answer("can you explain the voice to tax part in our thesis")

        self.assertEqual(answer, "Voice document answer.")
        self.assertEqual(llm.document_calls, 1)
        self.assertIn("voice to text", retriever.queries[0].lower())

    def test_voice_query_normalizes_common_stt_mishearings(self) -> None:
        self.assertEqual(
            _normalize_voice_query_text("explain the voice to tax part"),
            "explain the voice to text part",
        )
        self.assertEqual(
            _normalize_voice_query_text("problem and system overview of the pieces"),
            "problem and system overview of the thesis",
        )


if __name__ == "__main__":
    unittest.main()
