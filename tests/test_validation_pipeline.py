from __future__ import annotations

import unittest

from assistant_app.eyes import EyeController
from assistant_app.llm import LocalLLM
from assistant_app.pipeline import AssistantPipeline
from assistant_app.retrieval import DocumentRetriever
from assistant_app.state import ResponseStateManager
from assistant_app.stt import SpeechToText
from assistant_app.tts import TextToSpeech
from assistant_app.validation import GroundedAnswerValidator


class SilentEyes(EyeController):
    def set_expression(self, expression: str) -> None:
        pass


class CapturingTTS(TextToSpeech):
    def __init__(self) -> None:
        self.messages: list[str] = []

    def say(self, text: str) -> None:
        self.messages.append(text)


class StaticSTT(SpeechToText):
    def listen(self) -> str:
        return ""


class EmptyRetriever(DocumentRetriever):
    def search(self, query: str, limit: int = 4) -> list[str]:
        return []


class StaticRetriever(DocumentRetriever):
    def search(self, query: str, limit: int = 4) -> list[str]:
        return ["[manual.txt, section 1]\nThe system runs offline."]


class CountingLLM(LocalLLM):
    def __init__(self) -> None:
        self.calls = 0

    def stream_answer(self, question: str, context: str):
        self.calls += 1
        yield "The system runs offline."

    def stream_general_answer(self, question: str):
        yield ""


class GroundedValidationPipelineTests(unittest.TestCase):
    def test_no_context_refuses_without_calling_llm(self) -> None:
        tts = CapturingTTS()
        llm = CountingLLM()
        pipeline = AssistantPipeline(
            state_manager=ResponseStateManager(tts=tts, eyes=SilentEyes()),
            stt=StaticSTT(),
            retriever=EmptyRetriever(),
            llm=llm,
            answer_tts=tts,
        )

        answer = pipeline.answer("Where is the policy?")

        self.assertEqual(answer, GroundedAnswerValidator.no_context_message)
        self.assertEqual(llm.calls, 0)
        self.assertIn(GroundedAnswerValidator.no_context_message, tts.messages)

    def test_context_allows_llm_answer(self) -> None:
        tts = CapturingTTS()
        llm = CountingLLM()
        pipeline = AssistantPipeline(
            state_manager=ResponseStateManager(tts=tts, eyes=SilentEyes()),
            stt=StaticSTT(),
            retriever=StaticRetriever(),
            llm=llm,
            answer_tts=tts,
        )

        answer = pipeline.answer("How does it run?")

        self.assertEqual(answer, "The system runs offline.")
        self.assertEqual(llm.calls, 1)


if __name__ == "__main__":
    unittest.main()
