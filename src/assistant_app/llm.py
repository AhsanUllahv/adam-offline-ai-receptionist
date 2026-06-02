from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import requests

from assistant_app.config import AssistantConfig


class LocalLLM(ABC):
    @abstractmethod
    def stream_answer(self, question: str, context: str) -> Iterable[str]: ...

    @abstractmethod
    def stream_general_answer(self, question: str) -> Iterable[str]: ...

    def stream_voice_answer(self, question: str, context: str) -> Iterable[str]:
        return self.stream_answer(question, context)

    def stream_voice_general_answer(self, question: str) -> Iterable[str]:
        return self.stream_general_answer(question)


@dataclass
class OllamaLLM(LocalLLM):
    model: str
    base_url: str
    timeout_seconds: float = 120.0
    num_predict: int = 96
    num_ctx: int = 2048
    temperature: float = 0.1

    @classmethod
    def from_config(cls, config: AssistantConfig) -> "OllamaLLM":
        return cls(
            model=config.ollama_model,
            base_url=config.ollama_url.rstrip("/"),
            num_predict=config.ollama_num_predict,
            num_ctx=config.ollama_num_ctx,
            temperature=config.ollama_temperature,
        )

    def warmup(self) -> None:
        """Load the model into memory now so the first real query has no cold-start penalty."""
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": " ", "stream": False,
                      "keep_alive": -1, "options": {"num_predict": 1}},
                timeout=60.0,
            )
        except Exception:
            pass

    def stream_answer(self, question: str, context: str) -> Iterable[str]:
        prompt = self._build_prompt(question, context)
        try:
            yield from self._stream_from_ollama(prompt)
        except requests.RequestException:
            yield from self._fallback_answer(question, context)

    def stream_voice_answer(self, question: str, context: str) -> Iterable[str]:
        prompt = self._build_voice_prompt(question, context)
        try:
            yield from self._stream_from_ollama(prompt, num_predict=150, num_ctx=1536)
        except requests.RequestException:
            yield from self._fallback_answer(question, context)

    def stream_voice_general_answer(self, question: str) -> Iterable[str]:
        prompt = self._build_voice_general_prompt(question)
        try:
            yield from self._stream_from_ollama(prompt, num_predict=150, num_ctx=1536)
        except requests.RequestException:
            yield "Sorry, I can't reach the language model right now. Please make sure Ollama is running."

    def stream_general_answer(self, question: str) -> Iterable[str]:
        prompt = self._build_general_prompt(question)
        try:
            yield from self._stream_from_ollama(prompt)
        except requests.RequestException:
            yield "I can help with general questions when the local model is running. "
            yield "Please make sure Ollama is started and the selected model is loaded."

    def _stream_from_ollama(self, prompt: str, num_predict: int | None = None, num_ctx: int | None = None) -> Iterable[str]:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "keep_alive": -1,
                "options": {
                    "num_predict": num_predict if num_predict is not None else self.num_predict,
                    "num_ctx": num_ctx if num_ctx is not None else self.num_ctx,
                    "temperature": self.temperature,
                },
            },
            stream=True,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            payload = json.loads(line.decode("utf-8"))
            token = payload.get("response", "")
            if token:
                yield token
            if payload.get("done"):
                break

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "You are a knowledgeable assistant. Use ONLY the knowledge context below to answer. "
            "Be direct and clear. Give a complete answer — include key facts, numbers, steps, or "
            "specifications the user asked about. Use plain sentences or bullet points, whichever "
            "is clearer. Do not start with filler like 'Sure' or 'Great question'. "
            "If a specific detail is not in the context, say so briefly at the end. "
            "Informal terms are fine: 'robo eyes' = robot eye modules or eye displays.\n\n"
            f"Knowledge context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def _build_general_prompt(self, question: str) -> str:
        return (
            "You are a helpful, friendly assistant. Answer the question directly and naturally. "
            "Be conversational and concise. Do not start with filler like 'Sure' or 'Of course'. "
            "For greetings, respond warmly in one sentence. For general knowledge questions, "
            "answer clearly. If asked about a specific institution, fee, policy, or official "
            "detail you have no data on, briefly say so and suggest uploading that document.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def _build_voice_prompt(self, question: str, context: str) -> str:
        return (
            "You are a warm, friendly voice assistant having a natural spoken conversation. "
            "Answer in 1 or 2 short sentences maximum. "
            "Use plain spoken language — no bullet points, no markdown, no headers, no bold text. "
            "Sound like a helpful friend talking, not a textbook. "
            "Use ONLY the knowledge context below. If a detail is missing, say so in one natural sentence. "
            "Informal terms are fine.\n\n"
            f"Knowledge context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def _build_voice_general_prompt(self, question: str) -> str:
        return (
            "You are a warm, friendly voice assistant having a natural spoken conversation. "
            "Answer in 1 or 2 short sentences maximum. "
            "Use plain spoken language — no bullet points, no markdown, no headers. "
            "Sound like a helpful friend talking, not a textbook. "
            "For greetings, respond warmly in one sentence. "
            "If you don't have specific data, say so naturally and offer to help another way.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def _fallback_answer(self, question: str, context: str) -> Iterable[str]:
        if context:
            yield "I found relevant document context, but Ollama is not reachable. "
            yield "Start Ollama and keep the selected model warm for full grounded answers."
        else:
            yield "I could not find that clearly in your documents."

