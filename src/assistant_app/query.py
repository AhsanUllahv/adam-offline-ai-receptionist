from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    DOCUMENT_QUERY = "document_query"
    EXIT = "exit"
    EMPTY = "empty"


@dataclass(frozen=True)
class ProcessedQuery:
    original: str
    text: str
    intent: Intent


@dataclass
class ConversationManager:
    max_turns: int = 6
    turns: deque[tuple[str, str]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.turns = deque(self.turns, maxlen=self.max_turns)

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append((question, answer))

    def last_question(self) -> str | None:
        if not self.turns:
            return None
        return self.turns[-1][0]


class IntentRouter:
    def route(self, text: str) -> Intent:
        cleaned = text.strip().lower()
        if not cleaned:
            return Intent.EMPTY
        if cleaned in {"exit", "quit", "stop"}:
            return Intent.EXIT
        return Intent.DOCUMENT_QUERY


@dataclass
class QueryReformulator:
    followup_prefixes: tuple[str, ...] = (
        "what about ",
        "and ",
        "how about ",
        "what is it",
        "explain it",
    )

    def reformulate(self, question: str, conversation: ConversationManager) -> str:
        cleaned = " ".join(question.split())
        if not cleaned:
            return ""

        lowered = cleaned.lower()
        last_question = conversation.last_question()
        if last_question and lowered.startswith(self.followup_prefixes):
            return f"Previous question: {last_question}. Follow-up question: {cleaned}"
        return cleaned


@dataclass
class QueryProcessor:
    router: IntentRouter = field(default_factory=IntentRouter)
    reformulator: QueryReformulator = field(default_factory=QueryReformulator)
    conversation: ConversationManager = field(default_factory=ConversationManager)

    def process(self, question: str) -> ProcessedQuery:
        intent = self.router.route(question)
        if intent is not Intent.DOCUMENT_QUERY:
            return ProcessedQuery(original=question, text=question.strip(), intent=intent)
        text = self.reformulator.reformulate(question, self.conversation)
        return ProcessedQuery(original=question, text=text, intent=intent)

    def remember(self, question: str, answer: str) -> None:
        if question.strip() and answer.strip():
            self.conversation.add_turn(question, answer)
