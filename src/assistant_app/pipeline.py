from __future__ import annotations

import os
import queue as _queue_mod
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from assistant_app.llm import LocalLLM
from assistant_app.query import Intent, QueryProcessor
from assistant_app.retrieval import DocumentRetriever
from assistant_app.state import ResponseState, ResponseStateManager
from assistant_app.stt import SpeechToText, TextInputSTT
from assistant_app.tts import TextToSpeech
from assistant_app.validation import GroundedAnswerValidator
from assistant_app.wakeword import AlwaysReadyDetector, WakeWordDetector

# Human-like wake acknowledgments — varied so it never sounds scripted
_WAKE_GREETINGS = [
    "Yes?",
    "I'm listening.",
    "Go ahead.",
    "I'm here.",
]

# Filler phrases for timed feedback — mimic human "thinking aloud"
_FILLER_SCHEDULE = [
    (0.6, "One moment."),
    (1.6, "Let me check that..."),
    (2.8, "I'm searching through the documents..."),
    (5.5, "This is taking a moment — still looking..."),
    (9.0, "Almost there, bear with me..."),
]


class AsyncTTS:
    """
    Wraps a TextToSpeech so that say() is non-blocking.
    Sentences are queued and spoken by a background thread,
    allowing LLM streaming to continue while audio plays.
    """

    def __init__(self, tts: "TextToSpeech") -> None:
        self._tts = tts
        self._q: _queue_mod.Queue = _queue_mod.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        if text:
            self._q.put(text)

    def interrupt(self) -> None:
        drained: list = []
        while True:
            try:
                drained.append(self._q.get_nowait())
            except _queue_mod.Empty:
                break
        for item in drained:
            if isinstance(item, threading.Event):
                item.set()
        self._tts.interrupt()

    def wait(self) -> None:
        """Block until all currently queued sentences have finished playing."""
        done = threading.Event()
        self._q.put(done)
        done.wait(timeout=60.0)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if isinstance(item, threading.Event):
                item.set()
                continue
            if item is None:
                break
            self._tts.say(item)


_GENERAL_STOPWORDS = {
    "a", "an", "and", "are", "can", "for", "from", "about",
    "do", "does", "how", "i", "is", "it", "of", "on", "or",
    "know", "mean", "the", "this", "to", "what", "with", "you",
    "your", "who", "write", "essay", "question",
}

_SPECIFIC_INFO_TERMS = {
    "admission", "address", "campus", "college", "contact", "course",
    "courses", "fee", "fees", "location", "phone", "policy",
    "principal", "school", "schedule", "timing", "university",
}

_COLOR_TERMS = {
    "black", "blue", "brown", "color", "colour", "gray", "green", "grey",
    "orange", "purple", "red", "silver", "white", "yellow",
}


def _keyword_terms(text: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(term) > 2 and term not in _GENERAL_STOPWORDS
    ]


def _expand_query_terms(terms: list[str]) -> set[str]:
    expanded = set(terms)
    if "eyes" in expanded:
        expanded.add("eye")
    if "robo" in expanded:
        expanded.add("robot")
    if "display" in expanded:
        expanded.add("screen")
    return expanded


def _split_context(context: str) -> tuple[str, str]:
    lines = context.splitlines()
    if lines and lines[0].startswith("[") and lines[0].endswith("]"):
        return lines[0].strip("[]"), " ".join(lines[1:]).strip()
    return "", " ".join(lines).strip()


def _has_relevant_document_context(query: str, contexts: list[str]) -> bool:
    if not contexts:
        return False
    query_terms = _expand_query_terms(_keyword_terms(query))
    if not query_terms:
        return False
    combined_terms: set[str] = set()
    combined_text_parts: list[str] = []
    for context in contexts[:3]:
        source, body = _split_context(context)
        combined_text_parts.append(body.lower())
        combined_terms.update(_keyword_terms(source))
        combined_terms.update(_keyword_terms(body))
    combined_text = " ".join(combined_text_parts)
    query_specific_terms = query_terms - {"color", "colour"}
    if {"flag", "pakistan", "pakistani"} & query_terms:
        return bool(({"flag", "pakistan", "pakistani"} & combined_terms))
    if {"color", "colour"} & query_terms and _COLOR_TERMS & combined_terms:
        if query_specific_terms and not (query_specific_terms & combined_terms):
            return False
        return True
    if "look" in query_terms and {"design", "appearance", "white", "black", "blue"} & combined_terms:
        return True
    if "robot" in query_terms and "design intent" in combined_text:
        return True
    overlap = query_terms & combined_terms
    if len(query_terms) == 1:
        term = next(iter(query_terms))
        return len(term) >= 3 and term in combined_terms
    if len(overlap) >= 2:
        return True
    return len(overlap) / max(len(query_terms), 1) >= 0.6


def _needs_official_knowledge_answer(query: str) -> bool:
    terms = set(_keyword_terms(query))
    if not terms & _SPECIFIC_INFO_TERMS:
        return False
    return len(terms - _SPECIFIC_INFO_TERMS) >= 1


def _official_knowledge_missing_answer() -> str:
    return (
        "I do not have official local information about that in the uploaded documents yet. "
        "Upload the brochure, policy, or contact file and I can answer from it."
    )


def _normalized_words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _normalize_voice_query_text(text: str) -> str:
    normalized = " ".join(text.split())
    replacements = (
        (r"\bvoice\s+to\s+tax\b", "voice to text"),
        (r"\bspeech\s+to\s+tax\b", "speech to text"),
        (r"\btext\s+to\s+voice\b", "text to speech"),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    if re.search(r"\b(problem|system|overview|thesis)\b", normalized, re.IGNORECASE):
        normalized = re.sub(r"\bpieces\b", "thesis", normalized, flags=re.IGNORECASE)
    return normalized


def _needs_generated_voice_answer(question: str) -> bool:
    words = _normalized_words(question)
    if words & {"explain", "overview", "summarize", "summary", "describe", "methodology", "theory", "background"}:
        return True
    if "how" in words and words & {"work", "works", "process", "system", "implemented"}:
        return True
    if "part" in words and words & {"thesis", "section", "chapter", "problem", "system"}:
        return True
    return False


def _extract_color_fact(body: str) -> str | None:
    compact = " ".join(body.split())
    patterns = (
        r"The head and body are .*?(?:branding\.|$)",
        r"Body finish areas .*?(?:perimeter\.|$)",
        r"Design intent: .*?(?:branding\.|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            fact = match.group(0).strip()
            if fact.lower().startswith("design intent:"):
                color_match = re.search(
                    r"The head and body are .*?(?:branding\.|$)",
                    fact,
                    flags=re.IGNORECASE,
                )
                if color_match:
                    return color_match.group(0).strip()
            return fact
    return None


def _extract_labeled_fact(body: str, labels: tuple[str, ...]) -> str | None:
    compact = " ".join(body.split())
    boundaries = (
        "Top section / head",
        "Middle section",
        "Bottom section / base",
        "Connection flow",
        "Inner-design area",
    )
    for label in labels:
        pattern = rf"{re.escape(label)}\s+(.*?)(?=" + "|".join(
            re.escape(boundary) for boundary in boundaries if boundary != label
        ) + r"|$)"
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if not match:
            continue
        fact = match.group(1).strip(" :-")
        if fact:
            return fact
    return None


def _simple_voice_answer(question: str) -> str | None:
    normalized = " ".join(question.lower().strip().split()).rstrip("?.!")
    if not normalized:
        return "I didn't catch that. Could you say it again?"
    words = set(normalized.split())
    if words & {"hi", "hello", "hey", "salam", "yo"} and len(normalized) <= 30:
        return "Hello! How can I help?"
    if normalized in {"how are you", "how r u", "how are you doing", "hows it going"}:
        return "I am ready and listening. Ask me anything."
    if normalized in {"thanks", "thank you", "thank you very much", "thx", "ty"}:
        return "You're welcome."
    if normalized in {"ok", "okay", "got it", "noted", "alright"}:
        return "Sure."
    return None


@dataclass
class _VoiceLogAccumulator:
    wake_detected_at: str = ""
    transcript: str = ""
    answer: str = ""
    sources: tuple[str, ...] = ()
    knowledge_source: str = "general"
    filler_count: int = 0
    barge_in: bool = False
    answer_truncated: bool = False
    stt_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    total_seconds: float = 0.0
    word_count: int = 0
    source_count: int = 0
    cpu_load: float = 0.0
    memory_percent: float = 0.0
    temperature_c: float | None = None
    error: str = ""


def _sys_cpu_load() -> float:
    try:
        return round(os.getloadavg()[0], 2)
    except Exception:
        return 0.0


def _sys_memory_percent() -> float:
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
        total = values.get("MemTotal", 0)
        avail = values.get("MemAvailable", 0)
        return round((total - avail) / total * 100, 1) if total else 0.0
    except Exception:
        return 0.0


def _sys_temperature_c() -> float | None:
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            value = int(path.read_text().strip())
            if value > 0:
                return round(value / 1000, 1)
        except Exception:
            continue
    return None


def _push(event_type: str, **payload: object) -> None:
    try:
        from assistant_app import events
        events.push_event(event_type, **payload)
    except Exception:
        pass


@dataclass
class AssistantPipeline:
    state_manager: ResponseStateManager
    stt: SpeechToText
    retriever: DocumentRetriever
    llm: LocalLLM
    answer_tts: TextToSpeech
    validator: GroundedAnswerValidator = field(default_factory=GroundedAnswerValidator)
    query_processor: QueryProcessor = field(default_factory=QueryProcessor)
    wakeword_detector: WakeWordDetector = field(default_factory=AlwaysReadyDetector)
    enable_barge_in: bool = False
    sqlite_path: str | None = None
    no_rag: bool = False
    continuous_conversation: bool = True
    followup_timeout_seconds: float = 45.0
    wake_greeting_enabled: bool = False
    voice_answer_mode: str = "fast"
    voice_retrieval_limit: int = 2
    voice_context_chars: int = 900
    _is_voice: bool = field(default=False, init=False, repr=False)
    _active_feedback: TimedFeedback | None = field(default=None, init=False, repr=False)
    _current_log: _VoiceLogAccumulator | None = field(default=None, init=False, repr=False)
    _last_sources: list[str] = field(default_factory=list, init=False, repr=False)
    _last_barge_in: bool = field(default=False, init=False, repr=False)

    def run_text_demo(self) -> None:
        """Main loop: wake -> listen -> answer -> follow up -> repeat."""
        is_voice = not isinstance(self.stt, TextInputSTT)
        self._is_voice = is_voice
        if is_voice:
            print("Virtual receptionist ready. Say 'Adam' to wake me.")
        else:
            print("Offline assistant ready. Type a question, or 'exit' to quit.")
        self.state_manager.transition(ResponseState.IDLE)

        wait_for_wake = True
        while True:
            if wait_for_wake:
                self.wakeword_detector.wait_for_wake_word()

                if is_voice and self.sqlite_path:
                    from datetime import datetime
                    self._current_log = _VoiceLogAccumulator(
                        wake_detected_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )

                if is_voice and self.wake_greeting_enabled:
                    greeting = random.choice(_WAKE_GREETINGS)
                    self.answer_tts.say(greeting)
                    _push("wake_response", text=greeting)
            else:
                print("\n[conversation] Listening for a follow-up...", flush=True)
                _push(
                    "state",
                    state="follow_up_listening",
                    message="Listening for a follow-up question",
                )
                if is_voice and self.sqlite_path and self._current_log is None:
                    from datetime import datetime
                    self._current_log = _VoiceLogAccumulator(
                        wake_detected_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )

            self.state_manager.transition(ResponseState.LISTENING)
            timeout = None if wait_for_wake else self.followup_timeout_seconds
            stt_start = time.monotonic()
            question = self._listen_for_question(timeout_seconds=timeout)
            stt_seconds = time.monotonic() - stt_start
            _push("transcript", text=question)

            if self._current_log is not None:
                self._current_log.transcript = question
                self._current_log.stt_seconds = round(stt_seconds, 3)

            processed = self.query_processor.process(question)
            if processed.intent is Intent.EXIT:
                self.state_manager.transition(ResponseState.IDLE)
                self._current_log = None
                if is_voice:
                    self.answer_tts.say("Goodbye, have a great day!")
                print("Goodbye.")
                return
            if processed.intent is Intent.EMPTY:
                self.state_manager.transition(ResponseState.IDLE)
                self._current_log = None
                wait_for_wake = True
                continue

            answer_text = self.answer(processed.original)
            if is_voice and self.sqlite_path and answer_text:
                self._log_voice_interaction(processed.original, answer_text)

            wait_for_wake = not (is_voice and self.continuous_conversation)

    def _listen_for_question(self, timeout_seconds: float | None = None) -> str:
        try:
            return self.stt.listen(timeout_seconds=timeout_seconds)
        except TypeError:
            return self.stt.listen()

    def answer(self, question: str) -> str:
        filler = TimedFeedback(self.state_manager, self.answer_tts if self._is_voice else None)
        filler.start()
        self._active_feedback = filler
        started_at = time.monotonic()
        retrieval_seconds = 0.0
        generation_seconds = 0.0
        knowledge_source = "general"
        self._last_barge_in = False

        try:
            processed = self.query_processor.process(question)
            if processed.intent is Intent.EMPTY:
                return ""
            if processed.intent is Intent.EXIT:
                return "Goodbye."

            self.state_manager.transition(ResponseState.PROCESSING)
            answer_query = (
                _normalize_voice_query_text(processed.text)
                if self._is_voice
                else processed.text
            )

            quick_answer = _simple_voice_answer(answer_query) if self._is_voice else None
            if quick_answer:
                knowledge_source = "quick"
                answer = self._speak_static_answer(quick_answer)
            elif self.no_rag:
                gen_start = time.monotonic()
                answer = self._stream_general_answer(answer_query)
                generation_seconds = time.monotonic() - gen_start
            else:
                ret_start = time.monotonic()
                contexts = self._retrieve_contexts(
                    answer_query,
                    limit=self.voice_retrieval_limit if self._is_voice else None,
                )
                retrieval_seconds = time.monotonic() - ret_start

                if _has_relevant_document_context(answer_query, contexts):
                    knowledge_source = "documents"
                    use_fast_extract = (
                        self._is_voice
                        and self.voice_answer_mode == "fast"
                        and not _needs_generated_voice_answer(answer_query)
                    )
                    fast_answer = (
                        self._build_fast_voice_answer(answer_query, contexts)
                        if use_fast_extract
                        else None
                    )
                    if fast_answer:
                        answer = self._speak_static_answer(fast_answer)
                    else:
                        gen_start = time.monotonic()
                        context = (
                            self._compact_voice_contexts(contexts)
                            if self._is_voice
                            else "\n\n".join(contexts)
                        )
                        answer = self._stream_answer(answer_query, context)
                        generation_seconds = time.monotonic() - gen_start
                elif self._is_voice and _needs_official_knowledge_answer(answer_query):
                    knowledge_source = "missing"
                    answer = self._speak_static_answer(_official_knowledge_missing_answer())
                else:
                    gen_start = time.monotonic()
                    answer = self._stream_general_answer(answer_query)
                    generation_seconds = time.monotonic() - gen_start

            self.query_processor.remember(processed.original, answer)
            _push("answer_complete", text=answer)
            return answer
        except Exception as exc:
            self.state_manager.transition(ResponseState.ERROR)
            message = f"I couldn't complete that clearly: {exc}"
            self.answer_tts.say(message)
            _push("error", message=message)
            if self._current_log is not None:
                self._current_log.error = str(exc)
            return message
        finally:
            filler.stop()
            if self._active_feedback is filler:
                self._active_feedback = None
            elapsed = time.monotonic() - started_at
            print(f"\n[latency] {elapsed:.2f}s")
            _push("latency", seconds=round(elapsed, 3))

            if self._current_log is not None and self._is_voice:
                log = self._current_log
                log.retrieval_seconds = round(retrieval_seconds, 3)
                log.generation_seconds = round(generation_seconds, 3)
                # Voice logs show answer-computation latency; STT and audio playback are shown separately.
                log.total_seconds = round(retrieval_seconds + generation_seconds, 3)
                log.sources = tuple(self._last_sources)
                log.source_count = len(self._last_sources)
                log.knowledge_source = knowledge_source
                log.barge_in = self._last_barge_in
                log.answer_truncated = self._last_barge_in
                log.filler_count = filler.count
                log.cpu_load = _sys_cpu_load()
                log.memory_percent = _sys_memory_percent()
                log.temperature_c = _sys_temperature_c()

            self.state_manager.transition(ResponseState.IDLE)

    def _log_voice_interaction(self, question: str, answer: str) -> None:
        try:
            from assistant_app.audit import AuditStore
            store = AuditStore(self.sqlite_path)
            log = self._current_log
            sources = log.sources if log else ()
            store.log_interaction(
                question=question,
                answer=answer,
                sources=sources,
                answer_mode="voice",
                model="voice",
                retrieval_seconds=log.retrieval_seconds if log else 0.0,
                generation_seconds=log.generation_seconds if log else 0.0,
                session_id="voice",
            )
            if log is not None:
                log.answer = answer
                log.word_count = len(answer.split()) if answer.strip() else 0
                store.save_voice_log(
                    wake_detected_at=log.wake_detected_at,
                    transcript=question,
                    answer=answer,
                    sources=log.sources,
                    knowledge_source=log.knowledge_source,
                    filler_count=log.filler_count,
                    barge_in=log.barge_in,
                    answer_truncated=log.answer_truncated,
                    stt_seconds=log.stt_seconds,
                    retrieval_seconds=log.retrieval_seconds,
                    generation_seconds=log.generation_seconds,
                    total_seconds=log.total_seconds,
                    word_count=log.word_count,
                    source_count=log.source_count,
                    cpu_load=log.cpu_load,
                    memory_percent=log.memory_percent,
                    temperature_c=log.temperature_c,
                    error=log.error,
                )
        except Exception:
            pass
        finally:
            self._current_log = None

    def _retrieve_context(self, question: str) -> str:
        return "\n\n".join(self._retrieve_contexts(question))

    def _retrieve_contexts(self, question: str, limit: int | None = None) -> list[str]:
        self.state_manager.transition(ResponseState.SEARCHING)
        results = self.retriever.search(question, limit=limit or 4)
        self._last_sources = []
        if results:
            for ctx in results:
                line = ctx.splitlines()[0] if ctx.splitlines() else ""
                if line.startswith("[") and line.endswith("]"):
                    self._last_sources.append(line.strip("[]"))
            if self._last_sources:
                _push("sources", sources=self._last_sources)
        return results

    def _compact_voice_contexts(self, contexts: list[str]) -> str:
        compacted: list[str] = []
        remaining = max(200, self.voice_context_chars)
        for context in contexts[: self.voice_retrieval_limit]:
            if remaining <= 0:
                break
            source, body = _split_context(context)
            prefix = f"[{source}]\n" if source else ""
            budget = max(0, remaining - len(prefix))
            if budget <= 0:
                break
            snippet = body[:budget].rsplit(" ", 1)[0] or body[:budget]
            compacted.append(f"{prefix}{snippet}".strip())
            remaining -= len(compacted[-1]) + 2
        return "\n\n".join(compacted)

    def _build_fast_voice_answer(self, question: str, contexts: list[str]) -> str | None:
        if not contexts:
            return None
        source, body = _split_context(contexts[0])
        query_words = _normalized_words(question)
        answer = self._topic_fact_for_fast_voice(query_words, body)
        if not answer:
            return None
        answer = " ".join(answer.split())
        if len(answer) > 240:
            answer = answer[:240].rsplit(" ", 1)[0].rstrip(".,;:") + "."
        return answer

    def _topic_fact_for_fast_voice(self, query_words: set[str], body: str) -> str | None:
        if {"color", "colour"} & query_words or ({"look", "appearance", "design"} & query_words and "robot" in query_words):
            color_fact = _extract_color_fact(body)
            if color_fact:
                return color_fact

        labels: tuple[str, ...] = ()
        if {"head", "top"} & query_words or ("had" in query_words and {"robot", "board"} & query_words):
            labels = ("Top section / head",)
        elif "middle" in query_words:
            labels = ("Middle section",)
        elif {"bottom", "base"} & query_words:
            labels = ("Bottom section / base",)
        elif "jetson" in query_words:
            labels = ("Bottom section / base", "Middle section")
        elif "speaker" in query_words or "speakers" in query_words or "audio" in query_words:
            labels = ("Middle section",)
        elif "battery" in query_words:
            labels = ("Top section / head",)
        if not labels:
            return None
        fact = _extract_labeled_fact(body, labels)
        if not fact:
            return None
        return fact

    def _best_extractive_sentence(self, question: str, body: str) -> str:
        cleaned = re.sub(r"Robot 3D Design Brief[^.?!]*[.?!]?", " ", body, flags=re.IGNORECASE)
        cleaned = re.sub(r"Full internal reference image[^.?!]*[.?!]?", " ", cleaned, flags=re.IGNORECASE)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned.strip())
        query_terms = _expand_query_terms(_keyword_terms(question))

        def score(sentence: str) -> tuple[int, int, int]:
            terms = set(_keyword_terms(sentence))
            is_titleish = int(sentence.lower().startswith(("page ", "section ", "full inner-design", "inner-design area")))
            return (len(query_terms & terms), -is_titleish, -len(sentence))

        candidates = [sentence.strip() for sentence in sentences if sentence.strip()]
        best = max(candidates, key=score) if candidates else body.strip()
        if score(best)[0] == 0 and body.strip():
            best = body.strip()
        return best

    def _speak_static_answer(self, answer: str) -> str:
        self.state_manager.transition(ResponseState.SPEAKING)
        self._stop_timed_feedback()
        print(f"\nAssistant: {answer}", flush=True)
        self.answer_tts.say(answer)
        return answer

    def _stream_answer(self, question: str, context: str) -> str:
        stream = (
            self.llm.stream_voice_answer(question, context)
            if self._is_voice
            else self.llm.stream_answer(question, context)
        )
        return self._stream_response(stream, label="Assistant: ")

    def _stream_general_answer(self, question: str) -> str:
        stream = (
            self.llm.stream_voice_general_answer(question)
            if self._is_voice
            else self.llm.stream_general_answer(question)
        )
        return self._stream_response(stream, label="Assistant (general): ")

    def _stream_response(self, stream: Iterable[str], label: str) -> str:
        self.state_manager.transition(ResponseState.SPEAKING)
        chunks: list[str] = []
        sentence_buffer: list[str] = []
        barge_in = threading.Event()
        monitor: BargeinMonitor | None = None

        # In voice mode wrap TTS so say() is non-blocking — LLM keeps streaming
        # while the previous sentence is being synthesised and played.
        tts: object = AsyncTTS(self.answer_tts) if self._is_voice else self.answer_tts

        print(f"\n{label}", end="", flush=True)
        try:
            for chunk in stream:
                chunks.append(chunk)
                sentence_buffer.append(chunk)
                print(chunk, end="", flush=True)
                _push("token", text=chunk)
                if self._sentence_ready(sentence_buffer):
                    sentence = "".join(sentence_buffer).strip()
                    sentence_buffer.clear()
                    if self.enable_barge_in and monitor is None:
                        monitor = BargeinMonitor(tts, barge_in)
                        monitor.start()
                    if barge_in.is_set():
                        print("\n[barge-in] User interrupted.", flush=True)
                        _push("barge_in")
                        break
                    self._stop_timed_feedback(interrupt_audio=True)
                    tts.say(sentence)

            if sentence_buffer and not barge_in.is_set():
                self._stop_timed_feedback(interrupt_audio=True)
                tts.say("".join(sentence_buffer).strip())

            # Wait for the audio queue to drain (skipped on barge-in)
            if self._is_voice and not barge_in.is_set():
                tts.wait()
        finally:
            if monitor:
                monitor.stop()

        print()
        self._last_barge_in = barge_in.is_set()
        return "".join(chunks).strip()

    def _stop_timed_feedback(self, interrupt_audio: bool = False) -> None:
        if self._active_feedback is not None:
            self._active_feedback.stop(interrupt_audio=interrupt_audio)
            self._active_feedback = None

    def _sentence_ready(self, pieces: list[str]) -> bool:
        text = "".join(pieces).strip()
        return len(text) > 80 or text.endswith((".", "!", "?"))


class BargeinMonitor:
    """
    Runs a VAD monitor in a background thread during TTS playback.
    When the user speaks (≥3 consecutive VAD-positive frames ≈ 90 ms),
    it calls tts.interrupt() and sets the barge_in event so the pipeline
    stops streaming.
    """

    _FRAME_MS = 30
    _CONSECUTIVE_SPEECH_NEEDED = 3

    def __init__(
        self,
        tts: TextToSpeech,
        barge_in: threading.Event,
        sample_rate: int = 16000,
    ) -> None:
        self._tts = tts
        self._barge_in = barge_in
        self._sample_rate = sample_rate
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)

    def _run(self) -> None:
        try:
            import sounddevice as sd
            import webrtcvad
        except ImportError:
            return

        vad = webrtcvad.Vad(2)
        frame_size = int(self._sample_rate * self._FRAME_MS / 1000) * 2
        audio_q: queue.Queue[bytes] = queue.Queue()

        def callback(indata: bytes, frames: int, time_info: object, status: object) -> None:
            audio_q.put(bytes(indata))

        try:
            with sd.RawInputStream(
                samplerate=self._sample_rate,
                blocksize=frame_size // 2,
                dtype="int16",
                channels=1,
                callback=callback,
            ):
                consecutive_speech = 0
                while not self._stop.is_set():
                    try:
                        frame = audio_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if len(frame) != frame_size:
                        continue

                    if vad.is_speech(frame, self._sample_rate):
                        consecutive_speech += 1
                        if consecutive_speech >= self._CONSECUTIVE_SPEECH_NEEDED:
                            self._tts.interrupt()
                            self._barge_in.set()
                            _push("barge_in")
                            return
                    else:
                        consecutive_speech = 0
        except Exception:
            pass


class TimedFeedback:
    def __init__(self, state_manager: ResponseStateManager, tts: "TextToSpeech | None" = None) -> None:
        self.state_manager = state_manager
        self._tts = tts
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.count: int = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self, interrupt_audio: bool = False) -> None:
        self._stop.set()
        if interrupt_audio and self._tts is not None:
            self._tts.interrupt()
        self._thread.join(timeout=0.5)

    def _run(self) -> None:
        started = time.monotonic()
        for delay, message in _FILLER_SCHEDULE:
            remaining = delay - (time.monotonic() - started)
            if remaining > 0 and self._stop.wait(remaining):
                return
            if self._stop.is_set():
                return
            if self.state_manager.current_state in {
                ResponseState.PROCESSING,
                ResponseState.SEARCHING,
                ResponseState.SPEAKING,
            }:
                print(f"\n[assistant] {message}", file=sys.stderr)
                _push("filler", text=message)
                self.count += 1
                if self._tts is not None:
                    if self._stop.is_set():
                        return
                    self._tts.say(message)
