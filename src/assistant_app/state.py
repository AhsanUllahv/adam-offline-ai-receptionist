from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from assistant_app.eyes import EyeController
from assistant_app.tts import TextToSpeech


class ResponseState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SEARCHING = "searching_documents"
    ANSWER_READY = "answer_ready"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass(frozen=True)
class StateFeedback:
    voice: str | None
    eyes: str


STATE_FEEDBACK = {
    ResponseState.IDLE: StateFeedback(voice=None, eyes="normal blinking"),
    ResponseState.LISTENING: StateFeedback(
        voice=None, eyes="focused eyes"
    ),
    ResponseState.PROCESSING: StateFeedback(
        voice=None, eyes="thinking animation"
    ),
    ResponseState.SEARCHING: StateFeedback(
        voice=None, eyes="scanning animation"
    ),
    ResponseState.ANSWER_READY: StateFeedback(
        voice=None, eyes="bright active eyes"
    ),
    ResponseState.SPEAKING: StateFeedback(voice=None, eyes="talking animation"),
    ResponseState.ERROR: StateFeedback(
        voice=None, eyes="confused expression"
    ),
}


class ResponseStateManager:
    def __init__(self, tts: TextToSpeech, eyes: EyeController) -> None:
        self.tts = tts
        self.eyes = eyes
        self.current_state = ResponseState.IDLE

    def transition(self, state: ResponseState) -> None:
        self.current_state = state
        feedback = STATE_FEEDBACK[state]
        self.eyes.set_expression(feedback.eyes)
        if feedback.voice:
            self.tts.say(feedback.voice)
        _push_state_event(state)


def _push_state_event(state: ResponseState) -> None:
    try:
        from assistant_app import events
        events.push_event("state", state=state.value)
    except Exception:
        pass
