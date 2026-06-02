from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    allowed: bool
    message: str | None = None


class GroundedAnswerValidator:
    no_context_message = "I could not find that clearly in your documents."

    def validate_context(self, context: str) -> ValidationResult:
        if not context.strip():
            return ValidationResult(allowed=False, message=self.no_context_message)
        return ValidationResult(allowed=True)
