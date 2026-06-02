from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class EyeController(ABC):
    @abstractmethod
    def set_expression(self, expression: str) -> None: ...


@dataclass
class ConsoleEyeController(EyeController):
    """Console stand-in for robotic eye hardware."""

    def set_expression(self, expression: str) -> None:
        print(f"[eyes] {expression}")


EYE_COMMANDS = {
    "normal blinking": "IDLE",
    "focused eyes": "LISTENING",
    "thinking animation": "THINKING",
    "scanning animation": "SEARCHING",
    "bright active eyes": "READY",
    "talking animation": "SPEAKING",
    "confused expression": "ERROR",
}


@dataclass
class SerialEyeController(EyeController):
    port: str
    baudrate: int = 115200
    timeout: float = 1.0

    def __post_init__(self) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for ESP32-S3 eye control. Install with: pip install -e '.[all]'"
            ) from exc
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

    def set_expression(self, expression: str) -> None:
        command = EYE_COMMANDS.get(expression, "IDLE")
        self._serial.write(f"{command}\n".encode("utf-8"))

    def close(self) -> None:
        self._serial.close()


def build_eye_controller(port: str | None, baudrate: int = 115200) -> EyeController:
    if port and port.strip():
        return SerialEyeController(port=port, baudrate=baudrate)
    return ConsoleEyeController()
