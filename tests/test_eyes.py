from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from assistant_app.eyes import ConsoleEyeController, SerialEyeController, build_eye_controller


class EyeControllerTests(unittest.TestCase):
    def test_missing_port_uses_console_controller(self) -> None:
        self.assertIsInstance(build_eye_controller(None), ConsoleEyeController)
        self.assertIsInstance(build_eye_controller(""), ConsoleEyeController)

    def test_serial_controller_writes_state_command(self) -> None:
        serial_instance = MagicMock()
        with patch("serial.Serial", return_value=serial_instance):
            controller = SerialEyeController(port="/dev/ttyUSB0")
            controller.set_expression("scanning animation")

        serial_instance.write.assert_called_once_with(b"SEARCHING\n")

    def test_unknown_expression_defaults_to_idle(self) -> None:
        serial_instance = MagicMock()
        with patch("serial.Serial", return_value=serial_instance):
            controller = SerialEyeController(port="/dev/ttyUSB0")
            controller.set_expression("unknown")

        serial_instance.write.assert_called_once_with(b"IDLE\n")


if __name__ == "__main__":
    unittest.main()
