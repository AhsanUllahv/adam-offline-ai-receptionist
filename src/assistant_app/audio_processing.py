from __future__ import annotations

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only on minimal installs
    np = None


class AudioPreprocessor:
    """
    Applies AGC (Automatic Gain Control) and NS (Noise Suppression) to raw
    int16 PCM audio frames before they reach the speech recognizer.

    AEC (Acoustic Echo Cancellation) is not implementable in pure Python without
    the reference speaker signal. For production use, enable it at the OS level:
      - PulseAudio: load module-echo-cancel aec_method=webrtc
      - PipeWire:   module-echo-cancel with webrtc filter
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        target_rms: float = 3000.0,
        agc_max_gain: float = 8.0,
        ns_strength: float = 1.5,
        warmup_frames: int = 15,
    ) -> None:
        self._target_rms = target_rms
        self._agc_max_gain = agc_max_gain
        self._ns_strength = ns_strength
        self._warmup_frames = warmup_frames
        self._noise_floor: float | None = None
        self._frame_count = 0

    def process(self, frame: bytes) -> bytes:
        if np is None:
            raise RuntimeError("Audio preprocessing requires numpy installed.")
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        samples = self._apply_ns(samples)
        samples = self._apply_agc(samples)
        return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

    def _apply_agc(self, samples: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(samples ** 2))) + 1e-6
        gain = min(self._target_rms / rms, self._agc_max_gain)
        return samples * gain

    def _apply_ns(self, samples: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(samples ** 2)))

        if self._frame_count < self._warmup_frames:
            # Use the first N frames (assumed quiet) to estimate the noise floor.
            if self._noise_floor is None:
                self._noise_floor = rms
            else:
                self._noise_floor = 0.85 * self._noise_floor + 0.15 * rms
            self._frame_count += 1
            return samples

        noise_threshold = (self._noise_floor or 1.0) * self._ns_strength
        if rms < noise_threshold:
            # Soft gate: square-law attenuation below the noise floor
            attenuation = max(0.0, rms / noise_threshold) ** 2
            return samples * attenuation
        return samples
