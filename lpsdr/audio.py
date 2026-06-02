"""Audio queue playback helpers."""

from __future__ import annotations

import queue
from typing import Any

import numpy as np


def load_sounddevice() -> Any:
    """Load sounddevice only when audio playback is started."""
    import sounddevice

    return sounddevice


class AudioPlayer:
    """Play mono float32 chunks received from a queue."""

    def __init__(
        self,
        audio_queue: "queue.Queue[np.ndarray]",
        sample_rate_hz: float,
        block_size: int,
    ) -> None:
        self._audio_queue = audio_queue
        sounddevice = load_sounddevice()
        self._stream = sounddevice.OutputStream(
            samplerate=sample_rate_hz,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            callback=self._callback,
        )
        self._pending = np.empty(0, dtype=np.float32)

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()

    def close(self) -> None:
        self._stream.close()

    def _callback(self, outdata, frames, time, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            print(status)

        while self._pending.size < frames:
            try:
                chunk = self._audio_queue.get_nowait()
            except queue.Empty:
                break
            self._pending = np.concatenate((self._pending, chunk))

        if self._pending.size >= frames:
            outdata[:, 0] = self._pending[:frames]
            self._pending = self._pending[frames:]
        else:
            available = self._pending.size
            outdata[:available, 0] = self._pending
            outdata[available:, 0] = 0.0
            self._pending = np.empty(0, dtype=np.float32)

    def __enter__(self) -> "AudioPlayer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.stop()
        self.close()
