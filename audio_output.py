"""Bounded audio sample queue and playback thread.

The processing stage can push mono floating-point samples into
``AudioSampleQueue`` whenever it has audio to emit. ``AudioPlaybackThread`` drains
that queue in fixed-size blocks and writes them to the system's default audio
output at the sample rate supplied during initialization.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


class AudioSampleQueue:
    """Thread-safe bounded FIFO queue for mono float audio samples."""

    def __init__(self, max_samples: int) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be greater than zero")

        self._max_samples = max_samples
        self._samples: deque[float] = deque(maxlen=max_samples)
        self._condition = threading.Condition()

    @property
    def max_samples(self) -> int:
        """Maximum number of audio samples retained by this queue."""
        return self._max_samples

    def push_samples(self, samples: "np.ndarray[Any, Any]") -> None:
        """Append mono samples, dropping oldest queued samples if necessary.

        Samples are converted to ``float32`` and clipped to the usual soundcard
        range of ``[-1.0, 1.0]`` before they are queued.
        """
        import numpy as np

        audio_samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio_samples.size == 0:
            return

        clipped_samples = np.clip(audio_samples, -1.0, 1.0)
        with self._condition:
            self._samples.extend(float(sample) for sample in clipped_samples)
            self._condition.notify_all()

    def drain_for_playback(
        self,
        sample_count: int,
        stop_event: threading.Event,
        timeout: float = 0.05,
    ) -> "np.ndarray[Any, Any]":
        """Return ``sample_count`` samples for playback, zero-padding underruns.

        The audio device must be fed steadily. If the producer has not queued a
        full block by ``timeout`` or shutdown begins, this method returns the
        queued samples plus enough silence to make the block the requested size.
        """
        import numpy as np

        if sample_count <= 0:
            raise ValueError("sample_count must be greater than zero")
        if sample_count > self._max_samples:
            raise ValueError(
                f"sample_count ({sample_count}) cannot exceed max_samples "
                f"({self._max_samples})"
            )

        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self._samples) < sample_count and not stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)

            samples_to_drain = min(sample_count, len(self._samples))
            block = np.zeros(sample_count, dtype=np.float32)
            if samples_to_drain:
                block[:samples_to_drain] = np.fromiter(
                    (self._samples.popleft() for _ in range(samples_to_drain)),
                    dtype=np.float32,
                    count=samples_to_drain,
                )
            return block

    def wake_waiters(self) -> None:
        """Wake the playback thread so it can notice shutdown."""
        with self._condition:
            self._condition.notify_all()


class AudioPlaybackThread(threading.Thread):
    """Drain an ``AudioSampleQueue`` to the system audio output."""

    def __init__(
        self,
        sample_queue: AudioSampleQueue,
        sample_rate: int,
        block_size: int,
        stop_event: threading.Event,
        device: str | int | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if block_size <= 0:
            raise ValueError("block_size must be greater than zero")
        if block_size > sample_queue.max_samples:
            raise ValueError("block_size cannot exceed the queue's max_samples")

        super().__init__(name="audio-playback", daemon=True)
        self.sample_queue = sample_queue
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.stop_event = stop_event
        self.device = device

    def run(self) -> None:
        import sounddevice as sd

        try:
            self._play_samples(sd)
        except Exception as exc:
            print(f"Audio playback stopped: {exc}", file=sys.stderr)
        finally:
            self.sample_queue.wake_waiters()

    def _play_samples(self, sounddevice: Any) -> None:
        with sounddevice.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            device=self.device,
        ) as stream:
            while not self.stop_event.is_set():
                samples = self.sample_queue.drain_for_playback(
                    self.block_size,
                    self.stop_event,
                )
                stream.write(samples.reshape(-1, 1))
