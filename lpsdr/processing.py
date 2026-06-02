"""Processing thread that converts SDR IQ chunks into audio chunks."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np

from .radio import SDRSampleSource


@dataclass(frozen=True)
class ProcessingConfig:
    """Controls how many SDR samples are processed per chunk."""

    chunk_size: int


class ProcessingThread(threading.Thread):
    """Wait for SDR chunks, process them, and enqueue audio samples."""

    def __init__(
        self,
        sample_source: SDRSampleSource,
        audio_queue: "queue.Queue[np.ndarray]",
        config: ProcessingConfig,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="sdr-processing")
        self._sample_source = sample_source
        self._audio_queue = audio_queue
        self._config = config
        self._stop_event = stop_event
        self.done_event = threading.Event()
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                iq_samples = self._sample_source.next_samples(self._config.chunk_size)
                audio_samples = self._process(iq_samples)
                self._enqueue_audio(audio_samples)
        except BaseException as exc:
            self.error = exc
            self._stop_event.set()
        finally:
            self.done_event.set()

    def _enqueue_audio(self, audio_samples: np.ndarray) -> None:
        while not self._stop_event.is_set():
            try:
                self._audio_queue.put(audio_samples, timeout=0.1)
                return
            except queue.Full:
                continue

    def _process(self, iq_samples: np.ndarray) -> np.ndarray:
        """Placeholder DSP stage: use the real part as mono float32 audio."""
        return np.real(iq_samples).astype(np.float32, copy=False)
