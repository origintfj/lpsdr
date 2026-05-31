"""Processing thread for routing SDR samples to downstream consumers."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from audio_output import AudioSampleQueue
from sdr_reader import IQSampleBuffer

if TYPE_CHECKING:
    import numpy as np

    from waterfall import RadioConfig


class ProcessingThread(threading.Thread):
    """Process new I/Q samples and publish them to display/audio buffers."""

    def __init__(
        self,
        config: RadioConfig,
        sample_buffer: IQSampleBuffer,
        waterfall_queue: IQSampleBuffer | None,
        time_domain_queue: IQSampleBuffer | None,
        stop_event: threading.Event,
        audio_queue: AudioSampleQueue | None = None,
        audio_sample_rate: int | None = None,
        audio_gain: float = 0.2,
    ) -> None:
        super().__init__(name="processing", daemon=True)
        self.config = config
        self.sample_buffer = sample_buffer
        self.waterfall_queue = waterfall_queue
        self.time_domain_queue = time_domain_queue
        self.stop_event = stop_event
        self.audio_queue = audio_queue
        self.audio_sample_rate = audio_sample_rate
        self.audio_gain = audio_gain

    def run(self) -> None:
        while not self.stop_event.is_set():
            block = self.sample_buffer.wait_for_samples(
                self.config.display_update_sample_count,
                self.stop_event,
            )
            if block is None:
                continue

            # These are intentionally independent handoff points. A processing
            # stage can choose to push different blocks into each display, while
            # the sample rate represented by both remains the SDR sample rate.
            if self.waterfall_queue is not None:
                self.waterfall_queue.append(block)
            if self.time_domain_queue is not None:
                self.time_domain_queue.append(block)
            self._push_audio_samples(block)

    def _push_audio_samples(self, block: "np.ndarray[Any, Any]") -> None:
        """Push a simple downsampled monitor stream to the audio queue, if enabled."""
        import numpy as np

        if self.audio_queue is None or self.audio_sample_rate is None:
            return

        # This is a lightweight audio monitor path rather than a full AM/FM/SSB
        # demodulator. It lets downstream code hear a bounded, real-valued view
        # of processed samples while keeping playback rate independent of SDR
        # sample rate.
        decimation = max(1, round(self.config.sample_rate / self.audio_sample_rate))
        audio_samples = np.real(block[::decimation]).astype(np.float32)
        self.audio_queue.push_samples(audio_samples * self.audio_gain)

