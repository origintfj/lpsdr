"""Processing thread for routing SDR samples to downstream consumers."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from audio_output import AudioSampleQueue
from sdr_reader import IQSampleBuffer

if TYPE_CHECKING:
    import numpy as np

    from waterfall import RadioConfig

import numpy as np

class SampleStream:
    def __init__(self) -> None:
        self.stream = None
        return None

    def append(self, iframe: np.ndarray) -> int:
        if self.stream is None:
            self.stream = iframe
        else:
            self.stream = np.concatenate((self.stream, iframe))
        return len(iframe)

    def get(self, oldest_n: int) -> np.ndarray:
        return self.stream[:oldest_n]

    def remove(self, oldest_n: int) -> None:
        self.stream = self.stream[oldest_n:]

    def convolve_rev(self, fir_vec_rev: np.ndarray) -> tuple[np.ndarray, int]:
        if self.stream is None:
            return np.array([], dtype=self.stream.dtype), 0
        if len(self.stream) < len(fir_vec_rev):
            return np.array([], dtype=self.stream.dtype), 0

        out_stream_len = len(self.stream) - len(fir_vec_rev) + 1
        out_stream = np.empty(out_stream_len, dtype=self.stream.dtype)

        for i in range(out_stream_len):
            out_stream[i] = np.dot(self.stream[i:i+len(fir_vec_rev)], fir_vec_rev)

        return out_stream, out_stream_len

    def dphase(self) -> tuple[np.ndarray, int]:
        if self.stream is None:
            return np.array([], dtype=self.stream.dtype), 0

        stream_arg = np.angle(self.stream)

        out_stream_len = len(self.stream) - 1
        out_stream = np.empty(out_stream_len, dtype=np.float32)

        for i in range(out_stream_len):
            darg = stream_arg[i] - stream_arg[i+1]
            if darg < -np.pi:
                darg = darg + 2 * np.pi
            elif darg > np.pi:
                darg = darg - 2 * np.pi
            out_stream[i] = darg/4

        return out_stream, out_stream_len


class ProcessingThread(threading.Thread):
    """Process new I/Q samples and publish them to display/audio buffers."""

    bb_stream = SampleStream()
    bb_lpf_stream = SampleStream()
    def _design_lpf(self, cutoff_hz: float, fs_hz: float, num_taps: int = 257) -> np.ndarray:
        """
        Windowed-sinc real FIR low-pass. Suitable for complex IQ (applies to I and Q together).
        cutoff_hz is ~3 dB cutoff; choose taps to trade stopband vs compute.
        """
        n = np.arange(num_taps) - (num_taps - 1) / 2
        h = 2 * cutoff_hz / fs_hz * np.sinc(2 * cutoff_hz * n / fs_hz)
        h *= np.hamming(num_taps)
        h /= np.sum(h)
        return h.astype(np.float32)

    def __init__(
        self,
        config: RadioConfig,
        sample_buffer: IQSampleBuffer,
        waterfall_queue: IQSampleBuffer | None,
        time_domain_queue: IQSampleBuffer | None,
        stop_event: threading.Event,
        audio_queue: AudioSampleQueue | None = None,
    ) -> None:
        super().__init__(name="processing", daemon=True)
        self.config = config
        self.sample_buffer = sample_buffer
        self.waterfall_queue = waterfall_queue
        self.time_domain_queue = time_domain_queue
        self.stop_event = stop_event
        self.audio_queue = audio_queue

    def run(self) -> None:
        lpf = self._design_lpf(75e3, 2.4e6, 257)
        lpf_rev = lpf

        while not self.stop_event.is_set():
            block = self.sample_buffer.wait_for_samples(
                1000,#self.config.display_update_sample_count,
                self.stop_event,
            )
            if block is None:
                continue

            x = np.convolve(block, lpf, mode="same")  # complex64, length = need_raw

            #n = self.bb_stream.append(block)
            #bb_frame, n = self.bb_stream.convolve_rev(lpf_rev)  # complex64, length = need_raw
            #self.bb_stream.remove(n)
            #x = bb_frame

            phase    = np.angle(x)
            phase_uw = np.unwrap(phase)
            phase_uw_lpf = phase_uw
            dp = np.diff(phase_uw_lpf[::50])
            y = dp / 4 / np.pi

            #bb_frame = bb_frame[::50]
            #bb_frame = block[::50]

            #self.bb_lpf_stream.append(bb_frame)
            #bb_frame, n = self.bb_lpf_stream.dphase()
            #self.bb_lpf_stream.remove(n)

            bb_frame = y

            # These are intentionally independent handoff points. A processing
            # stage can choose to push different blocks into each display, while
            # the sample rate represented by both remains the SDR baseband sample rate.
            #if self.waterfall_queue is not None:
            #    self.waterfall_queue.append(block)
            if self.time_domain_queue is not None:
                self.time_domain_queue.append(bb_frame)
            self._push_audio_samples(bb_frame)

    def _push_audio_samples(self, block: "np.ndarray[Any, Any]") -> None:
        """Push the real part of each processed sample to the audio queue."""
        import numpy as np

        if self.audio_queue is None:
            return

        audio_samples = np.real(block).astype(np.float32)
        self.audio_queue.push_samples(audio_samples)

