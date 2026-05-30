#!/usr/bin/env python3
"""Threaded RTL-SDR waterfall display skeleton.

This application uses one thread to continuously collect I/Q samples from an
RTL-SDR device, a second thread to turn sample blocks into FFT power spectra,
and the main thread to render a live waterfall display.

Dependencies:
    pip install numpy matplotlib pyrtlsdr

Example:
    python3 waterfall.py --center-freq 100.1e6 --sample-rate 2.4e6 --gain auto
"""

from __future__ import annotations

import argparse
import queue
import signal
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.image import AxesImage

from rtlsdr import RtlSdr


@dataclass(frozen=True)
class RadioConfig:
    """Runtime configuration for the SDR and FFT pipeline."""

    center_freq: float
    sample_rate: float
    gain: str | float
    fft_size: int
    read_size: int
    waterfall_rows: int
    min_db: float
    max_db: float


class IQSampleBuffer:
    """Thread-safe buffer for complex I/Q samples from the SDR."""

    def __init__(self, max_samples: int) -> None:
        self._samples: Deque[np.complex64] = deque(maxlen=max_samples)
        self._condition = threading.Condition()
        self._total_written = 0

    def append(self, samples: np.ndarray) -> None:
        """Append a batch of complex samples and notify waiting consumers."""
        complex_samples = np.asarray(samples, dtype=np.complex64)
        with self._condition:
            self._samples.extend(complex_samples)
            self._total_written += len(complex_samples)
            self._condition.notify_all()

    def wait_for_block(
        self,
        block_size: int,
        last_total_seen: int,
        stop_event: threading.Event,
        timeout: float = 0.25,
    ) -> tuple[Optional[np.ndarray], int]:
        """Wait until enough new samples are available and return one FFT block.

        The returned block is copied out of the shared buffer so processing can
        happen without holding the condition lock. ``last_total_seen`` lets this
        method wait for genuinely new samples rather than reprocessing the same
        buffer contents repeatedly.
        """
        with self._condition:
            while not stop_event.is_set():
                enough_samples = len(self._samples) >= block_size
                enough_new_samples = self._total_written - last_total_seen >= block_size
                if enough_samples and enough_new_samples:
                    block = np.array(list(self._samples)[-block_size:], dtype=np.complex64)
                    return block, self._total_written
                self._condition.wait(timeout=timeout)
        return None, last_total_seen

    def wake_all(self) -> None:
        """Wake consumers so they can notice shutdown."""
        with self._condition:
            self._condition.notify_all()


class SDRReaderThread(threading.Thread):
    """Continuously read samples from the RTL-SDR into the I/Q buffer."""

    def __init__(
        self,
        config: RadioConfig,
        sample_buffer: IQSampleBuffer,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="sdr-reader", daemon=True)
        self.config = config
        self.sample_buffer = sample_buffer
        self.stop_event = stop_event

    def run(self) -> None:
        sdr = RtlSdr()
        try:
            sdr.sample_rate = self.config.sample_rate
            sdr.center_freq = self.config.center_freq
            if isinstance(self.config.gain, str) and self.config.gain.lower() == "auto":
                sdr.gain = "auto"
            else:
                sdr.gain = float(self.config.gain)

            while not self.stop_event.is_set():
                samples = sdr.read_samples(self.config.read_size)
                self.sample_buffer.append(samples)
        finally:
            sdr.close()
            self.sample_buffer.wake_all()


class FFTProcessorThread(threading.Thread):
    """Wait for new I/Q samples, compute spectra, and publish waterfall rows."""

    def __init__(
        self,
        config: RadioConfig,
        sample_buffer: IQSampleBuffer,
        spectra_queue: "queue.Queue[np.ndarray]",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="fft-processor", daemon=True)
        self.config = config
        self.sample_buffer = sample_buffer
        self.spectra_queue = spectra_queue
        self.stop_event = stop_event
        self.window = np.hanning(config.fft_size).astype(np.float32)

    def run(self) -> None:
        last_total_seen = 0
        while not self.stop_event.is_set():
            block, last_total_seen = self.sample_buffer.wait_for_block(
                self.config.fft_size,
                last_total_seen,
                self.stop_event,
            )
            if block is None:
                continue

            windowed = block * self.window
            spectrum = np.fft.fftshift(np.fft.fft(windowed, n=self.config.fft_size))
            power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

            try:
                self.spectra_queue.put(power_db.astype(np.float32), timeout=0.1)
            except queue.Full:
                # Drop frames if the GUI cannot keep up; the newest rows matter most.
                pass


class WaterfallDisplay:
    """Matplotlib-based waterfall display updated from processed FFT rows."""

    def __init__(self, config: RadioConfig, spectra_queue: "queue.Queue[np.ndarray]") -> None:
        self.config = config
        self.spectra_queue = spectra_queue
        self.waterfall = np.full(
            (config.waterfall_rows, config.fft_size),
            config.min_db,
            dtype=np.float32,
        )
        self.figure, self.axis = plt.subplots()
        self.image: Optional[AxesImage] = None

    def _frequency_extent_mhz(self) -> list[float]:
        half_span = self.config.sample_rate / 2.0
        return [
            (self.config.center_freq - half_span) / 1e6,
            (self.config.center_freq + half_span) / 1e6,
            0,
            self.config.waterfall_rows,
        ]

    def start(self, stop_event: threading.Event) -> None:
        self.image = self.axis.imshow(
            self.waterfall,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=self._frequency_extent_mhz(),
            vmin=self.config.min_db,
            vmax=self.config.max_db,
            cmap="viridis",
        )
        self.axis.set_title(
            f"RTL-SDR Waterfall centered at {self.config.center_freq / 1e6:.6f} MHz"
        )
        self.axis.set_xlabel("Frequency (MHz)")
        self.axis.set_ylabel("Time (newest at top)")
        self.figure.colorbar(self.image, ax=self.axis, label="Power (dB)")

        def update(_: int) -> list[AxesImage]:
            updated = False
            while True:
                try:
                    row = self.spectra_queue.get_nowait()
                except queue.Empty:
                    break
                self.waterfall = np.roll(self.waterfall, shift=-1, axis=0)
                self.waterfall[-1, :] = row
                updated = True

            if updated and self.image is not None:
                self.image.set_data(self.waterfall)
            return [self.image] if self.image is not None else []

        def on_close(_: object) -> None:
            stop_event.set()

        self.figure.canvas.mpl_connect("close_event", on_close)
        self.animation = FuncAnimation(
            self.figure,
            update,
            interval=50,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


def parse_gain(value: str) -> str | float:
    """Parse gain as either the literal 'auto' or a tuner gain in dB."""
    if value.lower() == "auto":
        return "auto"
    return float(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Threaded RTL-SDR waterfall display skeleton")
    parser.add_argument(
        "--center-freq",
        type=float,
        required=True,
        help="Center frequency in Hz, e.g. 100.1e6",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=2.4e6,
        help="RTL-SDR sample rate in samples/sec",
    )
    parser.add_argument(
        "--gain",
        type=parse_gain,
        default="auto",
        help="Tuner gain in dB or 'auto'",
    )
    parser.add_argument(
        "--fft-size",
        type=int,
        default=2048,
        help="Number of samples per FFT row",
    )
    parser.add_argument(
        "--read-size",
        type=int,
        default=16_384,
        help="Samples read from the SDR at a time",
    )
    parser.add_argument(
        "--buffer-blocks",
        type=int,
        default=32,
        help="Number of FFT-sized blocks retained in the I/Q buffer",
    )
    parser.add_argument(
        "--waterfall-rows",
        type=int,
        default=300,
        help="Number of rows in the waterfall history",
    )
    parser.add_argument(
        "--min-db",
        type=float,
        default=-80.0,
        help="Waterfall color scale minimum",
    )
    parser.add_argument(
        "--max-db",
        type=float,
        default=20.0,
        help="Waterfall color scale maximum",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = RadioConfig(
        center_freq=args.center_freq,
        sample_rate=args.sample_rate,
        gain=args.gain,
        fft_size=args.fft_size,
        read_size=args.read_size,
        waterfall_rows=args.waterfall_rows,
        min_db=args.min_db,
        max_db=args.max_db,
    )

    stop_event = threading.Event()
    sample_buffer = IQSampleBuffer(max_samples=args.fft_size * args.buffer_blocks)
    spectra_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=args.waterfall_rows)

    def request_shutdown(signum: int, _: object) -> None:
        print(f"Received signal {signum}; shutting down...")
        stop_event.set()
        sample_buffer.wake_all()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    reader = SDRReaderThread(config, sample_buffer, stop_event)
    processor = FFTProcessorThread(config, sample_buffer, spectra_queue, stop_event)
    reader.start()
    processor.start()

    try:
        WaterfallDisplay(config, spectra_queue).start(stop_event)
    finally:
        stop_event.set()
        sample_buffer.wake_all()
        reader.join(timeout=2.0)
        processor.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
