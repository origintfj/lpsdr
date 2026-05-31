#!/usr/bin/env python3
"""Threaded RTL-SDR waterfall and time-domain display skeleton.

This application uses one thread to continuously collect I/Q samples from an
RTL-SDR device, a processing thread to route sample blocks into display/audio
queues, and the main Matplotlib GUI thread to render both a live time-domain I/Q
plot and a live waterfall display.

The reader uses the standard ``rtl_sdr`` command-line program rather than the
``pyrtlsdr`` Python bindings. That keeps the skeleton compatible with systems
where the installed Python package expects newer ``librtlsdr`` symbols than the
host library provides.

Dependencies:
    pip install numpy matplotlib

Example:
    python3 waterfall.py --center-freq 100.1e6 --sample-rate 2.4e6 --gain auto
"""

from __future__ import annotations

import argparse
import importlib.util
import queue
import signal
import sys
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from audio_output import AudioPlaybackThread, AudioSampleQueue
from sdr_reader import IQSampleBuffer, SDRReaderThread

if TYPE_CHECKING:
    # These imports are only needed by type checkers. Keeping the real runtime
    # imports inside the classes/functions below lets `python3 waterfall.py --help`
    # work even before the user has installed NumPy and Matplotlib.
    import numpy as np
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D


@dataclass(frozen=True)
class RadioConfig:
    """Runtime configuration for the SDR and display pipeline."""

    # Frequencies and rates are stored in Hz so the values can be passed to the
    # rtl_sdr command without unit conversion surprises.
    center_freq: float
    sample_rate: float
    gain: str | float
    # One FFT is computed from this many complex I/Q samples. Larger values give
    # better frequency resolution but update the waterfall more slowly.
    fft_size: int
    # Number of complex samples shown in the time-domain I/Q graph.
    iq_display_sample_count: int
    # Number of fresh complex samples routed to displays per processing pass.
    # This controls GUI update granularity independently from the FFT size used
    # to build waterfall rows.
    display_update_sample_count: int
    # Number of complex samples the reader tries to pull from rtl_sdr per read.
    read_size: int
    # Number of time-history rows kept in the Matplotlib image.
    waterfall_rows: int
    min_db: float
    max_db: float
    # Name or full path of the native rtl_sdr capture program.
    rtl_sdr_path: str


class IQDisplaySampleQueue:
    """Bounded queue of complex I/Q sample blocks for GUI displays.

    Producers call :meth:`push_samples` with an arbitrary-sized NumPy-compatible
    complex sample batch. The GUI thread calls :meth:`drain_available` to take
    ownership of any queued blocks without blocking. If the queue fills, oldest
    blocks are dropped so live displays prefer recent samples over stale backlog.
    """

    def __init__(self, max_blocks: int) -> None:
        if max_blocks <= 0:
            raise ValueError("max_blocks must be greater than zero")
        self._blocks: "queue.Queue[np.ndarray[Any, Any]]" = queue.Queue(
            maxsize=max_blocks
        )

    @property
    def max_blocks(self) -> int:
        """Maximum number of queued sample blocks retained."""
        return self._blocks.maxsize

    def push_samples(self, samples: "np.ndarray[Any, Any]") -> None:
        """Queue a complex sample block, dropping the oldest block if full."""
        import numpy as np

        iq_samples = np.asarray(samples, dtype=np.complex64).reshape(-1)
        if iq_samples.size == 0:
            return

        while True:
            try:
                self._blocks.put_nowait(iq_samples.copy())
                return
            except queue.Full:
                try:
                    self._blocks.get_nowait()
                except queue.Empty:
                    continue

    def drain_available(self) -> list["np.ndarray[Any, Any]"]:
        """Return all currently queued sample blocks without blocking."""
        blocks: list["np.ndarray[Any, Any]"] = []
        while True:
            try:
                blocks.append(self._blocks.get_nowait())
            except queue.Empty:
                return blocks


class SampleRouterThread(threading.Thread):
    """Wait for new I/Q samples and publish them to display/audio queues."""

    def __init__(
        self,
        config: RadioConfig,
        sample_buffer: IQSampleBuffer,
        waterfall_queue: IQDisplaySampleQueue | None,
        time_domain_queue: IQDisplaySampleQueue | None,
        stop_event: threading.Event,
        audio_queue: AudioSampleQueue | None = None,
        audio_sample_rate: int | None = None,
        audio_gain: float = 0.2,
    ) -> None:
        super().__init__(name="sample-router", daemon=True)
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

            # These are intentionally independent queues. A processing stage can
            # choose to push different blocks into each display, while the sample
            # rate represented by both queues remains the SDR sample rate.
            if self.waterfall_queue is not None:
                self.waterfall_queue.push_samples(block)
            if self.time_domain_queue is not None:
                self.time_domain_queue.push_samples(block)
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


class WaterfallDisplay:
    """Matplotlib GUI for time-domain I/Q and waterfall displays."""

    def __init__(
        self,
        config: RadioConfig,
        waterfall_queue: IQDisplaySampleQueue,
        time_domain_queue: IQDisplaySampleQueue,
    ) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        self.config = config
        self.waterfall_queue = waterfall_queue
        self.time_domain_queue = time_domain_queue
        # The image array is pre-filled with the low end of the color scale so
        # the plot opens immediately, before the first FFT row arrives.
        self.waterfall = np.full(
            (config.waterfall_rows, config.fft_size),
            config.min_db,
            dtype=np.float32,
        )
        self._waterfall_pending = np.empty(0, dtype=np.complex64)
        self._time_domain = np.zeros(
            config.iq_display_sample_count,
            dtype=np.complex64,
        )
        self.window = np.hanning(config.fft_size).astype(np.float32)
        self.figure, (self.time_axis, self.waterfall_axis) = plt.subplots(
            2,
            1,
            gridspec_kw={"height_ratios": [1, 2]},
            constrained_layout=True,
        )
        self.image: Optional["AxesImage"] = None
        self.i_line: Optional["Line2D"] = None
        self.q_line: Optional["Line2D"] = None
        self.animation: Any = None
        self._plt = plt

    def _frequency_extent_mhz(self) -> list[float]:
        # Matplotlib uses this extent to label the x-axis in actual RF frequency
        # rather than FFT-bin number. With complex I/Q data, the visible span is
        # roughly center_freq +/- sample_rate/2.
        half_span = self.config.sample_rate / 2.0
        return [
            (self.config.center_freq - half_span) / 1e6,
            (self.config.center_freq + half_span) / 1e6,
            0,
            self.config.waterfall_rows,
        ]

    def _time_axis_ms(self) -> "np.ndarray[Any, Any]":
        import numpy as np

        return (
            np.arange(self.config.iq_display_sample_count, dtype=np.float32)
            / self.config.sample_rate
            * 1_000.0
        )

    def start(self, stop_event: threading.Event) -> None:
        from matplotlib.animation import FuncAnimation

        time_axis_ms = self._time_axis_ms()
        (self.i_line,) = self.time_axis.plot(
            time_axis_ms,
            self._time_domain.real,
            label="I",
            linewidth=1.0,
        )
        (self.q_line,) = self.time_axis.plot(
            time_axis_ms,
            self._time_domain.imag,
            label="Q",
            linewidth=1.0,
        )
        self.time_axis.set_title("Time-domain I/Q samples")
        self.time_axis.set_xlabel("Time (ms)")
        self.time_axis.set_ylabel("Amplitude")
        self.time_axis.set_ylim(-1.1, 1.1)
        self.time_axis.grid(True, alpha=0.3)
        self.time_axis.legend(loc="upper right")

        self.image = self.waterfall_axis.imshow(
            self.waterfall,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=self._frequency_extent_mhz(),
            vmin=self.config.min_db,
            vmax=self.config.max_db,
            cmap="viridis",
        )
        self.waterfall_axis.set_title(
            f"RTL-SDR Waterfall centered at {self.config.center_freq / 1e6:.6f} MHz"
        )
        self.waterfall_axis.set_xlabel("Frequency (MHz)")
        self.waterfall_axis.set_ylabel("Time (newest at top)")
        self.figure.colorbar(self.image, ax=self.waterfall_axis, label="Power (dB)")

        def update(_: int) -> list[Any]:
            artists: list[Any] = []
            time_domain_updated = (
                self._update_time_domain()
                and self.i_line is not None
                and self.q_line is not None
            )
            if time_domain_updated:
                artists.extend([self.i_line, self.q_line])
            if self._update_waterfall() and self.image is not None:
                artists.append(self.image)
            return artists

        def on_close(_: object) -> None:
            # Closing the plot window is treated the same as Ctrl-C: all worker
            # threads should observe the stop event and exit.
            stop_event.set()

        self.figure.canvas.mpl_connect("close_event", on_close)
        # Keep a reference to the animation object. If it is only a local
        # variable, Matplotlib may garbage-collect it and stop GUI updates.
        self.animation = FuncAnimation(
            self.figure,
            update,
            interval=50,
            blit=False,
            cache_frame_data=False,
        )
        self._plt.show()

    def _update_time_domain(self) -> bool:
        import numpy as np

        blocks = self.time_domain_queue.drain_available()
        if not blocks:
            return False

        samples = np.concatenate([self._time_domain, *blocks]).astype(
            np.complex64,
            copy=False,
        )
        self._time_domain = samples[-self.config.iq_display_sample_count :]

        if self.i_line is not None and self.q_line is not None:
            self.i_line.set_ydata(self._time_domain.real)
            self.q_line.set_ydata(self._time_domain.imag)
        return True

    def _update_waterfall(self) -> bool:
        import numpy as np

        blocks = self.waterfall_queue.drain_available()
        if blocks:
            self._waterfall_pending = np.concatenate(
                [self._waterfall_pending, *blocks]
            ).astype(np.complex64, copy=False)

        updated = False
        while self._waterfall_pending.size >= self.config.fft_size:
            block = self._waterfall_pending[: self.config.fft_size]
            self._waterfall_pending = self._waterfall_pending[self.config.fft_size :]
            # FFT bins are shifted so negative offsets appear on the left, the
            # tuned center frequency appears in the middle, and positive offsets
            # appear on the right.
            windowed = block * self.window
            spectrum = np.fft.fftshift(np.fft.fft(windowed, n=self.config.fft_size))
            # A tiny offset prevents log10(0). This is relative dB power, not a
            # calibrated dBm measurement.
            power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
            # Roll older rows toward the bottom and place the newest FFT row at
            # the top edge of the live history.
            self.waterfall = np.roll(self.waterfall, shift=-1, axis=0)
            self.waterfall[-1, :] = power_db.astype(np.float32)
            updated = True

        if updated and self.image is not None:
            self.image.set_data(self.waterfall)
        return updated


def missing_runtime_dependencies(enable_audio: bool = False) -> list[str]:
    """Return Python packages needed to run the live waterfall that are missing."""
    # Checking with importlib keeps argument parsing usable in a fresh checkout;
    # the expensive imports happen only after we know the packages are present.
    required_packages = ["numpy", "matplotlib"]
    if enable_audio:
        required_packages.append("sounddevice")
    return [
        package
        for package in required_packages
        if importlib.util.find_spec(package) is None
    ]


def parse_gain(value: str) -> str | float:
    """Parse gain as either the literal 'auto' or a tuner gain in dB."""
    if value.lower() == "auto":
        return "auto"
    return float(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Threaded RTL-SDR waterfall and time-domain display skeleton"
    )
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
        "--time-domain-samples",
        "--iq-display-samples",
        dest="iq_display_sample_count",
        type=int,
        default=2048,
        help="Number of most recent I/Q samples shown in the time-domain graph",
    )
    parser.add_argument(
        "--display-update-samples",
        type=int,
        default=1024,
        help=(
            "Fresh I/Q samples routed to GUI display queues per processing pass; "
            "independent of --fft-size"
        ),
    )
    parser.add_argument(
        "--read-size",
        type=int,
        default=16_384,
        help="Complex samples read from rtl_sdr at a time",
    )
    parser.add_argument(
        "--buffer-blocks",
        type=int,
        default=32,
        help="Number of FFT-sized blocks retained in the I/Q buffer",
    )
    parser.add_argument(
        "--display-queue-blocks",
        type=int,
        default=8,
        help="Maximum queued sample blocks retained per GUI display",
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
    parser.add_argument(
        "--rtl-sdr-path",
        default="rtl_sdr",
        help="Path to the rtl_sdr executable used for sample capture",
    )
    parser.add_argument(
        "--enable-audio",
        action="store_true",
        help="Enable a simple mono audio monitor from processed samples",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=48_000,
        help="Audio playback sample rate in samples/sec",
    )
    parser.add_argument(
        "--audio-block-size",
        type=int,
        default=1024,
        help="Audio samples written to the sound device per block",
    )
    parser.add_argument(
        "--audio-buffer-seconds",
        type=float,
        default=0.5,
        help="Maximum queued audio backlog in seconds",
    )
    parser.add_argument(
        "--audio-gain",
        type=float,
        default=0.2,
        help="Linear gain applied before samples are queued for audio playback",
    )
    parser.add_argument(
        "--audio-device",
        default=None,
        help="Optional sounddevice output device name or index",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    missing_packages = missing_runtime_dependencies(enable_audio=args.enable_audio)
    if missing_packages:
        print(
            "Missing Python package(s): "
            + ", ".join(missing_packages)
            + ". Install them with: python3 -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    if args.fft_size <= 0:
        print("--fft-size must be greater than zero", file=sys.stderr)
        return 2
    if args.buffer_blocks <= 0:
        print("--buffer-blocks must be greater than zero", file=sys.stderr)
        return 2
    if args.iq_display_sample_count <= 0:
        print("--time-domain-samples must be greater than zero", file=sys.stderr)
        return 2
    if args.display_update_samples <= 0:
        print("--display-update-samples must be greater than zero", file=sys.stderr)
        return 2
    max_buffer_samples = args.fft_size * args.buffer_blocks
    if args.display_update_samples > max_buffer_samples:
        print(
            "--display-update-samples cannot exceed --fft-size * --buffer-blocks",
            file=sys.stderr,
        )
        return 2
    if args.display_queue_blocks <= 0:
        print("--display-queue-blocks must be greater than zero", file=sys.stderr)
        return 2

    config = RadioConfig(
        center_freq=args.center_freq,
        sample_rate=args.sample_rate,
        gain=args.gain,
        fft_size=args.fft_size,
        iq_display_sample_count=args.iq_display_sample_count,
        display_update_sample_count=args.display_update_samples,
        read_size=args.read_size,
        waterfall_rows=args.waterfall_rows,
        min_db=args.min_db,
        max_db=args.max_db,
        rtl_sdr_path=args.rtl_sdr_path,
    )

    stop_event = threading.Event()
    # The sample buffer bridges the reader and processing thread. The two GUI
    # queues are independent handoff points: processing code can push one sample
    # block stream for waterfall FFTs and another stream for the time-domain plot.
    sample_buffer = IQSampleBuffer(max_samples=max_buffer_samples)
    waterfall_queue = IQDisplaySampleQueue(max_blocks=args.display_queue_blocks)
    time_domain_queue = IQDisplaySampleQueue(max_blocks=args.display_queue_blocks)
    audio_queue: AudioSampleQueue | None = None
    audio_thread: AudioPlaybackThread | None = None
    if args.enable_audio:
        max_audio_samples = max(
            args.audio_block_size,
            round(args.audio_sample_rate * args.audio_buffer_seconds),
        )
        audio_queue = AudioSampleQueue(max_samples=max_audio_samples)
        audio_thread = AudioPlaybackThread(
            audio_queue,
            sample_rate=args.audio_sample_rate,
            block_size=args.audio_block_size,
            stop_event=stop_event,
            device=args.audio_device,
        )

    def request_shutdown(signum: int, _: object) -> None:
        print(f"Received signal {signum}; shutting down...")
        stop_event.set()
        # Terminating the child process unblocks the reader if it is stuck in a
        # blocking stdout read, while wake_waiters releases blocking conditions.
        reader.stop()
        sample_buffer.wake_waiters()
        if audio_queue is not None:
            audio_queue.wake_waiters()

    reader = SDRReaderThread(config, sample_buffer, stop_event)
    processor = SampleRouterThread(
        config,
        sample_buffer,
        waterfall_queue,
        time_domain_queue,
        stop_event,
        audio_queue=audio_queue,
        audio_sample_rate=args.audio_sample_rate if args.enable_audio else None,
        audio_gain=args.audio_gain,
    )

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    reader.start()
    processor.start()
    if audio_thread is not None:
        audio_thread.start()

    try:
        WaterfallDisplay(config, waterfall_queue, time_domain_queue).start(stop_event)
    finally:
        # The GUI runs on the main thread. When it exits, always ask the worker
        # threads to stop and wait briefly so the rtl_sdr process is cleaned up.
        stop_event.set()
        reader.stop()
        sample_buffer.wake_waiters()
        if audio_queue is not None:
            audio_queue.wake_waiters()
        reader.join(timeout=2.0)
        processor.join(timeout=2.0)
        if audio_thread is not None:
            audio_thread.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
