#!/usr/bin/env python3
"""Threaded RTL-SDR waterfall display skeleton.

This application uses one thread to continuously collect I/Q samples from an
RTL-SDR device, a second thread to turn sample blocks into FFT power spectra,
and the main thread to render a live waterfall display.

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
import shutil
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    # These imports are only needed by type checkers. Keeping the real runtime
    # imports inside the classes/functions below lets `python3 waterfall.py --help`
    # work even before the user has installed NumPy and Matplotlib.
    import numpy as np
    from matplotlib.image import AxesImage


@dataclass(frozen=True)
class RadioConfig:
    """Runtime configuration for the SDR and FFT pipeline."""

    # Frequencies and rates are stored in Hz so the values can be passed to the
    # rtl_sdr command without unit conversion surprises.
    center_freq: float
    sample_rate: float
    gain: str | float
    # One FFT is computed from this many complex I/Q samples. Larger values give
    # better frequency resolution but update the waterfall more slowly.
    fft_size: int
    # Number of complex samples the reader tries to pull from rtl_sdr per read.
    read_size: int
    # Number of time-history rows kept in the Matplotlib image.
    waterfall_rows: int
    min_db: float
    max_db: float
    # Name or full path of the native rtl_sdr capture program.
    rtl_sdr_path: str


class IQSampleBuffer:
    """Thread-safe buffer for complex I/Q samples from the SDR."""

    def __init__(self, max_samples: int) -> None:
        # The deque acts as a rolling capture buffer. Once it reaches maxlen,
        # the oldest samples are discarded automatically so memory use stays
        # bounded during long runs.
        self._samples: deque[complex] = deque(maxlen=max_samples)
        # The condition protects the deque and lets the processor sleep until
        # the reader has appended enough fresh samples for another FFT.
        self._condition = threading.Condition()
        # Monotonic counter used to distinguish "new samples arrived" from
        # "there are still old samples sitting in the rolling buffer".
        self._total_written = 0

    def append(self, samples: "np.ndarray[Any, Any]") -> None:
        """Append a batch of complex samples and notify waiting consumers."""
        import numpy as np

        complex_samples = np.asarray(samples, dtype=np.complex64)
        with self._condition:
            self._samples.extend(complex_samples)
            self._total_written += len(complex_samples)
            # Wake the FFT thread if it was blocked waiting for a full block.
            self._condition.notify_all()

    def wait_for_block(
        self,
        block_size: int,
        last_total_seen: int,
        stop_event: threading.Event,
        timeout: float = 0.25,
    ) -> tuple[Optional["np.ndarray[Any, Any]"], int]:
        """Wait until enough new samples are available and return one FFT block.

        The returned block is copied out of the shared buffer so processing can
        happen without holding the condition lock. ``last_total_seen`` lets this
        method wait for genuinely new samples rather than reprocessing the same
        buffer contents repeatedly.
        """
        import numpy as np

        with self._condition:
            while not stop_event.is_set():
                enough_samples = len(self._samples) >= block_size
                enough_new_samples = self._total_written - last_total_seen >= block_size
                if enough_samples and enough_new_samples:
                    # Copy the newest block out while holding the lock, then let
                    # the caller do the expensive FFT without blocking the reader.
                    block = np.array(list(self._samples)[-block_size:], dtype=np.complex64)
                    return block, self._total_written
                # Timed waits avoid hanging forever if shutdown happens between
                # notifications or if the SDR process exits unexpectedly.
                self._condition.wait(timeout=timeout)
        return None, last_total_seen

    def wake_all(self) -> None:
        """Wake consumers so they can notice shutdown."""
        with self._condition:
            self._condition.notify_all()


class SDRReaderThread(threading.Thread):
    """Continuously read samples from rtl_sdr into the I/Q buffer."""

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
        self._process: subprocess.Popen[bytes] | None = None
        self._process_lock = threading.Lock()

    def run(self) -> None:
        try:
            self._read_from_rtl_sdr()
        except Exception as exc:
            # Any capture failure should stop the whole pipeline; otherwise the
            # FFT and GUI threads would sit idle waiting for samples forever.
            print(f"SDR reader stopped: {exc}", file=sys.stderr)
            self.stop_event.set()
        finally:
            self.stop()
            self.sample_buffer.wake_all()

    def stop(self) -> None:
        """Terminate the rtl_sdr child process, if it is still running."""
        with self._process_lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        # Ask rtl_sdr to exit gracefully first. If it is blocked in a USB read,
        # fall back to kill so the Python process can still shut down promptly.
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)

    def _read_from_rtl_sdr(self) -> None:
        import numpy as np

        rtl_sdr = shutil.which(self.config.rtl_sdr_path)
        if rtl_sdr is None:
            raise RuntimeError(
                f"Could not find '{self.config.rtl_sdr_path}'. Install rtl-sdr "
                "or pass --rtl-sdr-path with the full executable path."
            )

        # `rtl_sdr ... -` writes raw I/Q bytes to stdout. The frequency and
        # sample-rate arguments are rounded because the CLI expects integer Hz.
        command = [
            rtl_sdr,
            "-f",
            str(round(self.config.center_freq)),
            "-s",
            str(round(self.config.sample_rate)),
        ]
        if not (isinstance(self.config.gain, str) and self.config.gain.lower() == "auto"):
            command.extend(["-g", str(self.config.gain)])
        command.append("-")

        # Only stdout is piped. rtl_sdr's status messages remain on stderr so
        # users can still see hardware errors and tuning messages in the shell.
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        with self._process_lock:
            self._process = process

        # rtl_sdr emits two unsigned bytes per complex sample: I, then Q.
        bytes_per_read = self.config.read_size * 2
        while not self.stop_event.is_set():
            if process.stdout is None:
                raise RuntimeError("rtl_sdr stdout pipe was not created")

            raw = process.stdout.read(bytes_per_read)
            if not raw:
                exit_code = process.poll()
                raise RuntimeError(f"rtl_sdr exited before samples were available: {exit_code}")
            if len(raw) < bytes_per_read:
                continue

            # Convert [I0, Q0, I1, Q1, ...] from unsigned 8-bit integers into
            # complex64 samples centered near 0.0. The 127.5 midpoint maps the
            # byte range 0..255 to approximately -1.0..+1.0.
            interleaved = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            normalized = (interleaved - 127.5) / 127.5
            iq_samples = normalized[0::2] + 1j * normalized[1::2]
            self.sample_buffer.append(iq_samples)


class FFTProcessorThread(threading.Thread):
    """Wait for new I/Q samples, compute spectra, and publish waterfall rows."""

    def __init__(
        self,
        config: RadioConfig,
        sample_buffer: IQSampleBuffer,
        spectra_queue: "queue.Queue[np.ndarray[Any, Any]]",
        stop_event: threading.Event,
    ) -> None:
        import numpy as np

        super().__init__(name="fft-processor", daemon=True)
        self.config = config
        self.sample_buffer = sample_buffer
        self.spectra_queue = spectra_queue
        self.stop_event = stop_event
        # A Hann window reduces FFT spectral leakage so narrow signals smear
        # less into neighboring bins in the displayed spectrum.
        self.window = np.hanning(config.fft_size).astype(np.float32)

    def run(self) -> None:
        import numpy as np

        last_total_seen = 0
        while not self.stop_event.is_set():
            block, last_total_seen = self.sample_buffer.wait_for_block(
                self.config.fft_size,
                last_total_seen,
                self.stop_event,
            )
            if block is None:
                continue

            # FFT bins are shifted so negative offsets appear on the left, the
            # tuned center frequency appears in the middle, and positive offsets
            # appear on the right.
            windowed = block * self.window
            spectrum = np.fft.fftshift(np.fft.fft(windowed, n=self.config.fft_size))
            # A tiny offset prevents log10(0). This is relative dB power, not a
            # calibrated dBm measurement.
            power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

            try:
                self.spectra_queue.put(power_db.astype(np.float32), timeout=0.1)
            except queue.Full:
                # Drop frames if the GUI cannot keep up; the newest rows matter most.
                pass


class WaterfallDisplay:
    """Matplotlib-based waterfall display updated from processed FFT rows."""

    def __init__(
        self,
        config: RadioConfig,
        spectra_queue: "queue.Queue[np.ndarray[Any, Any]]",
    ) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        self.config = config
        self.spectra_queue = spectra_queue
        # The image array is pre-filled with the low end of the color scale so
        # the plot opens immediately, before the first FFT row arrives.
        self.waterfall = np.full(
            (config.waterfall_rows, config.fft_size),
            config.min_db,
            dtype=np.float32,
        )
        self.figure, self.axis = plt.subplots()
        self.image: Optional["AxesImage"] = None
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

    def start(self, stop_event: threading.Event) -> None:
        import numpy as np
        from matplotlib.animation import FuncAnimation

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

        def update(_: int) -> list["AxesImage"]:
            updated = False
            while True:
                try:
                    row = self.spectra_queue.get_nowait()
                except queue.Empty:
                    break
                # Roll older rows toward the bottom and place the newest FFT row
                # at the top edge of the live history.
                self.waterfall = np.roll(self.waterfall, shift=-1, axis=0)
                self.waterfall[-1, :] = row
                updated = True

            if updated and self.image is not None:
                self.image.set_data(self.waterfall)
            return [self.image] if self.image is not None else []

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


def missing_runtime_dependencies() -> list[str]:
    """Return Python packages needed to run the live waterfall that are missing."""
    # Checking with importlib keeps argument parsing usable in a fresh checkout;
    # the expensive imports happen only after we know the packages are present.
    return [
        package
        for package in ("numpy", "matplotlib")
        if importlib.util.find_spec(package) is None
    ]


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
        help="Complex samples read from rtl_sdr at a time",
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
    parser.add_argument(
        "--rtl-sdr-path",
        default="rtl_sdr",
        help="Path to the rtl_sdr executable used for sample capture",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    missing_packages = missing_runtime_dependencies()
    if missing_packages:
        print(
            "Missing Python package(s): "
            + ", ".join(missing_packages)
            + ". Install them with: python3 -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    config = RadioConfig(
        center_freq=args.center_freq,
        sample_rate=args.sample_rate,
        gain=args.gain,
        fft_size=args.fft_size,
        read_size=args.read_size,
        waterfall_rows=args.waterfall_rows,
        min_db=args.min_db,
        max_db=args.max_db,
        rtl_sdr_path=args.rtl_sdr_path,
    )

    stop_event = threading.Event()
    # The sample buffer bridges the reader and processor threads; the spectra
    # queue bridges the processor and GUI. Keeping these as separate handoff
    # points makes it clear which thread owns each stage of the pipeline.
    sample_buffer = IQSampleBuffer(max_samples=args.fft_size * args.buffer_blocks)
    spectra_queue: "queue.Queue[np.ndarray[Any, Any]]" = queue.Queue(maxsize=args.waterfall_rows)

    def request_shutdown(signum: int, _: object) -> None:
        print(f"Received signal {signum}; shutting down...")
        stop_event.set()
        # Terminating the child process unblocks the reader if it is stuck in a
        # blocking stdout read, while wake_all releases the processor condition.
        reader.stop()
        sample_buffer.wake_all()

    reader = SDRReaderThread(config, sample_buffer, stop_event)
    processor = FFTProcessorThread(config, sample_buffer, spectra_queue, stop_event)

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    reader.start()
    processor.start()

    try:
        WaterfallDisplay(config, spectra_queue).start(stop_event)
    finally:
        # The GUI runs on the main thread. When it exits, always ask the worker
        # threads to stop and wait briefly so the rtl_sdr process is cleaned up.
        stop_event.set()
        reader.stop()
        sample_buffer.wake_all()
        reader.join(timeout=2.0)
        processor.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
