"""SDR capture thread and bounded I/Q sample handoff buffer.

The classes in this module form the producer side of the waterfall pipeline:
``SDRReaderThread`` reads raw bytes from the ``rtl_sdr`` command-line tool,
converts them to complex I/Q samples, and appends them to ``IQSampleBuffer``.
Consumers can then make one blocking call that states exactly how many fresh
samples they need before processing can continue, or drain whatever samples are
currently available for non-blocking display updates.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from collections import deque
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np


class SDRReaderConfig(Protocol):
    """Configuration attributes required by ``SDRReaderThread``."""

    center_freq: float
    sample_rate: float
    gain: str | float
    read_size: int
    rtl_sdr_path: str


class IQSampleBuffer:
    """Thread-safe bounded FIFO buffer for complex I/Q samples.

    ``max_samples`` fixes the largest backlog retained by the buffer. If a
    producer appends samples faster than a consumer drains them, the oldest
    unconsumed samples are discarded by the underlying rolling deque so memory
    usage stays bounded. The same buffer type is used for the capture handoff,
    waterfall handoff, and time-domain display handoff.
    """

    def __init__(self, max_samples: int) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be greater than zero")

        self._max_samples = max_samples
        self._samples: deque[complex] = deque(maxlen=max_samples)
        self._condition = threading.Condition()

    @property
    def max_samples(self) -> int:
        """Maximum number of complex samples retained by this buffer."""
        return self._max_samples

    def append(self, samples: "np.ndarray[Any, Any]") -> None:
        """Append a batch of complex samples and notify waiting consumers."""
        import numpy as np

        complex_samples = np.asarray(samples, dtype=np.complex64).reshape(-1)
        if complex_samples.size == 0:
            return

        with self._condition:
            self._samples.extend(complex_samples)
            self._condition.notify_all()

    def wait_for_samples(
        self,
        sample_count: int,
        stop_event: threading.Event,
        timeout: float = 0.25,
    ) -> "np.ndarray[Any, Any] | None":
        """Block until ``sample_count`` new samples are ready, then consume them.

        The returned samples are removed from the buffer, so the next call waits
        for and returns samples that have not previously been handed to the
        consumer. ``sample_count`` must not exceed ``max_samples``; otherwise the
        request can never be satisfied by this bounded buffer.
        """
        import numpy as np

        if sample_count <= 0:
            raise ValueError("sample_count must be greater than zero")
        if sample_count > self._max_samples:
            raise ValueError(
                f"sample_count ({sample_count}) cannot exceed max_samples "
                f"({self._max_samples})"
            )

        with self._condition:
            while not stop_event.is_set():
                if len(self._samples) >= sample_count:
                    return np.fromiter(
                        (self._samples.popleft() for _ in range(sample_count)),
                        dtype=np.complex64,
                        count=sample_count,
                    )
                self._condition.wait(timeout=timeout)
        return None

    def drain_available(self) -> "np.ndarray[Any, Any] | None":
        """Return and consume all samples currently retained, if any.

        This method is intentionally non-blocking for GUI update paths. If the
        buffer has overflowed since the previous drain, the returned array
        contains the newest samples still retained by ``max_samples``.
        """
        import numpy as np

        with self._condition:
            sample_count = len(self._samples)
            if sample_count == 0:
                return None
            return np.fromiter(
                (self._samples.popleft() for _ in range(sample_count)),
                dtype=np.complex64,
                count=sample_count,
            )

    def wake_waiters(self) -> None:
        """Wake blocked consumers so they can notice shutdown."""
        with self._condition:
            self._condition.notify_all()


class SDRReaderThread(threading.Thread):
    """Continuously read samples from rtl_sdr into an ``IQSampleBuffer``."""

    def __init__(
        self,
        config: SDRReaderConfig,
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
            self.sample_buffer.wake_waiters()

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
        gain_is_auto = (
            isinstance(self.config.gain, str)
            and self.config.gain.lower() == "auto"
        )
        if not gain_is_auto:
            command.extend(["-g", str(self.config.gain)])
        command.append("-")

        # Only stdout is piped. rtl_sdr's status messages remain on stderr so
        # users can still see hardware errors and tuning messages in the shell.
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        with self._process_lock:
            self._process = process

        # rtl_sdr emits two unsigned bytes per complex sample: I, then Q. For
        # RTL-SDR dongles this is not a selectable output depth in this script;
        # it reflects the native 8-bit sample format exposed by the RTL2832U
        # hardware/rtl_sdr tool. Different SDR hardware may support wider ADCs,
        # but this rtl_sdr-based reader is intentionally written for the common
        # 8-bit RTL-SDR byte stream.
        bytes_per_read = self.config.read_size * 2
        while not self.stop_event.is_set():
            if process.stdout is None:
                raise RuntimeError("rtl_sdr stdout pipe was not created")

            raw = process.stdout.read(bytes_per_read)
            if not raw:
                exit_code = process.poll()
                raise RuntimeError(
                    f"rtl_sdr exited before samples were available: {exit_code}"
                )
            if len(raw) < bytes_per_read:
                continue

            # Convert [I0, Q0, I1, Q1, ...] from unsigned 8-bit integers into
            # complex64 samples centered near 0.0. The 127.5 midpoint maps the
            # byte range 0..255 to approximately -1.0..+1.0; converting to
            # float/complex64 makes the later FFT math convenient, but it does
            # not add resolution beyond the original 8-bit measurements.
            interleaved = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            normalized = (interleaved - 127.5) / 127.5
            iq_samples = normalized[0::2] + 1j * normalized[1::2]
            self.sample_buffer.append(iq_samples)
