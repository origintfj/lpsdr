"""SDR capture thread and bounded I/Q sample handoff buffer.

The classes in this module form the producer side of the waterfall pipeline:
``SDRReaderThread`` streams complex I/Q samples directly from ``pyrtlsdr``
and appends them to ``IQSampleBuffer``.
Consumers can then make one blocking call that states exactly how many fresh
samples they need before processing can continue, or drain whatever samples are
currently available for non-blocking display updates.
"""

from __future__ import annotations

import sys
import threading
from collections import deque
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np


class SDRReaderConfig(Protocol):
    """Configuration attributes required by ``SDRReaderThread``."""

    center_freq: float
    bb_sample_rate: int
    gain: str | float
    read_size: int


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
    """Continuously read samples from pyrtlsdr into an ``IQSampleBuffer``."""

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
        self._sdr: Any | None = None
        self._sdr_lock = threading.Lock()

    def run(self) -> None:
        try:
            self._read_from_pyrtlsdr()
        except Exception as exc:
            # Any capture failure should stop the whole pipeline; otherwise the
            # FFT and GUI threads would sit idle waiting for samples forever.
            print(f"SDR reader stopped: {exc}", file=sys.stderr)
            self.stop_event.set()
        finally:
            self.stop()
            self._close_sdr()
            self.sample_buffer.wake_waiters()

    def stop(self) -> None:
        """Cancel any active pyrtlsdr async read."""
        with self._sdr_lock:
            sdr = self._sdr
        if sdr is None:
            return

        try:
            if not getattr(sdr, "read_async_canceling", False):
                sdr.cancel_read_async()
        except Exception:
            # The async read may already have returned because the device was
            # unplugged or the callback requested cancellation. The reader
            # thread will still close the SDR handle from its own finally block.
            pass

    def _close_sdr(self) -> None:
        with self._sdr_lock:
            sdr = self._sdr
            self._sdr = None
        if sdr is not None:
            sdr.close()

    def _read_from_pyrtlsdr(self) -> None:
        from rtlsdr import RtlSdr

        sdr = RtlSdr()
        with self._sdr_lock:
            self._sdr = sdr

        # Configure the dongle through librtlsdr directly instead of launching
        # the rtl_sdr command-line program and reading from a stdout pipe. The
        # pyrtlsdr async reader keeps libusb's transfer queue inside the driver
        # path, avoiding the extra process/pipe hop that can create gaps when
        # Python waits between fixed-size stdout reads.
        sdr.sample_rate = self.config.bb_sample_rate
        sdr.center_freq = round(self.config.center_freq)
        sdr.gain = self.config.gain

        def append_samples(samples: Any, context: Any) -> None:
            del context
            if self.stop_event.is_set():
                sdr.cancel_read_async()
                return
            self.sample_buffer.append(samples)

        # read_samples_async blocks until cancel_read_async() is called. The
        # callback receives already-normalized complex samples from pyrtlsdr, so
        # this thread only has to append them into the bounded handoff buffer.
        sdr.read_samples_async(append_samples, self.config.read_size)
