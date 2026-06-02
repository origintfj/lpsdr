"""RTL-SDR sample source helpers."""

from dataclasses import dataclass

import numpy as np
from rtlsdr import RtlSdr


@dataclass(frozen=True)
class RadioConfig:
    """Configuration values applied to the RTL-SDR device."""

    center_freq_hz: float
    sample_rate_hz: float
    gain: float | str = "auto"
    device_index: int = 0


class SDRSampleSource:
    """Small wrapper exposing a blocking `next_samples` method."""

    def __init__(self, config: RadioConfig) -> None:
        self.config = config
        self._sdr = RtlSdr(device_index=config.device_index)
        self._configure(config)

    def _configure(self, config: RadioConfig) -> None:
        self._sdr.sample_rate = config.sample_rate_hz
        self._sdr.center_freq = config.center_freq_hz
        self._sdr.gain = config.gain

    def next_samples(self, n_samples: int) -> np.ndarray:
        """Return the next `n_samples` complex IQ samples from the radio."""
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than zero")
        return self._sdr.read_samples(n_samples)

    def close(self) -> None:
        """Release the RTL-SDR device."""
        self._sdr.close()

    def __enter__(self) -> "SDRSampleSource":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()
