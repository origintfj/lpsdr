"""RTL-SDR sample source helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

_PKG_RESOURCES_WARNING = r"pkg_resources is deprecated as an API\..*"


@dataclass(frozen=True)
class RadioConfig:
    """Configuration values applied to the RTL-SDR device."""

    center_freq_hz: float
    sample_rate_hz: float
    gain: float | str = "auto"
    device_index: int = 0


def load_rtlsdr_class() -> type[Any]:
    """Load pyrtlsdr's device class while silencing its setuptools warning.

    The pyrtlsdr release currently imports ``pkg_resources`` from its package
    ``__init__``. Newer Python 3.13+ virtual environments with recent
    setuptools versions emit a deprecation warning for that import. Keep the
    warning local to this compatibility shim so the application can use that
    pyrtlsdr version without noisy startup output.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_PKG_RESOURCES_WARNING,
            category=UserWarning,
        )
        from rtlsdr import RtlSdr

    if RtlSdr is None:
        raise RuntimeError("pyrtlsdr did not provide an RtlSdr device class")
    return RtlSdr


class SDRSampleSource:
    """Small wrapper exposing a blocking `next_samples` method."""

    def __init__(self, config: RadioConfig) -> None:
        self.config = config
        self._sdr = load_rtlsdr_class()(device_index=config.device_index)
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
