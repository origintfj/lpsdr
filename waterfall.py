#!/usr/bin/env python3
"""Threaded RTL-SDR waterfall and time-domain display skeleton.

This application uses one thread to continuously collect I/Q samples from an
RTL-SDR device, a processing thread to route sample blocks into display/audio
buffers, and the main Matplotlib GUI thread to render both a live time-domain I/Q
plot and a live waterfall display.

The reader uses the ``pyrtlsdr`` Python bindings to stream samples directly
from librtlsdr instead of launching the ``rtl_sdr`` command-line program.

Dependencies:
    pip install numpy matplotlib pyrtlsdr

Example:
    python3 waterfall.py --center-freq 100.1e6 --bb-sample-rate 2400000 --gain auto
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import importlib.util
from pathlib import Path
import signal
import sys
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from audio_output import AudioPlaybackThread, AudioSampleQueue
from processing_thread import ProcessingThread
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
    """Runtime configuration for the SDR, display, and optional audio pipeline."""

    # Frequencies and rates are stored in Hz so values can be passed directly to
    # pyrtlsdr/librtlsdr without unit conversion surprises.
    center_freq: float
    # Baseband sample rate from the SDR, distinct from audio playback sample rate.
    bb_sample_rate: int
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
    # Number of complex samples requested for each pyrtlsdr async callback.
    read_size: int
    # Number of FFT-sized blocks retained in the capture buffer.
    buffer_blocks: int
    # Number of display-update blocks retained for waterfall display rendering.
    display_queue_blocks: int
    # Number of time-history rows kept in the Matplotlib image.
    waterfall_rows: int
    min_db: float
    max_db: float
    # Deprecated compatibility option retained for older YAML/CLI configs.
    rtl_sdr_path: str
    enable_audio: bool
    audio_sample_rate: int
    audio_block_size: int
    audio_buffer_seconds: float
    audio_device: str | int | None


DEFAULT_CONFIG: dict[str, Any] = {
    "center_freq": None,
    "bb_sample_rate": 2_400_000,
    "gain": "auto",
    "fft_size": 2048,
    "iq_display_sample_count": 2048,
    "display_update_sample_count": 1024,
    "read_size": 16_384,
    "buffer_blocks": 32,
    "display_queue_blocks": 8,
    "waterfall_rows": 300,
    "min_db": -80.0,
    "max_db": 20.0,
    "rtl_sdr_path": "rtl_sdr",
    "enable_audio": False,
    "audio_sample_rate": 48_000,
    "audio_block_size": 1024,
    "audio_buffer_seconds": 0.5,
    "audio_device": None,
}

CONFIG_SECTION_KEYS: dict[str, dict[str, str]] = {
    "radio": {
        "center_freq": "center_freq",
        "bb_sample_rate": "bb_sample_rate",
        "sample_rate": "bb_sample_rate",
        "gain": "gain",
        "read_size": "read_size",
        "rtl_sdr_path": "rtl_sdr_path",
    },
    "processing": {
        "fft_size": "fft_size",
        "buffer_blocks": "buffer_blocks",
        "display_update_samples": "display_update_sample_count",
        "display_update_sample_count": "display_update_sample_count",
    },
    "display": {
        "time_domain_samples": "iq_display_sample_count",
        "iq_display_samples": "iq_display_sample_count",
        "iq_display_sample_count": "iq_display_sample_count",
        "display_queue_blocks": "display_queue_blocks",
        "waterfall_rows": "waterfall_rows",
        "min_db": "min_db",
        "max_db": "max_db",
    },
    "audio": {
        "enable": "enable_audio",
        "enabled": "enable_audio",
        "enable_audio": "enable_audio",
        "sample_rate": "audio_sample_rate",
        "audio_sample_rate": "audio_sample_rate",
        "block_size": "audio_block_size",
        "audio_block_size": "audio_block_size",
        "buffer_seconds": "audio_buffer_seconds",
        "audio_buffer_seconds": "audio_buffer_seconds",
        "device": "audio_device",
        "audio_device": "audio_device",
    },
}

FLAT_CONFIG_KEYS: dict[str, str] = {
    key: key for key in DEFAULT_CONFIG
}
FLAT_CONFIG_KEYS.update(
    {
        "sample_rate": "bb_sample_rate",
        "time_domain_samples": "iq_display_sample_count",
        "iq_display_samples": "iq_display_sample_count",
        "display_update_samples": "display_update_sample_count",
    }
)


class RadioGui:
    """Matplotlib GUI for time-domain I/Q and waterfall displays."""

    def __init__(
        self,
        config: RadioConfig,
        waterfall_queue: IQSampleBuffer,
        time_domain_queue: IQSampleBuffer,
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
        # roughly center_freq +/- bb_sample_rate/2.
        half_span = self.config.bb_sample_rate / 2.0
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
            / self.config.bb_sample_rate
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

        samples = self.time_domain_queue.drain_available()
        if samples is None:
            return False

        if samples.size >= self.config.iq_display_sample_count:
            self._time_domain = samples[-self.config.iq_display_sample_count :]
        else:
            self._time_domain = np.concatenate([self._time_domain, samples])[
                -self.config.iq_display_sample_count :
            ]

        if self.i_line is not None and self.q_line is not None:
            self.i_line.set_ydata(self._time_domain.real)
            self.q_line.set_ydata(self._time_domain.imag)
        return True

    def _update_waterfall(self) -> bool:
        import numpy as np

        samples = self.waterfall_queue.drain_available()
        if samples is not None:
            self._waterfall_pending = np.concatenate(
                [self._waterfall_pending, samples],
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


class ConfigError(ValueError):
    """Raised when a YAML configuration file cannot be used."""


def _normalize_config_key(key: object) -> str:
    if not isinstance(key, str):
        raise ConfigError(f"Configuration key {key!r} must be a string")
    return key.replace("-", "_")


def _strip_yaml_comment(line: str) -> str:
    in_single_quote = False
    in_double_quote = False
    escaped = False
    for index, char in enumerate(line):
        if char == "\\" and in_double_quote and not escaped:
            escaped = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote and not escaped:
            in_double_quote = not in_double_quote
        elif char == "#" and not in_single_quote and not in_double_quote:
            return line[:index].rstrip()
        escaped = False
    return line.rstrip()


def _parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False

    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_simple_yaml_mapping(text: str) -> dict[str, Any]:
    """Parse the small YAML mapping subset used by the bundled template.

    PyYAML is used when installed. This fallback intentionally supports the
    configuration style this application writes and documents: top-level scalar
    keys plus one level of nested mapping sections. It rejects lists and deeper
    structures so unsupported YAML does not get misread silently.
    """
    root: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    current_section_name: str | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = _strip_yaml_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ConfigError(
                f"YAML line {line_number}: indentation must use pairs of spaces"
            )
        if line.lstrip().startswith("-"):
            raise ConfigError(
                f"YAML line {line_number}: lists are not supported here"
            )
        if ":" not in line:
            raise ConfigError(f"YAML line {line_number}: expected a key/value pair")

        key, value = line.strip().split(":", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"YAML line {line_number}: key cannot be empty")

        if indent == 0:
            if value.strip() == "":
                current_section = {}
                current_section_name = key
                root[key] = current_section
            else:
                current_section = None
                current_section_name = None
                root[key] = _parse_yaml_scalar(value)
            continue

        if indent == 2 and current_section is not None:
            if value.strip() == "":
                raise ConfigError(
                    f"YAML line {line_number}: nested sections below "
                    f"{current_section_name!r} are not supported"
                )
            current_section[key] = _parse_yaml_scalar(value)
            continue

        raise ConfigError(
            f"YAML line {line_number}: only top-level keys and one nested mapping "
            "level are supported without PyYAML"
        )

    return root


def load_yaml_config(config_path: str | None) -> dict[str, Any]:
    """Load radio settings from a YAML file and return canonical config keys."""
    if config_path is None:
        return {}

    path = Path(config_path).expanduser()
    text = path.read_text(encoding="utf-8")
    if importlib.util.find_spec("yaml") is not None:
        import yaml

        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse YAML: {exc}") from exc
    else:
        loaded = parse_simple_yaml_mapping(text)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError("Top-level YAML content must be a mapping")

    config: dict[str, Any] = {}
    for raw_key, value in loaded.items():
        key = _normalize_config_key(raw_key)
        if key in CONFIG_SECTION_KEYS:
            if not isinstance(value, dict):
                raise ConfigError(f"YAML section {raw_key!r} must be a mapping")
            for section_raw_key, section_value in value.items():
                section_key = _normalize_config_key(section_raw_key)
                try:
                    canonical_key = CONFIG_SECTION_KEYS[key][section_key]
                except KeyError as exc:
                    raise ConfigError(
                        f"Unknown YAML option {section_raw_key!r} in section "
                        f"{raw_key!r}"
                    ) from exc
                config[canonical_key] = section_value
            continue

        try:
            canonical_key = FLAT_CONFIG_KEYS[key]
        except KeyError as exc:
            raise ConfigError(f"Unknown YAML option {raw_key!r}") from exc
        config[canonical_key] = value

    return config


def _coerce_bool(value: Any, option_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ConfigError(f"{option_name} must be true or false")


def _coerce_positive_int(value: Any, option_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{option_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{option_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ConfigError(f"{option_name} must be greater than zero")
    return parsed


def _coerce_float(value: Any, option_name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{option_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{option_name} must be a number") from exc


def _coerce_integer_hz(value: Any, option_name: str) -> int:
    try:
        return parse_integer_hz(str(value))
    except argparse.ArgumentTypeError as exc:
        raise ConfigError(f"{option_name} must be an integer number of Hz") from exc


def _coerce_gain(value: Any) -> str | float:
    if isinstance(value, str):
        try:
            return parse_gain(value)
        except ValueError as exc:
            raise ConfigError("gain must be 'auto' or a tuner gain in dB") from exc
    return _coerce_float(value, "gain")


def cli_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Return only command-line options explicitly supplied by the user."""
    return {
        key: value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }


def build_config(args: argparse.Namespace) -> RadioConfig:
    """Merge defaults, YAML settings, and CLI overrides into RadioConfig."""
    merged = DEFAULT_CONFIG | load_yaml_config(args.config) | cli_config_overrides(args)
    if merged["center_freq"] is None:
        raise ConfigError(
            "center_freq is required; set it in YAML or with --center-freq"
        )

    center_freq = _coerce_float(merged["center_freq"], "center_freq")
    bb_sample_rate = _coerce_integer_hz(merged["bb_sample_rate"], "bb_sample_rate")
    gain = _coerce_gain(merged["gain"])
    fft_size = _coerce_positive_int(merged["fft_size"], "fft_size")
    iq_display_sample_count = _coerce_positive_int(
        merged["iq_display_sample_count"], "iq_display_sample_count"
    )
    display_update_sample_count = _coerce_positive_int(
        merged["display_update_sample_count"], "display_update_sample_count"
    )
    read_size = _coerce_positive_int(merged["read_size"], "read_size")
    buffer_blocks = _coerce_positive_int(merged["buffer_blocks"], "buffer_blocks")
    display_queue_blocks = _coerce_positive_int(
        merged["display_queue_blocks"], "display_queue_blocks"
    )
    waterfall_rows = _coerce_positive_int(merged["waterfall_rows"], "waterfall_rows")
    min_db = _coerce_float(merged["min_db"], "min_db")
    max_db = _coerce_float(merged["max_db"], "max_db")
    rtl_sdr_path = str(merged["rtl_sdr_path"])
    enable_audio = _coerce_bool(merged["enable_audio"], "enable_audio")
    audio_sample_rate = _coerce_positive_int(
        merged["audio_sample_rate"], "audio_sample_rate"
    )
    audio_block_size = _coerce_positive_int(
        merged["audio_block_size"], "audio_block_size"
    )
    audio_buffer_seconds = _coerce_float(
        merged["audio_buffer_seconds"], "audio_buffer_seconds"
    )
    if audio_buffer_seconds <= 0:
        raise ConfigError("audio_buffer_seconds must be greater than zero")
    audio_device = merged["audio_device"]

    max_buffer_samples = fft_size * buffer_blocks
    if display_update_sample_count > max_buffer_samples:
        raise ConfigError(
            "display_update_sample_count cannot exceed fft_size * buffer_blocks"
        )

    return RadioConfig(
        center_freq=center_freq,
        bb_sample_rate=bb_sample_rate,
        gain=gain,
        fft_size=fft_size,
        iq_display_sample_count=iq_display_sample_count,
        display_update_sample_count=display_update_sample_count,
        read_size=read_size,
        buffer_blocks=buffer_blocks,
        display_queue_blocks=display_queue_blocks,
        waterfall_rows=waterfall_rows,
        min_db=min_db,
        max_db=max_db,
        rtl_sdr_path=rtl_sdr_path,
        enable_audio=enable_audio,
        audio_sample_rate=audio_sample_rate,
        audio_block_size=audio_block_size,
        audio_buffer_seconds=audio_buffer_seconds,
        audio_device=audio_device,
    )


def missing_runtime_dependencies(enable_audio: bool = False) -> list[str]:
    """Return Python packages needed to run the live waterfall that are missing."""
    # Checking with importlib keeps argument parsing usable in a fresh checkout;
    # the expensive imports happen only after we know the packages are present.
    required_packages = ["numpy", "matplotlib", "rtlsdr"]
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


def parse_integer_hz(value: str) -> int:
    """Parse a frequency/rate argument as an integer number of Hz."""
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid number") from exc

    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer number of Hz")
    return int(parsed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Threaded RTL-SDR waterfall and time-domain display skeleton"
    )
    parser.add_argument(
        "--config",
        help="YAML configuration file; command-line options override YAML values",
    )
    parser.add_argument(
        "--center-freq",
        type=float,
        default=None,
        help="Center frequency in Hz, e.g. 100.1e6",
    )
    parser.add_argument(
        "--bb-sample-rate",
        dest="bb_sample_rate",
        type=parse_integer_hz,
        default=None,
        help="RTL-SDR baseband sample rate in samples/sec",
    )
    parser.add_argument(
        "--sample-rate",
        dest="bb_sample_rate",
        type=parse_integer_hz,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gain",
        type=parse_gain,
        default=None,
        help="Tuner gain in dB or 'auto'",
    )
    parser.add_argument(
        "--fft-size",
        type=int,
        default=None,
        help="Number of samples per FFT row",
    )
    parser.add_argument(
        "--time-domain-samples",
        "--iq-display-samples",
        dest="iq_display_sample_count",
        type=int,
        default=None,
        help="Number of most recent I/Q samples shown in the time-domain graph",
    )
    parser.add_argument(
        "--display-update-samples",
        type=int,
        default=None,
        help=(
            "Fresh I/Q samples routed to GUI display handoff points per "
            "processing pass; "
            "independent of --fft-size"
        ),
    )
    parser.add_argument(
        "--read-size",
        type=int,
        default=None,
        help="Complex samples requested for each pyrtlsdr async callback",
    )
    parser.add_argument(
        "--buffer-blocks",
        type=int,
        default=None,
        help="Number of FFT-sized blocks retained in the I/Q buffer",
    )
    parser.add_argument(
        "--display-queue-blocks",
        type=int,
        default=None,
        help=(
            "Waterfall display buffer capacity measured in "
            "--display-update-samples blocks"
        ),
    )
    parser.add_argument(
        "--waterfall-rows",
        type=int,
        default=None,
        help="Number of rows in the waterfall history",
    )
    parser.add_argument(
        "--min-db",
        type=float,
        default=None,
        help="Waterfall color scale minimum",
    )
    parser.add_argument(
        "--max-db",
        type=float,
        default=None,
        help="Waterfall color scale maximum",
    )
    parser.add_argument(
        "--rtl-sdr-path",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--enable-audio",
        action="store_true",
        default=None,
        help="Enable a simple mono audio monitor from processed samples",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=None,
        help="Audio playback sample rate in samples/sec",
    )
    parser.add_argument(
        "--audio-block-size",
        type=int,
        default=None,
        help="Audio samples written to the sound device per block",
    )
    parser.add_argument(
        "--audio-buffer-seconds",
        type=float,
        default=None,
        help="Maximum queued audio backlog in seconds",
    )
    parser.add_argument(
        "--audio-device",
        default=None,
        help="Optional sounddevice output device name or index",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        config = build_config(args)
    except (ConfigError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    missing_packages = missing_runtime_dependencies(enable_audio=config.enable_audio)
    if missing_packages:
        print(
            "Missing Python package(s): "
            + ", ".join(missing_packages)
            + ". Install them with: python3 -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    max_buffer_samples = config.fft_size * config.buffer_blocks

    stop_event = threading.Event()
    # All I/Q handoffs use the same bounded sample-buffer type. The capture
    # buffer is sized for processing backlog, the waterfall buffer is sized from
    # the display handoff block count, and the time-domain buffer is sized only
    # by the number of samples shown in that graph.
    sample_buffer = IQSampleBuffer(max_samples=max_buffer_samples)
    waterfall_queue = IQSampleBuffer(
        max_samples=config.display_update_sample_count * config.display_queue_blocks,
    )
    time_domain_queue = IQSampleBuffer(max_samples=config.iq_display_sample_count)
    audio_queue: AudioSampleQueue | None = None
    audio_thread: AudioPlaybackThread | None = None
    if config.enable_audio:
        max_audio_samples = max(
            config.audio_block_size,
            round(config.audio_sample_rate * config.audio_buffer_seconds),
        )
        audio_queue = AudioSampleQueue(max_samples=max_audio_samples)
        audio_thread = AudioPlaybackThread(
            audio_queue,
            sample_rate=config.audio_sample_rate,
            block_size=config.audio_block_size,
            stop_event=stop_event,
            device=config.audio_device,
        )

    def request_shutdown(signum: int, _: object) -> None:
        print(f"Received signal {signum}; shutting down...")
        stop_event.set()
        # Cancelling the async read unblocks the reader, while wake_waiters
        # releases blocking conditions.
        reader.stop()
        sample_buffer.wake_waiters()
        if audio_queue is not None:
            audio_queue.wake_waiters()

    reader = SDRReaderThread(config, sample_buffer, stop_event)
    processor = ProcessingThread(
        config,
        sample_buffer,
        waterfall_queue,
        time_domain_queue,
        stop_event,
        audio_queue=audio_queue,
    )

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    reader.start()
    processor.start()
    if audio_thread is not None:
        audio_thread.start()

    try:
        RadioGui(config, waterfall_queue, time_domain_queue).start(stop_event)
    finally:
        # The GUI runs on the main thread. When it exits, always ask the worker
        # threads to stop and wait briefly so the pyrtlsdr device is cleaned up.
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
