"""Application entry point for the lightweight RTL-SDR audio skeleton."""

from __future__ import annotations

import argparse
import queue
import signal
import threading

import numpy as np

from .audio import AudioPlayer
from .processing import ProcessingConfig, ProcessingThread
from .radio import RadioConfig, SDRSampleSource


def initialise_radio(args: argparse.Namespace) -> SDRSampleSource:
    """Create and configure the RTL-SDR sample source from CLI arguments."""
    config = RadioConfig(
        center_freq_hz=args.center_freq,
        sample_rate_hz=args.sample_rate,
        gain=args.gain,
        device_index=args.device_index,
    )
    return SDRSampleSource(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream RTL-SDR real samples to audio output."
    )
    parser.add_argument(
        "--center-freq", type=float, default=100_000_000, help="RF center frequency in Hz"
    )
    parser.add_argument(
        "--sample-rate", type=float, default=48_000, help="SDR/audio sample rate in Hz"
    )
    parser.add_argument("--gain", default="auto", help="RTL-SDR gain value, or 'auto'")
    parser.add_argument("--device-index", type=int, default=0, help="RTL-SDR device index")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Samples to read and process at a time",
    )
    parser.add_argument(
        "--queue-size", type=int, default=16, help="Maximum queued audio chunks"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stop_event = threading.Event()

    def request_stop(signum, frame) -> None:  # type: ignore[no-untyped-def]
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=args.queue_size)
    processing_config = ProcessingConfig(chunk_size=args.chunk_size)

    with initialise_radio(args) as sample_source:
        processor = ProcessingThread(
            sample_source, audio_queue, processing_config, stop_event
        )
        processor.start()

        with AudioPlayer(audio_queue, args.sample_rate, args.chunk_size):
            print("lpsdr running; press Ctrl-C to stop", flush=True)
            while not stop_event.is_set():
                if processor.done_event.wait(timeout=0.25):
                    break

        processor.join(timeout=2.0)
        if processor.error is not None:
            raise RuntimeError("SDR processing thread stopped unexpectedly") from processor.error


if __name__ == "__main__":
    main()
