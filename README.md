# lpsdr

A small Python RTL-SDR waterfall display skeleton.

## Setup

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The application expects an RTL-SDR device supported by `pyrtlsdr` and the
underlying `librtlsdr` native library to be available on the host system.

## Run

Select the center frequency in Hz:

```bash
python3 waterfall.py --center-freq 100.1e6
```

Useful options:

- `--sample-rate`: sample rate in samples/sec, default `2.4e6`.
- `--gain`: tuner gain in dB or `auto`, default `auto`.
- `--fft-size`: samples used for each FFT waterfall row, default `2048`.
- `--waterfall-rows`: displayed waterfall history rows, default `300`.
- `--min-db` / `--max-db`: color scale bounds.

## Threading model

- `SDRReaderThread` continuously reads complex I/Q samples from the RTL-SDR and
  appends them to a thread-safe `IQSampleBuffer`.
- `FFTProcessorThread` waits until the buffer has at least one new FFT-sized
  block, computes a windowed FFT, and publishes power spectra to a queue.
- The main thread owns the Matplotlib GUI and consumes spectra from the queue to
  update the waterfall display.
