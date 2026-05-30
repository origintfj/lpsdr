# lpsdr

A small Python RTL-SDR waterfall display skeleton that reads samples through the standard `rtl_sdr` command-line program.

## Setup

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The application expects an RTL-SDR device and the `rtl_sdr` executable from the native `rtl-sdr` tools to be available on the host system. It intentionally does not use the `pyrtlsdr` Python bindings, avoiding version mismatches such as missing `librtlsdr` symbols.

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
- `--rtl-sdr-path`: executable used for capture, default `rtl_sdr`.

## Threading model

- `SDRReaderThread` launches `rtl_sdr`, converts interleaved unsigned 8-bit I/Q
  bytes into complex samples, and appends them to a thread-safe `IQSampleBuffer`.
- `FFTProcessorThread` waits until the buffer has at least one new FFT-sized
  block, consumes that block from the buffer, computes a windowed FFT, and
  publishes power spectra to a queue.
- The main thread owns the Matplotlib GUI and consumes spectra from the queue to
  update the waterfall display.
