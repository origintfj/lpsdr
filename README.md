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

## I/Q sample resolution

The `rtl_sdr` tool used by this example streams interleaved unsigned 8-bit I/Q
bytes: one byte for I, then one byte for Q, repeated for each complex sample.
For ordinary RTL-SDR dongles this is effectively the hardware/tool sample
format, not a resolution chosen by this Python script. The script converts those
8-bit byte values into floating-point complex numbers before the FFT so NumPy can
process them conveniently, but that conversion does not add ADC resolution.

Other SDR families can have wider ADCs or different host sample formats. To use
one of those devices, replace `SDRReaderThread` with a reader for that hardware's
API/stream format and keep the rest of the buffer, FFT, and display pipeline.

## Threading model

- `SDRReaderThread` launches `rtl_sdr`, converts interleaved unsigned 8-bit I/Q
  bytes into complex samples, and appends them to a thread-safe `IQSampleBuffer`.
- `FFTProcessorThread` waits until the buffer has at least one new FFT-sized
  block, consumes that block from the buffer, computes a windowed FFT, and
  publishes power spectra to a queue.
- The main thread owns the Matplotlib GUI and consumes spectra from the queue to
  update the waterfall display.
