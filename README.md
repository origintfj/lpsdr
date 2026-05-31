# lpsdr

A small Python RTL-SDR waterfall and time-domain I/Q display skeleton that reads samples through the standard `rtl_sdr` command-line program.

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

You can also put the radio, processing, display, and audio settings in a YAML
file and run with `--config`:

```bash
cp radio_config.template.yaml radio_config.yaml
# Edit radio_config.yaml, especially radio.center_freq.
python3 waterfall.py --config radio_config.yaml
```

Command-line options remain available and override values loaded from YAML, so
you can keep a baseline file and temporarily tune individual settings:

```bash
python3 waterfall.py --config radio_config.yaml --center-freq 102.5e6 --gain auto
```

The template file, `radio_config.template.yaml`, documents every supported YAML
option. The preferred structure uses `radio`, `processing`, `display`, and
`audio` sections. Hyphenated keys are accepted as aliases for underscored keys.
The application can parse this documented mapping format without extra packages;
if PyYAML is installed, it will be used automatically for broader YAML syntax.

Useful options:

- `--config`: YAML configuration file. Values provided on the command line
  override values from the file.

- `--bb-sample-rate`: RTL-SDR baseband sample rate in samples/sec, default `2400000`.
- `--gain`: tuner gain in dB or `auto`, default `auto`.
- `--fft-size`: samples used for each FFT waterfall row, default `2048`.
- `--time-domain-samples` / `--iq-display-samples`: most recent I/Q samples shown in the time-domain graph above the waterfall, default `2048`.
- `--display-update-samples`: fresh I/Q samples routed to GUI display handoff points per processing pass, independent of `--fft-size`, default `1024`.
- `--display-queue-blocks`: waterfall display buffer capacity measured in `--display-update-samples` blocks, default `8`. The time-domain I/Q buffer is sized only by `--time-domain-samples` / `--iq-display-samples`.
- `--waterfall-rows`: displayed waterfall history rows, default `300`.
- `--min-db` / `--max-db`: color scale bounds.
- `--rtl-sdr-path`: executable used for capture, default `rtl_sdr`.
- `--enable-audio`: start an audio playback thread that monitors processed
  samples through the speakers.
- `--audio-sample-rate`: speaker playback rate, default `48000`.
- `--audio-block-size`: audio samples written to the sound device per block,
  default `1024`.
- `--audio-buffer-seconds`: maximum queued audio backlog, default `0.5`.
- `--audio-device`: optional `sounddevice` output device name or index.

## I/Q sample resolution

The `rtl_sdr` tool used by this example streams interleaved unsigned 8-bit I/Q
bytes: one byte for I, then one byte for Q, repeated for each complex sample.
For ordinary RTL-SDR dongles this is effectively the hardware/tool sample
format, not a resolution chosen by this Python script. The script converts those
8-bit byte values into floating-point complex numbers before the FFT so NumPy can
process them conveniently, but that conversion does not add ADC resolution.

Other SDR families can have wider ADCs or different host sample formats. To use
one of those devices, replace `SDRReaderThread` in `sdr_reader.py` with a
reader for that hardware's API/stream format and keep the rest of the buffer,
FFT, and display pipeline.

## Threading model

- `sdr_reader.SDRReaderThread` launches `rtl_sdr`, converts interleaved unsigned
  8-bit I/Q bytes into complex samples, and appends them to a thread-safe
  `sdr_reader.IQSampleBuffer`.
- `IQSampleBuffer` is initialized with a fixed maximum sample count (the
  `--fft-size` multiplied by `--buffer-blocks`) so capture backlog memory remains
  bounded. Its `wait_for_samples(sample_count, stop_event)` method lets a
  processing thread block until exactly the requested number of fresh, previously
  unconsumed samples is available.
- `processing_thread.ProcessingThread` asks the capture buffer for
  `--display-update-samples` fresh samples per pass and appends that block to
  independent `sdr_reader.IQSampleBuffer` instances for waterfall generation and
  the time-domain I/Q graph. Waterfall FFT rows are still computed from
  `--fft-size` accumulated samples, so GUI handoff granularity is independent
  of the waterfall block size. This mirrors the intended processing-thread
  handoff model: downstream code can choose whether to append a block for the
  waterfall display, the time-domain display, both displays, or neither.
- Both GUI handoff paths carry samples at the SDR baseband sample rate configured by
  `--bb-sample-rate`. They are separate `IQSampleBuffer` instances, so they do not
  have to contain the same sample blocks even though their x-axes use the same
  baseband-sample-rate basis.
- The waterfall `IQSampleBuffer` is sized to
  `--display-update-samples * --display-queue-blocks` and drops oldest samples
  if the GUI falls behind, keeping waterfall generation recent and memory use
  bounded. The time-domain `IQSampleBuffer` is bounded independently by
  `--time-domain-samples` / `--iq-display-samples`; the plot keeps its own
  rolling view of exactly that many most-recent samples.
- `audio_output.AudioPlaybackThread` drains `AudioSampleQueue` in fixed-size
  blocks and writes them to the configured `sounddevice` output at the audio
  sample rate chosen during initialization. The processing thread pushes the real
  part of each processed sample block directly into this queue. The audio queue
  is bounded by `--audio-buffer-seconds` so speaker latency and memory use stay
  bounded; if producers append more samples than fit, the oldest queued samples
  are discarded first.
- The main thread owns `waterfall.RadioGui`, the Matplotlib GUI facade. It
  drains the waterfall buffer to compute FFT rows and drains the time-domain
  buffer into the plot's rolling view so the I/Q graph always shows the last
  `--time-domain-samples` samples retained for that display.
