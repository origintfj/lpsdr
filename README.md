# lpsdr

A lightweight Python skeleton for streaming samples from an RTL-SDR device to an audio output.

## Structure

- `lpsdr.radio.SDRSampleSource` configures `pyrtlsdr` and exposes `next_samples(n)`.
- `lpsdr.processing.ProcessingThread` blocks on `next_samples(n)`, performs placeholder processing, and pushes the real component into an audio queue.
- `lpsdr.audio.AudioPlayer` drains queued mono `float32` chunks and plays them through `sounddevice`.
- `lpsdr.main.initialise_radio()` is called from `main()` to keep radio configuration explicit and lightweight.

## Install

```bash
python -m pip install -e .
```

You will also need RTL-SDR system libraries and an available audio device.

Current `pyrtlsdr` releases still import `pkg_resources`, which can emit a deprecation warning in Python 3.13 virtual environments with recent `setuptools` versions. The package pins `setuptools<81`, and `lpsdr.radio` also loads `rtlsdr` lazily while suppressing only that compatibility warning at runtime.

## Run

```bash
lpsdr --center-freq 100000000 --sample-rate 48000 --gain auto --chunk-size 1024
```

## Runtime behavior

After the SDR and audio stream open successfully, the application prints `lpsdr running; press Ctrl-C to stop` and remains active until it receives `SIGINT` or `SIGTERM`. If it exits before that message, startup failed while opening the RTL-SDR or audio device. If the processing thread stops after startup, `main()` re-raises that thread error instead of silently exiting.
