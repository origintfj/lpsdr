import sdr_prod
import threading
import queue
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

import sounddevice as sd

# -------------------------
# Config
# -------------------------
SDR_Q_MAX = 64                 # queue depth in chunks
SDR_FFT_N = 4096               # FFT length for display
TARGET_BASEBAND_HZ = 500_000   # aim to decimate near this rate for the GUI
GUI_REFRESH_HZ = 30            # UI update cadence

# Time-domain scope window (seconds of decimated samples to display)
TIME_SCOPE_SEC = 10000e-3 # FIXME this isn't in sec anymore

# Waterfall appearance
N_ROWS   = 400                 # height of the waterfall
COLORMAP = 'viridis'
VMIN_DB  = -60.0
VMAX_DB  = 0.0

SAMPLE_RATE = 48000
CHANNELS = 1
DTYPE = "float32"
BLOCK_SIZE = 4096           # audio frames per block
OUT_Q_MAX = 32

# -------------------------
# Shared state
# -------------------------
sdr_q = queue.Queue(maxsize=SDR_Q_MAX)
out_q = queue.Queue(maxsize=OUT_Q_MAX)
stop_event = threading.Event()
tune_up_event = threading.Event()
tune_dn_event = threading.Event()
gui_lock = threading.Lock()

# Latest SDR spectrum for GUI (waterfall)
latest_sdr_freqs = None      # np.ndarray, Hz (absolute by default)
latest_sdr_mag   = None      # np.ndarray, dBFS
latest_sdr_seq   = 0         # increments whenever a new spectrum is published

# Latest time-domain (decimated) window for scope
latest_td_i   = None         # np.ndarray, float32 (I)
latest_td_q   = None         # np.ndarray, float32 (Q)
latest_td_fs  = None         # float: decimated sample rate (Hz)
latest_td_seq = 0            # increments when time-domain is updated

# -------------------------
# Processing thread
# -------------------------
def _design_lpf(cutoff_hz: float, fs_hz: float, num_taps: int = 257) -> np.ndarray:
    """
    Windowed-sinc real FIR low-pass. Suitable for complex IQ (applies to I and Q together).
    cutoff_hz is ~3 dB cutoff; choose taps to trade stopband vs compute.
    """
    n = np.arange(num_taps) - (num_taps - 1) / 2
    h = 2 * cutoff_hz / fs_hz * np.sinc(2 * cutoff_hz * n / fs_hz)
    h *= np.hamming(num_taps)
    h /= np.sum(h)
    return h.astype(np.float32)


class Serialiser:
    # serial_buffer
    # max_size

    def __init__(self, max_size: int):
        self.serial_buffer = np.empty(0, dtype=np.complex64)
        self.max_size = max_size

    def serialise(self, parallel_buffer):
        pulled = False
        while True:
            try:
                chunk = parallel_buffer.get_nowait()  # np.complex64, shape (N,)
            except queue.Empty:
                break
            pulled = True
            if self.serial_buffer.size == 0:
                self.serial_buffer = chunk
            else:
                self.serial_buffer = np.concatenate((self.serial_buffer, chunk))
            if self.serial_buffer.size > self.max_size:
                self.serial_buffer = self.serial_buffer[-self.max_size:]
        return self.serial_buffer, pulled

class SampleStream:
    def __init__(self) -> None:
        self.stream = None
        return None

    def append(self, iframe: np.ndarray) -> int:
        if self.stream is None:
            self.stream = iframe
        else:
            self.stream = np.concatenate((self.stream, iframe))
        return len(iframe)

    def get(self, oldest_n: int) -> np.ndarray:
        return self.stream[:oldest_n]

    def remove(self, oldest_n: int) -> None:
        self.stream = self.stream[oldest_n:]

    def convolve_rev(self, fir_vec_rev: np.ndarray) -> tuple[np.ndarray, int]:
        if self.stream is None:
            return np.array([], dtype=self.stream.dtype), 0
        if len(self.stream) < len(fir_vec_rev):
            return np.array([], dtype=self.stream.dtype), 0

        out_stream_len = len(self.stream) - len(fir_vec_rev) + 1
        out_stream = np.empty(out_stream_len, dtype=self.stream.dtype)

        for i in range(out_stream_len):
            out_stream[i] = np.dot(self.stream[i:i+len(fir_vec_rev)], fir_vec_rev)

        return out_stream, out_stream_len

    def dphase(self) -> tuple[np.ndarray, int]:
        if self.stream is None:
            return np.array([], dtype=self.stream.dtype), 0

        stream_arg = np.angle(self.stream)

        out_stream_len = len(self.stream) - 1
        out_stream = np.empty(out_stream_len, dtype=np.float32)

        for i in range(out_stream_len):
            darg = stream_arg[i] - stream_arg[i+1]
            if darg < -np.pi:
                darg = darg + 2 * np.pi
            elif darg > np.pi:
                darg = darg - 2 * np.pi
            out_stream[i] = darg/4

        return out_stream, out_stream_len

def _design_lpf(self, cutoff_hz: float, fs_hz: float, num_taps: int = 257) -> np.ndarray:
    """
    Windowed-sinc real FIR low-pass. Suitable for complex IQ (applies to I and Q together).
    cutoff_hz is ~3 dB cutoff; choose taps to trade stopband vs compute.
    """
    n = np.arange(num_taps) - (num_taps - 1) / 2
    h = 2 * cutoff_hz / fs_hz * np.sinc(2 * cutoff_hz * n / fs_hz)
    h *= np.hamming(num_taps)
    h /= np.sum(h)
    return h.astype(np.float32)

def processing_thread():
    """
    Drain SDR chunks, optional TIME-DOMAIN LPF, decimate for GUI,
    publish: (a) spectrum for waterfall, (b) latest time-domain window for scope.
    """
    global latest_sdr_freqs, latest_sdr_mag, latest_sdr_seq
    global latest_td_i, latest_td_q, latest_td_fs, latest_td_seq

    # --- Time-domain LPF (optional): keep up to ~100 kHz around DC pre-decimation
    LPF_CUTOFF_HZ = 75_000.0
    LPF_TAPS = 257
    lpf = _design_lpf(LPF_CUTOFF_HZ, sdr_prod.SDR_RATE_HZ, LPF_TAPS)

    # Decimation to target a friendly display rate
    decim = max(1, int(sdr_prod.SDR_RATE_HZ // TARGET_BASEBAND_HZ))
    fs_view = sdr_prod.SDR_RATE_HZ / decim  # sample rate after decimation

    # --- Buffers & rates
    # Keep enough raw samples to build several filtered/decimated frames
    max_keep = (SDR_FFT_N * 4) * decim + (LPF_TAPS - 1)
    ser = Serialiser(max_keep)


    # --- FFT window (on decimated segment)
    win = np.hanning(SDR_FFT_N).astype(np.float32)

    # How many decimated samples to show in the time-domain scope
    scope_len_decim = max(1, int(TIME_SCOPE_SEC * fs_view))

    first_print = True

    bb_stream = SampleStream()
    bb_lpf_stream = SampleStream()

    lpf = _design_lpf(75e3, 2.4e6, 257)
    lpf_rev = lpf

    while not stop_event.is_set():
        # 1) Drain any available SDR chunks
        sdr_buf, pulled = ser.serialise(sdr_q)

        if not pulled:
            time.sleep(0.001)
            continue

        # 2) Build one filtered+decimated frame for FFT and a time-domain window for scope
        need_decimated = SDR_FFT_N
        need_raw = need_decimated * decim
        need_with_transient = need_raw + (LPF_TAPS - 1)

        if sdr_buf.size >= need_with_transient:
            # Take the newest window of raw data big enough for one decimated FFT frame
            block = sdr_buf[-need_with_transient:]  # complex64

            x = np.convolve(block, lpf, mode="same")  # complex64, length = need_raw

            #n = bb_stream.append(block)
            #bb_frame, n = bb_stream.convolve_rev(lpf_rev)  # complex64, length = need_raw
            #bb_stream.remove(n)
            #x = bb_frame

            phase    = np.angle(x)
            phase_uw = np.unwrap(phase)
            phase_uw_lpf = phase_uw
            dp = np.diff(phase_uw_lpf[::50])
            y = dp / 2 / np.pi

            bb_frame = y

            #n = bb_stream.append(block)
            #bb_frame, n = bb_stream.convolve_rev(lpf_rev)  # complex64, length = need_raw
            #bb_stream.remove(n)

            #bb_frame = bb_frame[::50]
            #bb_frame = block[::50]

            #bb_lpf_stream.append(bb_frame)
            #bb_frame, n = bb_lpf_stream.dphase()
            #bb_lpf_stream.remove(n)

            a = bb_frame.copy().astype(np.float32)
            out_q.put(a, timeout=0.1)

            # ---------- Time-domain scope (top) ----------
            # Decimate for GUI
            view = bb_frame / 2 / np.pi # complex, fs = fs_view
            # Take the most recent TIME_SCOPE_SEC seconds of I & Q
            td = view[-scope_len_decim:] if view.size >= scope_len_decim else view
            td_i = td.real.astype(np.float32, copy=False)
            td_q = td.imag.astype(np.float32, copy=False)

            with gui_lock:
                latest_td_i  = td_i.copy()
                latest_td_q  = td_q.copy()
                latest_td_fs = float(fs_view)
                latest_td_seq += 1

            # ---------- Spectrum + Waterfall (bottom) ----------
            # Decimate for GUI
            view = block[::decim] # complex, fs = fs_view
            if view.size >= SDR_FFT_N:
                seg = view[-SDR_FFT_N:]  # complex, length Nfft @ fs_view
                X = np.fft.fftshift(np.fft.fft(seg * win, n=SDR_FFT_N))
                mag_db = 20 * np.log10(np.maximum(np.abs(X) / (SDR_FFT_N / 2), 1e-12)).astype(np.float32)

                # Frequency axis (absolute RF by default)
                freqs = np.fft.fftshift(np.fft.fftfreq(SDR_FFT_N, d=1.0 / fs_view)) + sdr_prod.SDR_CENTER_HZ

                with gui_lock:
                    latest_sdr_freqs = freqs
                    latest_sdr_mag   = mag_db
                    latest_sdr_seq  += 1

                if first_print:
                    print("[SDR] First spectrum + time-domain window published")
                    first_print = False

        # Optional: drop some already-used raw samples to keep latency bounded
        N = need_with_transient
        if sdr_buf.size >= N:
            sdr_buf = sdr_buf[N:]


def output_thread():
    """Consumes from output FIFO and plays to speakers."""
    try:
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE
        ) as ostream:
            silence = np.zeros((BLOCK_SIZE, CHANNELS), dtype=DTYPE)
            while not stop_event.is_set():
                try:
                    block = out_q.get(timeout=0.05)
                except queue.Empty:
                    block = silence
                try:
                    ostream.write(block)
                except sd.PortAudioError as e:
                    # Try to continue on underflow/overflow; abort on fatal errors
                    print(f"[Output] PortAudioError: {e}")
    except Exception as e:
        print(f"[Output] Exception: {e}")
        stop_event.set()


# -------------------------
# GUI
# -------------------------
def run_gui():
    fig, (ax_time, ax_wf) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1.2])
    fig.canvas.manager.set_window_title("RTL-SDR Time-Domain Scope + Waterfall (press 'q' to quit)")

    # Time-domain scope (top): I and Q traces
    time_i_line, = ax_time.plot([], [], label="I (real)")
    time_q_line, = ax_time.plot([], [], label="Q (imag)")
    ax_time.set_title("Time-Domain (most recent window, decimated)")
    ax_time.set_xlabel("Time [ms]")
    ax_time.set_ylabel("Amplitude")
    ax_time.legend(loc="upper right")
    ax_time.set_ylim(-1.2, 1.2)  # typical IQ range; adjust if needed

    # Waterfall image (bottom)
    waterfall = np.full((N_ROWS, SDR_FFT_N), VMIN_DB, dtype=np.float32)
    im = ax_wf.imshow(
        waterfall,
        aspect="auto",
        origin="lower",    # new rows appended at the bottom
        extent=[0, 1, 0, N_ROWS],  # x extent will be updated once we see freqs
        cmap=COLORMAP,
        vmin=VMIN_DB, vmax=VMAX_DB
    )
    cb = fig.colorbar(im, ax=ax_wf, label="Power [dBFS]")
    ax_wf.set_title("Waterfall (most recent at bottom)")
    ax_wf.set_xlabel("Frequency [Hz]")
    ax_wf.set_ylabel("Time →")

    # Track last sequences so we only redraw on fresh data
    last_spec_seq = -1
    last_td_seq   = -1

    # Quit on 'q'
    def on_key(event):
        if event.key in ("q", "Q"):
            stop_event.set()
            plt.close(fig)
        # If you're wiring tuning events, handle here.
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame):
        nonlocal last_spec_seq, last_td_seq, waterfall

        # ----------- Time-domain scope update -----------
        with gui_lock:
            td_i = None if latest_td_i is None else latest_td_i.copy()
            td_q = None if latest_td_q is None else latest_td_q.copy()
            td_fs = latest_td_fs
            td_seq = latest_td_seq

        if td_i is not None and td_q is not None and td_fs and td_seq != last_td_seq:
            last_td_seq = td_seq
            N = td_i.size
            # Time axis in milliseconds (most recent window, right-aligned)
            t_ms = (np.arange(-N, 0) / td_fs) * 1e3
            time_i_line.set_data(t_ms, td_i)
            time_q_line.set_data(t_ms, td_q)
            ax_time.set_xlim(t_ms[0], t_ms[-1])

        # ----------- Waterfall update -----------
        with gui_lock:
            fs = latest_sdr_freqs
            ms = latest_sdr_mag
            spec_seq = latest_sdr_seq

        if fs is not None and ms is not None and spec_seq != last_spec_seq:
            last_spec_seq = spec_seq

            # Roll waterfall up and insert new row
            if ms.size == waterfall.shape[1]:
                waterfall[:-1, :] = waterfall[1:, :]
                waterfall[-1, :] = ms
            else:
                # If FFT length changed (shouldn't), resize waterfall
                waterfall = np.full((N_ROWS, ms.size), VMIN_DB, dtype=np.float32)
                waterfall[-1, :] = ms

            # Update the image data and x extent (frequency axis)
            im.set_data(waterfall)
            ax_wf.set_xlim(fs.min(), fs.max())
            im.set_extent([fs.min(), fs.max(), 0, N_ROWS])

        return (time_i_line, time_q_line, im)

    interval_ms = int(1000.0 / GUI_REFRESH_HZ)
    ani = FuncAnimation(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)
    fig._ani = ani  # keep a strong reference

    plt.tight_layout()
    plt.show()

# -------------------------
# Main
# -------------------------
def main():
    print("Starting SDR and display... (press 'q' to quit)")

    # Start SDR producer (fills sdr_q with complex64 chunks)
    # NOTE: This assumes your sdr_prod.producer_thread accepts the same args you already wired.
    tsdr = threading.Thread(target=sdr_prod.producer_thread,
                            args=(sdr_q, [stop_event, tune_up_event, tune_dn_event]),
                            daemon=True)
    tout = threading.Thread(target=output_thread, daemon=True)
    tsdr.start()
    tout.start()

    # Start processing thread (drains sdr_q, publishes time-domain + FFT for GUI)
    tproc = threading.Thread(target=processing_thread, daemon=True)
    tproc.start()

    try:
        run_gui()
    finally:
        stop_event.set()
        # Allow threads to exit cleanly
        for _ in range(50):
            if not (tsdr.is_alive() or tproc.is_alive()):
                break
            time.sleep(0.02)
        print("Shutting down.")

if __name__ == "__main__":
    main()

