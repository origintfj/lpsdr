import queue
import threading
from rtlsdr import RtlSdr

# --- SDR config ---
DAB_11C = 220.352e6 # DAB 11C: 220.352 MHz
DAB_12D = 229.072e6 # DAB 12D: 229.072 MHz
FM_CAM  = 97.2e6    # CAM FM

FM_TOM  = 30.0e6    # CAM FM
FM_TOM  = 96e6    # CAM FM

SDR_CENTER_HZ = FM_TOM
SDR_RATE_HZ   = 2.4e6
SDR_PPM       = 60
SDR_GAIN_DB   = None#80#49.6      # or 'auto' by setting to None
SDR_CHUNK     = 16384     # samples per async callback

center_freq = SDR_CENTER_HZ

def _setup_sdr() -> RtlSdr:
    global center_freq

    sdr = RtlSdr()
    sdr.sample_rate = SDR_RATE_HZ
    sdr.center_freq = center_freq
    sdr.freq_correction = SDR_PPM
    sdr.gain = 'auto' if SDR_GAIN_DB is None else SDR_GAIN_DB
    return sdr

def producer_thread(sdr_q: queue.Queue, event_list):
    """
    Produce complex64 baseband chunks into sdr_q using librtlsdr's async API.
    """

    sdr = _setup_sdr()

    def _cb(samples, _ctx):
        global center_freq

        if event_list[0].is_set(): # stop event
            # Ask librtlsdr to stop as soon as we can
            try:
                sdr.cancel_read_async()
            except Exception:
                pass
            return

        #if event_list[1].is_set():
        #    event_list[1].clear()
        #    center_freq += 1e6
        #    sdr.center_freq = center_freq
        #elif event_list[2].is_set():
        #    event_list[2].clear()
        #    center_freq -= 1e6
        #    sdr.center_freq = center_freq

        # Non-blocking put; drop oldest on overflow
        try:
            sdr_q.put_nowait(samples)  # dtype complex64, len SDR_CHUNK
        except queue.Full:
            try:
                _ = sdr_q.get_nowait()
            except queue.Empty:
                pass
            try:
                sdr_q.put_nowait(samples)
            except queue.Full:
                pass

    try:
        # This blocks inside C until cancel_read_async() is called.
        sdr.read_samples_async(_cb, num_samples=SDR_CHUNK)
    except Exception as e:
        if not event_list[0].is_set():
            print(f"[SDR] read_samples_async error: {e}")
    finally:
        try:
            sdr.cancel_read_async()
        except Exception:
            pass
        sdr.close()

