"""
ENCS5323 Project - Shared helpers for the ADALM-PLUTO 2.4 GHz sensing/generation system.

Everything that both Part 1 (sensing) and Part 2 (generation) need lives here:
device discovery and configuration, the 2.4 GHz Wi-Fi channel plan, and the
calibrated power-spectrum estimator.
"""

import gc
import os
import sys
import time
import numpy as np

# ----------------------------------------------------------------------------
# Band plan
# ----------------------------------------------------------------------------
BAND_START_HZ = 2400e6      # lower edge of the 2.4 GHz ISM band we must display
BAND_STOP_HZ  = 2500e6      # upper edge

# 802.11b/g/n channel centres in the 2.4 GHz band.
# Channels 1..13 are spaced 5 MHz apart starting at 2412 MHz; channel 14 is special.
WIFI_CHANNELS = {n: 2412e6 + 5e6 * (n - 1) for n in range(1, 14)}
WIFI_CHANNELS[14] = 2484e6

# A standard 802.11g/n channel occupies 20 MHz of bandwidth (22 MHz for legacy
# 802.11b spectral mask). We use 20 MHz because that is what our generator emits.
WIFI_CHANNEL_BW_HZ = 20e6

# The three classic non-overlapping channels - useful as generator candidates.
NON_OVERLAPPING = (1, 6, 11)


def channel_edges(ch, bw_hz=WIFI_CHANNEL_BW_HZ):
    """Return (low, high) frequency edges in Hz for a Wi-Fi channel number."""
    fc = WIFI_CHANNELS[ch]
    return fc - bw_hz / 2.0, fc + bw_hz / 2.0


# ----------------------------------------------------------------------------
# Device connection
# ----------------------------------------------------------------------------
# Common URIs a Pluto shows up as. The USB-Ethernet gadget is 192.168.2.1 by
# default; a second unit is usually re-addressed to 192.168.3.1 so that two
# boards can be used at once.
DEFAULT_URIS = (
    "ip:192.168.2.1",
    "ip:192.168.3.1",
    "ip:pluto.local",
    "usb:",
)


def discover_plutos(verbose=True):
    """Scan for attached PlutoSDR units. Returns a list of URI strings."""
    import iio

    found = []
    # Local USB / network contexts that libiio can enumerate for us.
    try:
        scan = iio.scan_contexts()
        for uri, desc in scan.items():
            if "PlutoSDR" in desc or "ad9361" in desc.lower() or "Analog Devices" in desc:
                found.append(uri)
            elif verbose:
                print(f"  (ignoring non-Pluto context: {uri} -> {desc})")
    except Exception as exc:  # pragma: no cover - depends on host libiio build
        if verbose:
            print(f"  context scan unavailable ({exc}); falling back to fixed URIs")

    # Always probe the well-known addresses too, because the USB-Ethernet
    # gadget is frequently not returned by scan_contexts() on macOS.
    for uri in DEFAULT_URIS:
        if uri == "usb:" or uri in found:
            continue
        try:
            ctx = iio.Context(uri)
            if any("ad936" in d.name for d in ctx.devices if d.name):
                found.append(uri)
            del ctx
        except Exception:
            pass

    return found


class Pluto:
    """Thin wrapper around adi.Pluto with the settings this project relies on.

    The wrapper exists so that Part 1 and Part 2 configure the radio in exactly
    the same way, which is what makes their power readings comparable.
    """

    def __init__(self, uri=None, sample_rate=20e6, rx_gain_db=40,
                 gain_mode="manual", buffer_size=8192, verbose=True):
        import adi

        if uri is None:
            candidates = discover_plutos(verbose=verbose)
            if not candidates:
                raise RuntimeError(
                    "No ADALM-PLUTO found. Check that the USB cable is in the "
                    "port labelled 'USB' (not 'PWR') and that the PlutoSDR "
                    "drive / 192.168.2.1 network interface appears on the host."
                )
            uri = candidates[0]
            if verbose:
                print(f"[pluto] auto-selected {uri}")

        self.uri = uri
        self.verbose = verbose
        self.sample_rate = float(sample_rate)
        self.rx_gain_db = float(rx_gain_db)
        self.buffer_size = int(buffer_size)

        # Opening AND configuring are retried as one unit. Retrying only the
        # open is not enough: on this board the failure regularly lands on the
        # first attribute write instead ("Input/output error" while setting
        # rf_bandwidth), which left runs dying halfway through a measurement
        # matrix. The radio is also not released instantly when the previous
        # process exits, so back-to-back scripts need the same patience.
        last = None
        for attempt in range(6):
            try:
                self.sdr = adi.Pluto(uri=uri)
                self._configure(gain_mode)
                break
            except Exception as exc:
                last = exc
                if verbose:
                    print(f"[pluto] {uri} not ready ({exc}); retry {attempt+1}/6")
                try:
                    self.sdr = None
                    gc.collect()
                except Exception:
                    pass
                time.sleep(2.5)
        else:
            raise RuntimeError(
                f"Could not open and configure {uri} after several attempts. "
                f"Last error: {last}. If this persists, unplug both the USB and "
                f"PWR cables, wait 10 s, reconnect power first, then USB."
            )

        if verbose:
            print(f"[pluto] {uri}: fs={self.sample_rate/1e6:.3f} MSPS, "
                  f"rx_bw={self.rf_bandwidth/1e6:.3f} MHz, "
                  f"rx_gain={self.rx_gain_db:.1f} dB, N={self.buffer_size}")

    def _configure(self, gain_mode):
        """Apply the settings both parts rely on, so their readings match.

        A stock Pluto carries an AD9363 (20 MHz RF bandwidth, 325 MHz-3.8 GHz);
        many are "unlocked" to AD9364 behaviour (56 MHz, 70 MHz-6 GHz), so the
        bandwidth is requested and then read back rather than assumed.
        """
        self.sdr.sample_rate = int(self.sample_rate)
        self.sdr.rx_rf_bandwidth = int(min(self.sample_rate * 0.9, 56e6))
        self.rf_bandwidth = float(self.sdr.rx_rf_bandwidth)

        # Manual gain is essential: AGC would silently re-scale each capture and
        # destroy the comparability of the occupancy measurements over time.
        self.sdr.gain_control_mode_chan0 = gain_mode
        if gain_mode == "manual":
            self.sdr.rx_hardwaregain_chan0 = float(self.rx_gain_db)

        self.sdr.rx_buffer_size = int(self.buffer_size)

    # ------------------------------------------------------------------
    def set_rx_freq(self, f_hz, settle_s=0.004, flush_buffers=1):
        """Retune the receive LO and throw away the transient.

        After an LO step the AD936x needs a moment to settle and the buffer
        already in flight still holds samples from the *old* frequency, so we
        discard it. Skipping this is the classic cause of ghost signals
        appearing in a swept spectrum.
        """
        self.sdr.rx_lo = int(f_hz)
        if settle_s:
            time.sleep(settle_s)
        for _ in range(flush_buffers):
            self.sdr.rx()

    def capture(self):
        """Return one complex baseband buffer as float64 complex."""
        return np.asarray(self.sdr.rx(), dtype=np.complex128)

    def close(self):
        """Release the radio, including the underlying libiio context.

        Destroying the buffers is not enough: while the adi.Pluto object is
        still referenced the USB context stays claimed, and the next process or
        the next `Pluto(...)` in this one fails with "No device found". Dropping
        the reference and forcing a collection is what actually frees it.
        """
        for meth in ("rx_destroy_buffer", "tx_destroy_buffer"):
            try:
                getattr(self.sdr, meth)()
            except Exception:
                pass
        self.sdr = None
        gc.collect()
        time.sleep(0.5)


# ----------------------------------------------------------------------------
# Spectrum estimation
# ----------------------------------------------------------------------------
# The Pluto's 12-bit ADC is presented by libiio as int16 samples left-aligned to
# 2048 full scale. Converting to dBFS and then applying the known receive gain
# gives a power in dBm that is accurate to within a few dB once CAL_OFFSET_DB is
# trimmed against a known source. It is a *relative* measurement, which is all
# the occupancy decision needs, but reporting dBm makes the plots readable.
ADC_FULL_SCALE = 2048.0
CAL_OFFSET_DB = float(os.environ.get("PLUTO_CAL_OFFSET_DB", -10.0))


def welch_psd(samples, fs, nfft=1024, rx_gain_db=0.0, window="hann",
              combine="mean"):
    """Averaged periodogram of a complex baseband capture.

    Returns (freq_offsets_hz, psd_dbm) with the spectrum shifted so that index 0
    is the most negative frequency offset from the LO, i.e. ready to be pasted
    onto an absolute frequency axis.
    """
    from scipy.signal import get_window

    x = np.asarray(samples, dtype=np.complex128) / ADC_FULL_SCALE
    nfft = int(min(nfft, len(x)))
    win = get_window(window, nfft)
    # Coherent power gain of the window, so the result is independent of choice.
    win_power = np.sum(win ** 2)

    n_seg = len(x) // nfft
    if n_seg == 0:
        raise ValueError("capture shorter than one FFT segment")

    # `combine` decides what a "burst" means to us. Averaging the segments
    # gives the steadiest noise floor, but it also dilutes a short Wi-Fi burst
    # across the whole buffer - a frame filling one segment in eight is pulled
    # down by ~9 dB and disappears under the threshold. Taking the maximum
    # instead preserves the burst, which is what occupancy sensing needs.
    acc = np.zeros(nfft)
    for k in range(n_seg):
        seg = x[k * nfft:(k + 1) * nfft] * win
        spec = np.fft.fftshift(np.fft.fft(seg, nfft))
        power = np.abs(spec) ** 2
        if combine == "max":
            acc = np.maximum(acc, power)
        else:
            acc += power
    if combine != "max":
        acc /= n_seg

    # Normalise to power spectral density, then to dBm referenced to 50 ohm.
    psd = acc / (win_power * fs)
    psd_db = 10.0 * np.log10(psd + 1e-30)
    psd_dbm = psd_db + 10.0 * np.log10(fs / nfft)      # per-bin power, not per-Hz
    psd_dbm = psd_dbm - rx_gain_db + CAL_OFFSET_DB      # de-embed the RX gain

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))
    return freqs, psd_dbm


def noise_floor_dbm(psd_dbm, seed_percentile=15.0, margin_db=6.0):
    """Estimate the thermal noise floor of a spectrum that contains signals.

    This is the single most important number in the whole detector: every
    occupancy decision is "is this bin more than X dB above the floor?", so a
    biased floor silently breaks everything downstream.

    Two obvious estimators both fail here:
      * the plain median sits *inside* a signal, because three active Wi-Fi
        networks cover ~66 of the 100 MHz we display;
      * iterative sigma-clipping from above never starts, because when the
        majority of bins are signal, nothing looks like an outlier.
    Measured on our own band that error was ~26 dB, which made a busy band read
    as empty.

    So we work from the bottom instead. A low percentile is guaranteed to land
    in noise as long as some part of the band is quiet - in 2.4 GHz there is
    always at least the region above channel 11 - and we then average every bin
    within `margin_db` of it to get a stable estimate.
    """
    x = np.asarray(psd_dbm, dtype=float)
    seed = float(np.percentile(x, seed_percentile))
    quiet = x <= seed + margin_db
    if quiet.sum() < max(8, int(0.02 * x.size)):
        return seed
    return float(np.median(x[quiet]))


def build_sweep_plan(fs, band_start=BAND_START_HZ, band_stop=BAND_STOP_HZ,
                     usable_fraction=0.75):
    """Choose the LO centres needed to tile the whole band.

    Only the middle `usable_fraction` of each capture is kept: the outer part of
    the baseband spectrum sits on the analogue filter roll-off and would read
    artificially low, producing fake "empty" channels at every seam.
    """
    usable_bw = fs * usable_fraction
    span = band_stop - band_start
    n_steps = int(np.ceil(span / usable_bw))
    centres = band_start + usable_bw * (np.arange(n_steps) + 0.5)
    return centres, usable_bw


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# Emitter detection
# ----------------------------------------------------------------------------
# Naively thresholding per Wi-Fi channel does not work in the 2.4 GHz band,
# because the channels are 20 MHz wide but only 5 MHz apart. A single
# transmitter on channel 1 puts energy inside the nominal bands of channels 1-5,
# so a per-channel test reports five "occupied" channels for one Wi-Fi network.
#
# Instead we find the contiguous regions of spectrum that are actually above the
# noise floor, measure each one's centre frequency and bandwidth, and only then
# attribute it to the nearest Wi-Fi channel. That yields the answer a spectrum
# analyst would give - and it is what the channel-hopping logic in Part 2 needs.

def smooth_psd(psd_dbm, df_hz, smooth_hz=2e6):
    """Moving-average smoother, edge-padded so the band edges stay honest."""
    w = max(1, int(round(smooth_hz / df_hz)))
    if w < 2:
        return np.asarray(psd_dbm, dtype=float)
    pad = w // 2
    x = np.pad(np.asarray(psd_dbm, dtype=float), pad, mode="edge")
    return np.convolve(x, np.ones(w) / w, mode="same")[pad:pad + len(psd_dbm)]


def _walk_edge(sm, anchor, step, limit, edge_level, rise_stop_db):
    """Grow a carrier outward from `anchor` until it really ends.

    Falling below `edge_level` is not a sufficient stopping rule on its own. A
    weak carrier next to a strong one clamps its edge level to the absolute
    detection threshold, and the walk then runs straight through the valley and
    up the side of the strong neighbour - a weak channel 11 was being absorbed
    into channel 6 and discarded as a duplicate.

    So we also watch for the spectrum climbing again: once it rises more than
    `rise_stop_db` above the lowest point seen so far on this walk, we have
    crossed the valley into somebody else's signal and stop there.
    """
    i = anchor
    run_min = sm[anchor]
    while (i - anchor) * step < abs(limit - anchor):
        j = i + step
        if j < 0 or j > len(sm) - 1 or (step > 0 and j > limit) or (step < 0 and j < limit):
            break
        v = sm[j]
        if v <= edge_level:
            break
        if v > run_min + rise_stop_db:
            break          # climbing into the neighbouring carrier
        run_min = min(run_min, v)
        i = j
    return i


def detect_emitters(freq_grid, psd_dbm, threshold_db=8.0,
                    flat_half_hz=9e6, min_separation_hz=15e6, min_bw_hz=4e6,
                    edge_drop_db=12.0, search_half_hz=25e6, smooth_hz=1e6,
                    rise_stop_db=6.0):
    """Find the distinct transmissions present in a stitched spectrum.

    Two simpler detectors were tried first and both failed on this band:

      * Thresholding and taking contiguous runs. The 802.11 mask only falls to
        -20 dBr at 11 MHz offset, so access points on channels 1 and 6 remain
        joined above the threshold and were reported as one 75 MHz signal.
      * Peak picking on the smoothed spectrum. An OFDM carrier is flat-topped,
        not peaked, so the maxima landed on the shoulders - channel 1 was
        reported at 2420 MHz instead of 2412 MHz - because two neighbours'
        skirts add up and lift the inner edges slightly.

    What works is to use the structure we already know: Wi-Fi carriers sit on a
    fixed 5 MHz channel grid and are flat across their middle 18 MHz. So we
    score every candidate channel by the median power over its flat region,
    which is exactly a matched filter for the OFDM rectangle, and keep the
    strongest scoring channels while suppressing their overlapping neighbours.

    Returns (emitters, noise_floor_dbm).
    """
    psd_dbm = np.asarray(psd_dbm, dtype=float)
    nf = noise_floor_dbm(psd_dbm)
    df = float(freq_grid[1] - freq_grid[0])
    sm = smooth_psd(psd_dbm, df, smooth_hz)

    # --- score every channel on the grid --------------------------------------
    scores = {}
    for ch, fc in WIFI_CHANNELS.items():
        sel = np.abs(freq_grid - fc) <= flat_half_hz
        if np.count_nonzero(sel) >= 8:
            scores[ch] = float(np.median(psd_dbm[sel]))

    # --- locate carriers, strongest first --------------------------------------
    # The scores of channels 5, 6 and 7 are nearly identical, because all three
    # of their flat windows fall inside the same 18 MHz OFDM top - so whichever
    # one wins is decided by noise, and the reported channel flips run to run.
    # The measured centre frequency does not suffer from this, so the channel is
    # assigned from the centroid of the carrier rather than from the scores.
    candidates = [c for c, sc in scores.items() if sc > nf + threshold_db]

    emitters = []
    claimed = []          # centre frequencies already accounted for
    anchor_half_hz = 10e6
    span_bins = int(search_half_hz / df)

    for ch in sorted(candidates, key=lambda c: -scores[c]):
        fc0 = WIFI_CHANNELS[ch]
        # Suppression is by frequency, not channel number: channels 13 and 14
        # are adjacent by index but 12 MHz apart, so an index rule gets them
        # wrong. The window must exceed the 10 MHz of a two-channel step, or one
        # access point on channel 1 is also reported on channel 3.
        if any(abs(fc0 - c) < min_separation_hz for c in claimed):
            continue

        # Anchor on the strongest bin near this channel, then grow outward to
        # the carrier's own -12 dB points, stopping in the valley between
        # neighbouring carriers.
        win = np.flatnonzero(np.abs(freq_grid - fc0) <= anchor_half_hz)
        if win.size == 0:
            continue
        anchor = int(win[np.argmax(sm[win])])
        edge_level = max(nf + threshold_db, sm[anchor] - edge_drop_db)

        lo_lim = max(0, anchor - span_bins)
        hi_lim = min(len(sm) - 1, anchor + span_bins)
        a = _walk_edge(sm, anchor, -1, lo_lim, edge_level, rise_stop_db)
        b = _walk_edge(sm, anchor, +1, hi_lim, edge_level, rise_stop_db)

        bw = float((b - a) * df)
        if bw < min_bw_hz:
            continue  # degenerate match: no real carrier width here

        seg = psd_dbm[a:b + 1]
        w = 10 ** (seg / 10.0)
        f_c = float(np.sum(freq_grid[a:b + 1] * w) / np.sum(w))

        # Re-check against what we already found, now using the measured centre.
        if any(abs(f_c - c) < min_separation_hz for c in claimed):
            continue
        claimed.append(f_c)

        nearest = min(WIFI_CHANNELS, key=lambda c: abs(WIFI_CHANNELS[c] - f_c))
        emitters.append({
            "f_lo": float(freq_grid[a]), "f_hi": float(freq_grid[b]),
            "f_center": f_c, "bw_hz": bw,
            "peak_dbm": float(np.max(seg)), "mean_dbm": float(np.mean(seg)),
            "channel_power_dbm": scores[ch],
            "nearest_channel": int(nearest),
        })

    emitters.sort(key=lambda e: e["f_center"])
    return emitters, nf


def occupied_channels(emitters):
    """Wi-Fi channel numbers attributed to the detected emitters."""
    return sorted({e["nearest_channel"] for e in emitters})


def occupied_centers_hz(emitters):
    """Centre frequencies of the detected emitters, in Hz."""
    return [e["f_center"] for e in emitters]
