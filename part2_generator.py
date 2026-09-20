#!/usr/bin/env python3
"""
ENCS5323 Project - Part 2: Wi-Fi-like signal generator with dynamic channel hopping.

Generates an OFDM signal with the same structure and bandwidth as a 2.4 GHz
Wi-Fi channel, periodically senses the band with its own receiver, and hops to
whichever channel keeps the greatest distance from the occupied ones.

  # look only, transmit nothing (safe default)
  python part2_generator.py --duration 60

  # actually transmit
  python part2_generator.py --transmit --duration 60 --tx-gain -30

  # a single fixed configuration, for the centre-frequency / bandwidth sweep
  python part2_generator.py --transmit --fixed-channel 6 --bw 20e6 --duration 30

RESPONSIBILITY: transmitting is opt-in via --transmit, the TX gain is limited,
and the run is time-capped. Operate indoors, toward your own equipment only,
for the minimum time needed - as required by the project brief.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

from pluto_common import (
    BAND_START_HZ, BAND_STOP_HZ, WIFI_CHANNELS, WIFI_CHANNEL_BW_HZ,
    NON_OVERLAPPING, Pluto, detect_emitters, occupied_channels,
)

# Safety envelope. The AD936x accepts -89.75..0 dB of TX gain; anything near the
# top is far more power than a tabletop experiment needs.
TX_GAIN_MIN_DB = -89.75
# Hard ceiling regardless of what is passed in. 0 dB is the AD936x maximum,
# roughly +7 dBm (5 mW) at the connector - about 1/20th of a normal access
# point and well inside the 2.4 GHz ISM limits. The ceiling previously sat at
# -10 dB, but at that level (~0.5 mW) the interferer could not be resolved
# above the ambient variation of a congested band, so the required
# performance-versus-configuration measurement was not possible.
TX_GAIN_MAX_DB = 0.0
MAX_RUN_SECONDS = 300.0     # a runaway transmitter is the thing to avoid


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------
def make_ofdm_waveform(n_symbols=64, n_fft=64, n_cp=16, duty=0.5, rng=None):
    """Build one 802.11a/g-structured OFDM burst followed by an idle gap.

    The subcarrier layout is the real one: 64-point IFFT, subcarriers -26..+26
    with DC nulled and four pilots, and a 16-sample cyclic prefix. At a sample
    rate of 20 MSPS that gives 312.5 kHz spacing, a 4 us symbol and roughly
    16.6 MHz of occupied bandwidth inside a 20 MHz channel - which is what makes
    the generated signal look like Wi-Fi to the sensing side rather than like a
    plain carrier.

    The trailing gap makes the signal bursty like real traffic; `duty` sets the
    fraction of time the transmitter is actually on.
    """
    rng = rng or np.random.default_rng()

    pilot_idx = np.array([-21, -7, 7, 21])
    data_idx = np.array([i for i in range(-26, 27)
                         if i != 0 and i not in pilot_idx])

    symbols = []
    for _ in range(n_symbols):
        X = np.zeros(n_fft, dtype=complex)
        # QPSK on the data subcarriers, fixed BPSK pilots.
        phases = rng.integers(0, 4, size=data_idx.size)
        X[data_idx % n_fft] = np.exp(1j * (np.pi / 4 + phases * np.pi / 2))
        X[pilot_idx % n_fft] = 1.0
        x = np.fft.ifft(X) * np.sqrt(n_fft)
        symbols.append(np.concatenate([x[-n_cp:], x]))   # prepend cyclic prefix

    burst = np.concatenate(symbols)
    if duty >= 1.0:
        frame = burst
    else:
        gap_len = int(len(burst) * (1.0 - duty) / max(duty, 1e-6))
        frame = np.concatenate([burst, np.zeros(gap_len, dtype=complex)])

    # Scale to a safe peak so the DAC never clips (clipping would splatter
    # energy into neighbouring channels and corrupt our own measurements).
    peak = np.max(np.abs(frame))
    if peak > 0:
        frame = frame / peak * 0.7
    return frame


def occupied_bandwidth(samples, fs, fraction=0.99):
    """Measured occupied bandwidth of the generated waveform (x% power)."""
    spec = np.abs(np.fft.fftshift(np.fft.fft(samples))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1.0 / fs))
    total = np.sum(spec)
    csum = np.cumsum(spec) / total
    lo = freqs[np.searchsorted(csum, (1 - fraction) / 2)]
    hi = freqs[np.searchsorted(csum, 1 - (1 - fraction) / 2)]
    return float(hi - lo)


# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------
def select_channel(occupied_centers_hz, candidates=None, bw_hz=WIFI_CHANNEL_BW_HZ):
    """Pick the channel that sits furthest from every occupied transmission.

    The rule the brief asks for is a maximin: choose the candidate whose
    *closest* occupied neighbour is as far away as possible. Ties are broken
    toward the centre of the band, which is what produces the behaviour
    described in the brief - with signals at both ends, the generator settles in
    the middle of the remaining gap rather than hugging one edge.

    Returns (channel, distance_to_nearest_occupied_hz).
    """
    if candidates is None:
        # Channels 1-13 only. Channel 14 is Japan-only and is not permitted
        # under the ETSI rules that apply here, so it must never be selected
        # even though it is the furthest point from a crowded low end.
        candidates = [c for c in sorted(WIFI_CHANNELS) if c <= 13]
    band_mid = (BAND_START_HZ + BAND_STOP_HZ) / 2.0

    if not occupied_centers_hz:
        # Empty band: start in the middle so there is room to move either way.
        ch = min(candidates, key=lambda c: abs(WIFI_CHANNELS[c] - band_mid))
        return ch, float("inf")

    def score(ch):
        fc = WIFI_CHANNELS[ch]
        nearest = min(abs(fc - f) for f in occupied_centers_hz)
        # Primary: maximise clearance. Secondary: prefer the band centre.
        return (nearest, -abs(fc - band_mid))

    best = max(candidates, key=score)
    return best, float(min(abs(WIFI_CHANNELS[best] - f)
                           for f in occupied_centers_hz))


def clearance_of(channel, occupied_centers_hz):
    """Distance from a channel centre to the nearest occupied transmission."""
    if not occupied_centers_hz:
        return float("inf")
    return float(min(abs(WIFI_CHANNELS[channel] - f) for f in occupied_centers_hz))


def choose_with_hysteresis(occupied_centers_hz, current_channel, candidates,
                           hysteresis_hz=3e6):
    """Pick the next channel, but only move if it is worth moving.

    The maximin rule on its own is unstable. With channels 1, 6 and 11 busy,
    channels 4, 8 and 13 all sit exactly 10 MHz from their nearest neighbour, so
    measurement noise of a few hundred kHz in the estimated centre frequencies
    is enough to change the winner - the generator then hops every single
    evaluation without the band having changed at all, which is both useless and
    very visible in the logs.

    Requiring a new channel to beat the current one by a clear margin removes
    the thrashing while still reacting immediately to a real change, because a
    genuine change moves the clearance by far more than the margin.

    Returns (channel, clearance_hz, moved_for_a_reason).
    """
    best, best_clear = select_channel(occupied_centers_hz, candidates)
    if current_channel is None:
        return best, best_clear, True
    if best == current_channel:
        return current_channel, best_clear, False

    current_clear = clearance_of(current_channel, occupied_centers_hz)
    if best_clear - current_clear >= hysteresis_hz:
        return best, best_clear, True
    return current_channel, current_clear, False


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class Generator:
    """Owns the transmit side of one Pluto and the hop decisions."""

    def __init__(self, sdr, waveform, tx_gain_db=-30.0, bw_hz=WIFI_CHANNEL_BW_HZ,
                 transmit=False, verbose=True):
        self.sdr = sdr
        self.waveform = waveform
        self.bw_hz = bw_hz
        self.transmit = transmit
        self.verbose = verbose
        self.current_channel = None
        self.tx_active = False

        self.tx_gain_db = float(np.clip(tx_gain_db, TX_GAIN_MIN_DB, TX_GAIN_MAX_DB))
        if transmit and self.tx_gain_db != tx_gain_db:
            print(f"[safety] TX gain clamped {tx_gain_db:+.1f} -> "
                  f"{self.tx_gain_db:+.1f} dB")

        if transmit:
            self.sdr.sdr.tx_rf_bandwidth = int(bw_hz)
            self.sdr.sdr.tx_hardwaregain_chan0 = self.tx_gain_db
            self.sdr.sdr.tx_cyclic_buffer = True

        # pyadi expects int16-scaled samples.
        self.tx_samples = (waveform * (2 ** 14)).astype(np.complex64)

    def start_tx(self, channel):
        """(Re)key the transmitter on `channel`.

        Retried, because allocating the cyclic TX buffer over USB fails
        intermittently on this board ("Input/output error") and an unhandled
        failure kills a measurement matrix halfway through.
        """
        fc = WIFI_CHANNELS[channel]
        self.current_channel = channel
        if not self.transmit:
            self.tx_active = False
            return

        last = None
        for attempt in range(5):
            try:
                if self.tx_active:
                    self.sdr.sdr.tx_destroy_buffer()
                    self.tx_active = False
                self.sdr.sdr.tx_lo = int(fc)
                self.sdr.sdr.tx(self.tx_samples)
                self.tx_active = True
                return
            except Exception as exc:
                last = exc
                if self.verbose:
                    print(f"[gen] TX start failed ({exc}); retry {attempt+1}/5")
                try:
                    self.sdr.sdr.tx_destroy_buffer()
                except Exception:
                    pass
                self.tx_active = False
                time.sleep(1.5)
        raise RuntimeError(f"Could not key the transmitter on channel "
                           f"{channel}: {last}")

    def stop_tx(self):
        if self.tx_active:
            self.sdr.sdr.tx_destroy_buffer()
            self.tx_active = False


class _SimulatedGenerator:
    """Stand-in for Generator when running without hardware: records the channel
    decisions so the hopping logic can be exercised, and transmits nothing."""

    def __init__(self):
        self.current_channel = None
        self.tx_active = False

    def start_tx(self, channel):
        self.current_channel = channel

    def stop_tx(self):
        self.tx_active = False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=None)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--bw", type=float, default=20e6,
                    help="generated signal bandwidth in Hz (sets the sample rate)")
    ap.add_argument("--tx-gain", type=float, default=-30.0,
                    help=f"TX hardware gain dB (clamped to <= {TX_GAIN_MAX_DB})")
    ap.add_argument("--rx-gain", type=float, default=40.0)
    ap.add_argument("--buffer", type=int, default=131072,
                    help="RX buffer samples; sets the dwell per LO step")
    ap.add_argument("--duty", type=float, default=0.5,
                    help="transmit duty cycle of the burst pattern")
    ap.add_argument("--sense-period", type=float, default=3.0,
                    help="seconds between spectrum re-evaluations")
    ap.add_argument("--threshold", type=float, default=8.0)
    ap.add_argument("--max-hold", type=int, default=4,
                    help="captures max-held per LO step")
    ap.add_argument("--sense-sweeps", type=int, default=1,
                    help="sweeps max-held per evaluation before deciding")
    ap.add_argument("--hysteresis", type=float, default=3e6,
                    help="Hz a new channel must beat the current one by to hop")
    ap.add_argument("--candidates", default="all",
                    help="'all' (1-13) or 'non-overlapping' (1,6,11)")
    ap.add_argument("--fixed-channel", type=int, default=None,
                    help="disable hopping and stay on this channel")
    ap.add_argument("--transmit", action="store_true",
                    help="actually key the transmitter (default: sense only)")
    ap.add_argument("--out", default="logs/part2")
    ap.add_argument("--simulate", action="store_true",
                    help="synthetic band, no hardware and no transmission")
    args = ap.parse_args()

    if args.duration > MAX_RUN_SECONDS:
        print(f"[safety] duration capped at {MAX_RUN_SECONDS:.0f} s")
        args.duration = MAX_RUN_SECONDS

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    candidates = (list(NON_OVERLAPPING) if args.candidates == "non-overlapping"
                  else sorted(c for c in WIFI_CHANNELS if c <= 13))

    # --- waveform ------------------------------------------------------------
    fs = args.bw          # 64-subcarrier OFDM: occupied BW tracks the sample rate
    wave = make_ofdm_waveform(duty=args.duty)
    meas_bw = occupied_bandwidth(wave, fs)
    print(f"[gen] OFDM waveform: {len(wave)} samples @ {fs/1e6:.1f} MSPS, "
          f"duty {args.duty:.0%}, occupied BW {meas_bw/1e6:.2f} MHz")

    if args.transmit:
        print("\n" + "=" * 70)
        print("  TRANSMITTER ENABLED - indoor, own equipment, minimum duration.")
        print(f"  TX gain {min(args.tx_gain, TX_GAIN_MAX_DB):+.1f} dB, "
              f"run {args.duration:.0f} s.")
        print("=" * 70 + "\n")
    else:
        print("[gen] sense-only run (pass --transmit to key the radio)\n")

    # --- radio ---------------------------------------------------------------
    if args.simulate:
        from part1_sensor import SimulatedSweeper
        sdr = None
        sweeper = SimulatedSweeper(nfft=1024, sample_rate=fs)
        gen = _SimulatedGenerator()
    else:
        sdr = Pluto(uri=args.uri, sample_rate=fs, rx_gain_db=args.rx_gain,
                    buffer_size=args.buffer)
        from part1_sensor import BandSweeper
        sweeper = BandSweeper(sdr, nfft=1024, max_hold=args.max_hold)
        gen = Generator(sdr, wave, tx_gain_db=args.tx_gain, bw_hz=args.bw,
                        transmit=args.transmit)

    log_path = f"{args.out}_hops.csv"
    log = open(log_path, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["t_s", "occupied_channels", "occupied_centers_mhz",
                     "chosen_channel", "chosen_center_mhz", "clearance_mhz",
                     "hopped", "sense_duration_s", "reaction_s"])

    t0 = time.time()
    n_hops = 0
    reactions = []
    try:
        while time.time() - t0 < args.duration:
            # --- sense -------------------------------------------------------
            # The transmitter is muted while sensing, otherwise the generator
            # would detect its own signal and refuse to stay anywhere.
            t_sense = time.time()
            gen.stop_tx()

            # Decide on several sweeps, not one. A single sweep is a snapshot of
            # a bursty band: measured on the live band the generator saw only
            # channel 11 busy in one unlucky sweep and hopped onto channel 1,
            # which was in fact the most crowded channel in the room. Max-holding
            # a few sweeps makes the occupied set stable enough to act on.
            psd = None
            for _ in range(max(1, args.sense_sweeps)):
                one = sweeper.sweep()
                psd = one if psd is None else np.maximum(psd, one)
            emitters, nf = detect_emitters(sweeper.freq_grid, psd, args.threshold)
            centers = [e["f_center"] for e in emitters]
            busy = occupied_channels(emitters)
            sense_dur = time.time() - t_sense

            # --- decide ------------------------------------------------------
            if args.fixed_channel:
                chosen, clearance = args.fixed_channel, float("nan")
            else:
                chosen, clearance, _ = choose_with_hysteresis(
                    centers, gen.current_channel, candidates, args.hysteresis)

            hopped = (gen.current_channel is not None
                      and chosen != gen.current_channel)

            # --- act ---------------------------------------------------------
            gen.start_tx(chosen)
            reaction = time.time() - t_sense     # sense -> back on air
            if hopped or gen.current_channel is None:
                reactions.append(reaction)

            t_rel = time.time() - t0
            writer.writerow([
                f"{t_rel:.3f}", " ".join(map(str, busy)),
                " ".join(f"{c/1e6:.2f}" for c in centers),
                chosen, f"{WIFI_CHANNELS[chosen]/1e6:.1f}",
                "inf" if clearance == float("inf") else f"{clearance/1e6:.2f}",
                int(hopped), f"{sense_dur:.3f}", f"{reaction:.3f}"])
            log.flush()

            if hopped:
                n_hops += 1
            marker = "HOP ->" if hopped else "stay  "
            clr = ("free band" if clearance == float("inf")
                   else f"{clearance/1e6:5.1f} MHz clear")
            busy_str = str(busy) if busy else "[]"
            print(f"  t={t_rel:6.1f}s  occupied={busy_str:<14} "
                  f"{marker} ch{chosen:<3} ({WIFI_CHANNELS[chosen]/1e6:.0f} MHz, "
                  f"{clr})  reaction {reaction*1000:.0f} ms")

            # Hold this configuration until the next scheduled evaluation.
            sleep_left = args.sense_period - (time.time() - t_sense)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\n[gen] stopped by user")
    finally:
        gen.stop_tx()
        if sdr:
            sdr.close()
        log.close()

    print(f"\n[out] hop log -> {log_path}")
    print(f"[out] {n_hops} hops in {time.time()-t0:.0f} s")
    if reactions:
        r = np.array(reactions) * 1000
        print(f"[out] reaction time: mean {r.mean():.0f} ms, "
              f"min {r.min():.0f} ms, max {r.max():.0f} ms")


if __name__ == "__main__":
    sys.exit(main())
