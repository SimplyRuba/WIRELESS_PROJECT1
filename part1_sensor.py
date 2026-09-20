#!/usr/bin/env python3
"""
ENCS5323 Project - Part 1: Live Wi-Fi channel occupancy sensing.

Sweeps the whole 2400-2500 MHz ISM band with one ADALM-PLUTO, stitches the
captures into a single power spectrum, decides which 802.11 channels are
occupied, and displays both a labelled power-spectrum plot and a time/frequency
heatmap (waterfall) of the activity.

  python part1_sensor.py --duration 60
  python part1_sensor.py --uri ip:192.168.2.1 --duration 60 --out logs/run1
  python part1_sensor.py --simulate --duration 20     # no hardware needed

Every sweep is written to an .npz plus a per-channel occupancy .csv so that
analyze_logs.py can rebuild the report figures afterwards.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

from pluto_common import (
    BAND_START_HZ, BAND_STOP_HZ, WIFI_CHANNELS, WIFI_CHANNEL_BW_HZ,
    Pluto, welch_psd, noise_floor_dbm, build_sweep_plan, channel_edges,
    detect_emitters, occupied_channels,
)


# ---------------------------------------------------------------------------
# Sweeping
# ---------------------------------------------------------------------------
class BandSweeper:
    """Tiles 2400-2500 MHz with successive LO steps and stitches the result.

    The Pluto can only look at one `sample_rate`-wide slice at a time, so the
    "whole band" view the project asks for has to be assembled from several
    retunes. Each slice contributes only its middle portion, where the analogue
    filter response is flat.
    """

    def __init__(self, sdr, nfft=1024, band=(BAND_START_HZ, BAND_STOP_HZ),
                 usable_fraction=0.75, dc_notch_hz=150e3, max_hold=8,
                 combine="mean"):
        self.sdr = sdr
        self.nfft = nfft
        self.band = band
        self.dc_notch_hz = dc_notch_hz
        self.max_hold = max_hold
        fs = sdr.sample_rate
        self.spur_offsets_hz = (0.0, -fs / 8.0, +fs / 8.0)
        self.combine = combine
        self.centres, self.usable_bw = build_sweep_plan(
            sdr.sample_rate, band[0], band[1], usable_fraction)

        # Common output grid at the native FFT resolution of a single capture.
        self.df = sdr.sample_rate / nfft
        self.freq_grid = np.arange(band[0], band[1], self.df)
        self.n_bins = len(self.freq_grid)

        print(f"[sweep] {len(self.centres)} LO steps "
              f"({', '.join(f'{c/1e6:.1f}' for c in self.centres)} MHz), "
              f"usable {self.usable_bw/1e6:.1f} MHz each, "
              f"RBW {self.df/1e3:.1f} kHz, {self.n_bins} bins")

    def sweep(self):
        """One full pass over the band -> power spectrum on self.freq_grid."""
        grid = np.full(self.n_bins, np.nan)

        for centre in self.centres:
            self.sdr.set_rx_freq(centre)

            # Wi-Fi is bursty, and one dwell of a few hundred microseconds per
            # LO step usually lands between frames - measured on a live band, a
            # network known to be associated on channel 1 at -54 dBm went
            # undetected in every sweep. Max-holding several *captures* per step
            # raises the probability of intercept enough to see the traffic.
            #
            # Note this is deliberately a max over captures and an average
            # within each capture. Max-holding inside the buffer as well was
            # tried and made things worse: it takes the maximum of 32 noise
            # samples instead of 8, which inflates and destabilises the noise
            # floor and smeared detections across channels 1-9.
            psd = None
            for _ in range(self.max_hold):
                samples = self.sdr.capture()
                offsets, one = welch_psd(samples, self.sdr.sample_rate,
                                         nfft=self.nfft,
                                         rx_gain_db=self.sdr.rx_gain_db,
                                         combine=self.combine)
                psd = one if psd is None else np.maximum(psd, one)

            # Two artefacts of the radio itself sit at fixed offsets from the
            # LO and must be removed before the slice is used, or they are
            # indistinguishable from real signals:
            #   0 Hz      - the direct-conversion receiver leaking its own LO
            #   +/- fs/8  - a digital spur from the converter clock
            # Both showed up as constant vertical lines in the waterfall at
            # exactly 2450 and 2465 MHz. They are replaced by interpolation.
            bad = np.zeros_like(offsets, dtype=bool)
            for off in self.spur_offsets_hz:
                bad |= np.abs(offsets - off) < self.dc_notch_hz
            if np.any(bad) and not np.all(bad):
                psd[bad] = np.interp(offsets[bad], offsets[~bad], psd[~bad])

            # Keep only the flat middle of the slice.
            keep = np.abs(offsets) <= self.usable_bw / 2.0
            f_abs = centre + offsets[keep]
            p_abs = psd[keep]

            # Paste onto the common grid.
            lo, hi = f_abs[0], f_abs[-1]
            sel = (self.freq_grid >= lo) & (self.freq_grid <= hi)
            if np.any(sel):
                grid[sel] = np.interp(self.freq_grid[sel], f_abs, p_abs)

        # Any residual gaps (band edges that no slice reached) get filled by
        # nearest-neighbour so the plot stays continuous.
        if np.any(np.isnan(grid)):
            idx = np.arange(self.n_bins)
            good = ~np.isnan(grid)
            if np.any(good):
                grid[~good] = np.interp(idx[~good], idx[good], grid[good])
        return grid


class SimulatedSweeper:
    """Synthetic band used to develop and validate the processing chain when no
    radio is attached. Clearly marked in every output so simulated runs can
    never be mistaken for measurements."""

    def __init__(self, nfft=1024, band=(BAND_START_HZ, BAND_STOP_HZ),
                 sample_rate=20e6, occupied=(1, 6, 11), seed=0):
        self.df = sample_rate / nfft
        self.freq_grid = np.arange(band[0], band[1], self.df)
        self.n_bins = len(self.freq_grid)
        self.occupied = list(occupied)
        self.rng = np.random.default_rng(seed)
        self.t0 = time.time()
        print(f"[sweep] SIMULATION: channels {self.occupied} occupied, "
              f"{self.n_bins} bins, RBW {self.df/1e3:.1f} kHz")

    def sweep(self):
        # Thermal noise floor with realistic ripple.
        psd = -96.0 + self.rng.normal(0, 1.5, self.n_bins)

        # After 15 s the scene changes, so the hopping logic gets exercised.
        occupied = self.occupied
        if time.time() - self.t0 > 15:
            occupied = [1, 2, 3]

        for ch in occupied:
            fc = WIFI_CHANNELS[ch]
            # The real 802.11 transmit spectral mask, in dB relative to the
            # in-band PSD: flat to +/-9 MHz, -20 dBr at 11 MHz, -28 dBr at
            # 20 MHz, -45 dBr at 30 MHz. Using the true mask matters, because
            # its shallow shoulders are exactly why adjacent networks overlap
            # above the detection threshold.
            off_mhz = np.abs(self.freq_grid - fc) / 1e6
            shape = np.interp(off_mhz,
                              [0, 9, 11, 20, 30, 50],
                              [0, 0, -20, -28, -45, -60])
            level = -62.0 + self.rng.normal(0, 2.0)
            # Wi-Fi is bursty: roughly a quarter of the time a network is idle
            # between frames and only its beacons are on the air.
            if self.rng.random() >= 0.75:
                level -= 12.0
            contrib = level + shape
            psd = 10 * np.log10(10 ** (psd / 10) + 10 ** (contrib / 10))
        time.sleep(0.15)
        return psd


# ---------------------------------------------------------------------------
# Occupancy decision
# ---------------------------------------------------------------------------
def analyse_sweep(freq_grid, psd_dbm, threshold_db=8.0,
                  bw_hz=WIFI_CHANNEL_BW_HZ):
    """Full analysis of one sweep.

    Returns (emitters, busy_channels, band_power, noise_floor) where
      emitters      - the distinct transmissions actually found in the air
      busy_channels - Wi-Fi channel numbers those emitters sit on
      band_power    - per-channel median power, kept for the heatmap and for
                      reporting how much energy each nominal channel sees
    """
    emitters, nf = detect_emitters(freq_grid, psd_dbm, threshold_db)
    busy = occupied_channels(emitters)

    band_power = {}
    for ch in sorted(WIFI_CHANNELS):
        lo, hi = channel_edges(ch, bw_hz)
        sel = (freq_grid >= lo) & (freq_grid <= hi)
        band_power[ch] = float(np.median(psd_dbm[sel])) if np.any(sel) else float("nan")

    return emitters, busy, band_power, nf


# ---------------------------------------------------------------------------
# Live display
# ---------------------------------------------------------------------------
class LiveDisplay:
    """Two stacked axes: instantaneous spectrum on top, waterfall underneath."""

    def __init__(self, freq_grid, history=200, simulated=False):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.freq_grid = freq_grid
        self.history = history
        self.water = np.full((history, len(freq_grid)), -120.0)

        plt.ion()
        self.fig, (self.ax_psd, self.ax_wf) = plt.subplots(
            2, 1, figsize=(13, 8), height_ratios=[1, 1.15])
        title = "ENCS5323 - Live 2.4 GHz Wi-Fi Channel Occupancy (ADALM-PLUTO)"
        if simulated:
            title += "   [SIMULATED DATA - NOT A MEASUREMENT]"
        self.fig.suptitle(title, fontsize=12, fontweight="bold")

        f_mhz = freq_grid / 1e6
        (self.line,) = self.ax_psd.plot(f_mhz, np.full_like(f_mhz, -120.0),
                                        lw=0.8, color="#1f77b4")
        (self.nf_line,) = self.ax_psd.plot(
            [f_mhz[0], f_mhz[-1]], [-100, -100], "--", lw=1.0,
            color="#888888", label="noise floor")
        self.ax_psd.set_xlim(f_mhz[0], f_mhz[-1])
        self.ax_psd.set_ylim(-115, -30)
        self.ax_psd.set_xlabel("Frequency (MHz)")
        self.ax_psd.set_ylabel("Power (dBm / bin)")
        self.ax_psd.grid(alpha=0.3)

        # Mark every Wi-Fi channel centre so occupancy is readable at a glance.
        self.ch_labels = {}
        for ch, fc in sorted(WIFI_CHANNELS.items()):
            self.ax_psd.axvline(fc / 1e6, color="#cccccc", lw=0.6, zorder=0)
            self.ch_labels[ch] = self.ax_psd.text(
                fc / 1e6, -112, str(ch), ha="center", fontsize=7,
                color="#999999")
        self.ax_psd.legend(loc="upper right", fontsize=8)
        self._spans = []

        self.im = self.ax_wf.imshow(
            self.water, aspect="auto", origin="lower", cmap="viridis",
            extent=[f_mhz[0], f_mhz[-1], 0, history], vmin=-105, vmax=-50,
            interpolation="nearest")
        self.ax_wf.set_xlabel("Frequency (MHz)")
        self.ax_wf.set_ylabel("Sweep (older -> newer)")
        cb = self.fig.colorbar(self.im, ax=self.ax_wf, pad=0.01)
        cb.set_label("Power (dBm)")
        self.fig.tight_layout()

    def update(self, psd, emitters, busy, nf):
        self.line.set_ydata(psd)
        self.nf_line.set_ydata([nf, nf])

        for ch, label in self.ch_labels.items():
            on = ch in busy
            label.set_color("#d62728" if on else "#999999")
            label.set_fontweight("bold" if on else "normal")

        # Shade the span of every detected transmission.
        for patch in self._spans:
            patch.remove()
        self._spans = [
            self.ax_psd.axvspan(e["f_lo"] / 1e6, e["f_hi"] / 1e6,
                                color="#d62728", alpha=0.12, zorder=0)
            for e in emitters
        ]

        self.water = np.roll(self.water, -1, axis=0)
        self.water[-1] = psd
        self.im.set_data(self.water)

        desc = ", ".join(f"ch{e['nearest_channel']}@{e['f_center']/1e6:.1f}MHz"
                         f"/{e['bw_hz']/1e6:.0f}MHz" for e in emitters)
        self.ax_psd.set_title(
            f"Occupied: {desc if desc else 'none'}    "
            f"(noise floor {nf:.1f} dBm)", fontsize=10)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save(self, path):
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[out] figure -> {path}")

    def close(self):
        self.plt.ioff()
        self.plt.close(self.fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=None, help="Pluto URI, e.g. ip:192.168.2.1")
    ap.add_argument("--fs", type=float, default=20e6, help="sample rate (Hz)")
    ap.add_argument("--gain", type=float, default=40.0, help="RX gain (dB), manual")
    ap.add_argument("--nfft", type=int, default=1024)
    ap.add_argument("--buffer", type=int, default=8192)
    ap.add_argument("--max-hold", type=int, default=8,
                    help="captures max-held per LO step (probability of intercept)")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds to run")
    ap.add_argument("--threshold", type=float, default=8.0,
                    help="dB above noise floor to call a channel occupied")
    ap.add_argument("--out", default="logs/part1", help="output basename")
    ap.add_argument("--no-plot", action="store_true", help="headless logging only")
    ap.add_argument("--simulate", action="store_true",
                    help="synthetic band, no hardware required")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    sdr = None
    if args.simulate:
        sweeper = SimulatedSweeper(nfft=args.nfft, sample_rate=args.fs)
    else:
        sdr = Pluto(uri=args.uri, sample_rate=args.fs, rx_gain_db=args.gain,
                    buffer_size=args.buffer)
        sweeper = BandSweeper(sdr, nfft=args.nfft, max_hold=args.max_hold)

    display = None
    if not args.no_plot:
        display = LiveDisplay(sweeper.freq_grid, simulated=args.simulate)

    sweeps, stamps, occ_rows = [], [], []
    csv_path = f"{args.out}_occupancy.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["t_s", "noise_floor_dbm", "n_emitters", "busy_channels"]
                    + [f"ch{c}_dbm" for c in sorted(WIFI_CHANNELS)]
                    + [f"ch{c}_busy" for c in sorted(WIFI_CHANNELS)])

    # One row per detected transmission, so the report can quote real measured
    # centre frequencies and bandwidths rather than just channel numbers.
    em_path = f"{args.out}_emitters.csv"
    em_file = open(em_path, "w", newline="")
    em_writer = csv.writer(em_file)
    em_writer.writerow(["t_s", "f_center_mhz", "bw_mhz", "f_lo_mhz", "f_hi_mhz",
                        "peak_dbm", "mean_dbm", "nearest_channel"])

    t_start = time.time()
    n = 0
    print(f"[run] sensing for {args.duration:.0f} s - Ctrl-C to stop early")
    try:
        while time.time() - t_start < args.duration:
            psd = sweeper.sweep()
            t_rel = time.time() - t_start
            emitters, busy, powers, nf = analyse_sweep(
                sweeper.freq_grid, psd, args.threshold)

            sweeps.append(psd.astype(np.float32))
            stamps.append(t_rel)
            writer.writerow([f"{t_rel:.3f}", f"{nf:.2f}", len(emitters),
                             " ".join(str(c) for c in busy)]
                            + [f"{powers.get(c, float('nan')):.2f}" for c in sorted(WIFI_CHANNELS)]
                            + [int(c in busy) for c in sorted(WIFI_CHANNELS)])
            for e in emitters:
                em_writer.writerow([
                    f"{t_rel:.3f}", f"{e['f_center']/1e6:.3f}",
                    f"{e['bw_hz']/1e6:.3f}", f"{e['f_lo']/1e6:.3f}",
                    f"{e['f_hi']/1e6:.3f}", f"{e['peak_dbm']:.2f}",
                    f"{e['mean_dbm']:.2f}", e["nearest_channel"]])
            occ_rows.append(set(busy))

            if display:
                display.update(psd, emitters, busy, nf)
            n += 1
            if n % 20 == 0:
                rate = n / (time.time() - t_start)
                print(f"  {n} sweeps ({rate:.1f}/s)  occupied={busy}")
    except KeyboardInterrupt:
        print("\n[run] stopped by user")
    finally:
        csv_file.close()
        em_file.close()
        if sdr:
            sdr.close()

    elapsed = time.time() - t_start
    npz_path = f"{args.out}_sweeps.npz"
    np.savez_compressed(
        npz_path,
        freq_hz=sweeper.freq_grid.astype(np.float64),
        psd_dbm=np.asarray(sweeps, dtype=np.float32),
        t_s=np.asarray(stamps, dtype=np.float64),
        threshold_db=args.threshold,
        simulated=args.simulate,
    )
    print(f"[out] {n} sweeps in {elapsed:.1f} s ({n/max(elapsed,1e-9):.2f} sweeps/s)")
    print(f"[out] spectra   -> {npz_path}")
    print(f"[out] occupancy -> {csv_path}")
    print(f"[out] emitters  -> {em_path}")

    if display:
        display.save(f"{args.out}_live.png")
        display.close()

    if occ_rows:
        total = len(occ_rows)
        print("\nChannel duty cycle over the run:")
        for ch in sorted(WIFI_CHANNELS):
            frac = sum(ch in r for r in occ_rows) / total
            if frac > 0.01:
                bar = "#" * int(frac * 40)
                print(f"  ch{ch:<3} {frac*100:5.1f}%  {bar}")


if __name__ == "__main__":
    sys.exit(main())
