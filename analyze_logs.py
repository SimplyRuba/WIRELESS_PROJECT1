#!/usr/bin/env python3
"""
ENCS5323 Project - Turn the logged runs into the figures and numbers for the report.

  python analyze_logs.py --part1 logs/part1 --part2 logs/part2 --outdir figures

Produces:
  fig1_power_spectrum.png   mean + peak-hold spectrum over the whole 2.4 GHz band
  fig2_waterfall.png        time-frequency heatmap of the band
  fig3_occupancy.png        per-channel duty cycle and occupancy over time
  fig4_hops.png             generator channel decisions against the live band
and prints the summary tables to paste into the report.
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from pluto_common import WIFI_CHANNELS, BAND_START_HZ, BAND_STOP_HZ

# --- house style -------------------------------------------------------------
# Two validated categorical hues carry the line series; the heatmaps use a
# perceptually-uniform ramp with monotonic lightness so that a difference in
# colour always means a difference in power, including in greyscale print.
C_PRIMARY = "#2a78d6"    # blue   - slot 1
C_ACCENT = "#eb6834"     # orange - slot 2
C_THIRD = "#1baf7a"      # aqua   - slot 3 (direct-labelled only)
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d2"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.alpha": 0.7, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _channel_ticks(ax, y=None, every=1):
    """Label the Wi-Fi channel grid along the top of a frequency axis."""
    top = ax.secondary_xaxis("top")
    chs = [c for c in sorted(WIFI_CHANNELS) if c <= 13][::every]
    top.set_xticks([WIFI_CHANNELS[c] / 1e6 for c in chs])
    top.set_xticklabels([str(c) for c in chs], fontsize=8, color=INK_SOFT)
    top.set_xlabel("802.11 channel", fontsize=9, color=INK_SOFT)
    top.tick_params(length=2, colors=INK_SOFT)


# ---------------------------------------------------------------------------
def load_part1(base):
    npz = np.load(f"{base}_sweeps.npz")
    return {
        "freq": npz["freq_hz"], "psd": npz["psd_dbm"], "t": npz["t_s"],
        "threshold": float(npz["threshold_db"]),
        "simulated": bool(npz["simulated"]),
    }


def fig_power_spectrum(d, outdir):
    f = d["freq"] / 1e6
    mean = d["psd"].mean(axis=0)
    peak = d["psd"].max(axis=0)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(f, peak, lw=1.0, color=C_ACCENT, label="Peak hold")
    ax.plot(f, mean, lw=1.4, color=C_PRIMARY, label="Mean")
    ax.set_xlim(BAND_START_HZ / 1e6, BAND_STOP_HZ / 1e6)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (dBm per 19.5 kHz bin)")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_title(f"2.4 GHz band power spectrum over {d['t'][-1]:.0f} s "
                 f"({len(d['t'])} sweeps)" + ("  [SIMULATED]" if d["simulated"] else ""))
    for c in [ch for ch in sorted(WIFI_CHANNELS) if ch <= 13]:
        ax.axvline(WIFI_CHANNELS[c] / 1e6, color=GRID, lw=0.6, zorder=0)
    _channel_ticks(ax)
    ax.legend(loc="upper right")
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_power_spectrum.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_waterfall(d, outdir):
    f = d["freq"] / 1e6
    fig, ax = plt.subplots(figsize=(11, 5.2))
    vmin = float(np.percentile(d["psd"], 5))
    vmax = float(np.percentile(d["psd"], 99.9))
    im = ax.imshow(d["psd"], aspect="auto", origin="lower", cmap="viridis",
                   extent=[f[0], f[-1], 0, d["t"][-1]], vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Time (s)")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(False)
    ax.set_title("Channel activity over time"
                 + ("  [SIMULATED]" if d["simulated"] else ""))
    _channel_ticks(ax)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("Power (dBm)")
    cb.outline.set_edgecolor(GRID)
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_waterfall.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def load_occupancy(base):
    path = f"{base}_occupancy.csv"
    if not os.path.exists(path):
        return None
    ts, busy = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            ts.append(float(row["t_s"]))
            busy.append({c for c in range(1, 15) if row.get(f"ch{c}_busy") == "1"})
    return np.array(ts), busy


def fig_occupancy(base, outdir, simulated=False):
    got = load_occupancy(base)
    if not got:
        return None
    ts, busy = got
    chans = [c for c in sorted(WIFI_CHANNELS) if c <= 13]
    grid = np.array([[1 if c in b else 0 for b in busy] for c in chans])
    duty = grid.mean(axis=1) * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.4), height_ratios=[1, 1.25])

    ax1.bar([str(c) for c in chans], duty, color=C_PRIMARY, width=0.62)
    ax1.set_ylabel("Occupied (% of sweeps)")
    ax1.set_xlabel("802.11 channel")
    ax1.set_ylim(0, max(100, duty.max() * 1.15))
    ax1.set_title("Channel duty cycle" + ("  [SIMULATED]" if simulated else ""))
    # Direct labels on the meaningful bars: identity is never colour-alone.
    for i, v in enumerate(duty):
        if v > 1:
            ax1.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=8, color=INK_SOFT)

    ax2.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
               extent=[0, ts[-1], 0.5, len(chans) + 0.5],
               vmin=0, vmax=1, interpolation="nearest")
    ax2.set_yticks(range(1, len(chans) + 1))
    ax2.set_yticklabels([str(c) for c in chans], fontsize=8)
    ax2.set_ylabel("802.11 channel")
    ax2.set_xlabel("Time (s)")
    ax2.grid(False)
    ax2.set_title("Which channels were busy, sweep by sweep (yellow = occupied)")

    fig.tight_layout()
    p = os.path.join(outdir, "fig3_occupancy.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_hops(base, outdir):
    path = f"{base}_hops.csv"
    if not os.path.exists(path):
        return None, None
    t, chosen, occ, react, hopped = [], [], [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["t_s"]))
            chosen.append(int(row["chosen_channel"]))
            occ.append([int(x) for x in row["occupied_channels"].split()] if row["occupied_channels"] else [])
            react.append(float(row["reaction_s"]))
            hopped.append(row["hopped"] == "1")
    t = np.array(t)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    # Occupied channels sensed at each evaluation.
    ox = [t[i] for i, cs in enumerate(occ) for _ in cs]
    oy = [c for cs in occ for c in cs]
    ax.scatter(ox, oy, s=42, color=C_ACCENT, marker="s",
               label="Occupied (sensed)", zorder=2)
    ax.step(t, chosen, where="post", lw=2.0, color=C_PRIMARY,
            label="Generator channel", zorder=3)
    ax.scatter(t[np.array(hopped)], np.array(chosen)[np.array(hopped)],
               s=70, facecolor="white", edgecolor=C_PRIMARY, lw=2,
               zorder=4, label="Hop")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("802.11 channel")
    ax.set_yticks(range(1, 14))
    ax.set_ylim(0.5, 13.5)
    ax.set_title("Generator channel selection against the live band")
    # Below the axes: at the top right it overlapped the channel-13 trace.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    p = os.path.join(outdir, "fig4_hops.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "evaluations": len(t), "hops": int(np.sum(hopped)),
        "reaction_ms_mean": float(np.mean(react) * 1000),
        "reaction_ms_min": float(np.min(react) * 1000),
        "reaction_ms_max": float(np.max(react) * 1000),
    }
    return p, stats


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part1", default="logs/part1")
    ap.add_argument("--part2", default="logs/part2")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    made = []
    if os.path.exists(f"{args.part1}_sweeps.npz"):
        d = load_part1(args.part1)
        made += [fig_power_spectrum(d, args.outdir), fig_waterfall(d, args.outdir)]
        p = fig_occupancy(args.part1, args.outdir, d["simulated"])
        if p:
            made.append(p)

        print(f"\n=== Part 1 summary ({'SIMULATED' if d['simulated'] else 'measured'}) ===")
        print(f"sweeps: {len(d['t'])} over {d['t'][-1]:.1f} s "
              f"({len(d['t'])/max(d['t'][-1],1e-9):.2f} sweeps/s)")
        print(f"band:   {d['freq'][0]/1e6:.1f} - {d['freq'][-1]/1e6:.1f} MHz, "
              f"{len(d['freq'])} bins")
        got = load_occupancy(args.part1)
        if got:
            ts, busy = got
            print("\nchannel  duty cycle")
            for c in [ch for ch in sorted(WIFI_CHANNELS) if ch <= 13]:
                frac = sum(c in b for b in busy) / len(busy) * 100
                if frac > 0.5:
                    print(f"  ch{c:<4} {frac:5.1f}%")
    else:
        print(f"[warn] no Part 1 log at {args.part1}_sweeps.npz")

    p, stats = fig_hops(args.part2, args.outdir)
    if p:
        made.append(p)
        print("\n=== Part 2 summary ===")
        for k, v in stats.items():
            print(f"  {k:<18} {v:.1f}" if isinstance(v, float) else f"  {k:<18} {v}")
    else:
        print(f"[warn] no Part 2 log at {args.part2}_hops.csv")

    print("\nFigures written:")
    for m in made:
        print("  " + m)


if __name__ == "__main__":
    main()
