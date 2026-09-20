#!/usr/bin/env python3
"""
ENCS5323 Project - Part 2 experiment: vary the generated signal systematically
and record what it does to a working wireless connection.

The brief asks for the generated signal to be described by its centre frequency
and bandwidth, for those settings to be varied systematically, and for the
wireless connection's performance to be recorded for each configuration. This
script does exactly that and writes one CSV row per configuration.

  # rehearse the whole matrix without transmitting
  python experiment_sweep.py --dry-run --dwell 5

  # the real thing
  python experiment_sweep.py --transmit --channels 1,6,11 --bws 20e6,10e6 --dwell 30

Link performance is measured with tools that need no elevated privileges:
ping round-trip time and loss to the default gateway, plus the Wi-Fi interface's
own RSSI, noise and negotiated PHY rate. If iperf3 is installed and a server is
given, throughput is recorded too.
"""

import argparse
import csv
import os
import re
import subprocess
import time

import numpy as np

from pluto_common import WIFI_CHANNELS, Pluto


# ---------------------------------------------------------------------------
# Link measurement
# ---------------------------------------------------------------------------
def default_gateway():
    try:
        out = subprocess.run(["route", "-n", "get", "default"],
                             capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"gateway:\s*(\S+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def wifi_info():
    """RSSI / noise / PHY rate / channel of the active Wi-Fi link (macOS)."""
    info = {"rssi_dbm": np.nan, "noise_dbm": np.nan,
            "tx_rate_mbps": np.nan, "wifi_channel": ""}
    try:
        out = subprocess.run(["system_profiler", "SPAirPortDataType"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return info
    # Only the block under "Current Network Information" describes our link;
    # the rest of the output lists every network the radio can see.
    idx = out.find("Current Network Information:")
    if idx < 0:
        return info
    block = out[idx:idx + 1200]
    m = re.search(r"Signal\s*/\s*Noise:\s*(-?\d+)\s*dBm\s*/\s*(-?\d+)\s*dBm", block)
    if m:
        rssi, noise = float(m.group(1)), float(m.group(2))
        info["rssi_dbm"] = rssi
        # macOS reports a placeholder (e.g. -140 dBm) when the driver has no
        # real noise reading, which otherwise yields an impossible 84 dB SNR.
        info["noise_dbm"] = noise if -100.0 <= noise <= -60.0 else np.nan
    m = re.search(r"Transmit Rate:\s*([\d.]+)", block)
    if m:
        info["tx_rate_mbps"] = float(m.group(1))
    m = re.search(r"Channel:\s*(\d+)\s*\(([^)]*)\)", block)
    if m:
        info["wifi_channel"] = f"{m.group(1)} ({m.group(2)})"
    return info


def ping_stats(host, duration_s, interval=0.1):
    """Round-trip time and loss over `duration_s` of pinging."""
    count = max(3, int(duration_s / interval))
    res = {"rtt_avg_ms": np.nan, "rtt_max_ms": np.nan,
           "rtt_stddev_ms": np.nan, "loss_pct": np.nan}
    try:
        out = subprocess.run(
            ["ping", "-c", str(count), "-i", str(interval), host],
            capture_output=True, text=True,
            timeout=duration_s + 30).stdout
    except Exception:
        return res
    m = re.search(r"([\d.]+)% packet loss", out)
    if m:
        res["loss_pct"] = float(m.group(1))
    m = re.search(r"min/avg/max/stddev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)", out)
    if m:
        res["rtt_avg_ms"] = float(m.group(2))
        res["rtt_max_ms"] = float(m.group(3))
        res["rtt_stddev_ms"] = float(m.group(4))
    return res


def iperf3_throughput(server, duration_s):
    if not server:
        return np.nan
    try:
        out = subprocess.run(
            ["iperf3", "-c", server, "-t", str(int(duration_s)), "-f", "m"],
            capture_output=True, text=True, timeout=duration_s + 30).stdout
        hits = re.findall(r"([\d.]+)\s+Mbits/sec.*receiver", out)
        return float(hits[-1]) if hits else np.nan
    except Exception:
        return np.nan


def measure_link(gateway, duration_s, iperf_server=None):
    stats = ping_stats(gateway, duration_s) if gateway else {}
    stats.update(wifi_info())
    stats["throughput_mbps"] = iperf3_throughput(iperf_server, min(duration_s, 10))
    rssi, noise = stats.get("rssi_dbm", np.nan), stats.get("noise_dbm", np.nan)
    snr = rssi - noise if not (np.isnan(rssi) or np.isnan(noise)) else np.nan
    # A 2.4 GHz link cannot really show 70-80 dB of SNR; when it appears, the
    # driver handed us a placeholder noise figure rather than a measurement.
    stats["snr_db"] = snr if (not np.isnan(snr) and 0.0 < snr < 50.0) else np.nan
    return stats


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=None)
    ap.add_argument("--channels", default="1,6,11",
                    help="generator centre channels to test")
    ap.add_argument("--bws", default="20e6,10e6",
                    help="generated bandwidths in Hz")
    ap.add_argument("--dwell", type=float, default=30.0,
                    help="seconds per configuration (brief suggests 30-60)")
    ap.add_argument("--tx-gain", type=float, default=-30.0)
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--gateway", default=None)
    ap.add_argument("--iperf-server", default=None)
    ap.add_argument("--transmit", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="walk the matrix and measure, but never transmit")
    ap.add_argument("--reps", type=int, default=1,
                    help="repetitions of the whole matrix, in randomised order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paired", action="store_true", default=True,
                    help="measure a TX-off reference before each configuration")
    ap.add_argument("--ref-dwell", type=float, default=15.0,
                    help="seconds for each paired reference measurement")
    ap.add_argument("--out", default="logs/sweep")
    args = ap.parse_args()

    if args.dry_run:
        args.transmit = False

    channels = [int(c) for c in args.channels.split(",")]
    bws = [float(b) for b in args.bws.split(",")]
    gateway = args.gateway or default_gateway()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[exp] gateway {gateway}   Wi-Fi link: {wifi_info()}")
    print(f"[exp] {len(channels)}x{len(bws)} configurations, "
          f"{args.dwell:.0f} s each "
          f"(~{(len(channels)*len(bws)+1)*args.dwell/60:.1f} min total)")
    if args.transmit:
        print("\n" + "=" * 70)
        print("  TRANSMITTER ENABLED - indoor, own equipment, minimum duration.")
        print("=" * 70 + "\n")

    rows = []
    csv_path = f"{args.out}_configs.csv"
    fields = ["config", "tx_on", "channel", "center_mhz", "bw_mhz",
              "tx_gain_db", "duty", "rtt_avg_ms", "rtt_max_ms",
              "rtt_stddev_ms", "loss_pct", "rssi_dbm", "noise_dbm", "snr_db",
              "tx_rate_mbps", "throughput_mbps", "wifi_channel"]
    fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()

    # --- baseline with the transmitter off ----------------------------------
    # Without this every later number is uninterpretable: we need to know what
    # the link does when we are not interfering with it at all.
    print("[exp] baseline (transmitter off)...")
    base = measure_link(gateway, args.dwell, args.iperf_server)
    base.update({"config": "baseline", "tx_on": 0, "channel": "",
                 "center_mhz": "", "bw_mhz": "", "tx_gain_db": "",
                 "duty": ""})
    writer.writerow(base)
    fh.flush()
    rows.append(base)
    print(f"        RTT {base.get('rtt_avg_ms', float('nan')):.1f} ms, "
          f"loss {base.get('loss_pct', float('nan')):.1f}%, "
          f"SNR {base.get('snr_db', float('nan')):.0f} dB")

    # --- the configuration matrix -------------------------------------------
    sdr = gen = None
    try:
        # Randomised, repeated order. The ambient band drifts by tens of
        # milliseconds of RTT on its own, so a single pass in a fixed order
        # confounds the configuration with whatever the room happened to be
        # doing at that moment. Repeating in a shuffled order lets the effect be
        # separated from the drift by taking medians per configuration.
        import random
        rng = random.Random(args.seed)
        plan = [(bw, ch) for bw in bws for ch in channels] * max(1, args.reps)
        rng.shuffle(plan)
        # Group by bandwidth within the shuffled plan so the radio is only
        # reconfigured when it has to be (each change costs a reconnect).
        plan.sort(key=lambda t: t[0])

        current_bw = None
        for bw, ch in plan:
            if args.transmit and bw != current_bw:
                from part2_generator import Generator, make_ofdm_waveform
                if sdr is not None:
                    gen.stop_tx()
                    sdr.close()
                    sdr = gen = None
                # Bandwidth is set by the sample rate, so each bandwidth needs
                # the radio and the waveform rebuilt.
                sdr = Pluto(uri=args.uri, sample_rate=bw, rx_gain_db=40)
                wave = make_ofdm_waveform(duty=args.duty)
                gen = Generator(sdr, wave, tx_gain_db=args.tx_gain,
                                bw_hz=bw, transmit=True)
                current_bw = bw

            if True:
                name = f"ch{ch}_bw{bw/1e6:.0f}MHz"
                print(f"[exp] {name}: {'transmitting' if args.transmit else 'DRY RUN'} "
                      f"for {args.dwell:.0f} s...")
                if args.transmit:
                    gen.start_tx(ch)
                    time.sleep(0.3)      # let the PLL settle before measuring

                # A reference measured immediately before each configuration.
                # Ambient 2.4 GHz traffic drifts over the minutes this matrix
                # takes, so a single baseline at the start cannot separate our
                # interference from the room's own variation; a paired
                # before/after reading can.
                if args.paired and args.transmit:
                    gen.stop_tx()
                    time.sleep(0.3)
                    ref = measure_link(gateway, args.ref_dwell, None)
                    ref.update({"config": name + "_ref", "tx_on": 0,
                                "channel": ch,
                                "center_mhz": WIFI_CHANNELS[ch] / 1e6,
                                "bw_mhz": bw / 1e6, "tx_gain_db": "",
                                "duty": ""})
                    writer.writerow(ref)
                    fh.flush()
                    rows.append(ref)
                    gen.start_tx(ch)
                    time.sleep(0.3)

                st = measure_link(gateway, args.dwell, args.iperf_server)
                st.update({"config": name, "tx_on": int(args.transmit),
                           "channel": ch, "center_mhz": WIFI_CHANNELS[ch] / 1e6,
                           "bw_mhz": bw / 1e6, "tx_gain_db": args.tx_gain,
                           "duty": args.duty})
                if args.transmit:
                    gen.stop_tx()

                writer.writerow(st)
                fh.flush()
                rows.append(st)
                delta = ""
                if args.paired and args.transmit:
                    d = st.get("rtt_avg_ms", np.nan) - ref.get("rtt_avg_ms", np.nan)
                    delta = f" [ref {ref.get('rtt_avg_ms', float('nan')):.1f}, Δ{d:+.1f} ms]"
                print(f"        RTT {st.get('rtt_avg_ms', float('nan')):.1f} ms"
                      f"{delta} "
                      f"(max {st.get('rtt_max_ms', float('nan')):.1f}), "
                      f"loss {st.get('loss_pct', float('nan')):.1f}%, "
                      f"SNR {st.get('snr_db', float('nan')):.0f} dB, "
                      f"rate {st.get('tx_rate_mbps', float('nan')):.0f} Mbps")
    except KeyboardInterrupt:
        print("\n[exp] stopped by user")
    finally:
        if gen:
            gen.stop_tx()
        if sdr:
            sdr.close()
        fh.close()

    print(f"\n[out] {len(rows)} configurations -> {csv_path}")
    _plot(rows, args.out)


def _plot(rows, out_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = ["#2a78d6", "#eb6834", "#1baf7a"]
    INK_SOFT = "#52514e"
    GRID = "#d8d7d2"
    plt.rcParams.update({
        "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.7,
        "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": GRID,
        "legend.frameon": False, "font.size": 10,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    })

    cfgs = [r for r in rows if r.get("config") != "baseline"]
    base = next((r for r in rows if r.get("config") == "baseline"), None)
    if not cfgs:
        return

    bws = sorted({float(r["bw_mhz"]) for r in cfgs})
    chans = sorted({int(r["channel"]) for r in cfgs})

    # Packet loss leads, because once the interferer is strong enough to take
    # the link down the RTT is undefined - no packet returns to be timed - and
    # a missing RTT bar would otherwise read as "no effect" rather than "total
    # loss", which is the opposite of the truth.
    metrics = [("loss_pct", "Packet loss (%)"),
               ("rtt_avg_ms", "Mean RTT (ms)"),
               ("snr_db", "Link SNR (dB)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    width = 0.8 / max(len(bws), 1)

    for ax, (key, label) in zip(axes, metrics):
        for i, bw in enumerate(bws):
            xs, ys = [], []
            for j, ch in enumerate(chans):
                r = next((r for r in cfgs
                          if int(r["channel"]) == ch and float(r["bw_mhz"]) == bw), None)
                if r:
                    xs.append(j + (i - (len(bws) - 1) / 2) * width)
                    ys.append(float(r.get(key, np.nan)))
            ax.bar(xs, ys, width=width * 0.9, color=C[i % len(C)],
                   label=f"{bw:.0f} MHz")
            # Mark configurations where the link was down entirely.
            if key == "rtt_avg_ms":
                for x, y in zip(xs, ys):
                    if np.isnan(y):
                        ax.text(x, 0, "link\ndown", ha="center", va="bottom",
                                fontsize=7, color="#b03020", rotation=0)
        if base is not None and not np.isnan(float(base.get(key, np.nan))):
            ax.axhline(float(base[key]), ls="--", lw=1.2, color=INK_SOFT)
            ax.text(len(chans) - 0.5, float(base[key]), " baseline",
                    va="bottom", ha="right", fontsize=8, color=INK_SOFT)
        ax.set_xticks(range(len(chans)))
        ax.set_xticklabels([f"ch{c}" for c in chans])
        ax.set_xlabel("Generator centre channel")
        ax.set_ylabel(label)
        ax.set_title(label)

    axes[0].legend(title="Generated BW", loc="upper left")
    fig.suptitle("Wireless link performance vs generated signal configuration",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p = f"{out_base}_performance.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[out] figure -> {p}")


if __name__ == "__main__":
    main()
