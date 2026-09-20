#!/usr/bin/env python3
"""
ENCS5323 Project - Verify the generated signal and measure it.

The generator mutes its transmitter while it senses, so it can never observe its
own output. This script keys the transmitter and receives at the same time on
the same Pluto (the AD936x has independent TX and RX chains), which both proves
the signal is really on air and measures the two properties the brief asks the
generated signal to be described by: its centre frequency and its bandwidth.

  python verify_tx.py --channel 6 --bw 20e6 --transmit
"""
import argparse
import numpy as np

from pluto_common import Pluto, WIFI_CHANNELS, welch_psd, noise_floor_dbm


def occupied_bw_from_psd(freqs, psd_dbm, fraction=0.99):
    """x%-power occupied bandwidth and power-weighted centre of a measured PSD."""
    lin = 10 ** (psd_dbm / 10.0)
    lin = lin - np.median(lin)          # remove the noise pedestal
    lin[lin < 0] = 0
    total = lin.sum()
    if total <= 0:
        return np.nan, np.nan
    c = np.cumsum(lin) / total
    lo = freqs[np.searchsorted(c, (1 - fraction) / 2)]
    hi = freqs[np.searchsorted(c, 1 - (1 - fraction) / 2)]
    centre = float((freqs * lin).sum() / total)
    return float(hi - lo), centre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=None)
    ap.add_argument("--channel", type=int, default=6)
    ap.add_argument("--bw", type=float, default=20e6)
    ap.add_argument("--tx-gain", type=float, default=-30.0)
    ap.add_argument("--rx-gain", type=float, default=10.0,
                    help="low, because the RX antenna is centimetres from the TX")
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--transmit", action="store_true")
    ap.add_argument("--out", default="figures_real/fig5_generated_signal.png")
    args = ap.parse_args()

    from part2_generator import Generator, make_ofdm_waveform

    fc = WIFI_CHANNELS[args.channel]
    sdr = Pluto(uri=args.uri, sample_rate=args.bw, rx_gain_db=args.rx_gain,
                buffer_size=32768)
    sdr.set_rx_freq(fc)

    def grab(n=8):
        acc = None
        for _ in range(n):
            off, p = welch_psd(np.asarray(sdr.capture()), sdr.sample_rate,
                               nfft=2048, rx_gain_db=sdr.rx_gain_db)
            acc = p if acc is None else np.maximum(acc, p)
        return off, acc

    off, psd_off = grab()
    print(f"[verify] TX off: median {np.median(psd_off):.1f} dBm/bin, "
          f"peak {psd_off.max():.1f}")

    if not args.transmit:
        print("[verify] --transmit not given; nothing was radiated.")
        sdr.close()
        return

    wave = make_ofdm_waveform(duty=args.duty)
    gen = Generator(sdr, wave, tx_gain_db=args.tx_gain, bw_hz=args.bw,
                    transmit=True)
    gen.start_tx(args.channel)
    import time
    time.sleep(0.5)
    off, psd_on = grab()
    gen.stop_tx()
    sdr.close()

    print(f"[verify] TX on : median {np.median(psd_on):.1f} dBm/bin, "
          f"peak {psd_on.max():.1f}")

    rise = np.median(psd_on) - np.median(psd_off)
    print(f"[verify] in-band rise with transmitter on: {rise:+.1f} dB")

    freqs = fc + off
    bw, centre = occupied_bw_from_psd(freqs, psd_on)
    print(f"\n[measured] centre frequency : {centre/1e6:.3f} MHz "
          f"(channel {args.channel} nominal {fc/1e6:.0f} MHz)")
    print(f"[measured] occupied BW (99%): {bw/1e6:.2f} MHz "
          f"(configured {args.bw/1e6:.0f} MHz channel)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.grid": True, "grid.color": "#d8d7d2",
                         "grid.alpha": .7, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#d8d7d2", "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(freqs / 1e6, psd_off, lw=1.0, color="#eb6834", label="Transmitter off")
    ax.plot(freqs / 1e6, psd_on, lw=1.3, color="#2a78d6", label="Transmitter on")
    ax.axvline(fc / 1e6, color="#52514e", ls="--", lw=1)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (dBm per bin)")
    ax.set_title(f"Generated signal, channel {args.channel} "
                 f"({fc/1e6:.0f} MHz) — measured centre {centre/1e6:.2f} MHz, "
                 f"99% BW {bw/1e6:.2f} MHz")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"[out] figure -> {args.out}")


if __name__ == "__main__":
    main()
