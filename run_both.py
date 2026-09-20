#!/usr/bin/env python3
"""
ENCS5323 Project - Run Part 1 and Part 2 together on two ADALM-PLUTO units.

Part 2 generates and hops on one unit while Part 1 watches the whole 2.4 GHz
band on the other, so the generated signal can be seen appearing, moving and
disappearing in real time.

  python run_both.py --list                       # find the attached units
  python run_both.py --duration 60                # sense-only (safe default)
  python run_both.py --duration 60 --transmit     # generator actually on air

The two units must be on different URIs. A Pluto presents itself as a USB
network gadget at 192.168.2.1; to use two at once, one of them has to be
re-addressed (for example to 192.168.3.1) from its config.txt on the PlutoSDR
mass-storage drive.
"""

import argparse
import subprocess
import sys
import time

from pluto_common import discover_plutos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor-uri", default=None, help="Pluto running Part 1")
    ap.add_argument("--generator-uri", default=None, help="Pluto running Part 2")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--transmit", action="store_true")
    ap.add_argument("--tx-gain", type=float, default=-30.0)
    ap.add_argument("--bw", type=float, default=20e6)
    ap.add_argument("--sense-period", type=float, default=3.0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--outdir", default="logs")
    ap.add_argument("--list", action="store_true", help="just list attached units")
    args = ap.parse_args()

    found = discover_plutos()
    print(f"[both] Plutos found: {found if found else 'none'}")
    if args.list:
        return 0

    sensor_uri = args.sensor_uri
    gen_uri = args.generator_uri
    if sensor_uri is None or gen_uri is None:
        if len(found) < 2:
            print("\n[both] Two units are required to run the parts together.\n"
                  "       Found: " + (", ".join(found) if found else "none") + "\n"
                  "       Run each part on its own unit instead, or re-address\n"
                  "       the second Pluto so both are visible at once.")
            return 1
        sensor_uri = sensor_uri or found[0]
        gen_uri = gen_uri or found[1]

    print(f"[both] Part 1 (sensing)   -> {sensor_uri}")
    print(f"[both] Part 2 (generator) -> {gen_uri}")

    py = sys.executable
    p1 = [py, "part1_sensor.py", "--uri", sensor_uri,
          "--duration", str(args.duration),
          "--out", f"{args.outdir}/part1"]
    if args.no_plot:
        p1.append("--no-plot")

    p2 = [py, "part2_generator.py", "--uri", gen_uri,
          "--duration", str(args.duration), "--bw", str(args.bw),
          "--sense-period", str(args.sense_period),
          "--tx-gain", str(args.tx_gain),
          "--out", f"{args.outdir}/part2"]
    if args.transmit:
        p2.append("--transmit")

    # Start the sensor first so the very first sweeps capture the band *before*
    # the generator keys up - that "before" frame is what makes the effect of
    # the generated signal visible in the report.
    proc1 = subprocess.Popen(p1)
    time.sleep(3.0)
    proc2 = subprocess.Popen(p2)

    try:
        proc2.wait()
        proc1.wait()
    except KeyboardInterrupt:
        for p in (proc2, proc1):
            p.terminate()
        print("\n[both] stopped")

    print("\n[both] done. Build the figures with:")
    print(f"       {py} analyze_logs.py --part1 {args.outdir}/part1 "
          f"--part2 {args.outdir}/part2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
