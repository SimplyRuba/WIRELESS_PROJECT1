#!/usr/bin/env python3
"""
ENCS5323 Project - Demo control panel.

A small local web page that drives the same Python that runs the project, so the
live demo is done by clicking buttons instead of typing commands. It owns the
one radio and runs one mode at a time in a background thread:

  * Sensor    - Part 1: sweeps the whole 2.4 GHz band and shows the spectrum,
                waterfall, and which channels are occupied.
  * Generator - Part 2: senses the band and hops to the clearest channel;
                optionally transmits (opt-in, power-limited).

Run it:   .venv/bin/python demo/app.py        then open http://127.0.0.1:5000
Add --simulate defaults, or pick Simulate in the page, to rehearse with no radio.
"""
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np
from flask import Flask, jsonify, request, render_template

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pluto_common as PC
from pluto_common import detect_emitters, WIFI_CHANNELS
from part1_sensor import BandSweeper, SimulatedSweeper
from part2_generator import (Generator, make_ofdm_waveform,
                             choose_with_hysteresis, clearance_of)

app = Flask(__name__)

CANDIDATES = [c for c in sorted(WIFI_CHANNELS) if c <= 13]
N_POINTS = 720          # spectrum points sent to the browser

# --------------------------------------------------------------------------
STATE = {
    "running": False, "mode": None, "simulated": True, "transmit": False,
    "connected": False, "uri": None, "message": "Idle. Pick a mode to start.",
    "freqs": [], "psd": [], "noise_floor": None, "occupied": [], "emitters": [],
    "gen_channel": None, "gen_center_mhz": None, "clearance_mhz": None,
    "last_hop_t": None, "reaction_ms": None, "sweeps": 0, "elapsed": 0.0,
    "tx_gain": -10.0,
}
LOCK = threading.Lock()
WORKER = None

# --- live link test (the "send a packet" demo) ----------------------------
# Pings the router continuously and reports round-trip time and packet loss, so
# that when the generator transmits on the laptop's own channel you can watch
# the loss climb to 100% live.
PING = {"on": False, "gateway": None, "rtt_ms": None, "loss_pct": None, "up": None}
PING_THREAD = None


def _gateway():
    try:
        out = subprocess.run(["route", "-n", "get", "default"],
                             capture_output=True, text=True, timeout=4).stdout
        m = re.search(r"gateway:\s*(\S+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def _ping_once(host):
    try:
        out = subprocess.run(["ping", "-c", "1", "-t", "2", host],
                             capture_output=True, text=True, timeout=4).stdout
        m = re.search(r"time=([\d.]+)", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


class PingMonitor(threading.Thread):
    def __init__(self, gw):
        super().__init__(daemon=True)
        self.gw = gw
        self.stop_event = threading.Event()

    def run(self):
        window = []
        while not self.stop_event.is_set() and self.gw:
            window.append(_ping_once(self.gw))
            window = window[-6:]                 # last ~6 pings = the loss window
            got = [x for x in window if x is not None]
            loss = 100.0 * (len(window) - len(got)) / len(window)
            rtt = (sum(got) / len(got)) if got else None
            with LOCK:
                PING.update(rtt_ms=round(rtt, 1) if rtt is not None else None,
                            loss_pct=round(loss), up=(loss < 100.0))
            for _ in range(6):
                if self.stop_event.is_set():
                    break
                time.sleep(0.12)


def _downsample(freqs, psd, n=N_POINTS):
    """Max-pool the spectrum so bursts survive the reduction to n points."""
    L = len(psd)
    if L <= n:
        return list(map(float, freqs)), list(map(float, psd))
    k = L // n
    fz = np.asarray(freqs[:k * n]).reshape(n, k).mean(axis=1)
    pz = np.asarray(psd[:k * n]).reshape(n, k).max(axis=1)
    return fz.tolist(), pz.tolist()


class Worker(threading.Thread):
    def __init__(self, mode, simulate, transmit, tx_gain, uri=None,
                 fixed_channel=None):
        super().__init__(daemon=True)
        self.mode = mode
        self.simulate = simulate
        self.transmit = transmit and (mode == "generator") and (not simulate)
        self.tx_gain = tx_gain
        self.uri = uri
        self.fixed_channel = fixed_channel      # None = hop; else force this ch
        self.stop_event = threading.Event()

    def _set(self, **kw):
        with LOCK:
            STATE.update(kw)

    def run(self):
        sdr = gen = sweeper = None
        try:
            # ---- set up the radio (or the simulator) --------------------
            self._set(message="Connecting..." if not self.simulate
                      else "Starting simulation...", connected=False)
            if self.simulate:
                sweeper = SimulatedSweeper(nfft=1024, sample_rate=20e6)
                uri = "simulate"
            else:
                # Lighter USB load than the standalone scripts: a smaller buffer
                # and fewer max-hold captures per step means far fewer USB
                # transactions per sweep, which this old-firmware Pluto tolerates
                # much better in a long continuous demo run.
                sdr = PC.Pluto(uri=self.uri, sample_rate=20e6, rx_gain_db=50,
                               buffer_size=8192, verbose=False)
                sweeper = BandSweeper(sdr, nfft=1024, max_hold=2)
                uri = sdr.uri
            fds, _ = _downsample(sweeper.freq_grid / 1e6,
                                 np.zeros(len(sweeper.freq_grid)))
            self._set(connected=True, uri=uri, freqs=fds,
                      message=f"{self.mode.title()} running "
                              f"({'simulated' if self.simulate else uri}).")

            # ---- generator setup ----------------------------------------
            if self.mode == "generator" and self.transmit:
                wave = make_ofdm_waveform(duty=1.0)
                gen = Generator(sdr, wave, tx_gain_db=self.tx_gain,
                                bw_hz=20e6, transmit=True, verbose=False)

            current_channel = None
            t0 = time.time()
            sweeps = 0
            errors = 0                      # consecutive USB errors

            # ---- main loop ----------------------------------------------
            while not self.stop_event.is_set():
                t_sense = time.time()

                # One flaky sweep must not kill the whole run. The old firmware
                # drops the USB link now and then (Errno 5 / 60); we catch it,
                # keep the last good picture on screen, and retry. Only after
                # several failures in a row do we try a full radio reconnect,
                # and only if that also fails do we give up.
                try:
                    if self.mode == "generator" and gen is not None:
                        gen.stop_tx()
                    psd = sweeper.sweep()
                    errors = 0
                except Exception as exc:
                    errors += 1
                    self._set(message=f"Radio hiccup ({exc}); recovering "
                                      f"[{errors}]...")
                    if errors >= 4 and not self.simulate:
                        try:
                            sdr.close()
                        except Exception:
                            pass
                        try:
                            sdr = PC.Pluto(uri=self.uri, sample_rate=20e6,
                                           rx_gain_db=50, buffer_size=8192,
                                           verbose=False)
                            sweeper = BandSweeper(sdr, nfft=1024, max_hold=2)
                            gen = None if not (self.mode == "generator"
                                               and self.transmit) else \
                                Generator(sdr, make_ofdm_waveform(duty=1.0),
                                          tx_gain_db=self.tx_gain, bw_hz=20e6,
                                          transmit=True, verbose=False)
                            errors = 0
                            self._set(message="Reconnected. Running.")
                        except Exception as exc2:
                            self._set(message="The radio reset itself under "
                                              "load (old firmware). It usually "
                                              "comes back in a few seconds - "
                                              "press Reset, then Start again.",
                                      connected=False)
                            break
                    for _ in range(8):
                        if self.stop_event.is_set():
                            break
                        time.sleep(0.25)
                    continue

                emitters, nf = detect_emitters(sweeper.freq_grid, psd, 10.0)
                centers = [e["f_center"] for e in emitters]
                occ = sorted({e["nearest_channel"] for e in emitters})
                _, pds = _downsample(sweeper.freq_grid / 1e6, psd)
                sweeps += 1

                upd = dict(psd=pds, noise_floor=round(float(nf), 1),
                           occupied=occ, sweeps=sweeps,
                           elapsed=round(time.time() - t0, 1),
                           emitters=[{"ch": e["nearest_channel"],
                                      "center": round(e["f_center"] / 1e6, 1),
                                      "bw": round(e["bw_hz"] / 1e6, 1)}
                                     for e in emitters])

                if self.mode == "generator":
                    if self.fixed_channel:
                        # Forced channel (for the "jam my own Wi-Fi" demo): stay
                        # put and just report the clearance for context.
                        chosen = self.fixed_channel
                        clr = clearance_of(chosen, centers)
                    else:
                        chosen, clr, moved = choose_with_hysteresis(
                            centers, current_channel, CANDIDATES)
                    hopped = (current_channel is not None
                              and chosen != current_channel)
                    if gen is not None:
                        gen.start_tx(chosen)
                    reaction = time.time() - t_sense
                    if hopped or current_channel is None:
                        upd["reaction_ms"] = round(reaction * 1000)
                    if hopped:
                        upd["last_hop_t"] = round(time.time() - t0, 1)
                    current_channel = chosen
                    upd.update(gen_channel=chosen,
                               gen_center_mhz=round(WIFI_CHANNELS[chosen] / 1e6, 1),
                               clearance_mhz=(None if clr == float("inf")
                                              else round(clr / 1e6, 1)))

                self._set(**upd)

                # generator holds each configuration a few seconds; the sensor
                # paces itself so it does not hammer the USB link non-stop
                # (continuous back-to-back sweeps are what stress the old firmware).
                hold = 3.0 if self.mode == "generator" else (0.4 if not self.simulate else 0.0)
                while (time.time() - t_sense < hold
                       and not self.stop_event.is_set()):
                    time.sleep(0.1)

        except Exception as exc:            # surface any radio error in the UI
            self._set(message=f"Error: {exc}", connected=False)
        finally:
            try:
                if gen is not None:
                    gen.stop_tx()
            except Exception:
                pass
            try:
                if sdr is not None:
                    sdr.close()
            except Exception:
                pass
            self._set(running=False, mode=None,
                      message=STATE.get("message", "Stopped."))


def _stop_worker(wait=2.0):
    """Signal the worker to stop and stop waiting quickly.

    We only wait briefly: if the worker is wedged inside a blocking radio call
    it cannot be force-killed, so we abandon it (it is a daemon thread) and let
    the UI recover instead of freezing the request for many seconds.
    """
    global WORKER
    if WORKER is not None and WORKER.is_alive():
        WORKER.stop_event.set()
        WORKER.join(timeout=wait)
    WORKER = None


# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    with LOCK:
        s = dict(STATE)
        s["ping"] = dict(PING)
        return jsonify(s)


@app.route("/api/ping", methods=["POST"])
def api_ping():
    """Start or stop the continuous link test."""
    global PING_THREAD
    on = bool(request.get_json(force=True).get("on"))
    if PING_THREAD is not None and PING_THREAD.is_alive():
        PING_THREAD.stop_event.set()
    if on:
        gw = _gateway()
        with LOCK:
            PING.update(on=True, gateway=gw, rtt_ms=None, loss_pct=None, up=None)
        PING_THREAD = PingMonitor(gw)
        PING_THREAD.start()
    else:
        with LOCK:
            PING.update(on=False)
    return jsonify(ok=True, gateway=PING.get("gateway"))


@app.route("/api/start", methods=["POST"])
def api_start():
    global WORKER
    data = request.get_json(force=True)
    mode = data.get("mode", "sensor")
    simulate = bool(data.get("simulate", True))
    transmit = bool(data.get("transmit", False))
    tx_gain = float(data.get("tx_gain", -10.0))
    tx_gain = max(-89.0, min(0.0, tx_gain))     # hard safety clamp

    _stop_worker()
    with LOCK:
        STATE.update(running=True, mode=mode, simulated=simulate,
                     transmit=transmit and mode == "generator" and not simulate,
                     tx_gain=tx_gain, sweeps=0, elapsed=0.0, occupied=[],
                     emitters=[], gen_channel=None, gen_center_mhz=None,
                     clearance_mhz=None, last_hop_t=None, reaction_ms=None,
                     psd=[], noise_floor=None,
                     message=f"Starting {mode}...")
    fixed = data.get("fixed_channel")
    fixed = int(fixed) if fixed else None
    WORKER = Worker(mode, simulate, transmit, tx_gain,
                    uri=data.get("uri") or None, fixed_channel=fixed)
    WORKER.start()
    return jsonify(ok=True)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    _stop_worker()
    with LOCK:
        STATE.update(running=False, mode=None, message="Stopped.")
    return jsonify(ok=True)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Hard reset: stop any worker and clear all state, so the page recovers
    even if a previous run got stuck."""
    _stop_worker()
    with LOCK:
        STATE.update(running=False, mode=None, connected=False, transmit=False,
                     freqs=[], psd=[], noise_floor=None, occupied=[], emitters=[],
                     gen_channel=None, gen_center_mhz=None, clearance_mhz=None,
                     last_hop_t=None, reaction_ms=None, sweeps=0, elapsed=0.0,
                     message="Reset. Ready to start.")
    return jsonify(ok=True)


if __name__ == "__main__":
    if "--simulate" in sys.argv:
        STATE["simulated"] = True
    # Port 5000 is hijacked by macOS AirPlay Receiver, which silently
    # intercepts the browser before Flask ever sees the request, so the page
    # "never loads". Default to 8000 instead; override with a numeric argument.
    port = 8000
    for a in sys.argv[1:]:
        if a.isdigit():
            port = int(a)
    print(f"ENCS5323 demo -> http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
