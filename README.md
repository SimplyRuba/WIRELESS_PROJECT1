# ENCS5323 Project — Live Wi-Fi Channel Occupancy Sensing and Dynamic Channel-Hopping Interference Generation

Birzeit University — Wireless and Mobile Networks (ENCS5323), Dr. Mohammad K. Jubran

A two-part 2.4 GHz system built on the ADALM-PLUTO SDR:

* **Part 1 — sensing.** Sweeps the whole 2400–2500 MHz band, stitches the captures
  into one power spectrum, identifies which 802.11 channels are occupied, and
  shows how that changes over time as a heatmap.
* **Part 2 — generation.** Produces a Wi-Fi-like OFDM signal in a standard 20 MHz
  channel, periodically re-senses the band, and hops to whichever channel keeps
  the greatest distance from the occupied ones.

Each part runs on its own Pluto, and `run_both.py` runs them together so the
generated signal can be watched moving across the band in real time.

---

## 1. Setup

```bash
cd ENCS5323_Project
python3.11 -m venv .venv
.venv/bin/pip install numpy scipy matplotlib pylibiio pyadi-iio
```

`pyadi-iio` needs the native **libiio 0.25** library. There is no Homebrew
formula for it, so build it from source (this is what was done on this machine):

```bash
git clone --depth 1 --branch v0.25 https://github.com/analogdevicesinc/libiio.git
cd libiio
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/homebrew \
      -DWITH_USB_BACKEND=ON -DWITH_NETWORK_BACKEND=ON \
      -DOSX_FRAMEWORK=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j8 && cmake --install build
```

Check the radio is visible:

```bash
.venv/bin/python run_both.py --list      # -> Plutos found: ['usb:1.4.5']
```

**Attach both antennas before measuring.** Without them the receiver only picks
up case leakage and the whole band reads as noise.

---

## 2. Running it

### Part 1 — sense the band

```bash
.venv/bin/python part1_sensor.py --duration 60 --out logs/part1
```

| Option | Meaning |
|---|---|
| `--uri` | Pluto URI (auto-detected if omitted) |
| `--fs` | sample rate, default 20 MSPS — sets how wide each slice is |
| `--gain` | manual RX gain in dB, default 40 |
| `--threshold` | dB above the noise floor before a channel counts as occupied |
| `--no-plot` | log only, no live window |
| `--simulate` | synthetic band, no hardware needed |

Writes `*_sweeps.npz` (every spectrum), `*_occupancy.csv` (per-channel decisions)
and `*_emitters.csv` (one row per detected transmission, with its measured
centre frequency and bandwidth).

### Part 2 — generate and hop

```bash
# sense and decide, transmit nothing (default)
.venv/bin/python part2_generator.py --duration 60

# actually on air
.venv/bin/python part2_generator.py --transmit --duration 60 --tx-gain -30
```

| Option | Meaning |
|---|---|
| `--bw` | generated bandwidth in Hz (sets the sample rate), default 20 MHz |
| `--tx-gain` | TX gain in dB, clamped to ≤ −10 dB |
| `--duty` | duty cycle of the burst pattern |
| `--sense-period` | seconds between re-evaluations |
| `--hysteresis` | how much better a channel must be before hopping |
| `--fixed-channel` | disable hopping (used by the configuration sweep) |
| `--simulate` | synthetic band, no hardware, never transmits |

### Both together (two Plutos)

```bash
.venv/bin/python run_both.py --duration 60 --transmit
```

### The configuration experiment

```bash
.venv/bin/python experiment_sweep.py --transmit \
    --channels 1,6,11 --bws 20e6,10e6 --dwell 30
```

Measures a baseline with the transmitter off, then RTT, packet loss, RSSI, SNR
and PHY rate for each (centre frequency, bandwidth) configuration.

### Figures for the report

```bash
.venv/bin/python analyze_logs.py --part1 logs/part1 --part2 logs/part2
```

---

## 3. How it works

### Stitching the band

The Pluto sees only one `sample_rate`-wide slice at a time, so the 100 MHz band
is tiled from several LO steps. Only the middle 75 % of each slice is kept —
the outer part sits on the analogue filter roll-off and would read artificially
low, inventing an empty channel at every seam. After each retune the first
buffer is discarded, because it still holds samples from the previous
frequency. The LO leakage spike at 0 Hz offset is interpolated out; it is an
artefact of the direct-conversion receiver, not a signal in the air.

### Finding the noise floor

Every occupancy decision is "is this bin more than X dB above the floor?", so a
biased floor breaks everything downstream. Two obvious estimators both fail on
this band and were measured doing so:

* the plain **median** sits *inside* a signal, because three active networks
  cover ~66 of the 100 MHz displayed;
* **sigma-clipping from above** never starts, because when most bins are signal
  nothing looks like an outlier.

The error was ~26 dB — a busy band read as completely empty. The floor is
therefore estimated from the bottom of the distribution instead (a low
percentile, refined by averaging everything near it), which is reliable as long
as some part of the band is quiet.

### Identifying occupied channels

2.4 GHz channels are 20 MHz wide but only 5 MHz apart, so naive per-channel
thresholding reports one access point on five channels. Two better-looking
methods also failed:

* **contiguous runs above threshold** — the 802.11 mask only falls to −20 dBr at
  11 MHz offset, so channels 1 and 6 stay joined and were reported as a single
  75 MHz signal;
* **peak picking** — an OFDM carrier is flat-topped, not peaked, so the maxima
  landed on the shoulders (channel 1 reported at 2420 MHz instead of 2412 MHz),
  because neighbouring skirts add and lift the inner edges.

What works is to use the structure we already know: carriers sit on a fixed
5 MHz grid and are flat across their middle 18 MHz. Each candidate channel is
scored by the median power over that flat region — a matched filter for the
OFDM rectangle — and the strongest are kept while their overlapping neighbours
are suppressed. Each carrier is then grown outward to its own −12 dB points,
stopping once the spectrum climbs back up out of the valley into the next
carrier, and the channel is assigned from the measured centroid rather than
from the scores (scores for channels 5, 6 and 7 are near-identical inside one
flat top, so using them made the reported channel flip run to run).

On the synthetic band this identifies channels 1, 6 and 11 with their correct
centre frequencies and ~20.4 MHz bandwidths on 40 of 40 sweeps.

### Seeing bursty traffic at all (probability of intercept)

A swept receiver only looks at one slice at a time, so it can easily miss a
bursty signal. On the live band a network *known* to be associated on channel 1
at -54 dBm was detected in **no sweep at all**, because each LO step dwelled
only a few hundred microseconds and Wi-Fi frames are sparse.

Two things fix it, and both are about dwell time per step:

* **max-hold several captures per LO step** (`--max-hold`), and
* **a large RX buffer** (`--buffer`), which buys dwell in one USB transfer
  instead of many.

Note the max is taken *across captures* while each capture is still averaged
internally. Max-holding inside the buffer as well was tried and made things
worse — it takes the maximum of 32 noise samples rather than 8, which inflates
and destabilises the noise floor and smeared detections across channels 1-9.

For the generator this is a direct accuracy-versus-latency trade-off, measured
on the live band:

| Dwell per step | Reaction time | Detection quality |
|---|---|---|
| ~13 ms (max-hold 2) | ~0.55 s | misses the busy low band, reports an empty band |
| ~26 ms (max-hold 4) | ~0.95 s | reliably finds channels 1/2/5 — **default** |
| ~52 ms, 3 sweeps | ~5.3 s | best detection, too slow to react |

### Instrument artefacts

Two features of the spectrum come from the radio, not the air, and are notched
out at every LO step:

* **0 Hz offset** — the direct-conversion receiver leaking its own LO;
* **±fs/8 offset** — a digital spur from the converter clock.

Both appeared as constant vertical lines in the waterfall at exactly 2450 and
2465 MHz, standing 25 dB above their neighbours. After notching, the excess at
those frequencies is +0.4 dB and +0.2 dB.

The stitching itself was checked for seam discontinuities: across the six slice
boundaries the step in the averaged spectrum is ≤0.1 dB at four of them, so the
band is genuinely continuous rather than a set of mismatched blocks.

### Validation against independent ground truth

The detector's output was checked against macOS's own Wi-Fi scan, which lists
every network it can see and the channel each one uses:

| Channel | Networks seen by macOS | Duty cycle measured here |
|---|---|---|
| 1 | 3 (one 40 MHz wide) | 43 % |
| 2 | 7 | 30 % |
| 5 | — (the 40 MHz network on ch1 bonds ch1+ch5) | 26 % |
| 7, 8, 9, 10 | 1 each | 9–15 % |
| 11–13 | none | ≤ 3 % |

The measured occupancy tracks the independent scan, including the elevated
channel 5 caused by the 40 MHz network centred on channel 1.

### The generated signal

Real 802.11a/g structure: 64-point IFFT, subcarriers −26…+26 with DC nulled and
four pilots, 16-sample cyclic prefix. At 20 MSPS that is 312.5 kHz spacing, a
4 µs symbol and a measured 16.45 MHz occupied bandwidth inside the 20 MHz
channel — close to the 16.6 MHz of real Wi-Fi, with a 9.4 dB PAPR. Bandwidth is
varied by changing the sample rate, which scales the occupied bandwidth
proportionally. A trailing idle gap makes the signal bursty like real traffic.

### Choosing a channel

The rule is a maximin: pick the candidate whose *nearest* occupied neighbour is
furthest away, breaking ties toward the centre of the band. That reproduces the
behaviour the brief describes — with only channel 1 busy it runs to the far end;
with both ends busy it settles in the middle of the remaining gap.

Channels 1–13 only. Channel 14 is Japan-only and must never be selected even
though it is the furthest point from a crowded low end.

A plain maximin thrashes, though: with channels 1, 6 and 11 busy, channels 4, 8
and 13 all sit exactly 10 MHz clear, so a few hundred kHz of noise in the
estimated centres changed the winner and the generator hopped on *every*
evaluation without the band having changed. A new channel must now beat the
current one by a margin (default 3 MHz) before a hop happens — which removed the
thrashing while still reacting to a real change within one sense period.

---

### Measuring the effect on a real link

The brief asks for the wireless connection's performance to be recorded for
each configuration. Getting a trustworthy number here turned out to be harder
than generating the signal, for a reason worth reporting: **the band is already
so congested that its own variation swamps a low-power interferer.** Measured
round-trip times to the gateway with the transmitter *off* ranged from 9.8 ms
to 56.6 ms across the run — a spread far larger than the effect being measured.

Three things in `experiment_sweep.py` address that:

* **A paired reference per configuration.** A transmitter-off measurement is
  taken immediately before each transmitter-on measurement, so each
  configuration is compared against the band as it was seconds earlier rather
  than against a single baseline taken minutes ago.
* **Repetition in randomised order** (`--reps`). A single pass in a fixed order
  confounds the configuration with whatever the room happened to be doing at
  that moment; shuffling and repeating lets a median separate the effect from
  the drift.
* **A sanity check on the Wi-Fi noise reading.** macOS reports a placeholder
  (−140 dBm) when the driver has no real noise figure, which silently produced
  an impossible 84 dB SNR until it was filtered out.

The link metrics themselves (RTT, loss, RSSI, SNR, PHY rate) all come from
tools that need no elevated privileges, so the experiment is reproducible on
any Mac without sudo.

### Result: the interference is decisive when it overlaps the victim

The experiment was run at the hardware maximum TX gain (0 dB, roughly 5 mW) with
the antenna beside the laptop, 3 repetitions per configuration in randomised
order, each with its own transmitter-off reference. The result is clean and
deterministic:

| Generator centre | Offset from victim | Bandwidth | Packet loss | Link |
|---|---|---|---|---|
| channel 1  | 0 MHz  | 10 & 20 MHz | **100 % (6/6 reps)** | dead |
| channel 6  | 25 MHz | 10 & 20 MHz | 0–0.3 % | normal |
| channel 11 | 50 MHz | 10 & 20 MHz | 0 % | normal |

The laptop's own Wi-Fi is associated on channel 1 (2412 MHz). When the generator
sits on that same channel it takes the link down completely — 100 % packet loss
in every one of the six repetitions, both bandwidths. Moved 25 MHz away to
channel 6, or 50 MHz away to channel 11, the same signal has no measurable
effect: loss stays at zero and RTT sits on its baseline.

This is exactly the behaviour the channel-hopping generator in Part 2 is
designed around, now demonstrated end to end: co-channel interference is
destructive, and putting distance between the transmitter and the occupied
channel is what protects the link. It also shows why frequency offset, not
bandwidth, is the variable that matters here — 10 MHz and 20 MHz behaved
identically at every offset.

Two earlier findings from the low-power runs are kept because they are part of
the story: at the −10 dB ceiling (~0.5 mW) no effect could be resolved above the
congested band's own variation (median ΔRTT +0.3 ms, 9 of 18 configurations
"worse" — a coin flip). Raising the power to the hardware maximum and closing
the distance is what turned that into the result above.

## 4. Safety## 4. Safety

The brief makes the group responsible for the transmitter. The code enforces:

* transmitting is **opt-in** — `--transmit`; the default run only senses;
* TX gain is clamped to **≤ −10 dB** whatever is passed;
* run length is capped at 300 s, and the sweep uses 30 s dwells;
* the transmitter is muted whenever the generator senses.

Operate indoors, directed at your own equipment, for the minimum time needed.

---

## 5. Files

| File | Purpose |
|---|---|
| `pluto_common.py` | device setup, channel plan, PSD, noise floor, emitter detection |
| `part1_sensor.py` | band sweeper, live spectrum + waterfall, occupancy logging |
| `part2_generator.py` | OFDM waveform, channel selection, hopping, TX control |
| `run_both.py` | runs both parts on two units at once |
| `experiment_sweep.py` | centre-frequency / bandwidth matrix vs link performance |
| `analyze_logs.py` | builds the report figures and summary tables |
| `logs/`, `figures/` | outputs |

Runs marked `[SIMULATED]` used the synthetic band and are **not** measurements;
only unmarked runs belong in the report as results.
