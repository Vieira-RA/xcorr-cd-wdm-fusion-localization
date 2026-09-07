#!/usr/bin/env python
"""
Fused acquisition: Tektronix DPO7254 oscilloscope (4 detector channels)
+ Novoptel PM1000 reference polarimeter (Stokes vector).

Reads N measurements from both instruments back-to-back (same SOP per pair),
with matched integration windows (both on a 10 ns clock), and saves two
index-aligned CSVs sharing one timestamp.
"""

import sys
import csv
import gc
import os
import time
import statistics
from datetime import datetime

import numpy as np
import vxi11
from vxi11.vxi11 import Vxi11Exception

import Python_USB
import matlab

# ================= Configuration =================
# --- Oscilloscope ---
SCOPE_IP = "198.192.1.1"
CHANNELS = [1, 2, 3, 4]
CH_SCALES = {1: .02, 2: .02, 3: .02, 4: .02}   # V/div - adjust per channel if needed
SCOPE_TIMEOUT = 30.0

# --- Timing (matched between instruments) ---
# PM1000 samples at 100 MS/s -> 10 ns/sample; averaging window = 2^ATE * 10 ns.
HORIZONTAL_SCALE = 10e-6     # 10 µs/div → 100 µs window (was 1e-6)
RECORD_LENGTH    = 10000     # 10000 samples @ 10 ns (was 1000)
PM_SAMPLE_S = 10e-9         # PM1000 sample period
ATE_MIN, ATE_MAX = 0, 20

# --- PM1000 ---
PM_DESCR = "PM1000-100M-XL-FA-N20-D 82"

# --- Shared ---
DELAY_BETWEEN_S = 0.14       # wait between measurements so the scrambler changes SOP

# --- MATLAB Runtime stability (the bridge leaks arrays in long loops) ---
GC_EVERY = 10               # force Python gc every N measurements to free MATLAB arrays
PM_REINIT_EVERY = 0         # re-init the PM1000 every N measurements (0 = never)

# ================= Number of measurements =================
if len(sys.argv) > 1:
    NUM_MEASUREMENTS = int(sys.argv[1])      # e.g. python fused_acquisition.py 50
else:
    NUM_MEASUREMENTS = int(input("Enter number of measurements: "))

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SCOPE_CSV = f"detector_readings_{RUN_ID}.csv"
PM_CSV = f"stokes_{RUN_ID}.csv"


# ================= Oscilloscope =================
def read_curve(scope):
    """Read the current channel's CURVE? block and return raw int8 samples."""
    scope.write("CURVE?")
    hdr = scope.read_raw(2)
    if hdr[:1] != b"#":
        raise RuntimeError(f"Unexpected CURVE? header: {hdr!r}")
    n_digits = int(hdr[1:2])
    byte_count = int(scope.read_raw(n_digits).decode("ascii"))
    data = scope.read_raw(byte_count)

    saved = scope.timeout
    scope.timeout = 0.5
    try:
        scope.read_raw(1)          # consume optional trailing newline
    except Vxi11Exception:
        pass
    finally:
        scope.timeout = saved

    return np.frombuffer(data, dtype=np.int8).astype(float)


def read_scope_means(scope):
    """Read CH1..CH4 from the CURRENT (frozen) acquisition; {ch: mean volts}."""
    means = {}
    for ch in CHANNELS:
        scope.write(f"DATA:SOURCE CH{ch}")
        y_mult = float(scope.ask("WFMOUTPRE:YMULT?"))
        y_zero = float(scope.ask("WFMOUTPRE:YZERO?"))
        y_off  = float(scope.ask("WFMOUTPRE:YOFF?"))

        raw = read_curve(scope)
        volts = (raw - y_off) * y_mult + y_zero
        means[ch] = float(np.mean(volts))
    return means


def init_scope():
    """Configure the scope and return (scope, actual_time_window_s)."""
    scope = vxi11.Instrument(SCOPE_IP)
    scope.timeout = SCOPE_TIMEOUT
    print(scope.ask("*IDN?"))

    for ch in CHANNELS:
        scope.write(f"CH{ch}:STATE ON")
        scope.write(f"CH{ch}:SCALE {CH_SCALES[ch]}")
        scope.write(f"CH{ch}:POSITION 0.0")

    # deterministic timebase + record length (this is the "guarantee")
    scope.write(f"HORIZONTAL:SCALE {HORIZONTAL_SCALE}")
    scope.write(f"HORIZONTAL:RECORDLENGTH {RECORD_LENGTH}")

    scope.write("DATA:ENC SRI")
    scope.write("DATA:WIDTH 1")
    scope.write("WFMOUTPRE:BYT_NR 1")
    scope.write("DATA:START 1")
    scope.write("DATA:STOP 1e10")

    xincr = float(scope.ask("WFMOUTPRE:XINCR?"))
    n_pts = int(scope.ask("WFMOUTPRE:NR_PT?"))
    scope.write(f"DATA:STOP {n_pts}")

    scope.write("TRIGGER:A:MODE AUTO")
    scope.write("ACQUIRE:STOPAFTER RUNSTOP")
    scope.write("ACQUIRE:STATE RUN")
    time.sleep(1.0)

    scope_window_s = xincr * n_pts
    print(f"Scope: xincr = {xincr*1e9:.3f} ns, points = {n_pts}, "
          f"window = {scope_window_s*1e6:.3f} us")
    return scope, scope_window_s


# ================= PM1000 =================
def decode_stokes_from_regs(reg_values):
    S0_int, S0_frac, S1_int, S1_frac, S2_int, S2_frac, S3_int, S3_frac = reg_values
    S0 = S0_int + S0_frac / 65536.0

    def decode_si(int_val, frac_val):
        return (int_val - 32768) + frac_val / 65536.0

    return (S0,
            decode_si(S1_int, S1_frac),
            decode_si(S2_int, S2_frac),
            decode_si(S3_int, S3_frac))


# Pre-created register-address arrays (reused every read, instead of allocating
# a new matlab.double() per register per measurement, which crashes the MATLAB
# Runtime over long runs).
_STOKES_REGS = [
    (512+10, 'S0_int'), (512+11, 'S0_frac'),
    (512+12, 'S1_int'), (512+13, 'S1_frac'),
    (512+14, 'S2_int'), (512+15, 'S2_frac'),
    (512+16, 'S3_int'), (512+17, 'S3_frac'),
]
ADDR_ARRAYS = [(addr, name, matlab.double([addr], size=(1, 1)))
               for addr, name in _STOKES_REGS]


def read_stokes_vector(pm):
    raw = []
    for addr, name, addr_arr in ADDR_ARRAYS:
        val, ok = pm.readpm(addr_arr, nargout=2)
        if not ok:
            raise RuntimeError(f"Failed to read {name} at address {addr}")
        raw.append(int(val))
    return decode_stokes_from_regs(raw)


def init_pm1000(ate):
    pm = Python_USB.initialize()
    ok, diagnosis, handle, in_descr, out_descr = pm.initpm(
        matlab.uint16([3]), PM_DESCR, nargout=5)   # USB 3.0
    if not ok:
        pm.terminate()
        raise RuntimeError(f"PM1000 init failed: {diagnosis}")
    print("PM1000 initialized successfully.")

    ate_addr = matlab.double([512+1], size=(1, 1))
    if not pm.writepm(ate_addr, matlab.uint16([ate], size=(1, 1))):
        pm.terminate()
        raise RuntimeError("Failed to set ATE")

    pm.writepm(matlab.double([512+92], size=(1, 1)), matlab.uint16([0], size=(1, 1)))
    return pm


def close_pm1000(pm):
    try:
        pm.closepm()
        pm.terminate()
    except Exception:
        pass


# ================= Main =================
def main():
    scope = None
    pm = None
    try:
        scope, scope_window_s = init_scope()

        ate = int(round(np.log2(scope_window_s / PM_SAMPLE_S)))
        ate = max(ATE_MIN, min(ATE_MAX, ate))
        pm_window_s = (2 ** ate) * PM_SAMPLE_S

        pm = init_pm1000(ate)

        print(f"PM1000: ATE = {ate}, window = {pm_window_s*1e6:.3f} us")
        print(f"Integration match: {(pm_window_s/scope_window_s - 1)*100:+.1f}% "
              f"(scope {scope_window_s*1e6:.3f} us vs pm {pm_window_s*1e6:.3f} us)")

        with open(SCOPE_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["index", "ch1", "ch2", "ch3", "ch4"])
        with open(PM_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["index", "S0_uW", "S1_uW", "S2_uW", "S3_uW", "DOP"])

        stokes_rows = []
        for i in range(1, NUM_MEASUREMENTS + 1):
            # Optional safety net: reset the MATLAB bridge before it can crash.
            if PM_REINIT_EVERY > 0 and i > 1 and (i - 1) % PM_REINIT_EVERY == 0:
                close_pm1000(pm)
                pm = init_pm1000(ate)
                print(f"Re-initialized PM1000 at measurement {i}")

            # Freeze the scope so all 4 channels share ONE SOP instant
            # (otherwise the scrambler drifts between the 4 sequential reads).
            scope.write("ACQUIRE:STATE STOP")

            # Read the PM1000 immediately, closest to the frozen instant.
            S0, S1, S2, S3 = read_stokes_vector(pm)

            # Read the 4 scope channels from that frozen acquisition.
            scope_means = read_scope_means(scope)

            # Resume acquisition for the next measurement.
            scope.write("ACQUIRE:STATE RUN")

            dop = (S1**2 + S2**2 + S3**2) ** 0.5 / S0 if S0 > 0 else 0.0

            with open(SCOPE_CSV, "a", newline="") as f:
                csv.writer(f).writerow([i, scope_means[1], scope_means[2],
                                        scope_means[3], scope_means[4]])
            with open(PM_CSV, "a", newline="") as f:
                csv.writer(f).writerow([i, S0, S1, S2, S3, dop])
            stokes_rows.append((i, S0, S1, S2, S3, dop))

            print(f"Measurement {i}/{NUM_MEASUREMENTS}: "
                  f"ch1={scope_means[1]:.6e} ch2={scope_means[2]:.6e} "
                  f"ch3={scope_means[3]:.6e} ch4={scope_means[4]:.6e} | "
                  f"S0={S0:.3f} DOP={dop:.4f}")

            if i < NUM_MEASUREMENTS:
                time.sleep(DELAY_BETWEEN_S)

            # Free MATLAB array wrappers promptly (the runtime leaks them).
            if i % GC_EVERY == 0:
                gc.collect()

        print(f"\nSaved {NUM_MEASUREMENTS} measurements:")
        print(f"  scope : {os.path.abspath(SCOPE_CSV)}")
        print(f"  pm1000: {os.path.abspath(PM_CSV)}")

        if stokes_rows:
            s0 = [r[1] for r in stokes_rows]; s1 = [r[2] for r in stokes_rows]
            s2 = [r[3] for r in stokes_rows]; s3 = [r[4] for r in stokes_rows]
            dop = [r[5] for r in stokes_rows]
            print("\n--- PM1000 statistics across measurements ---")
            for name, vals in [("S0", s0), ("S1", s1), ("S2", s2), ("S3", s3), ("DOP", dop)]:
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                print(f"{name}: mean = {statistics.mean(vals):.4f}, std = {sd:.4f}")

    finally:
        if pm is not None:
            close_pm1000(pm)
        if scope is not None:
            scope.close()
        print("Done.")


if __name__ == "__main__":
    main()
