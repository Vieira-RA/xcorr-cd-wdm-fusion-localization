#!/usr/bin/env python
"""
Fused acquisition WITH scrambler control:
  - Tektronix DPO7254 oscilloscope  (4 detector channels)
  - Novoptel PM1000 reference polarimeter (Stokes vector)
  - Novoptel EPS1000 polarization scrambler (sets stable SOPs)

For each measurement the EPS1000 is moved to a random but STABLE state of
polarization (SOP), then the scope's 4 channels and the PM1000 Stokes vector are
read from that same SOP. Saves two index-aligned CSVs sharing one timestamp.
"""

import sys
import csv
import gc
import os
import time
import random
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
HORIZONTAL_SCALE = 10e-6     # s/div -> 100 us window (10 divisions)
RECORD_LENGTH    = 10000     # points @ 10 ns
PM_SAMPLE_S = 10e-9          # PM1000 sample period
ATE_MIN, ATE_MAX = 0, 20

# --- PM1000 ---
PM_DESCR = "PM1000-100M-XL-FA-N20-D 82"

# --- EPS1000 scrambler ---
EPS_DESCR = "EPS1000-20M-CL-S-LU-NN-D 332"
EPS_SETTLE_S = 0.3          # settle after position writes (device ~50 ns, USB ~ms)
DELAY_BETWEEN_S = 0       # let the scope's slow RUN/STOP transition settle between measurements

# --- MATLAB Runtime stability (the bridge leaks arrays in long loops) ---
GC_EVERY = 10               # force Python gc every N measurements to free MATLAB arrays

# ================= Number of measurements =================
if len(sys.argv) > 1:
    NUM_MEASUREMENTS = int(sys.argv[1])      # e.g. python fused_acquisition_controlled.py 50
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


# ================= EPS1000 scrambler =================
_EPS_ROT_REGS = list(range(0, 7))       # HWP + QWP0..QWP5 rotation enable/direction
_EPS_POS_REGS = list(range(40, 47))     # HWP + QWP0..QWP5 position index
_ROT_ADDR_ARRAYS = [matlab.double([r], size=(1, 1)) for r in _EPS_ROT_REGS]
_POS_ADDR_ARRAYS = [matlab.double([r], size=(1, 1)) for r in _EPS_POS_REGS]


def init_eps1000(pm):
    """Connect to the EPS1000 scrambler (USB 2.0) and stop all waveplates."""
    ok, diagnosis, handle, in_descr, out_descr = pm.initeps(
        matlab.uint16([2]), EPS_DESCR, nargout=5)
    if not ok:
        raise RuntimeError(f"EPS1000 init failed: {diagnosis}")
    print("EPS1000 initialized successfully.")

    # Stop rotation of all waveplates so the SOP holds still.
    for addr_arr in _ROT_ADDR_ARRAYS:
        pm.writeeps(addr_arr, matlab.double([0], size=(1, 1)))
    print("EPS1000 rotation disabled (stable-SOP mode).")


def set_random_sop(pm):
    """Write random 16-bit positions to all waveplates, then wait to settle."""
    for addr_arr in _POS_ADDR_ARRAYS:
        pm.writeeps(addr_arr, matlab.double([random.randint(0, 0xFFFF)], size=(1, 1)))
    time.sleep(EPS_SETTLE_S)


def close_eps1000(pm):
    try:
        pm.closeeps()
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
        init_eps1000(pm)

        print(f"PM1000: ATE = {ate}, window = {pm_window_s*1e6:.3f} us")
        print(f"Integration match: {(pm_window_s/scope_window_s - 1)*100:+.1f}% "
              f"(scope {scope_window_s*1e6:.3f} us vs pm {pm_window_s*1e6:.3f} us)")

        with open(SCOPE_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["index", "ch1", "ch2", "ch3", "ch4"])
        with open(PM_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["index", "S0_uW", "S1_uW", "S2_uW", "S3_uW", "DOP"])

        stokes_rows = []
        for i in range(1, NUM_MEASUREMENTS + 1):
            # Move the scrambler to a new random, stable SOP.
            set_random_sop(pm)

            # Freeze the scope so all 4 channels share ONE SOP instant.
            scope.write("ACQUIRE:STATE STOP")
            S0, S1, S2, S3 = read_stokes_vector(pm)    # reference SOP (stable now)
            scope_means = read_scope_means(scope)      # 4 detectors at same SOP
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

            # The scope's RUN->STOP transition is slow (~0.15 s); wait so the
            # next STOP freezes a fresh acquisition rather than a stale one.
            time.sleep(DELAY_BETWEEN_S)

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
            close_eps1000(pm)
            close_pm1000(pm)
        if scope is not None:
            scope.close()
        print("Done.")


if __name__ == "__main__":
    main()
