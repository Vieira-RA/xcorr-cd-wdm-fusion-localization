#!/usr/bin/env python3
"""Sweep the laser wavelength at a FROZEN input SOP and record both polarimeters.

This is the spectral-SOP-evolution experiment: the scrambler is parked on one
stable state and only lambda changes, so the trajectory on the Poincare sphere is
a pure function of wavelength. A polarimeter placed after a birefringent path
traces a circle about the PMD axis, one revolution per
    Delta_lambda = lambda^2 / (c * DGD)
which for the ~1.6 ps measured here is ~5 nm.

Both instruments are recorded at every point:
  * PM1000      -> reference Stokes vector (uW), the "truth" trajectory;
  * scope CH1-4 -> detector voltages, which a per-wavelength C turns into the
                   FBG polarimeter's own trajectory.

Runs in the pm1000_env interpreter (PyVISA for the scope, Python_USB for the
PM1000); the laser is tuned by calling ``laser_set.py`` in the interpreter that
has pyserial. Nothing in either environment is modified.

Usage
-----
    python sweep_trajectory.py --start 1535.0 --stop 1535.5 --step 0.02 \
        --out trajectory_1535.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scope_io import open_scope, ScopeError  # noqa: E402

CHANNELS = (1, 2, 3, 4)
HORIZONTAL_SCALE = 10e-6
RECORD_LENGTH = 10000
PM_DESCR = "PM1000-100M-XL-FA-N20-D 82"
EPS_DESCR = "EPS1000-20M-CL-S-LU-NN-D 332"
PM_SAMPLE_S = 10e-9
BASE = 512
STOKES_REGS = list(range(10, 18))


def read_curve(scope) -> bytes:
    return scope.read_block("CURVE?")


def read_scope_means(scope) -> Dict[int, float]:
    means = {}
    for channel in CHANNELS:
        scope.write("DATA:SOURCE CH{}".format(channel))
        y_mult = float(scope.ask("WFMOUTPRE:YMULT?"))
        y_zero = float(scope.ask("WFMOUTPRE:YZERO?"))
        y_off = float(scope.ask("WFMOUTPRE:YOFF?"))
        codes = np.frombuffer(read_curve(scope), dtype=np.int8).astype(float)
        means[channel] = float(np.mean((codes - y_off) * y_mult + y_zero))
    return means


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=float, default=1535.0)
    parser.add_argument("--stop", type=float, default=1535.5)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--out", default="trajectory.csv")
    parser.add_argument("--scope-ip", default=os.environ.get("PM_SCOPE_IP", "169.254.102.200"))
    parser.add_argument("--scope-transport", default=os.environ.get("PM_SCOPE_TRANSPORT", "pyvisa"))
    parser.add_argument("--com-port", default="COM4")
    parser.add_argument("--laser-python",
                        default=r"C:/Users/240404662/PhD_Code/phd_env/Scripts/python.exe")
    parser.add_argument("--pm-wavelength-register", type=int, default=69)
    parser.add_argument("--ate", type=int, default=13)
    parser.add_argument("--averages", type=int, default=6,
                        help="PM1000 reads averaged per point (the scope already averages 10k samples)")
    parser.add_argument("--ch-scales", default="0.02,0.02,0.02,0.02")
    parser.add_argument("--xincr", type=float, default=10e-9)
    args = parser.parse_args(argv)

    wavelengths = np.arange(args.start, args.stop + 1e-9, args.step)
    scales = [float(value) for value in args.ch_scales.split(",")]
    print("Sweeping {} wavelengths from {:.3f} to {:.3f} nm".format(
        len(wavelengths), wavelengths[0], wavelengths[-1]))

    os.environ["PM_SCOPE_IP"] = args.scope_ip
    os.environ["PM_SCOPE_TRANSPORT"] = args.scope_transport

    scope = open_scope(args.scope_ip, transport=args.scope_transport, timeout=60.0)
    print("scope:", scope.ask("*IDN?").strip())
    for index, channel in enumerate(CHANNELS):
        scope.write("CH{}:STATE ON".format(channel))
        scope.write("CH{}:SCALE {}".format(channel, scales[index]))
        scope.write("CH{}:POSITION 0.0".format(channel))
    scope.write("HORIZONTAL:SCALE {}".format(HORIZONTAL_SCALE))
    scope.write("HORIZONTAL:RECORDLENGTH {}".format(RECORD_LENGTH))
    scope.write("DATA:ENC SRI")
    scope.write("DATA:WIDTH 1")
    scope.write("WFMOUTPRE:BYT_NR 1")
    scope.write("DATA:START 1")
    scope.write("DATA:STOP 1e10")
    points = int(scope.ask("WFMOUTPRE:NR_PT?"))
    scope.write("DATA:STOP {}".format(points))
    scope.write("TRIGGER:A:MODE AUTO")
    scope.write("ACQUIRE:STOPAFTER RUNSTOP")
    scope.write("ACQUIRE:STATE RUN")
    time.sleep(1.0)

    import Python_USB
    import matlab

    pm = Python_USB.initialize()
    ok, diagnosis, *_ = pm.initpm(matlab.uint16([3]), PM_DESCR, nargout=5)
    if not ok:
        raise SystemExit("PM1000 init failed: {}".format(diagnosis))
    ok, diagnosis, *_ = pm.initeps(matlab.uint16([2]), EPS_DESCR, nargout=5)
    if not ok:
        raise SystemExit("EPS1000 init failed: {}".format(diagnosis))
    print("PM1000 + EPS1000 initialised")

    def _read_register(offset: int) -> int:
        # The USB bridge occasionally drops a read; retry a few times with the
        # small inter-command gap the vendor MATLAB code also enforces.
        for attempt in range(4):
            value, good = pm.readpm(matlab.double([BASE + offset], size=(1, 1)), nargout=2)
            if good:
                time.sleep(0.003)
                return int(value)
            time.sleep(0.02 * (attempt + 1))
        raise RuntimeError("readpm failed at {} after 4 attempts".format(offset))

    def read_stokes() -> List[float]:
        raw = [_read_register(offset) for offset in STOKES_REGS]
        s0 = raw[0] + raw[1] / 65536.0
        return [s0] + [(raw[2 * k] - 32768) + raw[2 * k + 1] / 65536.0 for k in (1, 2, 3)]

    agent = subprocess.Popen(
        [args.laser_python, str(HERE / "laser_agent.py"), args.com_port],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=str(HERE),
    )
    ready = json.loads(agent.stdout.readline() or "{}")
    if not ready.get("ready"):
        raise SystemExit("laser agent did not start: {}".format(ready))
    print("laser agent ready:", ready.get("idn"))

    def tune(wavelength_nm: float) -> None:
        print("{:.4f}".format(wavelength_nm), file=agent.stdin, flush=True)
        answer = json.loads(agent.stdout.readline() or "{}")
        if "error" in answer:
            raise SystemExit("laser tuning failed at {:.3f} nm: {}".format(
                wavelength_nm, answer["error"]))
        if abs(answer.get("wavelength_nm", 0.0) - wavelength_nm) > 0.005:
            raise SystemExit("laser settled at {!r} for a request of {:.3f} nm".format(
                answer.get("wavelength_nm"), wavelength_nm))

    rows: List[Dict[str, Any]] = []
    try:
        pm.writepm(matlab.double([BASE + 1], size=(1, 1)),
                   matlab.uint16([args.ate], size=(1, 1)))
        pm.writepm(matlab.double([BASE + 92], size=(1, 1)),
                   matlab.uint16([0], size=(1, 1)))
        # freeze the scrambler on one stable SOP for the whole sweep
        for reg in range(0, 7):
            pm.writeeps(matlab.double([reg], size=(1, 1)), matlab.double([0], size=(1, 1)))
        for reg in range(40, 47):
            pm.writeeps(matlab.double([reg], size=(1, 1)),
                        matlab.double([random.randint(0, 0xFFFF)], size=(1, 1)))
        print("scrambler parked on a fixed SOP")

        for index, wavelength in enumerate(wavelengths, start=1):
            tune(wavelength)
            if args.pm_wavelength_register:
                raw = int(round(299792.458 / float(wavelength) * 100.0))
                pm.writepm(matlab.double([BASE + args.pm_wavelength_register], size=(1, 1)),
                           matlab.double([raw], size=(1, 1)))
            time.sleep(0.2)
            scope.write("ACQUIRE:STATE STOP")
            samples = np.array([read_stokes() for _ in range(max(1, args.averages))])
            s0 = float(samples[:, 0].mean())
            vectors = samples[:, 1:]
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            unit = vectors / norms
            direction = unit.mean(axis=0)
            direction /= np.linalg.norm(direction)
            stokes = [s0] + list(direction * float(norms.mean()))
            means = read_scope_means(scope)
            scope.write("ACQUIRE:STATE RUN")
            time.sleep(0.15)          # let the acquisition restart before the next STOP

            row = {"wavelength_nm": float(wavelength), "S0": stokes[0],
                   "S1": stokes[1], "S2": stokes[2], "S3": stokes[3]}
            row.update({"ch{}".format(ch): means[ch] for ch in CHANNELS})
            row["dop"] = float(np.linalg.norm(stokes[1:]) / stokes[0]) if stokes[0] else 0.0
            rows.append(row)
            print("[{:3d}/{}] {:.3f} nm  S0={:7.2f} uW  DOP={:.4f}  ch={}".format(
                index, len(wavelengths), wavelength, row["S0"], row["dop"],
                " ".join("{:.1f}".format(means[ch] * 1e3) for ch in CHANNELS)))
    finally:
        try:
            print("q", file=agent.stdin, flush=True)
            agent.wait(timeout=5)
        except Exception:
            agent.kill()
        try:
            pm.closeeps()
        except Exception:
            pass
        try:
            pm.closepm()
        except Exception:
            pass
        try:
            pm.terminate()
        except Exception:
            pass
        try:
            scope.close()
        except Exception:
            pass

    out = Path(args.out)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {} ({} points)".format(out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
