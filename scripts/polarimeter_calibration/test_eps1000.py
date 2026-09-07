#!/usr/bin/env python
"""
Standalone EPS1000 scrambler test.

Initializes the PM1000 (to observe the SOP) and the EPS1000 (to set it),
disables waveplate rotation, writes two different position sets, and checks
(with the PM1000) whether the state of polarization actually changes.

All EPS1000 calls use matlab.double() (the vendor's own test does this; the
auto-generated samples used uint16, which may be silently ignored by the MEX).
"""

import time
import Python_USB
import matlab

PM_DESCR = "PM1000-100M-XL-FA-N20-D 82"
EPS_DESCR = "EPS1000-20M-CL-S-LU-NN-D"


def dbl(v):
    return matlab.double([v], size=(1, 1))


def read_stokes(pm):
    """Read the PM1000 Stokes vector (S0..S3) in uW."""
    raw = []
    for a in [512+10, 512+11, 512+12, 512+13, 512+14, 512+15, 512+16, 512+17]:
        val, ok = pm.readpm(dbl(a), nargout=2)
        if not ok:
            raise RuntimeError(f"readpm failed at address {a}")
        raw.append(int(val))
    S0 = raw[0] + raw[1] / 65536.0
    S1 = (raw[2] - 32768) + raw[3] / 65536.0
    S2 = (raw[4] - 32768) + raw[5] / 65536.0
    S3 = (raw[6] - 32768) + raw[7] / 65536.0
    return S0, S1, S2, S3


def read_eps(pm, addr):
    """Read one EPS1000 register, return (value, ok)."""
    res, ok = pm.readeps(dbl(addr), nargout=2)
    return int(res), ok


def write_eps(pm, addr, data):
    """Write one EPS1000 register, return ok."""
    return pm.writeeps(dbl(addr), dbl(data))


def main():
    pm = Python_USB.initialize()

    # --- PM1000 (to observe the SOP) ---
    ok, diag, h, i, o = pm.initpm(matlab.uint16([3]), PM_DESCR, nargout=5)
    print("PM1000 init ok:", ok, "diag:", diag)
    if not ok:
        pm.terminate()
        return

    # --- EPS1000 ---
    ok, diag, h, i, o = pm.initeps(matlab.uint16([2]), EPS_DESCR, nargout=5)
    print("EPS1000 init ok:", ok, "diag:", diag)
    if not ok:
        pm.closepm()
        pm.terminate()
        return

    # --- baseline state ---
    print("\n=== baseline EPS1000 state ===")
    for r in range(0, 7):
        print(f"  rotation reg {r}: {read_eps(pm, r)}")
    for r in [23, 24]:
        print(f"  mode/speed reg {r}: {read_eps(pm, r)}")
    for r in range(40, 47):
        print(f"  position reg {r}: {read_eps(pm, r)}")

    # --- disable rotation ---
    print("\n=== disabling rotation (regs 0..6 -> 0) ===")
    for r in range(0, 7):
        print(f"  writeeps({r}, 0) -> ok={write_eps(pm, r, 0)}")

    # --- set positions to 0, measure ---
    print("\n=== set positions to 0 ===")
    for r in range(40, 47):
        print(f"  writeeps({r}, 0) -> ok={write_eps(pm, r, 0)}")
    time.sleep(0.3)
    S_a = read_stokes(pm)
    print(f"  PM1000 (pos=0):    S0={S_a[0]:.3f} S1={S_a[1]:.3f} S2={S_a[2]:.3f} S3={S_a[3]:.3f}")

    # --- set positions to 0x8000 (180 deg apart from 0), measure ---
    print("\n=== set positions to 0x8000 ===")
    for r in range(40, 47):
        print(f"  writeeps({r}, 0x8000) -> ok={write_eps(pm, r, 0x8000)}")
    time.sleep(0.3)
    S_b = read_stokes(pm)
    print(f"  PM1000 (pos=8000): S0={S_b[0]:.3f} S1={S_b[1]:.3f} S2={S_b[2]:.3f} S3={S_b[3]:.3f}")

    # --- read back positions (should be 32768 if writes took) ---
    print("\n=== read back positions ===")
    for r in range(40, 47):
        print(f"  position reg {r}: {read_eps(pm, r)}")

    # --- verdict ---
    d = sum((a - b) ** 2 for a, b in zip(S_a[1:], S_b[1:])) ** 0.5
    print("\n=== RESULT ===")
    print(f"  |dS| in (S1,S2,S3) between the two settings: {d:.4f}")
    if d > 0.05:
        print("  => SOP CHANGED. EPS1000 control is working.")
    else:
        print("  => SOP did NOT change. Inspect the ok flags and read-backs above.")

    pm.closepm()
    pm.closeeps()
    pm.terminate()
    print("Done.")


if __name__ == "__main__":
    main()
