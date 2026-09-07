#!/usr/bin/env python
"""
Probe the PM1000 high-speed memory read (readhspm) to find the right start
address and confirm it returns a sensible Stokes trajectory.
"""
import Python_USB
import matlab
import numpy as np

PM_DESCR = "PM1000-100M-XL-FA-N20-D 82"


def main():
    pm = Python_USB.initialize()
    ok, diag, h, i, o = pm.initpm(matlab.uint16([3]), PM_DESCR, nargout=5)
    print("initpm ok:", ok, "diag:", diag)
    if not ok:
        pm.terminate()
        return

    # set ATE = 0 (raw 10 ns samples, no averaging)
    ok = pm.writepm(matlab.double([512+1], size=(1, 1)), matlab.uint16([0], size=(1, 1)))
    print("set ATE=0 ->", ok)

    for addr in [0, 64, 128, 256, 512, 1024]:
        try:
            d1, d2, d3, d4, ok = pm.readhspm(
                matlab.uint32([addr], size=(1, 1)),
                matlab.uint32([32], size=(1, 1)),
                nargout=5)
            a1 = np.array(d1).flatten()
            a2 = np.array(d2).flatten()
            a3 = np.array(d3).flatten()
            a4 = np.array(d4).flatten()
            print(f"\n--- readhspm(addr={addr}, num=32) ok={ok} ---")
            print("dout1[:8]:", a1[:8].tolist())
            print("dout2[:8]:", a2[:8].tolist())
            print("dout3[:8]:", a3[:8].tolist())
            print("dout4[:8]:", a4[:8].tolist())
            print("ranges -> d1:", a1.min(), a1.max(), " d2:", a2.min(), a2.max(),
                  " d3:", a3.min(), a3.max(), " d4:", a4.min(), a4.max())
        except Exception as e:
            print(f"readhspm(addr={addr}) FAILED:", e)

    pm.closepm()
    pm.terminate()
    print("\nDone.")


if __name__ == "__main__":
    main()
