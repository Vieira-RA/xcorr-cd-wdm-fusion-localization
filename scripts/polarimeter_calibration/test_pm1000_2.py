#!/usr/bin/env python
"""
Read the Stokes vector (S0, S1, S2, S3) from a Novoptel PM1000 polarimeter
via USB 3.0 using the MATLAB-compiled Python_USB package.
"""

import Python_USB
import matlab

def decode_stokes_from_regs(reg_values):
    """
    Decode the Stokes vector from the raw register values.
    reg_values: list of 8 integers in the order:
                [S0_int, S0_frac, S1_int, S1_frac, S2_int, S2_frac, S3_int, S3_frac]
    Returns: (S0, S1, S2, S3) as floats (µW).
    """
    S0_int, S0_frac, S1_int, S1_frac, S2_int, S2_frac, S3_int, S3_frac = reg_values

    # All fractions are 16-bit (denominator 65536)
    S0 = S0_int + S0_frac / 65536.0

    # S1..S3: integer part offset by 2^15
    def decode_si(int_val, frac_val):
        return (int_val - 32768) + frac_val / 65536.0

    S1 = decode_si(S1_int, S1_frac)
    S2 = decode_si(S2_int, S2_frac)
    S3 = decode_si(S3_int, S3_frac)

    return S0, S1, S2, S3

def main():
    my_pm = Python_USB.initialize()

    sel = matlab.uint16([3])          # USB 3.0
    descr = "PM1000-100M-XL-FA-N20-D 82"  # your device descriptor

    ok, diagnosis, handle, in_descr, out_descr = my_pm.initpm(sel, descr, nargout=5)
    if not ok:
        print("Initialization failed:", diagnosis)
        my_pm.terminate()
        return

    print("PM1000 initialized successfully.")

    # Read the 8 Stokes registers
    regs = {
        'S0_int': 512+10,
        'S0_frac': 512+11,
        'S1_int': 512+12,
        'S1_frac': 512+13,
        'S2_int': 512+14,
        'S2_frac': 512+15,
        'S3_int': 512+16,
        'S3_frac': 512+17,
    }

    raw_values = []
    for name, addr in regs.items():
        val, ok = my_pm.readpm(matlab.double([addr], size=(1,1)), nargout=2)
        if not ok:
            print(f"Failed to read {name}")
            my_pm.closepm()
            my_pm.terminate()
            return
        raw_values.append(int(val))

    # Decode
    S0, S1, S2, S3 = decode_stokes_from_regs(raw_values)

    print(f"Stokes vector in µW:")
    print(f"S0 (power) = {S0:.3f} µW")
    print(f"S1 = {S1:.6f}")
    print(f"S2 = {S2:.6f}")
    print(f"S3 = {S3:.6f}")

    print(f"Normalized Stokes vector in µW:")
    print(f"S1 = {S1 / S0:.6f}")
    print(f"S2 = {S2 / S0:.6f}")
    print(f"S3 = {S3 / S0:.6f}")

    # Compute DOP only if power is above a threshold (e.g., 0.1 µW)
    if S0 > 0.1:
        dop = (S1**2 + S2**2 + S3**2)**0.5 / S0
        print(f"DOP = {dop:.6f}")
    else:
        print("DOP undefined (power too low)")

    my_pm.closepm()
    my_pm.terminate()
    print("Connection closed.")
    print("Done.")

if __name__ == "__main__":
    main()