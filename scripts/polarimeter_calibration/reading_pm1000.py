#!/usr/bin/env python
"""
PM1000 Stokes vector reader using hardware averaging (ATE).
Saves averaged readings to CSV.
"""

import Python_USB
import matlab
import csv
import time
import statistics
import os
from datetime import datetime

def decode_stokes_from_regs(reg_values):
    """Decode Stokes vector from 8 raw register values (ints)."""
    S0_int, S0_frac, S1_int, S1_frac, S2_int, S2_frac, S3_int, S3_frac = reg_values
    S0 = S0_int + S0_frac / 65536.0
    def decode_si(int_val, frac_val):
        return (int_val - 32768) + frac_val / 65536.0
    S1 = decode_si(S1_int, S1_frac)
    S2 = decode_si(S2_int, S2_frac)
    S3 = decode_si(S3_int, S3_frac)
    return S0, S1, S2, S3

def read_stokes_vector(pm_handle):
    """Read the 8 registers and return decoded Stokes vector."""
    regs = [
        (512+10, 'S0_int'),
        (512+11, 'S0_frac'),
        (512+12, 'S1_int'),
        (512+13, 'S1_frac'),
        (512+14, 'S2_int'),
        (512+15, 'S2_frac'),
        (512+16, 'S3_int'),
        (512+17, 'S3_frac'),
    ]
    raw = []
    for addr, name in regs:
        val, ok = pm_handle.readpm(matlab.double([addr], size=(1,1)), nargout=2)
        if not ok:
            raise RuntimeError(f"Failed to read {name} at address {addr}")
        raw.append(int(val))
    return decode_stokes_from_regs(raw)

def init_pm1000(descr):
    """Initialize PM1000 and return handle."""
    pm = Python_USB.initialize()
    sel = matlab.uint16([3])  # USB 3.0
    ok, diagnosis, handle, in_descr, out_descr = pm.initpm(sel, descr, nargout=5)
    if not ok:
        pm.terminate()
        raise RuntimeError(f"Init failed: {diagnosis}")
    return pm

def main():
    # --- User parameters ---
    ATE = 7                    # Averaging time exponent: 2^ATE samples averaged
    N_BLOCKS = 50              # Number of averaged blocks (rows in CSV)
    DELAY_BETWEEN_BLOCKS = 0.1 # seconds
    DESCR = "PM1000-100M-XL-FA-N20-D 82"
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CSV_FILENAME = f"stokes_ate{ATE}_{timestamp}.csv"

    avg_samples = 2 ** ATE
    print(f"ATE = {ATE} → hardware averages {avg_samples} samples per read.")
    print(f"Saving to: {CSV_FILENAME}")

    # Initialize PM1000
    my_pm = init_pm1000(DESCR)
    print("PM1000 initialized successfully.")

    # Set ATE via register 512+1
    ate_addr = matlab.double([512+1], size=(1,1))
    ok = my_pm.writepm(ate_addr, matlab.uint16([ATE], size=(1,1)))
    if not ok:
        print("Failed to set ATE")
        my_pm.terminate()
        return
    print(f"ATE set to {ATE}.")

    # Disable external triggering (safe default)
    trig_addr = matlab.double([512+92], size=(1,1))
    my_pm.writepm(trig_addr, matlab.uint16([0], size=(1,1)))

    averaged_blocks = []

    print(f"\nStarting measurement: {N_BLOCKS} blocks...")
    for block_idx in range(1, N_BLOCKS + 1):
        try:
            S0, S1, S2, S3 = read_stokes_vector(my_pm)
            dop = (S1**2 + S2**2 + S3**2)**0.5 / S0 if S0 > 0 else 0
            averaged_blocks.append((block_idx, S0, S1, S2, S3, dop))
            print(f"Block {block_idx}/{N_BLOCKS}: S0 = {S0:.3f} µW, DOP = {dop:.4f}")
            time.sleep(DELAY_BETWEEN_BLOCKS)
        except Exception as e:
            print(f"Error in block {block_idx}: {e}")
            break

    # Save to CSV
    with open(CSV_FILENAME, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "S0_uW", "S1_uW", "S2_uW", "S3_uW", "DOP"])
        writer.writerows(averaged_blocks)

    print(f"\nResults saved to {os.path.abspath(CSV_FILENAME)}")

    # Statistics
    if averaged_blocks:
        s0_vals = [row[1] for row in averaged_blocks]
        s1_vals = [row[2] for row in averaged_blocks]
        s2_vals = [row[3] for row in averaged_blocks]
        s3_vals = [row[4] for row in averaged_blocks]
        dop_vals = [row[5] for row in averaged_blocks]
        
        print("\n--- Statistics across blocks ---")
        print(f"S0: mean = {statistics.mean(s0_vals):.3f} µW, std = {statistics.stdev(s0_vals):.3f}")
        print(f"S1: mean = {statistics.mean(s1_vals):.3f} µW, std = {statistics.stdev(s1_vals):.3f}")
        print(f"S2: mean = {statistics.mean(s2_vals):.3f} µW, std = {statistics.stdev(s2_vals):.3f}")
        print(f"S3: mean = {statistics.mean(s3_vals):.3f} µW, std = {statistics.stdev(s3_vals):.3f}")
        print(f"DOP: mean = {statistics.mean(dop_vals):.4f}, std = {statistics.stdev(dop_vals):.4f}")

    # Clean up
    my_pm.closepm()
    my_pm.terminate()
    print("Connection closed. Done.")

if __name__ == "__main__":
    main()