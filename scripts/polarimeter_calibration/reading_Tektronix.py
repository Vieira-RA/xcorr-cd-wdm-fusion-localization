import sys
import csv
import time
import numpy as np
import vxi11
from vxi11.vxi11 import Vxi11Exception

# ---------------- Configuration ----------------
SCOPE_IP = "198.192.1.1"
CSV_FILENAME = "detector_readings.csv"
CHANNELS = [1, 2, 3, 4]
CH_SCALES = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}   # V/div - adjust per channel if needed
TIMEOUT = 30.0
DELAY_BETWEEN_S = 0.5   # wait between measurements so the scrambler changes SOP

# ---------------- Number of measurements ----------------
if len(sys.argv) > 1:
    num_measurements = int(sys.argv[1])      # e.g. python reading_Tektronix.py 50
else:
    num_measurements = int(input("Enter number of measurements: "))

# ---------------- Connect ----------------
scope = vxi11.Instrument(SCOPE_IP)
scope.timeout = TIMEOUT
print(scope.ask("*IDN?"))

# ---------------- Ensure free-running acquisition ----------------
scope.write("TRIGGER:A:MODE AUTO")
scope.write("ACQUIRE:STOPAFTER RUNSTOP")
scope.write("ACQUIRE:STATE RUN")
time.sleep(1.0)

# ---------------- Enable & configure channels ----------------
for ch in CHANNELS:
    scope.write(f"CH{ch}:STATE ON")
    scope.write(f"CH{ch}:SCALE {CH_SCALES[ch]}")
    scope.write(f"CH{ch}:POSITION 0.0")

# ---------------- Waveform transfer setup ----------------
scope.write("DATA:ENC SRI")
scope.write("DATA:WIDTH 1")
scope.write("WFMOUTPRE:BYT_NR 1")
scope.write("DATA:START 1")
scope.write("DATA:STOP 1e10")
n_pts = int(scope.ask("WFMOUTPRE:NR_PT?"))
scope.write(f"DATA:STOP {n_pts}")
print(f"Record length: {n_pts}")


def read_curve(scope):
    """Read the current channel's CURVE? block and return raw int8 samples."""
    scope.write("CURVE?")
    hdr = scope.read_raw(2)                     # b'#<n>'
    if hdr[:1] != b"#":
        raise RuntimeError(f"Unexpected CURVE? header: {hdr!r}")
    n_digits = int(hdr[1:2])
    byte_count = int(scope.read_raw(n_digits).decode("ascii"))
    data = scope.read_raw(byte_count)

    # Consume an optional trailing newline after the binary block.
    saved = scope.timeout
    scope.timeout = 0.5
    try:
        scope.read_raw(1)
    except Vxi11Exception:
        pass
    finally:
        scope.timeout = saved

    return np.frombuffer(data, dtype=np.int8).astype(float)


# ---------------- Collect N measurements ----------------
# Write the CSV header once (fresh file each run).
with open(CSV_FILENAME, "w", newline="") as f:
    csv.writer(f).writerow(["index", "ch1", "ch2", "ch3", "ch4"])

for i in range(1, num_measurements + 1):
    means = {}
    for ch in CHANNELS:
        scope.write(f"DATA:SOURCE CH{ch}")
        y_mult = float(scope.ask("WFMOUTPRE:YMULT?"))
        y_zero = float(scope.ask("WFMOUTPRE:YZERO?"))
        y_off  = float(scope.ask("WFMOUTPRE:YOFF?"))

        raw = read_curve(scope)
        volts = (raw - y_off) * y_mult + y_zero
        means[ch] = float(np.mean(volts))

    row = [i, means[1], means[2], means[3], means[4]]
    with open(CSV_FILENAME, "a", newline="") as f:
        csv.writer(f).writerow(row)

    print(f"Measurement {i}/{num_measurements}: "
          f"ch1={means[1]:.6e} ch2={means[2]:.6e} "
          f"ch3={means[3]:.6e} ch4={means[4]:.6e}")

    if i < num_measurements:
        time.sleep(DELAY_BETWEEN_S)

print(f"Saved {num_measurements} measurements to {CSV_FILENAME}")

scope.close()
print("Done.")
