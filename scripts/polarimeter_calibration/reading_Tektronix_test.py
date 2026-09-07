import vxi11
import numpy as np
import matplotlib.pyplot as plt
import time

# ------------------------------------------------------------
# 1. Connect to the oscilloscope
# ------------------------------------------------------------
scope_ip = "198.192.1.1"
scope = vxi11.Instrument(scope_ip)

# Set a longer timeout (20 seconds) for large transfers
scope.timeout = 20.0

# Verify connection
print(scope.ask("*IDN?"))

# ------------------------------------------------------------
# 2. Reset and configure
# ------------------------------------------------------------
scope.write("*RST")
scope.write("*CLS")
time.sleep(0.5)

# ------------------------------------------------------------
# 3. Configure Channel 1
# ------------------------------------------------------------
ch = 1
scope.write(f"CH{ch}:COUPLING DC")
scope.write(f"CH{ch}:SCALE 1.0")
scope.write(f"CH{ch}:POSITION 0.0")
scope.write(f"CH{ch}:STATE ON")   # Ensure channel is enabled

# ------------------------------------------------------------
# 4. Set horizontal scale and trigger
# ------------------------------------------------------------
scope.write("HORIZONTAL:SCALE 1e-6")          # 1 µs/div
scope.write("TRIGGER:A:EDGE:SOURCE CH1")
scope.write("TRIGGER:A:EDGE:SLOPE RISE")
scope.write("TRIGGER:A:LEVEL 0.0")
scope.write("TRIGGER:A:MODE AUTO")            # AUTO mode triggers even without signal

# ------------------------------------------------------------
# 5. Set a reasonable record length (e.g., 1000 points for speed)
# ------------------------------------------------------------
scope.write("HORIZONTAL:RECORDLENGTH 1000")

# ------------------------------------------------------------
# 6. Acquire and force a trigger
# ------------------------------------------------------------
scope.write("ACQUIRE:STATE RUN")
time.sleep(0.2)                 # Allow acquisition to start
scope.write("TRIGGER:A:FORCE")  # Force a trigger event
time.sleep(0.2)                 # Let the scope complete the acquisition
scope.write("ACQUIRE:STATE STOP")

# Optional: Check if trigger occurred
trig_state = scope.ask("TRIGGER:STATE?")
print(f"Trigger state: {trig_state}")

# ------------------------------------------------------------
# 7. Request waveform data
# ------------------------------------------------------------
scope.write("DATA:SOURCE CH1")
scope.write("DATA:WIDTH 1")               # ASCII
scope.write("DATA:ENC RIB")               # ASCII format

# Query the curve data
print("Requesting waveform data...")
raw = scope.ask("CURVE?")
print("Data received!")

# Parse the ASCII response
header_len = int(raw[1])
num_points = int(raw[2:2+header_len])
data_str = raw[2+header_len:]
y_vals = np.array([float(v) for v in data_str.split(',')])

# Get time parameters
x_scale  = float(scope.ask("WFMP:SCALE?"))
x_zero   = float(scope.ask("WFMP:ZERO?"))
x_origin = float(scope.ask("WFMP:XORIGIN?"))
x_incr   = float(scope.ask("WFMP:XINCR?"))

x_vals = x_origin + np.arange(num_points) * x_incr

# ------------------------------------------------------------
# 8. Save and plot
# ------------------------------------------------------------
np.savetxt('waveform_data.csv', np.column_stack((x_vals, y_vals)),
           delimiter=',', header='Time(s),Voltage(V)', comments='')

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title(f'Channel {ch} Waveform')
plt.grid(True)
plt.show()

scope.close()
print("Done.")