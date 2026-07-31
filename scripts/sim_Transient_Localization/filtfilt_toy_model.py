"""
Toy model: phase‑preserving low‑pass filter with filtfilt.
- Gaussian pulse + delayed copy.
- White noise added.
- Low‑pass zero‑phase filter applied.
- Delay estimates compared: clean, noisy, filtered.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ============================================================
# T U N A B L E   P A R A M E T E R S
# ============================================================
BANDWIDTH_HZ = 500               # signal bandwidth (Hz)
FS = 10e3                        # sampling frequency (Hz)
TRUE_DELAY = 1.0e-3              # 1 ms delay
NOISE_STD = 0.15                 # noise amplitude
LOWPASS_CUTOFF = 3 * BANDWIDTH_HZ   # 1500 Hz – well above signal band
FILTER_ORDER = 4

# ============================================================
# Derived parameters & signals
# ============================================================
sigma_t = 0.3748 / BANDWIDTH_HZ
dt = 1.0 / FS
TOTAL_TIME = 80 * sigma_t
n_samples = int(TOTAL_TIME / dt) + 1
t = np.linspace(0, TOTAL_TIME, n_samples)
t0 = TOTAL_TIME / 2

pulse = np.exp(-0.5 * ((t - t0) / sigma_t) ** 2)
freq = np.fft.fftfreq(n_samples, d=dt)
delayed_pulse = np.fft.ifft(np.fft.fft(pulse) * np.exp(2j * np.pi * freq * TRUE_DELAY)).real

np.random.seed(42)
noise = NOISE_STD * np.random.randn(n_samples)
pulse_noisy = pulse + noise
delayed_noisy = delayed_pulse + noise

# Low‑pass filter
nyq = 0.5 * FS
b, a = butter(FILTER_ORDER, LOWPASS_CUTOFF/nyq, btype='low')
pulse_filt = filtfilt(b, a, pulse_noisy)
delayed_filt = filtfilt(b, a, delayed_noisy)

# ============================================================
# Delay estimation (same cross‑correlation + parabolic fit)
# ============================================================
def estimate_delay(ref, sig):
    N_corr = 2 * len(ref) - 1
    F_ref = np.fft.fft(ref, n=N_corr)
    F_sig = np.fft.fft(sig, n=N_corr)
    corr = np.fft.fftshift(np.fft.ifft(F_ref * np.conj(F_sig)).real)
    lags = np.arange(-len(sig)+1, len(ref)) * dt
    peak_idx = np.argmax(corr)
    if 0 < peak_idx < len(corr)-1:
        yl, yc, yr = corr[peak_idx-1], corr[peak_idx], corr[peak_idx+1]
        denom = yl - 2*yc + yr
        if abs(denom) > 1e-12:
            delta = (yl - yr) / (2 * denom)
            lag_sec = (lags[peak_idx] + delta * dt)
        else:
            lag_sec = lags[peak_idx]
    else:
        lag_sec = lags[peak_idx]
    return lag_sec, lags, corr

delay_clean, lags, corr_clean = estimate_delay(pulse, delayed_pulse)
delay_noisy, _, corr_noisy = estimate_delay(pulse_noisy, delayed_noisy)
delay_filt, _, corr_filt = estimate_delay(pulse_filt, delayed_filt)

print(f"True delay       : {TRUE_DELAY*1e6:.2f} µs")
print(f"Clean            : {delay_clean*1e6:.2f} µs")
print(f"Noisy            : {delay_noisy*1e6:.2f} µs")
print(f"Filtered         : {delay_filt*1e6:.2f} µs")

# ============================================================
# Plots (same as before, but with "Filtered (low‑pass)")
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Time domain
ax1 = axes[0,0]
ax1.plot(t*1e3, pulse_noisy, alpha=0.5, label='Noisy pulse')
ax1.plot(t*1e3, pulse_filt, label='Filtered pulse')
ax1.set_xlabel('Time (ms)')
ax1.set_title('Reference signal')
ax1.legend()

ax2 = axes[0,1]
ax2.plot(t*1e3, delayed_noisy, alpha=0.5, label='Noisy delayed')
ax2.plot(t*1e3, delayed_filt, label='Filtered delayed')
ax2.set_xlabel('Time (ms)')
ax2.set_title('Delayed signal')
ax2.legend()

# Correlation (zoom around peak)
ax3 = axes[1,0]
ax3.plot(lags*1e6, corr_noisy, alpha=0.5, label='Noisy')
ax3.plot(lags*1e6, corr_filt, label='Filtered')
ax3.axvline(delay_noisy*1e6, color='C0', linestyle='--', alpha=0.5)
ax3.axvline(delay_filt*1e6, color='C1', linestyle='--')
ax3.set_xlim((TRUE_DELAY - 10*dt)*1e6, (TRUE_DELAY + 10*dt)*1e6)
ax3.set_xlabel('Lag (µs)')
ax3.set_title('Cross‑correlation (zoom)')
ax3.legend()

# Delay error comparison
ax4 = axes[1,1]
methods = ['Clean', 'Noisy', 'Filtered']
errors = [abs(delay_clean - TRUE_DELAY)*1e6,
          abs(delay_noisy - TRUE_DELAY)*1e6,
          abs(delay_filt - TRUE_DELAY)*1e6]
ax4.bar(methods, errors)
ax4.set_ylabel('Absolute error (µs)')
ax4.set_title('Delay estimation error')

plt.tight_layout()
plt.savefig('toy_model_filtfilt.png', dpi=150)
plt.close()
print("Plot saved to toy_model_filtfilt.png")