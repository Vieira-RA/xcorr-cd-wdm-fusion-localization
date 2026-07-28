"""
Toy model: limits of 1D cross-correlation delay estimation.
- Gaussian pulse with controllable bandwidth.
- Exact fractional delay applied via Fourier phase shift.
- Cross-correlation via FFT, with/without parabolic sub-sample refinement.
- Tunable parameters at the top of the script.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# T U N A B L E   P A R A M E T E R S
# ============================================================
BANDWIDTH_HZ = .2e3          # signal bandwidth (Hz) → defines pulse width
FS = 20 * BANDWIDTH_HZ                  # sampling frequency (Hz)
TOTAL_TIME = 80 * (0.3748 / BANDWIDTH_HZ)   # capture enough of the pulse
TRUE_DELAY = 1.0e-6          # true delay in seconds (e.g., 1 µs)
NOISE_STD = 0.0              # standard deviation of additive white Gaussian noise

# ============================================================
# Derived parameters
# ============================================================
sigma_t = 0.3748 / BANDWIDTH_HZ    # Gaussian standard deviation
dt = 1.0 / FS
n_samples = int(TOTAL_TIME / dt) + 1
t = np.linspace(0, TOTAL_TIME, n_samples)
t0 = TOTAL_TIME / 2                 # centre of the pulse

print(f"sigma_t = {sigma_t*1e6:.2f} µs, dt = {dt*1e6:.3f} µs, n_samples = {n_samples}")

# ============================================================
# Generate Gaussian pulse (clean)
# ============================================================
pulse = np.exp(-0.5 * ((t - t0) / sigma_t) ** 2)

# Add noise if requested
if NOISE_STD > 0:
    noise = np.random.normal(0, NOISE_STD, size=n_samples)
    pulse += noise

# ============================================================
# Apply exact fractional delay using Fourier shift
# ============================================================
freq = np.fft.fftfreq(n_samples, d=dt)
phase_shift = np.exp(2j * np.pi * freq * TRUE_DELAY)   # positive for delay
delayed_pulse = -np.fft.ifft(np.fft.fft(pulse) * phase_shift).real

# ============================================================
# Cross-correlation via FFT (same logic as your 2D version)
# ============================================================
N_corr = 2 * n_samples - 1
F_ref = np.fft.fft(pulse, n=N_corr)
F_del = np.fft.fft(delayed_pulse, n=N_corr)
corr_raw = np.fft.ifft(F_ref * np.conj(F_del)).real
corr = np.fft.fftshift(corr_raw)                # zero lag at centre
lags_samples = np.arange(-n_samples + 1, n_samples)   # lags in samples

# Normalize to get correlation coefficient in [-1, 1]
corr_norm = corr / np.sqrt(np.sum(pulse**2) * np.sum(delayed_pulse**2))

# ============================================================
# Peak detection (integer sample)
# ============================================================
peak_idx = np.argmax(np.abs(corr_norm))
integer_lag_samples = lags_samples[peak_idx]
integer_delay = integer_lag_samples * dt
integer_peak_val = np.abs(corr_norm[peak_idx])   # correlation value at integer peak

# ============================================================
# Parabolic sub-sample refinement (if peak not at boundary)
# ============================================================
if 0 < peak_idx < len(corr_norm) - 1:
    y_left = np.abs(corr_norm[peak_idx - 1])
    y_center = np.abs(corr_norm[peak_idx])
    y_right = np.abs(corr_norm[peak_idx + 1])
    denom = y_left - 2*y_center + y_right
    if abs(denom) > 1e-15:
        delta_parab = (y_left - y_right) / (2 * denom)
    else:
        delta_parab = 0.0
else:
    delta_parab = 0.0

parabolic_lag_samples = lags_samples[peak_idx] + delta_parab
parabolic_delay = parabolic_lag_samples * dt

# ============================================================
# Print results
# ============================================================
print(f"\nTrue delay         : {TRUE_DELAY*1e9:.3f} ns")
print(f"Integer-sample lag : {integer_lag_samples} samples -> {integer_delay*1e9:.3f} ns")
print(f"Parabolic lag      : {parabolic_lag_samples:.4f} samples -> {parabolic_delay*1e9:.3f} ns")

# ============================================================
# Plot with fit points, parabola, and peak markers
# ============================================================
lags_us = lags_samples * dt * 1e6   # lags in µs
peak_lag_us = integer_delay * 1e6   # integer delay in µs
parabolic_lag_us = parabolic_delay * 1e6

plt.figure(figsize=(10, 4))
plt.plot(lags_us, np.abs(corr_norm), label='|Correlation|')

# Mark the three points used for parabolic fit
fit_lags = np.array([peak_idx-1, peak_idx, peak_idx+1])
fit_vals = np.abs(corr_norm[fit_lags])
plt.plot(lags_us[fit_lags], fit_vals, 'rs', markersize=8, label='Fit points')

# Draw the fitted parabola and its peak point
if abs(denom) > 1e-15:
    a = denom / 2.0
    b = (y_right - y_left) / 2.0
    c = y_center
    dx = np.linspace(-1.5, 1.5, 100)
    y_parab = a*dx**2 + b*dx + c
    x_lag_samples = lags_samples[peak_idx] + dx
    plt.plot(x_lag_samples * dt * 1e6, y_parab, 'm--', alpha=0.7, label='Fitted parabola')

    # Mark the interpolated peak as a point
    peak_val = a*delta_parab**2 + b*delta_parab + c
    plt.plot(parabolic_lag_us, peak_val, 'm*', markersize=12, label=f'Parabolic peak: {parabolic_lag_us*1e3:.2f} ns')

# Mark the integer peak on the correlation curve
plt.plot(peak_lag_us, integer_peak_val, 'go', markersize=5,
         label=f'Integer peak: {integer_delay*1e9:.2f} ns')

# True delay vertical line
plt.axvline(TRUE_DELAY*1e6, color='k', alpha=0.5,
            label=f'True delay: {TRUE_DELAY*1e9:.2f} ns')

plt.xlim([-2*(0.3748 / BANDWIDTH_HZ)*1e06, 2*(0.3748 / BANDWIDTH_HZ)*1e06])
plt.xlabel('Lag (µs)')
plt.ylabel('Normalized cross-correlation')
plt.title(f'Toy model – B={BANDWIDTH_HZ/1e3:.0f} kHz, fs={FS/1e3:.0f} kHz, delay={TRUE_DELAY*1e9:.0f} ns')
plt.legend()
plt.tight_layout()
plt.savefig('toy_model_crosscorr.png', dpi=150)
plt.close()
print("Plot saved to toy_model_crosscorr.png")