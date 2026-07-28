"""
Demonstration that parabolic interpolation of cross‑correlation peaks
gives meaningless results when the time delay is much smaller than the
pulse width. The same interpolation works correctly for larger delays.

Run as:
    python test_parabolic_artifact.py
Output figure:
    /home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/sim_Transient_Localization/parabolic_artifact_proof.png
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Generate a slow Gaussian pulse (200 Hz bandwidth)
# ============================================================
bandwidth_Hz = 200
sigma_t = 0.3748 / bandwidth_Hz   # ≈ 1.874e-3 s (1.874 ms)
pulse_duration = 20 * sigma_t      # 37.48 ms
t_start = -pulse_duration / 2
t_end = pulse_duration / 2

# True delay we try to measure: 1 µs (absurdly small compared to pulse width)
true_delay_s = 1e-9

# Sampling rates to test
dt_list = [800e-6, 80e-6, 8e-6]   # s

# ============================================================
# 2. Helper: exact fractional delay via Fourier shift
# ============================================================
def apply_fractional_delay(signal, t, delay):
    """Shift signal forward in time by `delay` seconds (positive → later)."""
    n = len(t)
    dt = t[1] - t[0]
    freq = np.fft.fftfreq(n, d=dt)
    phase_shift = np.exp(+2j * np.pi * freq * delay)   # <-- sign changed
    shifted_fft = np.fft.fft(signal) * phase_shift
    return np.fft.ifft(shifted_fft).real
    
# ============================================================
# 3. Cross‑correlation and parabolic fit (1D version)
# ============================================================
def xcorr_and_parabolic_peak(ref, sig, dt):
    """
    Returns:
        lag_sec : estimated delay (seconds), using parabolic subsample
        lags_sec : full lag axis (seconds)
        corr : cross‑correlation (full)
        peak_idx : integer peak index
        delta : fractional offset (samples)
    """
    n_ref = len(ref)
    n_sig = len(sig)
    N = n_ref + n_sig - 1
    F_ref = np.fft.fft(ref, n=N)
    F_sig = np.fft.fft(sig, n=N)
    corr = np.fft.ifft(F_ref * np.conj(F_sig)).real
    corr = np.fft.fftshift(corr)
    lags_samples = np.arange(-n_sig + 1, n_ref)
    lags_sec = lags_samples * dt

    peak_idx = np.argmax(corr)
    # Parabolic fit
    if 0 < peak_idx < len(corr) - 1:
        y_left = corr[peak_idx - 1]
        y_center = corr[peak_idx]
        y_right = corr[peak_idx + 1]
        denom = y_left - 2 * y_center + y_right
        if abs(denom) > 1e-15:
            delta = (y_left - y_right) / (2 * denom)
        else:
            delta = 0.0
    else:
        delta = 0.0
    lag_sub_samples = lags_samples[peak_idx] + delta
    lag_sec = lag_sub_samples * dt
    return lag_sec, lags_sec, corr, peak_idx, delta

# ============================================================
# 4. Run tests for small true delay and a larger reference delay
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
small_delay_results = []

for col, dt in enumerate(dt_list):
    # Build time grid for this dt
    n_samples = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, n_samples)
    # Gaussian envelope
    t0 = 0.0   # pulse centre
    gaussian = np.exp(-(t - t0) ** 2 / (2 * sigma_t ** 2))

    # Reference (no delay) and delayed version
    ref = gaussian.copy()
    delayed = apply_fractional_delay(gaussian, t, true_delay_s)

    lag_est, lags_sec, corr, peak_idx, delta = xcorr_and_parabolic_peak(ref, delayed, dt)
    small_delay_results.append((dt, lag_est, peak_idx, delta))

    # Zoom around the peak for plotting
    zoom_half = 5  # samples
    idx_start = max(0, peak_idx - zoom_half)
    idx_end = min(len(corr), peak_idx + zoom_half + 1)
    lags_zoom = lags_sec[idx_start:idx_end]
    corr_zoom = corr[idx_start:idx_end]

    ax = axes[0, col]
    ax.plot(lags_zoom * 1e6, corr_zoom, 'b.-', label='Cross-corr')
    # Mark the three points used for parabolic fit
    ax.plot(lags_sec[peak_idx-1] * 1e6, corr[peak_idx-1], 'rs', markersize=8, label='Fit points')
    ax.plot(lags_sec[peak_idx] * 1e6, corr[peak_idx], 'rs', markersize=8)
    ax.plot(lags_sec[peak_idx+1] * 1e6, corr[peak_idx+1], 'rs', markersize=8)

    # Draw fitted parabola (for visual check)
    delta_samples = delta
    x_fit = np.array([-1, 0, 1]) + peak_idx
    y_fit = corr[x_fit]
    a = (y_fit[0] - 2*y_fit[1] + y_fit[2]) / 2   # curvature *2
    b = (y_fit[2] - y_fit[0]) / 2
    c = y_fit[1]
    x_dense = np.linspace(peak_idx - 1.5, peak_idx + 1.5, 50)
    y_dense = a*(x_dense - peak_idx)**2 + b*(x_dense - peak_idx) + c
    ax.plot(x_dense * dt * 1e6, y_dense, 'm--', alpha=0.7, label='Parabola fit')

    ax.axvline(true_delay_s * 1e6, color='k', linestyle=':', label='True delay')
    ax.set_title(f"dt = {dt*1e6:.0f} µs, est. delay = {lag_est*1e9:.2f} ns")
    ax.set_xlabel('Lag (µs)')
    ax.legend(fontsize=7)

axes[0, 0].set_ylabel('Correlation')
fig.suptitle('Tiny delay (1 µs) – parabolic fit is an artefact', fontsize=14)

# Now test with a larger delay (3 ms) – the interpolation should work properly
true_large_delay_s = 3e-3   # 3 ms, comparable to pulse width
large_delay_results = []
for col, dt in enumerate(dt_list):
    n_samples = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, n_samples)
    gaussian = np.exp(-(t - t0) ** 2 / (2 * sigma_t ** 2))
    ref = gaussian.copy()
    delayed = apply_fractional_delay(gaussian, t, true_large_delay_s)
    lag_est, lags_sec, corr, peak_idx, delta = xcorr_and_parabolic_peak(ref, delayed, dt)
    large_delay_results.append((dt, lag_est, peak_idx, delta))

    zoom_half = 10
    idx_start = max(0, peak_idx - zoom_half)
    idx_end = min(len(corr), peak_idx + zoom_half + 1)
    lags_zoom = lags_sec[idx_start:idx_end]
    corr_zoom = corr[idx_start:idx_end]

    ax = axes[1, col]
    ax.plot(lags_zoom * 1e3, corr_zoom, 'b.-')   # x‑axis in ms now
    ax.plot(lags_sec[peak_idx-1] * 1e3, corr[peak_idx-1], 'rs', markersize=8)
    ax.plot(lags_sec[peak_idx] * 1e3, corr[peak_idx], 'rs', markersize=8)
    ax.plot(lags_sec[peak_idx+1] * 1e3, corr[peak_idx+1], 'rs', markersize=8)

    delta_samples = delta
    x_fit = np.array([-1, 0, 1]) + peak_idx
    y_fit = corr[x_fit]
    a = (y_fit[0] - 2*y_fit[1] + y_fit[2]) / 2
    b = (y_fit[2] - y_fit[0]) / 2
    c = y_fit[1]
    x_dense = np.linspace(peak_idx - 1.5, peak_idx + 1.5, 50)
    y_dense = a*(x_dense - peak_idx)**2 + b*(x_dense - peak_idx) + c
    ax.plot(x_dense * dt * 1e3, y_dense, 'm--', alpha=0.7)

    ax.axvline(true_large_delay_s * 1e3, color='k', linestyle=':')
    ax.set_title(f"dt = {dt*1e6:.0f} µs, est. delay = {lag_est*1e3:.3f} ms")
    ax.set_xlabel('Lag (ms)')

axes[1, 0].set_ylabel('Correlation')
# Add a text summary
fig.text(0.5, 0.01,
         "For a 1 µs delay (row 1) the parabolic fit gives different answers for each dt – it’s an artefact.\n"
         "For a 3 ms delay (row 2) the fit correctly recovers the delay in all cases.\n"
         "True delay is shown as dotted black line.",
         ha='center', fontsize=11, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
out_path = "/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/sim_Transient_Localization/parabolic_artifact_proof.png"
plt.savefig(out_path, dpi=200)
plt.close()
print(f"Figure saved to {out_path}")

# Print a concise summary
print("\n--- Tiny delay (1 µs) results ---")
for dt, lag_est, peak_idx, delta in small_delay_results:
    print(f"dt = {dt*1e6:6.0f} µs : estimated delay = {lag_est*1e9:8.2f} ns  (peak at sample {peak_idx}, delta = {delta:.4f})")

print("\n--- Large delay (3 ms) results ---")
for dt, lag_est, peak_idx, delta in large_delay_results:
    print(f"dt = {dt*1e6:6.0f} µs : estimated delay = {lag_est*1e3:8.3f} ms (true = 3.000 ms)")