"""
Toy model: cross-correlation delay estimation vs bandwidth.
- Single cross-correlation order: F_ref * conj(F_del)  → expected delay = +TRUE_DELAY
- Produces:
    1) an animation GIF (with integer, parabolic, phase‑slope estimates)
    2) an error‑vs‑bandwidth plot (all three methods)
    3) a delay‑value‑vs‑bandwidth plot (all three methods)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import os

# ============================================================
# T U N A B L E   P A R A M E T E R S
# ============================================================
TRUE_DELAY = 1.0e-6           # absolute delay (seconds)
NOISE_STD = 0.1              # additive noise standard deviation
FS_FACTOR = 15                # fs = FS_FACTOR * bandwidth

# Bandwidth sweep (log scale)
bandwidths = np.logspace(np.log10(200), np.log10(14e6), 60)

# ============================================================
# Helper: process one bandwidth
# ============================================================
def process_bandwidth(BANDWIDTH_HZ):
    """Run cross-correlation, parabolic fit, and phase‑slope estimation."""
    sigma_t = 0.3748 / BANDWIDTH_HZ
    FS = FS_FACTOR * BANDWIDTH_HZ
    dt = 1.0 / FS
    TOTAL_TIME = 4 * 20 * sigma_t
    n_samples = int(TOTAL_TIME / dt) + 1
    t = np.linspace(0, TOTAL_TIME, n_samples)
    t0 = TOTAL_TIME / 2

    # Generate pulse
    pulse = np.exp(-0.5 * ((t - t0) / sigma_t) ** 2)
    if NOISE_STD > 0:
        pulse += np.random.normal(0, NOISE_STD, size=n_samples)

    # Exact fractional delay
    freq = np.fft.fftfreq(n_samples, d=dt)
    phase_shift = np.exp(2j * np.pi * freq * TRUE_DELAY)
    delayed_pulse = np.fft.ifft(np.fft.fft(pulse) * phase_shift).real

    # ---- FFTs (full length for correlation, and raw for phase-slope) ----
    F_pulse = np.fft.fft(pulse)            # original length, for phase-slope
    F_delayed = np.fft.fft(delayed_pulse)

    # Cross-spectrum for phase-slope (positive frequencies)
    Sxy = F_pulse * np.conj(F_delayed)     # same length as freq
    # Only use frequencies where signal is significant (first half + zero)
    pos_mask = freq >= 0
    freq_pos = freq[pos_mask]
    Sxy_pos = Sxy[pos_mask]

    # Threshold by magnitude to avoid noise
    mag = np.abs(Sxy_pos)
    threshold = 0.05 * np.max(mag)
    mask = mag > threshold
    freq_used = freq_pos[mask]
    phase_used = np.unwrap(np.angle(Sxy_pos[mask]))

    # Fit a line: phase = slope * f + intercept
    A = np.vstack([freq_used, np.ones_like(freq_used)]).T
    coeff, _, _, _ = np.linalg.lstsq(A, phase_used, rcond=None)
    slope = coeff[0]
    # Cross-spectrum phase = -2π f τ  ⇒  τ = -slope/(2π)
    phase_slope_delay = -slope / (2 * np.pi)

    # ---- Cross-correlation via FFT (normal order) ----
    N_corr = 2 * n_samples - 1
    F_ref = np.fft.fft(pulse, n=N_corr)
    F_del = np.fft.fft(delayed_pulse, n=N_corr)
    corr_raw = np.fft.ifft(F_ref * np.conj(F_del)).real
    corr = np.fft.fftshift(corr_raw)
    lags_samples = np.arange(-n_samples + 1, n_samples)

    # Normalize
    corr_norm = corr / np.sqrt(np.sum(pulse**2) * np.sum(delayed_pulse**2))

    # Integer peak
    peak_idx = np.argmax(np.abs(corr_norm))
    integer_lag_samples = lags_samples[peak_idx]
    integer_delay = integer_lag_samples * dt

    # Parabolic fit
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

    # Expected delay
    expected_delay = TRUE_DELAY

    # Delays in ns
    int_delay_ns = integer_delay * 1e9
    par_delay_ns = parabolic_delay * 1e9
    ps_delay_ns = phase_slope_delay * 1e9

    # Errors (ns)
    int_err = (integer_delay - expected_delay) * 1e9
    par_err = (parabolic_delay - expected_delay) * 1e9
    ps_err = (phase_slope_delay - expected_delay) * 1e9

    # --- Build frame figure ---
    lags_us = lags_samples * dt * 1e6
    peak_lag_us = integer_delay * 1e6
    parabolic_lag_us = parabolic_delay * 1e6
    ps_lag_us = phase_slope_delay * 1e6
    integer_peak_val = np.abs(corr_norm[peak_idx])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lags_us, np.abs(corr_norm), label='|Correlation|')

    # Fit points
    fit_lags = np.array([peak_idx-1, peak_idx, peak_idx+1])
    fit_vals = np.abs(corr_norm[fit_lags])
    ax.plot(lags_us[fit_lags], fit_vals, 'rs', markersize=8, label='Fit points')

    # Parabola and parabolic peak
    if abs(denom) > 1e-15:
        a = denom / 2.0
        b = (y_right - y_left) / 2.0
        c = y_center
        dx = np.linspace(-1.5, 1.5, 100)
        y_parab = a*dx**2 + b*dx + c
        x_lag_samples = lags_samples[peak_idx] + dx
        ax.plot(x_lag_samples * dt * 1e6, y_parab, 'm--', alpha=0.7, label='Fitted parabola')
        peak_val = a*delta_parab**2 + b*delta_parab + c
        ax.plot(parabolic_lag_us, peak_val, 'm*', markersize=12, label='Parabolic peak')

    # Integer peak marker
    ax.plot(peak_lag_us, integer_peak_val, 'go', markersize=10,
            label=f'Int peak: {int_delay_ns:.2f} ns')

    # Phase‑slope line
    ax.axvline(ps_lag_us, color='c', linestyle='-', linewidth=2, 
               label=f'Phase slope: {ps_delay_ns:.2f} ns')

    # True delay line
    ax.axvline(expected_delay*1e6, color='k', alpha=0.5,
               label=f'True delay: {expected_delay*1e9:.2f} ns')

    ax.set_xlim(-5 * dt * 1e6, 5 * dt * 1e6)
    ax.set_xlabel('Lag (µs)')
    ax.set_ylabel('Normalized cross-correlation')
    ax.set_title(f'B = {BANDWIDTH_HZ/1e3:.1f} kHz, fs = {FS/1e3:.1f} kHz')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, int_err, par_err, ps_err, int_delay_ns, par_delay_ns, ps_delay_ns

# ============================================================
# Main sweep
# ============================================================
integer_errors_ns = []
parabolic_errors_ns = []
phase_slope_errors_ns = []
integer_delays_ns = []
parabolic_delays_ns = []
phase_slope_delays_ns = []
frame_files = []
frames_dir = "frames_tmp"
os.makedirs(frames_dir, exist_ok=True)

for idx, BANDWIDTH_HZ in enumerate(bandwidths):
    fig, int_err, par_err, ps_err, int_del_ns, par_del_ns, ps_del_ns = process_bandwidth(BANDWIDTH_HZ)
    integer_errors_ns.append(int_err)
    parabolic_errors_ns.append(par_err)
    phase_slope_errors_ns.append(ps_err)
    integer_delays_ns.append(int_del_ns)
    parabolic_delays_ns.append(par_del_ns)
    phase_slope_delays_ns.append(ps_del_ns)

    frame_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
    fig.savefig(frame_path, dpi=100)
    plt.close(fig)
    frame_files.append(frame_path)

    dt = 1.0 / (FS_FACTOR * BANDWIDTH_HZ)
    print(f"B={BANDWIDTH_HZ:.0f} Hz, dt={dt*1e6:.3f} µs, int={int_del_ns:.2f}, par={par_del_ns:.2f}, ps={ps_del_ns:.2f} ns")

# Assemble GIF
gif_path = "toy_model_animation.gif"
images = [Image.open(f) for f in frame_files]
images[0].save(gif_path, save_all=True, append_images=images[1:], duration=300, loop=0)
print(f"Animation saved to {gif_path}")

# Clean up frames
for f in frame_files:
    os.remove(f)
os.rmdir(frames_dir)

expected_delay_ns = TRUE_DELAY * 1e9

# ----- Error plot (all three methods) -----
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(bandwidths, integer_errors_ns, 'go-', label='Integer peak error')
ax.semilogx(bandwidths, parabolic_errors_ns, 'm*-', label='Parabolic peak error')
ax.semilogx(bandwidths, phase_slope_errors_ns, 'c.-', label='Phase‑slope error')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('Bandwidth (Hz)')
ax.set_ylabel('Delay error (ns)')
ax.set_title('Delay estimation error vs bandwidth')
ax.legend()
ax.set_ylim([-300, 300])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("toy_model_error_vs_bandwidth.png", dpi=150)
plt.close()
print("Error plot saved to toy_model_error_vs_bandwidth.png")

# ----- Delay value plot (all three methods) -----
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(bandwidths, integer_delays_ns, 'go-', label='Integer estimate')
ax.semilogx(bandwidths, parabolic_delays_ns, 'm*-', label='Parabolic estimate')
ax.semilogx(bandwidths, phase_slope_delays_ns, 'c.-', label='Phase‑slope estimate')
ax.axhline(expected_delay_ns, color='k', linestyle='--',
           label=f'True delay: {expected_delay_ns:.1f} ns')
ax.set_xlabel('Bandwidth (Hz)')
ax.set_ylabel('Estimated delay (ns)')
ax.set_title('Estimated delay vs bandwidth')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("toy_model_delay_vs_bandwidth.png", dpi=150)
plt.close()
print("Delay value plot saved to toy_model_delay_vs_bandwidth.png")

print("\nAll done!")