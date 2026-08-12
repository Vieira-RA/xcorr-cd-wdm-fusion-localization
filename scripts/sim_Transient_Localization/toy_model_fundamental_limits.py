#!/usr/bin/env python3
"""
Ideal 1‑D Gaussian delay estimation – bandwidth & SNR sweep.
Only parabolic cross‑correlation is used.
Standard‑deviation and bias heatmaps (in nanoseconds).
"""
import numpy as np
from tqdm import tqdm
from signal_processing import parabolic_corr_delay_1d
from plotting import plot_std_heatmap, plot_bias_heatmap
from scipy.signal import butter, filtfilt

# ======================== Parameters ========================
TRUE_DELAY_S = 1.0e-6                # 1 µs delay
BANDWIDTHS = np.logspace(np.log10(200), np.log10(2e6), 60)   # 30 bandwidth points
SNRS = np.linspace(30, 90, 60)       # 30 SNR points (dB)
N_REAL = 30                          # noise realisations per cell
FS_FACTOR = 20                       # fs = 30 * bandwidth
USE_FILTER = True                    # low‑pass filter on/off
FMAX_FACTOR = 2.0                    # PHAT mask (not used, but we keep f_max harmless)

# ======================== Storage ========================
shape = (len(BANDWIDTHS), len(SNRS), N_REAL)
est_parabolic = np.full(shape, np.nan)

# ======================== Sweep ========================
for i_bw, bw in enumerate(tqdm(BANDWIDTHS, desc="Bandwidth")):
    sigma_t = 0.3748 / bw
    fs = FS_FACTOR * bw
    dt = 1.0 / fs
    t_end = 160 * sigma_t
    n_samples = int(t_end / dt) + 1
    t = np.linspace(0, t_end, n_samples)
    t0 = t_end / 2

    # clean Gaussian pulse
    pulse = np.exp(-0.5 * ((t - t0) / sigma_t) ** 2)
    peak_power = np.max(pulse**2)

    # exact delayed copy
    freq = np.fft.fftfreq(n_samples, d=dt)
    delayed_pulse = np.fft.ifft(np.fft.fft(pulse) * np.exp(2j * np.pi * freq * TRUE_DELAY_S)).real

    # low‑pass filter design
    nyq = 0.5 * fs
    b_lp, a_lp = butter(4, 3*bw/nyq, btype='low')

    for i_snr, snr_db in enumerate(SNRS):
        for i_real in range(N_REAL):
            rng = np.random.RandomState(i_real)
            snr_lin = 10**(snr_db / 10.0)
            noise_var = peak_power / snr_lin

            ref_noisy = pulse + np.sqrt(noise_var) * rng.randn(n_samples)
            test_noisy = delayed_pulse + np.sqrt(noise_var) * rng.randn(n_samples)

            if USE_FILTER:
                ref_filt = filtfilt(b_lp, a_lp, ref_noisy)
                test_filt = filtfilt(b_lp, a_lp, test_noisy)
            else:
                ref_filt = ref_noisy
                test_filt = test_noisy

            # parabolic peak: ref = earlier, test = later → positive delay
            est_parabolic[i_bw, i_snr, i_real] = parabolic_corr_delay_1d(ref_filt, test_filt, dt)

# ======================== Metrics ========================
true_delay_ns = TRUE_DELAY_S * 1e9
est_ns = est_parabolic * 1e9            # convert all estimates to ns

std_ns = np.nanstd(est_ns, axis=2)       # standard deviation (ns)
bias_ns = np.nanmean(est_ns, axis=2) - true_delay_ns   # bias (ns)

# ======================== Heatmaps ========================
plot_std_heatmap(std_ns, SNRS, BANDWIDTHS,
                 'Parabolic peak std (ns)', 'ideal_std_parabolic.png', vmax=500)
plot_bias_heatmap(bias_ns, SNRS, BANDWIDTHS,
                  'Parabolic peak bias (ns)', 'ideal_bias_parabolic.png',
                  vmin=-100, vmax=100)

print("Heatmaps saved: ideal_std_parabolic.png, ideal_bias_parabolic.png")