#!/usr/bin/env python3
"""
Multi‑fibre, multi‑count post‑processing script for CD localisation sweep (parallelised).
Supports averaging over independent fibre PMD realisations.
Saves full 4‑D per‑fibre arrays for later fibre‑count analysis.
"""
import time
import os
import shutil
import datetime
import yaml
import numpy as np
from tqdm import tqdm
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Shared library imports
from quaternion import (
    jones_to_quaternion,
    regularize_signs,
    quaternion_to_rotation_vector,
    quaternion_rotate_stokes,
)
from signal_processing import (
    phase_slope_delay_2d,
    gcc_phat_2d,
    integer_corr_delay_2d,
    parabolic_corr_delay_2d,
    phase_slope_delay_1d,
    gcc_phat_1d,
    integer_corr_delay_1d,
    parabolic_corr_delay_1d,
)
from estimation import iqr_filter, adaptive_weighted_mean
from plotting import plot_std_heatmap, plot_bias_heatmap
from chromatic_dispersion import integrated_dispersion
from scipy.signal import butter, filtfilt

# Global publication‑ready style
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# ====================== load configuration ======================
with open("sweep_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# ---- create run folder and copy config ----
run_cfg = cfg.get("run", {})
run_name = run_cfg.get("name", "run")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder = f"results/{run_name}_{timestamp}"
os.makedirs(run_folder, exist_ok=True)
shutil.copy2("sweep_config.yaml", os.path.join(run_folder, "config_used.yaml"))
print(f"Output folder: {run_folder}")

# unpack configuration
fib = cfg["fibre"]
L_km = float(fib["L_km"]); L = L_km * 1e3
L_F = float(fib["L_F"]); D_pmd = float(fib["D_pmd"])
lambda0_nm = float(fib["lambda0_nm"]); lambda0 = lambda0_nm * 1e-9
c = 299792458.0; omega0 = 2 * np.pi * c / lambda0

A_pulse = float(cfg["transient"]["A_pulse"])
fs_factor = float(cfg["time"]["fs_factor"]); t_end_factor = float(cfg["time"]["t_end_factor"])

event_distance_km = float(cfg["event"]["distance_km"])

sweep_cfg = cfg["sweep"]
use_filter = cfg.get('filter', {}).get('enabled', True)

BANDWIDTHS = np.logspace(np.log10(float(sweep_cfg["bandwidths"]["start_Hz"])),
                         np.log10(float(sweep_cfg["bandwidths"]["stop_Hz"])),
                         int(sweep_cfg["bandwidths"]["num"]))
SNRS = np.linspace(float(sweep_cfg["snrs_dB"]["start"]),
                   float(sweep_cfg["snrs_dB"]["stop"]),
                   int(sweep_cfg["snrs_dB"]["num"]))
N_REAL = int(sweep_cfg["n_realisations"])
n_fibres = int(sweep_cfg.get("n_fibres", 1))

phat_cfg = cfg["phat"]
fmax_factor = float(phat_cfg["fmax_factor"]); mag_threshold = float(phat_cfg["mag_threshold"])
noise_base_seed = int(cfg["noise"]["base_seed"])
output_dir = cfg["output"]["jones_dir"]

# ====================== load metadata ======================
meta = np.load(os.path.join(output_dir, "metadata.npz"), allow_pickle=True)
full_wavelengths = meta["wavelengths_nm"]
full_delays = meta["channel_delays"]
N_full = len(full_wavelengths)
print(f"Full grid contains {N_full} channels.")

analysis_cfg = cfg.get("analysis", {})
channel_counts = analysis_cfg.get("channel_counts", [N_full])
if isinstance(channel_counts, int):
    channel_counts = [channel_counts]
selection = analysis_cfg.get("channel_selection", "alternating").lower()

# ====================== Parallel worker (unchanged) ======================
def process_one_realisation(bw, f_max, dt, b_lp, a_lp, stokes_clean, phi_mag_clean,
                            peak_power_stokes, peak_power_phi, integrals_abs,
                            pair_indices, n_pairs, ctrl_idx, channel_count,
                            i_snr, snr_db, i_real):
    rng = np.random.RandomState(noise_base_seed + i_real)
    n_channels = stokes_clean.shape[0]
    n_samples = stokes_clean.shape[1]

    # ---- Stokes (correct fair noise) ----
    snr_lin = 10**(snr_db / 10.0)
    noise_var_stokes = peak_power_stokes / (2.0 * snr_lin)   # factor 2 for two components
    stokes_noisy = np.zeros_like(stokes_clean)
    for ch in range(n_channels):
        noise = np.sqrt(noise_var_stokes[ch]) * rng.randn(n_samples, 3)
        stokes_noisy[ch] = stokes_clean[ch] + noise
        if use_filter:
            for comp in range(3):
                stokes_noisy[ch, :, comp] = filtfilt(b_lp, a_lp,
                                                    stokes_noisy[ch, :, comp])
    from rotations import rotate_centroid_to_north_pole
    stokes_rot = np.zeros_like(stokes_noisy)
    for ch in range(n_channels):
        S_rot, _ = rotate_centroid_to_north_pole(stokes_noisy[ch])
        stokes_rot[ch] = S_rot

    # ---- Rotation‑vector (already fair: peak power from AC) ----
    noise_var_phi = peak_power_phi / snr_lin
    phi_noisy = phi_mag_clean + np.sqrt(noise_var_phi[:, None]) * rng.randn(*phi_mag_clean.shape)
    phi_filt = np.zeros_like(phi_noisy)
    for ch in range(n_channels):
        if use_filter:
            phi_filt[ch] = filtfilt(b_lp, a_lp, phi_noisy[ch])
        else:
            phi_filt[ch] = phi_noisy[ch]
        phi_filt[ch] -= np.mean(phi_filt[ch])

    # ---- Delay estimation ----
    delays_2d = {k: np.zeros(n_pairs) for k in ['phase','phat','int','par']}
    delays_rv = {k: np.zeros(n_pairs) for k in ['phase','phat','int','par']}

    for idx, (i, j) in enumerate(pair_indices):
        sig_i_2d = stokes_rot[i, :, :2]; sig_j_2d = stokes_rot[j, :, :2]
        delays_2d['phase'][idx] = phase_slope_delay_2d(sig_j_2d, sig_i_2d, dt, True)
        delays_2d['phat'][idx]  = gcc_phat_2d(sig_j_2d, sig_i_2d, dt, f_max, mag_threshold)
        delays_2d['int'][idx]   = integer_corr_delay_2d(sig_j_2d, sig_i_2d, dt)
        delays_2d['par'][idx]   = parabolic_corr_delay_2d(sig_j_2d, sig_i_2d, dt)

        sig_i_1d = phi_filt[i]; sig_j_1d = phi_filt[j]
        delays_rv['phase'][idx] = phase_slope_delay_1d(sig_j_1d, sig_i_1d, dt, True)
        delays_rv['phat'][idx]  = gcc_phat_1d(sig_j_1d, sig_i_1d, dt, f_max, mag_threshold)
        delays_rv['int'][idx]   = integer_corr_delay_1d(sig_j_1d, sig_i_1d, dt)
        delays_rv['par'][idx]   = parabolic_corr_delay_1d(sig_j_1d, sig_i_1d, dt)

    dists_2d = {k: delays_2d[k] * 1e12 / integrals_abs for k in delays_2d}
    dists_rv = {k: delays_rv[k] * 1e12 / integrals_abs for k in delays_rv}

    # Control pair raw distances
    ctrl_2d = {}
    ctrl_rv = {}
    for k in ['phase','phat','int','par']:
        ctrl_2d[k] = dists_2d[k][ctrl_idx]
        ctrl_rv[k] = dists_rv[k][ctrl_idx]

    # Aggregated estimates (after IQR filter + weighted mean)
    est_2d = {}
    est_rv = {}
    for key in ['phase','phat','int','par']:
        mask_2d = iqr_filter(dists_2d[key])
        est_2d[key] = adaptive_weighted_mean(dists_2d[key], mask_2d, integrals_abs)[0]
        mask_rv = iqr_filter(dists_rv[key])
        est_rv[key] = adaptive_weighted_mean(dists_rv[key], mask_rv, integrals_abs)[0]

    return {
        'est_2d': est_2d,
        'est_rv': est_rv,
        'ctrl_2d': ctrl_2d,
        'ctrl_rv': ctrl_rv,
    }

# ====================== Main loop over channel counts ======================
for channel_count in channel_counts:
    print(f"\n{'='*60}\nRunning analysis for channel_count = {channel_count}\n{'='*60}")

    # ---------- Channel selection ----------
    if selection == "alternating":
        selected = []
        left, right = 0, N_full - 1
        while len(selected) < channel_count and left <= right:
            selected.append(right)
            if len(selected) < channel_count and left < right:
                selected.append(left)
            right -= 1; left += 1
        selected = sorted(selected[:channel_count])
    elif selection == "extremes":
        half = channel_count // 2
        selected = list(range(half)) + list(range(N_full - (channel_count - half), N_full))
        selected = sorted(selected)
    elif selection == "uniform":
        step = max(1, N_full // channel_count)
        selected = list(range(0, N_full, step))[:channel_count]
    elif selection == "first_n":
        selected = list(range(channel_count))
    else:
        raise ValueError(f"Unknown channel_selection: {selection}")

    channel_indices = np.array(selected)
    wavelengths_nm = full_wavelengths[channel_indices]
    channel_delays = full_delays[channel_indices]
    n_channels = len(channel_indices)

    pair_indices = list(itertools.combinations(range(n_channels), 2))
    n_pairs = len(pair_indices)
    integrals_abs = np.array([
        abs(integrated_dispersion(wavelengths_nm[i], wavelengths_nm[j]))
        for i, j in pair_indices
    ])
    ctrl_pair = (0, n_channels - 1)
    ctrl_idx = pair_indices.index(ctrl_pair)

    # ---------- 4D storage arrays (fibre, bandwidth, SNR, realisation) ----------
    shape_4d = (n_fibres, len(BANDWIDTHS), len(SNRS), N_REAL)
    phase_est_2d = np.full(shape_4d, np.nan);   phat_est_2d  = np.full(shape_4d, np.nan)
    int_est_2d   = np.full(shape_4d, np.nan);   par_est_2d   = np.full(shape_4d, np.nan)
    phase_est_rv = np.full(shape_4d, np.nan);   phat_est_rv  = np.full(shape_4d, np.nan)
    int_est_rv   = np.full(shape_4d, np.nan);   par_est_rv   = np.full(shape_4d, np.nan)

    ctrl_2d = { 'phase': np.full(shape_4d, np.nan), 'phat': np.full(shape_4d, np.nan),
                'int': np.full(shape_4d, np.nan),   'par': np.full(shape_4d, np.nan) }
    ctrl_rv = { 'phase': np.full(shape_4d, np.nan), 'phat': np.full(shape_4d, np.nan),
                'int': np.full(shape_4d, np.nan),   'par': np.full(shape_4d, np.nan) }

    bw_times = []

    # ---------- Fibre loop ----------
    for i_fibre in range(n_fibres):
        print(f"\n--- Fibre {i_fibre+1}/{n_fibres} ---")
        fibre_dir = os.path.join(output_dir, f"fibre_{i_fibre}")
        if not os.path.exists(fibre_dir):
            print(f"ERROR: Fibre directory {fibre_dir} not found! Skipping.")
            continue

        # ---------- Main sweep over bandwidths (same as before) ----------
        total_bw = len(BANDWIDTHS)
        for i_bw, bw in enumerate(tqdm(BANDWIDTHS, desc=f"BW ({channel_count}ch, fibre {i_fibre})")):
            bw_start = time.time()
            jones_file = os.path.join(fibre_dir, f"jones_bw_{bw:.0f}Hz.npz")
            data = np.load(jones_file, allow_pickle=True)
            U_all = data["U_all"]
            U_all = U_all[channel_indices, ...]
            dt = data["dt"].item()
            n_samples = data["n_samples"].item()

            s_in = np.array([1.0, 0.0, 0.0])

            # Clean Stokes
            stokes_clean = np.zeros((n_channels, n_samples, 3))
            for ch in range(n_channels):
                Q = np.zeros((n_samples, 4))
                for t in range(n_samples): Q[t] = jones_to_quaternion(U_all[ch, t])
                Q = regularize_signs(Q)
                stokes_clean[ch] = quaternion_rotate_stokes(Q, s_in)

            # Clean rotation‑vector magnitude
            phi_mag_clean = np.zeros((n_channels, n_samples))
            for ch in range(n_channels):
                Q = np.zeros((n_samples, 4))
                for t in range(n_samples): Q[t] = jones_to_quaternion(U_all[ch, t])
                Q = regularize_signs(Q)
                for t in range(n_samples):
                    phi_mag_clean[ch, t] = np.linalg.norm(quaternion_to_rotation_vector(Q[t]))

            # Fair SNR definitions
            stokes_clean_rot = np.zeros_like(stokes_clean)
            from rotations import rotate_centroid_to_north_pole
            for ch in range(n_channels):
                S_rot_clean, _ = rotate_centroid_to_north_pole(stokes_clean[ch])
                stokes_clean_rot[ch] = S_rot_clean
            peak_power_stokes = np.max(stokes_clean_rot[:,:,0]**2 + stokes_clean_rot[:,:,1]**2, axis=1)

            phi_ac = phi_mag_clean - np.mean(phi_mag_clean, axis=1, keepdims=True)
            peak_power_phi = np.max(phi_ac**2, axis=1)

            f_max = fmax_factor * bw
            nyq = 0.5 / dt
            b_lp, a_lp = butter(4, 3*bw/nyq, btype='low')

            # Parallel jobs
            jobs = []
            for i_snr, snr_db in enumerate(SNRS):
                for i_real in range(N_REAL):
                    jobs.append(delayed(process_one_realisation)(
                        bw, f_max, dt, b_lp, a_lp, stokes_clean, phi_mag_clean,
                        peak_power_stokes, peak_power_phi, integrals_abs,
                        pair_indices, n_pairs, ctrl_idx, channel_count,
                        i_snr, snr_db, i_real
                    ))

            print(f"  Processing {len(jobs)} jobs in parallel...")
            results = Parallel(n_jobs=-1, verbose=10)(jobs)

            # Unpack into 4D arrays (fibre index = i_fibre)
            for job_idx, (i_snr, snr_db) in enumerate([(i_snr, snr_db) for i_snr, snr_db in enumerate(SNRS) for _ in range(N_REAL)]):
                i_snr = job_idx // N_REAL
                i_real = job_idx % N_REAL
                res = results[job_idx]

                phase_est_2d[i_fibre, i_bw, i_snr, i_real] = res['est_2d']['phase']
                phat_est_2d[i_fibre, i_bw, i_snr, i_real]  = res['est_2d']['phat']
                int_est_2d[i_fibre, i_bw, i_snr, i_real]   = res['est_2d']['int']
                par_est_2d[i_fibre, i_bw, i_snr, i_real]   = res['est_2d']['par']

                phase_est_rv[i_fibre, i_bw, i_snr, i_real] = res['est_rv']['phase']
                phat_est_rv[i_fibre, i_bw, i_snr, i_real]  = res['est_rv']['phat']
                int_est_rv[i_fibre, i_bw, i_snr, i_real]   = res['est_rv']['int']
                par_est_rv[i_fibre, i_bw, i_snr, i_real]   = res['est_rv']['par']

                for k in ['phase','phat','int','par']:
                    ctrl_2d[k][i_fibre, i_bw, i_snr, i_real] = res['ctrl_2d'][k]
                    ctrl_rv[k][i_fibre, i_bw, i_snr, i_real] = res['ctrl_rv'][k]

            bw_times.append(time.time() - bw_start)
            remaining = np.mean(bw_times) * (total_bw - i_bw - 1)
            print(f"B={bw:.0f} Hz done. Est. remaining: {remaining/60:.1f} min")

    # ---------- SAVE FULL 4‑D ARRAYS (for later fibre‑count analysis) ----------
    save_4d = True      # set to False if you only want the averaged arrays
    if save_4d:
        np.savez_compressed(
            os.path.join(run_folder, f"full_4d_arrays_{channel_count}ch.npz"),
            phase_est_2d=phase_est_2d, phat_est_2d=phat_est_2d,
            int_est_2d=int_est_2d, par_est_2d=par_est_2d,
            phase_est_rv=phase_est_rv, phat_est_rv=phat_est_rv,
            int_est_rv=int_est_rv, par_est_rv=par_est_rv,
            ctrl_2d_phase=ctrl_2d['phase'], ctrl_2d_phat=ctrl_2d['phat'],
            ctrl_2d_int=ctrl_2d['int'],   ctrl_2d_par=ctrl_2d['par'],
            ctrl_rv_phase=ctrl_rv['phase'], ctrl_rv_phat=ctrl_rv['phat'],
            ctrl_rv_int=ctrl_rv['int'],   ctrl_rv_par=ctrl_rv['par'],
            BANDWIDTHS=BANDWIDTHS, SNRS=SNRS,
            event_distance_km=event_distance_km)

    # ---------- Average over fibres ----------
    if n_fibres > 1:
        print("Averaging results across fibres...")
        phase_est_2d_avg = np.nanmean(phase_est_2d, axis=0)
        phat_est_2d_avg  = np.nanmean(phat_est_2d,  axis=0)
        int_est_2d_avg   = np.nanmean(int_est_2d,   axis=0)
        par_est_2d_avg   = np.nanmean(par_est_2d,   axis=0)
        phase_est_rv_avg = np.nanmean(phase_est_rv, axis=0)
        phat_est_rv_avg  = np.nanmean(phat_est_rv,  axis=0)
        int_est_rv_avg   = np.nanmean(int_est_rv,   axis=0)
        par_est_rv_avg   = np.nanmean(par_est_rv,   axis=0)

        ctrl_2d_avg = {}
        ctrl_rv_avg = {}
        for k in ['phase','phat','int','par']:
            ctrl_2d_avg[k] = np.nanmean(ctrl_2d[k], axis=0)
            ctrl_rv_avg[k] = np.nanmean(ctrl_rv[k], axis=0)
    else:
        # Single fibre – just squeeze the first dimension
        phase_est_2d_avg = phase_est_2d[0]
        phat_est_2d_avg  = phat_est_2d[0]
        int_est_2d_avg   = int_est_2d[0]
        par_est_2d_avg   = par_est_2d[0]
        phase_est_rv_avg = phase_est_rv[0]
        phat_est_rv_avg  = phat_est_rv[0]
        int_est_rv_avg   = int_est_rv[0]
        par_est_rv_avg   = par_est_rv[0]

        ctrl_2d_avg = {k: ctrl_2d[k][0] for k in ['phase','phat','int','par']}
        ctrl_rv_avg = {k: ctrl_rv[k][0] for k in ['phase','phat','int','par']}

    # ---------- Compute metrics ----------
    true_dist = event_distance_km
    def compute_metrics(estimates):
        std = np.nanstd(estimates, axis=2)
        bias = np.nanmean(estimates, axis=2) - true_dist
        return std, bias

    methods = {}
    for name, arr in [('Phase‑slope 2D', phase_est_2d_avg), ('GCC‑PHAT 2D', phat_est_2d_avg),
                      ('Integer peak 2D', int_est_2d_avg), ('Parabolic peak 2D', par_est_2d_avg),
                      ('Phase‑slope RV', phase_est_rv_avg), ('GCC‑PHAT RV', phat_est_rv_avg),
                      ('Integer peak RV', int_est_rv_avg), ('Parabolic peak RV', par_est_rv_avg)]:
        methods[name] = compute_metrics(arr)
    for prefix, ctrl_dict in [('Ctrl 2D', ctrl_2d_avg), ('Ctrl RV', ctrl_rv_avg)]:
        for k, v in ctrl_dict.items():
            methods[f'{prefix} {k}'] = compute_metrics(v)

    # ---------- Save results ----------
    result_file = os.path.join(run_folder, f"sweep_results_{channel_count}ch.npz")
    np.savez_compressed(result_file,
                        BANDWIDTHS=BANDWIDTHS, SNRS=SNRS,
                        event_distance_km=event_distance_km,
                        phase_est_2d=phase_est_2d_avg, phat_est_2d=phat_est_2d_avg,
                        int_est_2d=int_est_2d_avg, par_est_2d=par_est_2d_avg,
                        phase_est_rv=phase_est_rv_avg, phat_est_rv=phat_est_rv_avg,
                        int_est_rv=int_est_rv_avg, par_est_rv=par_est_rv_avg,
                        ctrl_2d_phase=ctrl_2d_avg['phase'], ctrl_2d_phat=ctrl_2d_avg['phat'],
                        ctrl_2d_int=ctrl_2d_avg['int'], ctrl_2d_par=ctrl_2d_avg['par'],
                        ctrl_rv_phase=ctrl_rv_avg['phase'], ctrl_rv_phat=ctrl_rv_avg['phat'],
                        ctrl_rv_int=ctrl_rv_avg['int'], ctrl_rv_par=ctrl_rv_avg['par'],
                        bw_times=bw_times)

    # ---------- Generate figures ----------
    for method_name, (std, bias) in methods.items():
        safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        fig_std = os.path.join(run_folder, f'fig_std_{safe_name}_{channel_count}ch.png')
        fig_bias = os.path.join(run_folder, f'fig_bias_{safe_name}_{channel_count}ch.png')
        plot_std_heatmap(std, SNRS, BANDWIDTHS,
                         f'{method_name} std (km)', fig_std, vmax=100)
        plot_bias_heatmap(bias, SNRS, BANDWIDTHS,
                          f'{method_name} bias (km)', fig_bias, vmin=-50, vmax=50)

    print(f"Finished channel_count = {channel_count}")

print(f"\nAll channel counts processed. Outputs saved in: {run_folder}")