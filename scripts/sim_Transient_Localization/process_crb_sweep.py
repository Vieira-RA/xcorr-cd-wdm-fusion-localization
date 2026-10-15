#!/usr/bin/env python3
"""
CRB-study post-processor (2-D SOP only).

Pipeline per (fibre, bandwidth):
  * build clean 2-D Stokes (quaternion rotation of a fixed input Stokes),
    rotate the centroid to the north pole;
  * energy-SNR noise injection (sigma^2 = <per-component energy>/SNR_lin),
    common across channels (physically the receiver noise floor);
  * optional low-pass / band-pass filter;
  * per channel-pair delay via parabolic interpolation of the 2-D
    cross-correlation;
  * delay -> distance via |int D dlambda|;
  * combine pairs with inverse-variance (CRB) weights.

Stores per-realisation aggregated distances (three estimators), the per-pair
CRB arrays, sigma^2, beta^2, clean energies, and (optionally) all pair
distances for later re-analysis.

Run:  python process_crb_sweep.py sweep_config_crb_test.yaml
"""
import os
import sys
import time
import itertools
import numpy as np
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt

from quaternion import (
    jones_to_quaternion,
    regularize_signs,
    quaternion_rotate_stokes,
)
from rotations import rotate_centroid_to_north_pole
from signal_processing import parabolic_corr_delay_2d, gcc_phat_2d
from estimation import iqr_filter
from chromatic_dispersion import integrated_dispersion

import crb
from channel_selection import select_channels, local_indices


# ---------------------------------------------------------------------------
def clean_rot_2d(U_ch, s_in):
    """Clean 2-D Stokes (S1,S2) after rotating the centroid to the north pole."""
    n = U_ch.shape[0]
    Q = np.empty((n, 4))
    for t in range(n):
        Q[t] = jones_to_quaternion(U_ch[t])
    Q = regularize_signs(Q)
    S = quaternion_rotate_stokes(Q, s_in)          # (n, 3)
    S_rot, _ = rotate_centroid_to_north_pole(S)
    return S_rot[:, :2]


def design_filter(mode, bw, dt, order, low_factor, high_factor):
    nyq = 0.5 / dt
    if mode == "none":
        return None
    if mode == "lowpass":
        wn = min(high_factor * bw / nyq, 0.999)
        return butter(order, wn, btype="low")
    if mode == "bandpass":
        wn_lo = max(low_factor * bw / nyq, 1e-6)
        wn_hi = min(high_factor * bw / nyq, 0.999)
        return butter(order, [wn_lo, wn_hi], btype="band")
    raise ValueError("Unknown filter mode: %s" % mode)


def apply_filter(sig, ba):
    if ba is None:
        return sig
    b, a = ba
    out = np.empty_like(sig)
    if sig.ndim == 2:                      # (N, n_comp)
        for c in range(sig.shape[1]):
            out[:, c] = filtfilt(b, a, sig[:, c])
    else:                                  # (n_ch, N, n_comp) -> filter along time
        for ch in range(sig.shape[0]):
            for c in range(sig.shape[2]):
                out[ch, :, c] = filtfilt(b, a, sig[ch, :, c])
    return out


# ---------------------------------------------------------------------------
def process_one(clean2d, sigma2, pair_indices, D_abs, crb_dist_pair, w_gls,
                dt, ba, i_snr, i_real, seed,
                est_method="parabolic", f_max=0.0, mag_threshold=0.05):
    """One (SNR, realisation) for a given bandwidth. Returns estimates + Zs."""
    rng = np.random.RandomState(seed)
    noisy = clean2d + np.sqrt(sigma2) * rng.randn(*clean2d.shape)
    noisy = apply_filter(noisy, ba)

    n_pairs = len(pair_indices)
    Z = np.empty(n_pairs)
    for k, (i, j) in enumerate(pair_indices):
        # same argument order convention as process_sweep.py
        if est_method == "gcc_phat":
            delay = gcc_phat_2d(noisy[j], noisy[i], dt, f_max, mag_threshold)
        else:
            delay = parabolic_corr_delay_2d(noisy[j], noisy[i], dt)
        Z[k] = delay * 1e12 / D_abs[k]

    # inverse-variance (CRB) weights
    w_crb = 1.0 / crb_dist_pair ** 2
    est_crb = np.sum(w_crb * Z) / np.sum(w_crb)

    # covariance-aware (GLS) weights
    est_gls = np.sum(w_gls * Z) / np.sum(w_gls)

    # wavelength-separation weights (existing estimator)
    w_lam = D_abs ** 2
    est_lambda = np.sum(w_lam * Z) / np.sum(w_lam)

    # inverse-variance + IQR outlier pre-filter
    mask = iqr_filter(Z, factor=1.5)
    if np.any(mask):
        wv = w_crb[mask]
        est_crb_iqr = np.sum(wv * Z[mask]) / np.sum(wv)
    else:
        est_crb_iqr = np.nan

    return est_crb, est_gls, est_lambda, est_crb_iqr, Z


# ---------------------------------------------------------------------------
def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sw = cfg["sweep"]
    BANDWIDTHS = np.logspace(np.log10(float(sw["bandwidths"]["start_Hz"])),
                             np.log10(float(sw["bandwidths"]["stop_Hz"])),
                             int(sw["bandwidths"]["num"]))
    SNRS = np.linspace(float(sw["snrs_dB"]["start"]),
                       float(sw["snrs_dB"]["stop"]),
                       int(sw["snrs_dB"]["num"]))
    SNRS_LIN = 10.0 ** (SNRS / 10.0)
    N_REAL = int(sw["n_realisations"])
    n_fibres = int(sw["n_fibres"])

    analysis = cfg.get("analysis", {})
    counts = [int(c) for c in analysis.get("channel_counts", [4])]
    scheme = analysis.get("channel_selection", "alternating")

    fcfg = cfg.get("filter", {})
    f_mode = fcfg.get("mode", "none")
    f_order = int(fcfg.get("order", 4))
    f_low = float(fcfg.get("low_factor", 0.5))
    f_high = float(fcfg.get("high_factor", 3.0))

    noise_seed = int(cfg.get("noise", {}).get("base_seed", 0))
    jones_dir = cfg["output"]["jones_dir"]
    run_dir = cfg["output"].get("run_dir", "results_crb")
    save_pairs = cfg.get("output", {}).get("save_pairs", "all")
    # joblib workers: CRB_N_JOBS env var (HPC/SLURM) overrides the config
    n_jobs = int(os.environ.get("CRB_N_JOBS",
                                cfg.get("run", {}).get("n_jobs", -1)))
    est_method = analysis.get("estimator", "parabolic").lower()
    _phat = cfg.get("phat", {})
    fmax_factor = float(_phat.get("fmax_factor", 2.0))
    mag_threshold = float(_phat.get("mag_threshold", 0.05))
    print(f"Estimator: {est_method}")
    os.makedirs(run_dir, exist_ok=True)

    meta = np.load(os.path.join(jones_dir, "metadata.npz"), allow_pickle=True)
    full_wl = meta["wavelengths_nm"]
    full_delays = meta["channel_delays"]
    union_indices = meta["union_indices"]
    event_distance_km = float(meta["event_distance_km"])
    N_full = len(full_wl)

    s_in = np.array([1.0, 0.0, 0.0])
    n_bw = len(BANDWIDTHS)
    n_snr = len(SNRS)

    print(f"Config: {config_path}")
    print(f"  filter={f_mode} | counts={counts} | fibres={n_fibres} "
          f"| bw={n_bw} | snr={n_snr} | real={N_REAL} | save_pairs={save_pairs}")

    overall = time.time()
    for count in counts:
        print(f"\n{'='*64}\nChannel count = {count}\n{'='*64}")
        sel_full = np.array(select_channels(N_full, count, scheme))
        sel_local = local_indices(sel_full, union_indices)
        wl = full_wl[sel_full]
        n_ch = count
        pair_indices = list(itertools.combinations(range(n_ch), 2))
        n_pairs = len(pair_indices)
        D_abs = np.array([
            abs(integrated_dispersion(wl[i], wl[j])) for i, j in pair_indices
        ])

        store_pairs = save_pairs == "all"

        meta_common = dict(
            BANDWIDTHS=BANDWIDTHS, SNRS_dB=SNRS, SNRS_lin=SNRS_LIN,
            channel_indices_full=sel_full, channel_indices_local=sel_local,
            wavelengths_nm=wl, pair_indices=np.array(pair_indices),
            D_abs=D_abs, event_distance_km=event_distance_km,
            n_fibres=n_fibres, n_real=N_REAL, filter_mode=f_mode,
            filter_params=dict(order=f_order, low=f_low, high=f_high),
            cfg=cfg,
        )

        # ---- allocate / resume from checkpoint ----
        state_file = os.path.join(run_dir, f"crb_ckpt_{count}ch.npz")
        pair_shape = (n_fibres, n_bw, n_snr, N_REAL, n_pairs)
        if os.path.exists(state_file):
            st = np.load(state_file, allow_pickle=True)
            print(f"  Resuming {count}ch from checkpoint "
                  f"(fibres {sorted(int(x) for x in st['completed'])} done)")
            est_crb_a = st["est_crb"]
            est_lam_a = st["est_lambda"]
            est_crb_iqr_a = st["est_crb_iqr"]
            est_gls_a = (st["est_gls"] if "est_gls" in st.files else
                         np.full((n_fibres, n_bw, n_snr, N_REAL), np.nan))
            crb_delay_pairs = st["crb_delay_pairs"]
            crb_dist_pairs = st["crb_dist_pairs"]
            crb_eff_comb = st["crb_dist_combined_eff"]
            crb_det_comb = st["crb_dist_combined_det"]
            crb_gls_comb = (st["crb_dist_combined_gls"]
                            if "crb_dist_combined_gls" in st.files else
                            np.full((n_fibres, n_bw, n_snr), np.nan))
            sigma2_a = st["sigma2"]
            beta2_a = st["beta2"]
            energy_a = st["energy"]
            pair_Z_a = (st["pair_Z"] if (store_pairs and "pair_Z" in st.files)
                        else (np.full(pair_shape, np.nan, dtype=np.float32)
                              if store_pairs else None))
            completed = set(int(x) for x in st["completed"])
        else:
            shape = (n_fibres, n_bw, n_snr, N_REAL)
            est_crb_a = np.full(shape, np.nan)
            est_lam_a = np.full(shape, np.nan)
            est_crb_iqr_a = np.full(shape, np.nan)
            est_gls_a = np.full(shape, np.nan)
            crb_delay_pairs = np.full((n_fibres, n_bw, n_snr, n_pairs), np.nan)
            crb_dist_pairs = np.full((n_fibres, n_bw, n_snr, n_pairs), np.nan)
            crb_eff_comb = np.full((n_fibres, n_bw, n_snr), np.nan)
            crb_det_comb = np.full((n_fibres, n_bw, n_snr), np.nan)
            crb_gls_comb = np.full((n_fibres, n_bw, n_snr), np.nan)
            sigma2_a = np.full((n_fibres, n_bw, n_snr), np.nan)
            beta2_a = np.full((n_fibres, n_bw, n_ch), np.nan)
            energy_a = np.full((n_fibres, n_bw, n_ch), np.nan)
            pair_Z_a = (np.full(pair_shape, np.nan, dtype=np.float32)
                        if store_pairs else None)
            completed = set()

        def save_checkpoint(done):
            d = dict(meta_common)
            d.update(est_crb=est_crb_a, est_gls=est_gls_a,
                     est_lambda=est_lam_a,
                     est_crb_iqr=est_crb_iqr_a, crb_delay_pairs=crb_delay_pairs,
                     crb_dist_pairs=crb_dist_pairs,
                     crb_dist_combined_eff=crb_eff_comb,
                     crb_dist_combined_det=crb_det_comb,
                     crb_dist_combined_gls=crb_gls_comb,
                     sigma2=sigma2_a, beta2=beta2_a, energy=energy_a,
                     completed=np.array(sorted(done)))
            if store_pairs:
                d["pair_Z"] = pair_Z_a
            np.savez_compressed(state_file, **d)

        for i_f in range(n_fibres):
            if i_f in completed:
                print(f"  fibre {i_f} already complete, skipping.")
                continue
            fibre_dir = os.path.join(jones_dir, f"fibre_{i_f}")
            for i_bw, bw in enumerate(tqdm(BANDWIDTHS,
                                            desc=f"f{i_f} ({count}ch)")):
                data = np.load(os.path.join(
                    fibre_dir, f"jones_bw_{bw:.0f}Hz.npz"), allow_pickle=True)
                dt = float(data["dt"])
                U = data["U_all"][sel_local]           # (n_ch, N, 2, 2)

                clean2d = np.stack([clean_rot_2d(U[c], s_in)
                                    for c in range(n_ch)])   # (n_ch, N, 2)

                # energies (per-component) and bandwidths
                E_ch = np.mean(np.sum(clean2d ** 2, axis=1), axis=1)   # (n_ch,)
                base_energy = np.mean(E_ch)
                energy_a[i_f, i_bw] = E_ch
                beta2_a[i_f, i_bw] = [
                    crb.mean_square_bandwidth(clean2d[c], dt) for c in range(n_ch)
                ]

                ba = design_filter(f_mode, bw, dt, f_order, f_low, f_high)

                # ---- per-SNR CRB prep (sequential, cheap) ----
                snr_meta = []
                jobs = []
                for i_snr, snr_lin in enumerate(SNRS_LIN):
                    sigma2 = base_energy / snr_lin
                    sigma2_a[i_f, i_bw, i_snr] = sigma2

                    cd = np.empty(n_pairs)
                    cdet = np.empty(n_pairs)
                    for k, (i, j) in enumerate(pair_indices):
                        cd[k] = crb.crb_delay_effective(
                            clean2d[i], clean2d[j], dt, sigma2)
                        cdet[k] = crb.crb_delay_deterministic_pair(
                            clean2d[i], clean2d[j], dt, sigma2)
                    crb_delay_pairs[i_f, i_bw, i_snr] = cd
                    cdp = crb.crb_distance(cd, D_abs)
                    crb_dist_pairs[i_f, i_bw, i_snr] = cdp
                    crb_eff_comb[i_f, i_bw, i_snr] = crb.combined_crb_distance(
                        cd, D_abs)
                    crb_det_comb[i_f, i_bw, i_snr] = crb.combined_crb_distance(
                        cdet, D_abs)
                    gls_std, w_gls = crb.gls_combined_crb_distance(
                        cd, D_abs, pair_indices, n_ch)
                    crb_gls_comb[i_f, i_bw, i_snr] = gls_std

                    snr_meta.append((i_snr, sigma2, cdp, w_gls))
                    for r in range(N_REAL):
                        jobs.append(delayed(process_one)(
                            clean2d, sigma2, pair_indices, D_abs, cdp, w_gls,
                            dt, ba, i_snr, r, noise_seed + i_f * 100003 + r,
                            est_method, fmax_factor * bw, mag_threshold
                        ))

                results = Parallel(n_jobs=n_jobs, batch_size=1)(jobs)

                idx = 0
                for (i_snr, sigma2, cdp, w_gls) in snr_meta:
                    for r in range(N_REAL):
                        e_crb, e_gls, e_lam, e_iqr, Z = results[idx]
                        idx += 1
                        est_crb_a[i_f, i_bw, i_snr, r] = e_crb
                        est_gls_a[i_f, i_bw, i_snr, r] = e_gls
                        est_lam_a[i_f, i_bw, i_snr, r] = e_lam
                        est_crb_iqr_a[i_f, i_bw, i_snr, r] = e_iqr
                        if store_pairs:
                            pair_Z_a[i_f, i_bw, i_snr, r] = Z

            completed.add(i_f)
            save_checkpoint(completed)
            print(f"  checkpoint saved after fibre {i_f}.")

        out = os.path.join(run_dir, f"crb_results_{count}ch.npz")
        save_dict = dict(meta_common)
        save_dict.update(
            est_crb=est_crb_a, est_gls=est_gls_a, est_lambda=est_lam_a,
            est_crb_iqr=est_crb_iqr_a,
            crb_delay_pairs=crb_delay_pairs, crb_dist_pairs=crb_dist_pairs,
            crb_dist_combined_eff=crb_eff_comb,
            crb_dist_combined_det=crb_det_comb,
            crb_dist_combined_gls=crb_gls_comb,
            sigma2=sigma2_a, beta2=beta2_a, energy=energy_a,
        )
        if store_pairs:
            save_dict["pair_Z"] = pair_Z_a
        np.savez_compressed(out, **save_dict)
        if os.path.exists(state_file):
            os.remove(state_file)
        print(f"Saved {out}")

    print(f"\nAll done in {(time.time() - overall)/60:.1f} min -> {run_dir}/")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "sweep_config_crb_test.yaml"
    main(config)
