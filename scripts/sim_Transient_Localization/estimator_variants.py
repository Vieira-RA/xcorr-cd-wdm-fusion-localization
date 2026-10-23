#!/usr/bin/env python3
"""
Controlled test of peak/delay estimator variants on the REAL simulated signals.

Compares, per (bandwidth, SNR), the bias and std of the aggregated distance for
  parabolic      - current pipeline (|corr_x|+|corr_y|, 3-pt parabola)
  phase_slope    - magnitude-weighted phase slope (unbiased, higher variance)
  gcc_phat       - phase-transform GCC with bandwidth mask
  signed         - coherent sum (corr_x+corr_y), 3-pt parabola

Aggregation: inverse-variance (CRB) weighting over all pairs, same for all.

Run (small; needs the Jones data):  python estimator_variants.py <config> [count]
"""
import sys
import yaml
import numpy as np

from quaternion import (jones_to_quaternion, regularize_signs,
                        quaternion_rotate_stokes)
from rotations import rotate_centroid_to_north_pole
from signal_processing import (parabolic_corr_delay_2d, phase_slope_delay_2d,
                               gcc_phat_2d, integer_corr_delay_2d,
                               parabolic_fit)
import crb
from channel_selection import select_channels, local_indices


def clean_rot_2d(U_ch, s_in):
    n = U_ch.shape[0]
    Q = np.empty((n, 4))
    for t in range(n):
        Q[t] = jones_to_quaternion(U_ch[t])
    Q = regularize_signs(Q)
    S, _ = rotate_centroid_to_north_pole(quaternion_rotate_stokes(Q, s_in))
    return S[:, :2]


def signed_parabolic_delay(sig1, sig2, dt):
    N = len(sig1) + len(sig2) - 1
    Ax = np.fft.fft(sig1[:, 0], N); Ay = np.fft.fft(sig1[:, 1], N)
    Bx = np.fft.fft(sig2[:, 0], N); By = np.fft.fft(sig2[:, 1], N)
    cx = np.fft.ifft(Ax * np.conj(Bx)).real
    cy = np.fft.ifft(Ay * np.conj(By)).real
    corr = np.fft.fftshift(cx + cy)
    lags = np.arange(-len(sig2) + 1, len(sig1))
    k = int(np.argmax(np.abs(corr)))
    delta, _ = parabolic_fit(np.abs(corr), k)
    return (lags[k] + delta) * dt


METHODS = ["parabolic", "phase_slope", "gcc_phat", "signed"]


def pair_delay(method, a, b, dt, f_max, mag):
    if method == "parabolic":
        return parabolic_corr_delay_2d(a, b, dt)
    if method == "phase_slope":
        return phase_slope_delay_2d(a, b, dt, True)
    if method == "gcc_phat":
        return gcc_phat_2d(a, b, dt, f_max, mag)
    if method == "signed":
        return signed_parabolic_delay(a, b, dt)
    raise ValueError(method)


def main(config_path, count=15, n_bw=3, snrs=None,
         n_real=500, i_fibre=0):
    cfg = yaml.safe_load(open(config_path))
    if snrs is None:
        sw = cfg["sweep"]["snrs_dB"]
        snrs = np.linspace(float(sw["start"]), float(sw["stop"]), 4)
    jdir = cfg["output"]["jones_dir"]
    meta = np.load(jdir + "/metadata.npz", allow_pickle=True)
    wl = meta["wavelengths_nm"]; N_full = len(wl)
    union = meta["union_indices"]
    sel_full = np.array(select_channels(N_full, count,
                                        cfg["analysis"]["channel_selection"]))
    sel = local_indices(sel_full, union)
    wl_sel = wl[sel_full]
    pairs = list(__import__("itertools").combinations(range(count), 2))
    from chromatic_dispersion import integrated_dispersion
    D = np.array([abs(integrated_dispersion(wl_sel[i], wl_sel[j]))
                  for i, j in pairs])
    s_in = np.array([1.0, 0.0, 0.0])
    BW = meta["BANDWIDTHS"]
    bw_idx = [int(len(BW) * f) for f in (0.2, 0.5, 0.8)][:n_bw]
    event = float(meta["event_distance_km"])
    phat = cfg.get("phat", {})
    fmax_factor = float(phat.get("fmax_factor", 2.0))
    mag = float(phat.get("mag_threshold", 0.05))

    print("count=%d  pairs=%d  true Z=%.1f km" % (count, len(pairs), event))
    for ib in bw_idx:
        bw = BW[ib]
        d = np.load("%s/fibre_%d/jones_bw_%.0fHz.npz" % (jdir, i_fibre, bw),
                    allow_pickle=True)
        U = d["U_all"][sel]; dt = float(d["dt"])
        clean = np.stack([clean_rot_2d(U[c], s_in) for c in range(count)])
        # Mean per-component integrated energy across the selected channels.
        base_energy = np.mean(np.sum(clean ** 2, axis=1))
        print("--- nominal B = %.3g Hz ---" % bw)
        print("%5s | %s" % ("SNR", "  ".join("%-22s" % m for m in METHODS)))
        for snr in snrs:
            s2 = base_energy / 10 ** (snr / 10)
            cd = np.array([crb.crb_delay_effective(clean[i], clean[j], dt, s2)
                           for i, j in pairs])
            w = 1.0 / crb.crb_distance(cd, D) ** 2   # CRB IVW weights
            est = {m: np.empty(n_real) for m in METHODS}
            rng = np.random.RandomState(1234)
            for r in range(n_real):
                noisy = clean + np.sqrt(s2) * rng.randn(*clean.shape)
                for m in METHODS:
                    Z = np.empty(len(pairs))
                    for k, (i, j) in enumerate(pairs):
                        Z[k] = pair_delay(m, noisy[j], noisy[i], dt, fmax_factor * bw, mag) * 1e12 / D[k]
                    est[m][r] = np.sum(w * Z) / np.sum(w)
            line = "%5d |" % snr
            for m in METHODS:
                b = np.nanmean(est[m]) - event
                s = np.nanstd(est[m])
                line += "  %+7.3f +/- %7.3f " % (b, s)
            print(line)
    print("(bias +/- std in km)")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "sweep_config_crb_equispaced.yaml"
    cnt = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    main(cfg, cnt)
