#!/usr/bin/env python3
"""Held-out empirical-covariance GLS benchmark for pair fusion.

This is an empirical benchmark, not a CRB.  It estimates the pair-error
covariance from even-indexed noise realizations and evaluates the GLS weights
on odd-indexed realizations.  The 5% diagonal shrinkage is fixed before the
held-out evaluation.  The script reads saved ``pair_Z`` arrays and does not
regenerate Jones traces.

Run locally (the 15-channel pair_Z file is large):

    python empirical_covariance_gls.py results_crb_equispaced 15 --bandwidth 22000
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from estimator_comparison import combine_estimators
from snr_conventions import ENERGY_SNR_DEFINITION, energy_snr_axis_for_run


def within_fibre_covariance(errors):
    """Pool covariance after removing each fibre's pairwise mean error."""
    n_pair = errors.shape[-1]
    cov = np.zeros((n_pair, n_pair), dtype=float)
    dof = 0
    for fibre_errors in errors:
        fibre_errors = np.asarray(fibre_errors, dtype=float)
        fibre_errors -= np.mean(fibre_errors, axis=0, keepdims=True)
        cov += fibre_errors.T @ fibre_errors
        dof += len(fibre_errors) - 1
    return cov / dof


def summarize(errors):
    errors = np.asarray(errors, dtype=float).ravel()
    bias = float(np.mean(errors))
    std = float(np.std(errors, ddof=1))
    return float(np.sqrt(np.mean(errors ** 2))), std, bias


def main(run_dir="results_crb_equispaced", count=15, bandwidth_hz=22000.0,
         shrinkage=0.05):
    path = os.path.join(run_dir, "crb_results_%dch.npz" % count)
    d = np.load(path, allow_pickle=True)
    if "pair_Z" not in d.files:
        raise ValueError("pair_Z is required in %s" % path)

    pair_z = d["pair_Z"]
    bandwidths = d["BANDWIDTHS"]
    saved_snrs = d["SNRS_dB"]
    ib = int(np.argmin(np.abs(bandwidths - bandwidth_hz)))
    n_fibres, _, n_snr, n_real, n_pairs = pair_z.shape
    train_mask = np.arange(n_real) % 2 == 0
    test_mask = ~train_mask
    z_true = float(d["event_distance_km"])
    d_abs = np.asarray(d["D_abs"], dtype=float)
    d2 = d_abs ** 2
    kbest = int(np.argmax(d_abs))
    snrs = energy_snr_axis_for_run(run_dir, count, saved_snrs)

    stats = {name: np.full((3, n_snr), np.nan) for name in
             ("covariance_gls", "average_pool", "disp_pool", "robust_pool")}

    for is_ in range(n_snr):
        z = np.asarray(pair_z[:, ib, is_, :, :], dtype=np.float64)
        train = z[:, train_mask, :]
        test = z[:, test_mask, :]

        train_err = train - z_true
        covariance = within_fibre_covariance(train_err)
        diag = np.maximum(np.diag(covariance), np.finfo(float).tiny)
        covariance = ((1.0 - shrinkage) * covariance
                      + shrinkage * np.diag(diag))
        one = np.ones(n_pairs)
        precision = np.linalg.pinv(covariance, rcond=1e-10)
        weights = precision @ one
        denom = float(one @ weights)
        if not np.isfinite(denom) or denom <= 0.0:
            continue
        weights /= denom
        covariance_gls = np.einsum("frp,p->fr", test, weights) - z_true
        stats["covariance_gls"][:, is_] = summarize(covariance_gls)

        estimators = combine_estimators(test, d2, kbest)
        for name in ("average_pool", "disp_pool", "robust_pool"):
            stats[name][:, is_] = summarize(estimators[name] - z_true)

    d.close()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharex=True)
    styles = {
        "covariance_gls": ("#000000", "o-", "held-out covariance GLS"),
        "average_pool": ("#0072B2", "o--", "equal (filtered pool)"),
        "disp_pool": ("#009E73", "s--", r"$D^2$ (filtered pool)"),
        "robust_pool": ("#CC79A7", "D-", "robust (filtered pool)"),
    }
    ylabels = ["RMSE (km)", "Standard deviation (km)", "Absolute bias (km)"]
    for name, (color, style, label) in styles.items():
        axes[0].semilogy(snrs, stats[name][0], style, color=color, label=label)
        axes[1].semilogy(snrs, stats[name][1], style, color=color)
        axes[2].semilogy(snrs, np.abs(stats[name][2]), style, color=color)
    for ax, ylabel in zip(axes, ylabels):
        ax.set_xlabel("Energy SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=10)
    fig.suptitle("Held-out covariance GLS vs fusion rules (%d ch, nominal B=%.3g kHz)" %
                 (count, bandwidths[ib] / 1e3), fontsize=15)
    fig.subplots_adjust(wspace=0.48, top=0.86)
    out = os.path.join(run_dir, "figs_estimators",
                       "F9_empirical_covariance_gls.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    np.savez_compressed(
        os.path.join(run_dir, "empirical_covariance_gls_%dch.npz" % count),
        bandwidth_Hz=bandwidths[ib], SNRS_dB=snrs,
        snr_definition=ENERGY_SNR_DEFINITION, shrinkage=shrinkage,
        split="even train / odd test", **stats)
    print("saved", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default="results_crb_equispaced")
    parser.add_argument("count", nargs="?", type=int, default=15)
    parser.add_argument("--bandwidth", type=float, default=22000.0)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    args = parser.parse_args()
    main(args.run_dir, args.count, args.bandwidth, args.shrinkage)
