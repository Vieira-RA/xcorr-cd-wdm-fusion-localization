#!/usr/bin/env python3
"""Validate an additive channel-timing covariance reference on saved pair_Z data.

This is an offline, potentially memory-heavy check.  It uses even-indexed noise
realizations to estimate a regularized empirical pair covariance and odd
realizations to evaluate that GLS benchmark.  It also tests the separate
additive channel-timing covariance approximation.  Run locally (not on an HPC
login node):

    python validate_joint_crb.py results_crb_equispaced 15 --bandwidth 22000

The model reference is conditional on Var(tau_ij)=u_i+u_j and is not a
universal combined CRB.  The script reports the marginal-fit residual and
held-out covariance/estimator checks; it does not assume that those checks
pass.
"""
import argparse
import os
import sys

import numpy as np

import crb
from snr_conventions import energy_snr_axis_for_run


def pooled_within_fibre_covariance(errors):
    """Covariance pooled over fibres after centering each fibre separately."""
    n_pair = errors.shape[-1]
    cov = np.zeros((n_pair, n_pair), dtype=np.float64)
    dof = 0
    for fibre_errors in errors:
        fibre_errors = np.asarray(fibre_errors, dtype=np.float64)
        fibre_errors = fibre_errors - np.nanmean(
            fibre_errors, axis=0, keepdims=True)
        good = np.all(np.isfinite(fibre_errors), axis=1)
        fibre_errors = fibre_errors[good]
        if len(fibre_errors) < 2:
            continue
        cov += fibre_errors.T @ fibre_errors
        dof += len(fibre_errors) - 1
    return cov / max(dof, 1), dof


def shared_reference_sanity_check():
    """Three-channel example where independent-edge fusion is optimistic."""
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=int)
    pair_var = np.asarray([2.0e-24, 2.0e-24, 2.0e-24])  # seconds squared
    D = np.asarray([1.0, 2.0, 1.0])                    # ps/km
    joint_std, _ = crb.gls_combined_crb_distance(pair_var, D, pairs, 3)
    legacy_star = crb.star_combined_crb_distance(pair_var, D, pairs, 3)
    expected = np.sqrt(0.5)  # km, for channel errors with variance 1 ps^2
    if not np.isclose(joint_std, expected, rtol=1e-10):
        raise AssertionError("joint covariance sanity check failed")
    if not legacy_star < joint_std:
        raise AssertionError("example should expose the optimistic star sum")
    print("shared-reference check: covariance-aware std %.6f km; "
          "legacy independent-edge star %.6f km" % (joint_std, legacy_star))


def main(run_dir, count, bandwidth_hz, snr_index):
    shared_reference_sanity_check()
    path = os.path.join(run_dir, "crb_results_%dch.npz" % count)
    d = np.load(path, allow_pickle=True)
    if "pair_Z" not in d.files:
        raise ValueError("pair_Z is required: %s" % path)

    pair_z = d["pair_Z"]  # May be large; access only selected cells below.
    bw = d["BANDWIDTHS"]
    saved_snr = d["SNRS_dB"]
    ib = int(np.argmin(np.abs(bw - bandwidth_hz)))
    is_ = snr_index if snr_index >= 0 else len(saved_snr) + snr_index
    if not 0 <= is_ < len(saved_snr):
        raise IndexError("snr_index is outside the saved SNR grid")

    pair_indices = np.asarray(d["pair_indices"], dtype=int)
    D = np.asarray(d["D_abs"], dtype=float)
    n_channels = int(d["channel_indices_full"].shape[0])
    z_true = float(d["event_distance_km"])
    n_fibres, _, _, n_real, n_pairs = pair_z.shape
    train_idx = np.arange(n_real) % 2 == 0
    test_idx = ~train_idx

    # Delay errors in ps; multiply the range error by the pair CD coefficient.
    z_cell = np.asarray(pair_z[:, ib, is_, :, :], dtype=np.float64)
    train_distance_errors = z_cell[:, train_idx, :] - z_true
    test_delay_errors = (z_cell[:, test_idx, :] - z_true) * D[None, None, :]
    train_cov, train_dof = pooled_within_fibre_covariance(train_distance_errors)
    empirical_cov, test_dof = pooled_within_fibre_covariance(test_delay_errors)
    empirical_sd = np.sqrt(np.maximum(np.diag(empirical_cov), 0.0))

    # Theoretical model covariance, averaged over fibres after fitting u_i to
    # each fibre's marginal pair CRBs.  Units are converted from s^2 to ps^2.
    A = crb._incidence(pair_indices, n_channels)
    theoretical_cov = np.zeros((n_pairs, n_pairs), dtype=float)
    theoretical_pair_var = np.nanmean(
        d["crb_delay_pairs"][:, ib, is_, :], axis=0) * 1e24
    fitted_pair_var = np.zeros(n_pairs, dtype=float)
    bound_by_fibre = []
    heldout_errors = []
    for f in range(n_fibres):
        cd = d["crb_delay_pairs"][f, ib, is_, :]
        u = crb.per_channel_variances(cd, pair_indices, n_channels)
        cov_f = (A * u[None, :]) @ A.T
        theoretical_cov += cov_f / n_fibres
        fitted_pair_var += np.diag(cov_f) * 1e24 / n_fibres

        sigma_z, weights = crb.gls_combined_crb_distance(
            cd, D, pair_indices, n_channels)
        bound_by_fibre.append(sigma_z)
        wsum = np.sum(weights)
        if np.isfinite(sigma_z) and np.all(np.isfinite(weights)) and wsum != 0.0:
            estimates = z_cell[f, test_idx, :] @ weights / wsum
            heldout_errors.append(estimates - z_true)
        else:
            heldout_errors.append(np.full(np.sum(test_idx), np.nan))

    model_sd = np.sqrt(np.maximum(np.diag(theoretical_cov), 0.0))
    empirical_corr = empirical_cov / np.outer(empirical_sd, empirical_sd)
    model_corr = theoretical_cov / np.outer(model_sd, model_sd)
    upper = np.triu_indices(n_pairs, 1)
    finite = (np.isfinite(empirical_corr[upper]) &
              np.isfinite(model_corr[upper]))
    corr_rmse = np.sqrt(np.mean(
        (empirical_corr[upper][finite] - model_corr[upper][finite]) ** 2))
    relative_pair_var = np.abs(fitted_pair_var - theoretical_pair_var) / np.maximum(
        theoretical_pair_var, np.finfo(float).tiny)
    pair_std_ratio = empirical_sd / np.sqrt(np.maximum(
        theoretical_pair_var, np.finfo(float).tiny))

    heldout_errors = np.asarray(heldout_errors, dtype=float)
    fibre_std = np.full(n_fibres, np.nan)
    fibre_bias = np.full(n_fibres, np.nan)
    for f, errors in enumerate(heldout_errors):
        errors = errors[np.isfinite(errors)]
        if len(errors) >= 2:
            fibre_std[f] = np.std(errors, ddof=1)
            fibre_bias[f] = np.mean(errors)
    bound_mean = float(np.nanmean(bound_by_fibre))
    heldout_std = float(np.sqrt(np.nanmean(fibre_std ** 2)))
    valid_model_fibres = int(np.sum(np.isfinite(bound_by_fibre)))
    energy_snr = energy_snr_axis_for_run(run_dir, count, saved_snr)[is_]

    # Empirical-covariance GLS is an oracle-style benchmark, not a CRB.
    # Estimate the pair covariance on the training half, shrink it 5% toward
    # its diagonal for numerical stability, and evaluate on held-out samples.
    shrinkage = 0.05
    cov_diag = np.maximum(np.diag(train_cov), np.finfo(float).tiny)
    train_cov = ((1.0 - shrinkage) * train_cov
                 + shrinkage * np.diag(cov_diag))
    one = np.ones(n_pairs)
    precision = np.linalg.pinv(train_cov, rcond=1e-10)
    empirical_weights = precision @ one
    empirical_weights /= np.sum(empirical_weights)
    empirical_errors = z_cell[:, test_idx, :] @ empirical_weights - z_true
    emp_fibre_std = np.nanstd(empirical_errors, axis=1, ddof=1)
    emp_fibre_bias = np.nanmean(empirical_errors, axis=1)
    empirical_std = float(np.sqrt(np.nanmean(emp_fibre_std ** 2)))
    empirical_rmse = float(np.sqrt(np.nanmean(empirical_errors ** 2)))

    print("run/count:", run_dir, count)
    print("cell: B_nom=%.6g Hz, energy SNR=%.4f dB (saved axis %.4f dB)" %
          (bw[ib], energy_snr, saved_snr[is_]))
    print("held-out covariance degrees of freedom:", test_dof)
    print("training covariance degrees of freedom:", train_dof)
    print("held-out pair std / marginal CRB std p10/median/p90:",
          np.round(np.nanpercentile(pair_std_ratio, [10, 50, 90]), 3))
    print("marginal pair-variance fit: median abs relative error %.3f, p90 %.3f" %
          (np.nanmedian(relative_pair_var), np.nanpercentile(relative_pair_var, 90)))
    print("off-diagonal pair-correlation RMSE: %.4f" % corr_rmse)
    print("additive timing-model reference std: %.6g km (%d/%d fibres fit)" %
          (bound_mean, valid_model_fibres, n_fibres))
    print("held-out GLS std: %.6g km (ratio %.3f)" %
          (heldout_std, heldout_std / bound_mean))
    print("held-out mean bias: %.6g km (between-fibre SD %.6g km)" %
          (np.nanmean(fibre_bias), np.nanstd(fibre_bias, ddof=1)))
    print("held-out empirical-covariance GLS (shrinkage %.2f): "
          "RMSE %.6g km, std %.6g km, mean bias %.6g km" %
          (shrinkage, empirical_rmse, empirical_std,
           np.nanmean(emp_fibre_bias)))
    d.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default="results_crb_equispaced")
    parser.add_argument("count", nargs="?", type=int, default=15)
    parser.add_argument("--bandwidth", type=float, default=22000.0)
    parser.add_argument("--snr-index", type=int, default=-1,
                        help="saved SNR index; -1 selects the last point")
    args = parser.parse_args()
    main(args.run_dir, args.count, args.bandwidth, args.snr_index)
