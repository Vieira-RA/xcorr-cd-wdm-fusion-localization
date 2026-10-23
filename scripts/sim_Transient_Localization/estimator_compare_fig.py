#!/usr/bin/env python3
"""
Overlay figure: parabolic vs GCC-PHAT (same combiner), RMSE vs SNR at three
nominal bandwidths.

Run:  python estimator_compare_fig.py results_crb_equispaced results_crb_equispaced_gcc 15
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_conventions import energy_snr_axis_for_run

plt.rcParams.update({
    "font.size": 15, "font.family": "DejaVu Sans", "axes.titlesize": 16,
    "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 12, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    "lines.linewidth": 2.4, "lines.markersize": 8,
})
CP = {"parabolic": "#0072B2", "gcc_phat": "#D55E00"}


def load(run, n):
    return np.load(f"{run}/estcomp_{n}ch.npz", allow_pickle=True)


def main(run_par, run_gcc, n=15, est="robust_pool", out_dir=None):
    a = load(run_par, n); b = load(run_gcc, n)
    BW = a["BANDWIDTHS"]
    if ("snr_definition" in a.files and
            str(a["snr_definition"].item()) == crb.ENERGY_SNR_DEFINITION):
        SNRS = a["SNRS_dB"]
    else:
        SNRS = energy_snr_axis_for_run(run_par, n, a["SNRS_dB"])
    nb = len(BW)
    cols = [int(nb*0.2), int(nb*0.5), int(nb*0.8)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    for ax, ib in zip(axes, cols):
        ax.semilogy(SNRS, a["rmse_" + est][ib], "o-", color=CP["parabolic"],
                    label="parabolic")
        ax.semilogy(SNRS, b["rmse_" + est][ib], "s--", color=CP["gcc_phat"],
                    label="GCC-PHAT")
        ax.set_title(r"Nominal $B = %.3g$ kHz" % (BW[ib]/1e3))
        ax.set_xlabel("Energy SNR (dB)")
        ax.set_ylim(bottom=1e-3)
    axes[0].set_ylabel("RMSE (km)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Parabolic vs GCC-PHAT (%d ch, robust pool)" % n, y=1.02)
    out_dir = out_dir or os.path.join(run_gcc, "figs_estimators")
    os.makedirs(out_dir, exist_ok=True)
    f = os.path.join(out_dir, "F1_estimators_compare.png")
    fig.savefig(f); plt.close(fig)
    print("saved", f)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "results_crb_equispaced"
    b = sys.argv[2] if len(sys.argv) > 2 else "results_crb_equispaced_gcc"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    main(a, b, n)
