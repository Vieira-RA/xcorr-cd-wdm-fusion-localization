#!/usr/bin/env python3
"""Build the manuscript figures for the channel-fusion narrative.

The script only reads saved estimator and CRB result arrays; it does not rerun
Jones or noise simulations.  It writes F10/F11 to the equispaced result folder
and copies the same assets into ``paper/figures``.

Run from any directory:
    python scripts/sim_Transient_Localization/plot_fusion_narrative.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from estimator_comparison import snr_for_target
from snr_conventions import energy_snr_axis_for_run


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EQUISPACED = HERE / "results_crb_equispaced"
BANDEDGE = HERE / "results_crb_full_pairs"
PAPER_FIGURES = REPO / "paper" / "figures"
TARGET_BW_HZ = 22_000.0
TARGET_RMSE_KM = 5.0
POOL_ALPHA = 0.25

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 2.0,
})

METHODS = {
    "single": ("#666666", "x", r"single pair (largest $|D|$)"),
    "average": ("#0072B2", "o", "equal, all pairs"),
    "disp": ("#009E73", "s", r"$D^2$, all pairs"),
    "robust": ("#CC79A7", "D", "robust, all pairs"),
    "average_pool": ("#E69F00", "^", r"equal, filtered ($\alpha=0.25$)"),
}


def load_estcomp(run_dir, count):
    """Load one saved metric file and return its correctly labeled SNR axis."""
    path = run_dir / ("estcomp_%dch.npz" % count)
    with np.load(path, allow_pickle=False) as data:
        saved_snrs = np.asarray(data["SNRS_dB"], dtype=float)
        if ("snr_definition" in data.files and
                str(data["snr_definition"].item()) ==
                "mean_per_component_energy_over_noise_variance"):
            snrs = saved_snrs
        else:
            snrs = energy_snr_axis_for_run(str(run_dir), count, saved_snrs)
        return {
            "bandwidths": np.asarray(data["BANDWIDTHS"], dtype=float),
            "snrs": np.asarray(snrs, dtype=float),
            "rmse": {
                name: np.asarray(data["rmse_" + name], dtype=float)
                for name in METHODS
            },
            "pair_D_abs": np.asarray(data["pair_D_abs"], dtype=float),
        }


def nearest_bandwidth_index(bandwidths, target_hz):
    return int(np.argmin(np.abs(np.log(np.asarray(bandwidths) / target_hz))))


def save_both(fig, filename):
    result_path = EQUISPACED / "figs_estimators" / filename
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(result_path)
    fig.savefig(PAPER_FIGURES / filename)
    plt.close(fig)
    return result_path, PAPER_FIGURES / filename


def plot_fusion_strategy():
    counts = (5, 10, 15)
    result = {count: load_estcomp(EQUISPACED, count) for count in counts}
    bandwidths = result[15]["bandwidths"]
    ib = nearest_bandwidth_index(bandwidths, TARGET_BW_HZ)
    actual_bw = float(bandwidths[ib])
    snrs = result[15]["snrs"]

    fig, (ax_curve, ax_design) = plt.subplots(
        1, 2, figsize=(14.2, 5.3), gridspec_kw={"width_ratios": [1.2, 1]})

    # Representative RMSE curves: hold the grid, bandwidth, and estimator fixed.
    for name, (color, marker, label) in METHODS.items():
        ax_curve.semilogy(
            snrs, result[15]["rmse"][name][ib],
            color=color, marker=marker, markevery=5, markersize=5,
            label=label)
    ax_curve.axhline(TARGET_RMSE_KM, color="0.25", linestyle="--", linewidth=1.2)
    ax_curve.text(snrs[0] + 1.0, TARGET_RMSE_KM * 1.08,
                  "5-km target", color="0.25", fontsize=9)
    ax_curve.set_xlabel("Mean per-component energy SNR (dB)")
    ax_curve.set_ylabel("RMSE (km)")
    ax_curve.set_title("15 channels; nominal $B$ = %.2f kHz" % (actual_bw / 1e3))
    ax_curve.legend(loc="upper right", ncol=1)
    ax_curve.set_ylim(bottom=0.1)

    # Design comparison at one target RMSE across channel counts.
    for name, (color, marker, label) in METHODS.items():
        crossing = [
            snr_for_target(result[count]["rmse"][name][
                nearest_bandwidth_index(result[count]["bandwidths"], TARGET_BW_HZ)],
                result[count]["snrs"], TARGET_RMSE_KM)
            for count in counts
        ]
        ax_design.plot(counts, crossing, color=color, marker=marker,
                       markersize=6, label=label)
    ax_design.set_xticks(counts)
    ax_design.set_xlabel("Number of equispaced channels")
    ax_design.set_ylabel("Energy SNR for 5-km RMSE (dB)")
    ax_design.set_title("Matched-accuracy comparison")
    ax_design.text(0.03, 0.03, "A missing point means the target was not reached",
                   transform=ax_design.transAxes, fontsize=8, color="0.3")
    ax_design.legend(loc="best")
    fig.suptitle("Using WDM channel diversity: all-pair fusion versus hard selection",
                 y=1.03, fontsize=14)
    return save_both(fig, "F10_equispaced_fusion_strategy.png")


def weight_profile(pair_d):
    d = np.asarray(pair_d, dtype=float)
    d = np.abs(d[np.isfinite(d)])
    order = np.argsort(d)[::-1]
    d_norm = d[order] / np.max(d)
    w = d_norm ** 2
    cumulative = np.cumsum(w) / np.sum(w)
    keep = d_norm ** 2 >= POOL_ALPHA
    n_keep = int(np.count_nonzero(keep))
    retained_weight = float(np.sum(d_norm[keep] ** 2) / np.sum(d_norm ** 2))
    return d_norm, cumulative, n_keep, retained_weight


def plot_pair_weight_distribution():
    equi = load_estcomp(EQUISPACED, 15)
    edge = load_estcomp(BANDEDGE, 12)
    profiles = [
        ("Equispaced (15 ch)", weight_profile(equi["pair_D_abs"]), "#0072B2"),
        ("Alternating band-edge (12 ch)", weight_profile(edge["pair_D_abs"]),
         "#D55E00"),
    ]

    fig, (ax_disp, ax_mass) = plt.subplots(1, 2, figsize=(12.7, 4.8))
    for label, (d_norm, cumulative, n_keep, retained_weight), color in profiles:
        fraction = np.arange(1, len(d_norm) + 1) / len(d_norm)
        keep_fraction = n_keep / len(d_norm)
        ax_disp.semilogy(fraction, d_norm, color=color, linewidth=2,
                         label=label)
        ax_disp.axvline(keep_fraction, color=color, linestyle=":", alpha=0.85)
        ax_mass.plot(fraction, cumulative, color=color, linewidth=2,
                     label=label)
        ax_mass.axvline(keep_fraction, color=color, linestyle=":", alpha=0.85)
        ax_mass.plot(keep_fraction, retained_weight, "o", color=color,
                     markersize=6)
        ax_mass.annotate(
            "%d/%d pairs; %.3g%% weight excluded" %
            (n_keep, len(d_norm), 100.0 * (1.0 - retained_weight)),
            xy=(keep_fraction, retained_weight), xytext=(5, -16),
            textcoords="offset points", fontsize=8, color=color)

    ax_disp.axhline(np.sqrt(POOL_ALPHA), color="0.25", linestyle="--",
                    linewidth=1.1, label=r"cutoff $|D|/D_{\max}=0.5$")
    ax_disp.set_xlabel("Fraction of pairs, sorted by decreasing $|D|$")
    ax_disp.set_ylabel(r"Normalized dispersion $|D|/D_{\max}$")
    ax_disp.set_title("Pair-dispersion distribution")
    ax_disp.legend(loc="best", fontsize=8)

    ax_mass.set_xlabel("Fraction of highest-$|D|$ pairs included")
    ax_mass.set_ylabel(r"Cumulative share of total $D^2$ weight")
    ax_mass.set_title("Cumulative D-squared weight at alpha=0.25 cutoff")
    ax_mass.set_xlim(0, 1)
    ax_mass.set_ylim(0, 1.04)
    ax_mass.legend(loc="lower right", fontsize=8)
    fig.suptitle(r"A hard cutoff removes pairs; $D^2$ weighting changes their influence",
                 y=1.03, fontsize=14)
    return save_both(fig, "F11_pair_weight_distribution.png")


def main():
    outputs = [plot_fusion_strategy(), plot_pair_weight_distribution()]
    for result_path, paper_path in outputs:
        print("saved", result_path)
        print("copied", paper_path)


if __name__ == "__main__":
    main()
