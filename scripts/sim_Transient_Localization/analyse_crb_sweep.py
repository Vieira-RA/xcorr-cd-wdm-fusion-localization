#!/usr/bin/env python3
"""
CRB-study analysis and plots.

For each channel count saved by process_crb_sweep.py:
  * std / bias heatmaps with the conditional additive timing-model reference;
  * std vs SNR curves and multi-fibre scaling diagnostics.

Run:  python analyse_crb_sweep.py sweep_config_crb_test.yaml
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm, LogNorm
import yaml

import crb
from snr_conventions import energy_snr_axis_from_result

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "figure.dpi": 150,
    "savefig.dpi": 150, "savefig.bbox": "tight",
})


def nanstd(x, axis):
    return np.nanstd(x, axis=axis)


def load_runs(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "crb_results_*ch.npz")))
    runs = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        count = int(d["channel_indices_full"].shape[0])
        runs[count] = d
    return runs


# ---------------------------------------------------------------------------
def plot_std_with_crb(std, model_ref, SNRS, BW, title, filename,
                      vmax=100.0, gamma=0.4, crb_levels=None):
    """Std heatmap with conditional channel-timing-model contours."""
    fig, ax = plt.subplots(figsize=(8.2, 6))
    X, Y = np.meshgrid(SNRS, BW)
    data = np.clip(std, 0, vmax)
    c = ax.pcolormesh(X, Y, data, shading="auto", cmap="plasma",
                      norm=PowerNorm(gamma=gamma, vmin=0, vmax=vmax))
    ax.set_yscale("log")

    if crb_levels is None:
        crb_levels = [0.1, 0.3, 1, 3, 10, 30, 100]
    cs = ax.contour(X, Y, model_ref, levels=crb_levels, colors="k",
                    linewidths=1.0, linestyles="--")
    try:
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.3g")
    except Exception:
        pass
    ax.set_xlabel("Energy SNR (dB)")
    ax.set_ylabel("Nominal envelope parameter B (Hz)")
    ax.set_title(title)
    fig.colorbar(c, ax=ax, extend="max", label="Std (km)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_bias(bias, SNRS, BW, title, filename, vmin=-50, vmax=50):
    fig, ax = plt.subplots(figsize=(8.2, 6))
    X, Y = np.meshgrid(SNRS, BW)
    ax.pcolormesh(X, Y, bias, shading="auto", cmap="coolwarm",
                  norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
    ax.set_yscale("log")
    ax.set_xlabel("Energy SNR (dB)")
    ax.set_ylabel("Nominal envelope parameter B (Hz)")
    ax.set_title(title)
    fig.colorbar(ax.collections[0], ax=ax, extend="both", label="Bias (km)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_std_vs_snr(std, model_ref, SNRS, BW, bw_indices, title, filename):
    fig, ax = plt.subplots(figsize=(8.2, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(bw_indices)))
    for col, ib in zip(colors, bw_indices):
        ax.loglog(SNRS, std[ib], "o-", color=col, ms=3,
                  label=f"sim B={BW[ib]:.3g} Hz")
        ax.loglog(SNRS, model_ref[ib], "--", color=col, lw=1.2)
    ax.set_xlabel("Energy SNR (dB)")
    ax.set_ylabel("Std (km)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="k", ls="--", lw=1.2,
                      label="additive timing-model reference")]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
              fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_estimators(d, ib, SNRS, BW, title, filename):
    """Std vs SNR for the estimators at one bandwidth."""
    fig, ax = plt.subplots(figsize=(8.2, 6))
    series = [("est_crb", "pair-marginal inverse-variance", "o-"),
              ("est_gls", "additive timing-model GLS", "d-"),
              ("est_lambda", r"wavelength-separation ($D^2$)", "s--"),
              ("est_crb_iqr", "inverse-variance + IQR", "^:")]
    for key, lab, sty in series:
        if key not in d.files:
            continue
        std = np.nanmean(np.nanstd(d[key], axis=-1), axis=0)[ib]
        ax.loglog(SNRS, std, sty, ms=4, label=lab)
    ax.set_xlabel("Energy SNR (dB)")
    ax.set_ylabel("Std (km)")
    ax.set_title(title + "  (nominal B=%.3g Hz)" % BW[ib])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_vs_nfibres(std_by_nf, n_list, model_ref, ib, is_, title, filename):
    """std at a reference (bw,snr) vs number of fibres, with CRB/sqrt(N)."""
    vals = np.array([std_by_nf[n][ib, is_] for n in n_list])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(n_list, vals, "o-", label="simulation")
    ax.loglog(n_list, model_ref[ib, is_] / np.sqrt(n_list), "k--",
              label=r"CRB / $\sqrt{N}$")
    ax.set_xlabel("Number of fibres")
    ax.set_ylabel("Std (km)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_std_vs_nch(collected, ib, is_, title, filename):
    """std vs channel count at a reference (bw, snr), with joint bound."""
    counts = sorted(collected.keys())
    stds = [collected[c][0][ib, is_] for c in counts]
    crbs = [collected[c][1][ib, is_] for c in counts]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.loglog(counts, stds, "o-", label="simulation")
    ax.loglog(counts, crbs, "k--", label="additive timing-model reference")
    ax.set_xlabel("Number of channels")
    ax.set_ylabel("Std (km)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_counts_vs_snr(collected, ib, title, filename):
    """std vs SNR at one bandwidth for every channel count."""
    counts = sorted(collected.keys())
    fig, ax = plt.subplots(figsize=(8.2, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(counts)))
    for col, c in zip(colors, counts):
        std_f, crb_joint_f, bias_f, SNRS, BW = collected[c]
        ax.loglog(SNRS, std_f[ib], "o-", color=col, ms=3, label="%d ch" % c)
        ax.loglog(SNRS, crb_joint_f[ib], "--", color=col, lw=1.0)
    ax.set_xlabel("Energy SNR (dB)")
    ax.set_ylabel("Std (km)")
    ax.set_title(title + "  (nominal B=%.3g Hz)" % BW[ib])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ---------------------------------------------------------------------------
def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    run_dir = cfg["output"].get("run_dir", "results_crb")
    runs = load_runs(run_dir)
    if not runs:
        print("No result files found in", run_dir)
        return
    print("Loaded channel counts:", sorted(runs.keys()))

    collected = {}
    for count, d in sorted(runs.items()):
        SNRS = energy_snr_axis_from_result(d, d["SNRS_dB"], cfg)
        BW = d["BANDWIDTHS"]
        true_dist = float(d["event_distance_km"])
        est = d["est_crb"]                       # (nf, nbw, nsnr, nreal)
        crb_eff = d["crb_dist_combined_eff"]     # (nf, nbw, nsnr)
        crb_det = d["crb_dist_combined_det"]
        nf = est.shape[0]

        # Joint covariance-aware bound under the additive channel-timing model.
        cd_pairs = d["crb_delay_pairs"]
        D_abs = d["D_abs"]
        pair_idx = d["pair_indices"]
        crb_joint = np.full_like(crb_eff, np.nan)
        for f in range(nf):
            for ib in range(len(BW)):
                for is_ in range(len(SNRS)):
                    crb_joint[f, ib, is_] = crb.gls_combined_crb_distance(
                        cd_pairs[f, ib, is_], D_abs, pair_idx, count)[0]

        std = nanstd(est, axis=-1)               # (nf, nbw, nsnr)
        bias = np.nanmean(est, axis=-1) - true_dist

        # fibre-averaged std/bias and CRB
        std_f = np.nanmean(std, axis=0)
        bias_f = np.nanmean(bias, axis=0)
        crb_eff_f = np.nanmean(crb_eff, axis=0)
        crb_det_f = np.nanmean(crb_det, axis=0)
        crb_joint_f = np.nanmean(crb_joint, axis=0)
        collected[count] = (std_f, crb_joint_f, bias_f, SNRS, BW)

        prefix = os.path.join(run_dir, f"fig_{count}ch")
        plot_std_with_crb(std_f, crb_joint_f, SNRS, BW,
                          f"{count} ch | std (km), additive timing-model contours",
                          prefix + "_std_crb.png")
        plot_bias(bias_f, SNRS, BW,
                  f"{count} ch | bias (km)", prefix + "_bias.png")

        # std vs SNR at three bandwidths
        n_bw = len(BW)
        bw_idx = [int(n_bw * 0.2), int(n_bw * 0.5), int(n_bw * 0.8)]
        plot_std_vs_snr(std_f, crb_joint_f, SNRS, BW, bw_idx,
                        f"{count} ch | std vs SNR", prefix + "_std_vs_snr.png")
        plot_estimators(d, bw_idx[-1], SNRS, BW,
                        f"{count} ch | estimator comparison",
                        prefix + "_estimators.png")

        # ---- multi-fibre combining ----
        n_list = list(range(1, nf + 1))
        std_by_nf = {}
        for n in n_list:
            comb = np.nanmean(est[:n], axis=0)       # (nbw, nsnr, nreal)
            std_by_nf[n] = nanstd(comb, axis=-1)
        is_ref = int(np.argmin(np.abs(SNRS - 107.7815)))
        ib_ref = int(np.argmin(np.abs(np.log(BW) - np.log(np.sqrt(BW[0] * BW[-1])))))
        plot_vs_nfibres(std_by_nf, n_list, crb_joint_f, ib_ref, is_ref,
                        f"{count} ch | std vs fibres (B={BW[ib_ref]:.3g} Hz, "
                        f"energy SNR={SNRS[is_ref]:.1f} dB)",
                        prefix + "_vs_nfibres.png")

        # ---- save summary ----
        np.savez_compressed(
            os.path.join(run_dir, f"summary_{count}ch.npz"),
            SNRS_dB=SNRS, BANDWIDTHS=BW, true_dist=true_dist,
            std_per_fibre=std, bias_per_fibre=bias,
            std_fibre_avg=std_f, bias_fibre_avg=bias_f,
            crb_channel_model_fibre_avg=crb_joint_f,
            crb_independent_eff_legacy=crb_eff_f,
            crb_independent_det_legacy=crb_det_f,
            n_fibres_list=np.array(n_list),
        )
        print(f"[{count} ch] figures + summary written.")

    # ---- cross-channel-count comparison ----
    if collected:
        counts = sorted(collected)
        SNRS = collected[counts[0]][3]
        BW = collected[counts[0]][4]
        is_ref = int(np.argmin(np.abs(SNRS - 107.7815)))
        ib_ref = int(np.argmin(np.abs(np.log(BW)
                                      - np.log(np.sqrt(BW[0] * BW[-1])))))
        plot_std_vs_nch(
            collected, ib_ref, is_ref,
            "Std vs channels (B_nom=%.3g Hz, energy SNR=%.1f dB)"
            % (BW[ib_ref], SNRS[is_ref]),
            os.path.join(run_dir, "fig_std_vs_nchannels.png"))
        plot_counts_vs_snr(
            collected, ib_ref, "Std vs SNR by channel count",
            os.path.join(run_dir, "fig_counts_vs_snr.png"))
        print("Cross-count comparison figures written.")

    print("Analysis complete.")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "sweep_config_crb_test.yaml"
    main(config)
