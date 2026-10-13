#!/usr/bin/env python3
"""
Estimator comparison (all SNR-free) from per-pair distances.

Estimators
----------
  single   : one pair - the one with the largest |int D dlambda| (best pair)
  average  : unweighted mean over all pairs
  disp     : inverse-variance from dispersion, w ~ D^2
  robust   : w ~ D^2 * Tukey-bisquare on residuals from the median (smooth)

Reference
---------
  crb_gls  : covariance-aware (GLS) combined CRB (from crb_delay_pairs)

Figures (results/<run>/figs_estimators/)
  F1  RMSE vs SNR at three bandwidths (money plot)
  F2  improvement ratio maps  RMSE_single/RMSE_robust, RMSE_average/RMSE_robust
  F3  iso-precision: SNR needed for target RMSE vs bandwidth
  F4  dB gain vs number of channels
  F5  error distributions at a threshold cell
  F6  gap-to-CRB vs SNR

All derived arrays are saved to results/<run>/estcomp_<n>ch.npz so the figures
can be rebuilt without recomputation.

Run:  python estimator_comparison.py results_crb_full_pairs
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

import crb

# ---------------------------- publication style ---------------------------
plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.linewidth": 0.9, "lines.linewidth": 1.8, "legend.frameon": False,
})
COL = {"single": "#D55E00", "average": "#0072B2", "disp": "#009E73",
       "robust": "#CC79A7"}
LBL = {"single": "single best pair", "average": "pure average",
       "disp": r"$D^2$-weighted", "robust": r"$D^2$ + bisquare (robust)"}
MARK = {"single": "v", "average": "o", "disp": "s", "robust": "D"}


# ------------------------------ estimators --------------------------------
def combine_estimators(Zp, D2, kbest, c=2.5, iqr_factor=1.5):
    """Zp: (..., npairs) distance estimates. Returns dict of (..) arrays."""
    out = {}
    out["single"] = Zp[..., kbest]
    out["average"] = np.nanmean(Zp, axis=-1)
    out["disp"] = (np.nansum(D2 * Zp, axis=-1)
                   / np.nansum(np.broadcast_to(D2, Zp.shape), axis=-1))
    med = np.nanmedian(Zp, axis=-1, keepdims=True)
    q1 = np.nanpercentile(Zp, 25, axis=-1, keepdims=True)
    q3 = np.nanpercentile(Zp, 75, axis=-1, keepdims=True)
    scale = iqr_factor * (q3 - q1)
    r = np.abs(Zp - med)
    scale_safe = np.where(scale > 0, scale, 1.0)
    t = r / (c * scale_safe)
    wb = np.where(scale > 0, np.where(t < 1, (1.0 - t ** 2) ** 2, 0.0), 1.0)
    w = D2 * wb
    num = np.nansum(w * Zp, axis=-1)
    den = np.nansum(w, axis=-1)
    out["robust"] = np.where(den > 0, num / den, np.nan)
    return out


def metrics(E, Ztrue, tol):
    """E: (nf, nbw, nsnr, nreal). Returns dict of (nbw, nsnr) surfaces."""
    nf, nbw, nsnr, nre = E.shape
    err = E.transpose(0, 3, 1, 2).reshape(nf * nre, nbw, nsnr) - Ztrue
    mse = np.nanmean(err ** 2, axis=0)
    bias = np.nanmean(err, axis=0)
    std = np.sqrt(np.clip(mse - bias ** 2, 0.0, None))
    mae = np.nanmedian(np.abs(err), axis=0)
    succ = np.nanmean(np.abs(err) <= tol, axis=0)
    return dict(rmse=np.sqrt(mse), bias=bias, std=std, mae=mae, success=succ)


def gls_bound(d):
    cd = d["crb_delay_pairs"]; D = d["D_abs"]
    pairs = list(d["pair_indices"]); nch = int(d["channel_indices_full"].shape[0])
    nf, nbw, nsnr, _ = cd.shape
    out = np.full((nf, nbw, nsnr), np.nan)
    for f in range(nf):
        for ib in range(nbw):
            for is_ in range(nsnr):
                out[f, ib, is_] = crb.gls_combined_crb_distance(
                    cd[f, ib, is_], D, pairs, nch)[0]
    return out


# --------------------------------- plots ----------------------------------
def fig1_money(res, counts, BW, SNRS, embed_dir):
    key = counts[-1]
    r = res[key]
    nb = len(BW)
    bw_idx = [int(nb * 0.2), int(nb * 0.5), int(nb * 0.8)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, ib in zip(axes, bw_idx):
        for est in ["single", "average", "disp", "robust"]:
            ax.semilogy(SNRS, r["rmse"][est][ib], marker=MARK[est], ms=4,
                        color=COL[est], label=LBL[est])
        ax.semilogy(SNRS, r["crb"][ib], "k--", lw=1.4, label="CRB (GLS)")
        ax.set_title(r"$B = %.3g$ Hz" % BW[ib])
        ax.set_xlabel("SNR (dB)")
        ax.set_ylim(bottom=1e-3)
    axes[0].set_ylabel("RMSE (km)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Estimation error vs SNR  (%d channels)" % key, y=1.02)
    f = os.path.join(embed_dir, "F1_rmse_vs_snr.png")
    fig.savefig(f); plt.close(fig); return f


def fig2_ratio_maps(res, counts, BW, SNRS, embed_dir):
    key = counts[-1]; r = res[key]
    ratios = [r["rmse"]["single"] / r["rmse"]["robust"],
              r["rmse"]["average"] / r["rmse"]["robust"]]
    titles = ["single pair / robust", "pure average / robust"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    X, Y = np.meshgrid(SNRS, BW)
    for ax, R, t in zip(axes, ratios, titles):
        pc = ax.pcolormesh(X, Y, R, shading="auto", cmap="magma",
                           norm=LogNorm(vmin=1.0, vmax=np.nanmax(R)))
        ax.set_yscale("log"); ax.set_xlabel("SNR (dB)"); ax.set_title(t)
        fig.colorbar(pc, ax=ax, label="RMSE ratio")
    axes[0].set_ylabel("Bandwidth (Hz)")
    fig.suptitle("Improvement of robust fusion over baselines (%d ch)" % key, y=1.02)
    f = os.path.join(embed_dir, "F2_improvement_maps.png")
    fig.savefig(f); plt.close(fig); return f


def fig3_isoprecision(res, counts, BW, SNRS, embed_dir, target=1.0):
    key = counts[-1]; r = res[key]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for est in ["single", "average", "disp", "robust"]:
        need = np.full(len(BW), np.nan)
        for ib in range(len(BW)):
            y = r["rmse"][est][ib]
            if np.nanmin(y) <= target:          # interpolate in log y vs SNR
                ln = np.log10(y)
                need[ib] = np.interp(np.log10(target),
                                     ln[::-1], SNRS[::-1])
        ax.semilogx(BW, need, marker=MARK[est], ms=4, color=COL[est], label=LBL[est])
    # CRB prediction
    need = np.full(len(BW), np.nan)
    for ib in range(len(BW)):
        y = r["crb"][ib]
        if np.nanmin(y) <= target:
            need[ib] = np.interp(np.log10(target), np.log10(y)[::-1], SNRS[::-1])
    ax.semilogx(BW, need, "k--", lw=1.4, label="CRB (GLS)")
    ax.set_xlabel("Bandwidth (Hz)")
    ax.set_ylabel("SNR required for RMSE = %.3g km (dB)" % target)
    ax.set_title("%d channels" % key)
    ax.legend(loc="upper right")
    f = os.path.join(embed_dir, "F3_isoprecision.png")
    fig.savefig(f); plt.close(fig); return f


def fig4_db_gain(res, counts, BW, SNRS, embed_dir, ib=None):
    if ib is None:
        ib = len(BW) // 2
    is_ref = int(np.argmin(np.abs(SNRS - 80.0)))
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for est in ["single", "average", "disp"]:
        gains = [20 * np.log10(res[c]["rmse"][est][ib, is_ref]
                               / res[c]["rmse"]["robust"][ib, is_ref])
                 for c in counts]
        ax.plot(counts, gains, marker=MARK[est], color=COL[est], label=LBL[est])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Number of channels")
    ax.set_ylabel("SNR saving vs robust (dB)")
    ax.set_title("Fusion gain over baselines (B=%.3g Hz, SNR=%.0f dB)"
                 % (BW[ib], SNRS[is_ref]))
    ax.legend()
    f = os.path.join(embed_dir, "F4_db_gain_vs_channels.png")
    fig.savefig(f); plt.close(fig); return f


def fig5_distributions(d, pair_D, kbest, embed_dir, ib=None, is_=None,
                       c=2.5, iqr_factor=1.5):
    BW = d["BANDWIDTHS"]; SNRS = d["SNRS_dB"]; Ztrue = float(d["event_distance_km"])
    if ib is None:
        ib = len(BW) // 2
    if is_ is None:
        is_ = int(np.argmin(np.abs(SNRS - 35.0)))
    Zp = d["pair_Z"][:, ib, is_, :, :]           # (nf, nreal, npairs)
    D2 = pair_D ** 2
    est = combine_estimators(Zp, D2, kbest, c, iqr_factor)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    data = [np.ravel(est[k] - Ztrue) for k in ["single", "average", "disp", "robust"]]
    parts = ax.violinplot(data, showextrema=False, widths=0.85)
    for b, k in zip(parts["bodies"], ["single", "average", "disp", "robust"]):
        b.set_facecolor(COL[k]); b.set_alpha(0.6)
    for i, (k, dd) in enumerate(zip(["single", "average", "disp", "robust"], data)):
        ax.scatter([i + 1], [np.nanmedian(dd)], color="k", zorder=3, s=18)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([LBL[k] for k in ["single", "average", "disp", "robust"]],
                       rotation=15, ha="right")
    ax.set_ylabel("error  Z - Ztrue  (km)")
    ax.set_ylim(-200, 200)
    ax.set_title("Error distribution at threshold (B=%.3g Hz, SNR=%.0f dB)"
                 % (BW[ib], SNRS[is_]))
    f = os.path.join(embed_dir, "F5_error_distributions.png")
    fig.savefig(f); plt.close(fig); return f


def fig6_gap(res, counts, BW, SNRS, embed_dir):
    key = counts[-1]; r = res[key]; ib = len(BW) // 2
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for est in ["single", "average", "disp", "robust"]:
        ax.semilogy(SNRS, r["rmse"][est][ib] / r["crb"][ib],
                    marker=MARK[est], ms=4, color=COL[est], label=LBL[est])
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="CRB (GLS)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RMSE / CRB")
    ax.set_title("Gap to the CRB (%d ch, B=%.3g Hz)" % (key, BW[ib]))
    ax.legend()
    f = os.path.join(embed_dir, "F6_gap_to_crb.png")
    fig.savefig(f); plt.close(fig); return f


# ---------------------------------- main ----------------------------------
def main(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "crb_results_*ch.npz")))
    if not files:
        print("no results in", run_dir); return
    embed_dir = os.path.join(run_dir, "figs_estimators")
    os.makedirs(embed_dir, exist_ok=True)

    counts = []
    res = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "pair_Z" not in d.files:
            print("skip (no pair_Z):", f); continue
        n = int(d["channel_indices_full"].shape[0]); counts.append(n)
        BW = d["BANDWIDTHS"]; SNRS = d["SNRS_dB"]
        Ztrue = float(d["event_distance_km"])
        D = d["D_abs"]; D2 = D ** 2; kbest = int(np.argmax(D))
        tol = 0.01 * Ztrue          # 1% tolerance for success rate
        print("count %2d: computing estimators + GLS bound..." % n)

        est = combine_estimators(d["pair_Z"], D2, kbest)
        surf = {k: metrics(v, Ztrue, tol) for k, v in est.items()}
        crb_gls = gls_bound(d)
        crb_surf = np.nanmean(crb_gls, axis=0)

        res[n] = dict(BW=BW, SNRS=SNRS, toll=tol,
                      rmse={k: surf[k]["rmse"] for k in est},
                      bias={k: surf[k]["bias"] for k in est},
                      std={k: surf[k]["std"] for k in est},
                      mae={k: surf[k]["mae"] for k in est},
                      success={k: surf[k]["success"] for k in est},
                      crb=crb_surf)

        # save derived data (float32 for the per-realisation estimates)
        np.savez_compressed(
            os.path.join(run_dir, "estcomp_%dch.npz" % n),
            BANDWIDTHS=BW, SNRS_dB=SNRS, event_distance_km=Ztrue, tol=tol,
            pair_D_abs=D, kbest=kbest,
            **{("est_" + k): v.astype(np.float32) for k, v in est.items()},
            crb_gls=crb_gls,
            **{("rmse_" + k): surf[k]["rmse"] for k in est},
            **{("bias_" + k): surf[k]["bias"] for k in est},
            **{("std_" + k): surf[k]["std"] for k in est},
            **{("mae_" + k): surf[k]["mae"] for k in est},
            **{("success_" + k): surf[k]["success"] for k in est},
        )

        d.close()

    counts = sorted(counts)
    BW = res[counts[0]]["BW"]; SNRS = res[counts[0]]["SNRS"]
    made = [fig1_money(res, counts, BW, SNRS, embed_dir),
            fig2_ratio_maps(res, counts, BW, SNRS, embed_dir),
            fig3_isoprecision(res, counts, BW, SNRS, embed_dir),
            fig4_db_gain(res, counts, BW, SNRS, embed_dir),
            fig6_gap(res, counts, BW, SNRS, embed_dir)]
    df = np.load(os.path.join(run_dir, "crb_results_%dch.npz" % counts[-1]),
                 allow_pickle=True)
    made.append(fig5_distributions(df, df["D_abs"], int(np.argmax(df["D_abs"])),
                                   embed_dir))
    print("figures:")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_crb_full_pairs")
