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
    "font.size": 15, "font.family": "DejaVu Sans",
    "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.linewidth": 1.0, "lines.linewidth": 2.4, "legend.frameon": False,
    "lines.markersize": 8,
})
COL = {"single_worst": "#7F7F7F", "single": "#D55E00", "average": "#0072B2",
       "disp": "#009E73", "robust": "#CC79A7"}
LBL = {"single_worst": "single worst pair", "single": "single best pair",
       "average": "pure average", "disp": r"$D^2$-weighted",
       "robust": r"$D^2$ + bisquare (robust)"}
MARK = {"single_worst": "x", "single": "v", "average": "o", "disp": "s",
        "robust": "D"}
ESTS = ["single_worst", "single", "average", "disp", "robust"]
PLOT_ESTS = ["average", "disp", "robust"]      # curves shown in figures
MSS = {"average": 9, "disp": 14, "robust": 8}  # per-estimator marker sizes

TARGET_RMSE = 5.0          # km, used by the design figures (F3, F4)


def snr_for_target(rmse_row, SNRS, target):
    """First SNR (dB) at which RMSE crosses `target` (interpolated in log)."""
    y = np.asarray(rmse_row, dtype=float)
    ok = np.isfinite(y)
    if not np.any(ok) or np.nanmin(y) > target:
        return np.nan
    idx = np.where(ok & (y <= target))[0]
    j = int(idx[0])
    if j == 0:
        return float(SNRS[0])
    y0, y1 = np.log10(y[j - 1]), np.log10(y[j])
    lt = np.log10(target)
    return float(SNRS[j - 1] + (SNRS[j] - SNRS[j - 1]) * (lt - y0) / (y1 - y0))


# ------------------------------ estimators --------------------------------
def combine_estimators(Zp, D2, kbest, kworst=None, c=2.5, iqr_factor=1.5):
    """Zp: (..., npairs) distance estimates. Returns dict of (..) arrays."""
    if kworst is None:
        kworst = int(np.argmin(D2))
    out = {}
    out["single_worst"] = Zp[..., kworst]
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
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    for ax, ib in zip(axes, bw_idx):
        for est in PLOT_ESTS:
            mfc = "none" if est == "disp" else COL[est]
            ax.semilogy(SNRS, r["rmse"][est][ib], marker=MARK[est], ms=MSS[est],
                        color=COL[est], markerfacecolor=mfc, label=LBL[est])
        ax.semilogy(SNRS, r["crb"][ib], "k--", lw=1.6, label="CRB (GLS)")
        ax.set_title(r"$B = %.3g$ kHz" % (BW[ib] / 1e3))
        ax.set_xlabel("SNR (dB)")
        ax.set_ylim(bottom=1e-3)
    axes[0].set_ylabel("RMSE (km)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Estimation error vs SNR  (%d channels)" % key, y=1.02)
    f = os.path.join(embed_dir, "F1_rmse_vs_snr.png")
    fig.savefig(f); plt.close(fig); return f


def fig2_ratio_maps(res, counts, BW, SNRS, embed_dir):
    key = counts[-1]; r = res[key]
    dB = 20.0 * np.log10(r["rmse"]["average"] / r["rmse"]["robust"])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15.5, 5.2),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    X, Y = np.meshgrid(SNRS, BW)
    pc = ax0.pcolormesh(X, Y, dB, shading="auto", cmap="magma",
                        vmin=0, vmax=np.nanmax(dB))
    ax0.set_yscale("log"); ax0.set_xlabel("SNR (dB)")
    ax0.set_ylabel("Bandwidth (Hz)")
    ax0.set_title("pure average: improvement over robust")
    fig.colorbar(pc, ax=ax0, label="improvement (dB)")

    # diagnostic: improvement at the highest SNR, versus bandwidth
    is_hi = len(SNRS) - 1
    ax1.semilogx(BW, dB[:, is_hi], "o-", color=COL["disp"])
    ax1.set_xlabel("Bandwidth (Hz)")
    ax1.set_ylabel("improvement at SNR = %.0f dB (dB)" % SNRS[is_hi])
    ax1.set_title("high-SNR diagnostic")
    fig.suptitle("Robust vs pure averaging, %d ch" % key, y=1.02)
    f = os.path.join(embed_dir, "F2_improvement_maps.png")
    fig.savefig(f); plt.close(fig); return f


def fig3_snr_saving(res, counts, BW, SNRS, embed_dir, target=TARGET_RMSE):
    key = counts[-1]; r = res[key]
    navg = np.array([snr_for_target(r["rmse"]["average"][ib], SNRS, target)
                     for ib in range(len(BW))])
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for est in ["disp", "robust"]:
        n = np.array([snr_for_target(r["rmse"][est][ib], SNRS, target)
                      for ib in range(len(BW))])
        d = navg - n
        mfc = "none" if est == "disp" else COL[est]
        ax.semilogx(BW, d, marker=MARK[est], ms=MSS[est], color=COL[est],
                    markerfacecolor=mfc,
                    label="%s vs average (mean %.1f dB)" % (LBL[est], np.nanmean(d)))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Bandwidth (Hz)")
    ax.set_ylabel("SNR saved over pure averaging (dB)")
    ax.set_title("%d ch | SNR saving at RMSE = %.3g km" % (key, target))
    ax.legend(loc="best")
    f = os.path.join(embed_dir, "F3_isoprecision.png")
    fig.savefig(f); plt.close(fig); return f


def fig4_snr_vs_channels(res, counts, BW, SNRS, embed_dir, ib=None,
                         target=TARGET_RMSE):
    if ib is None:
        ib = len(BW) // 2
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for est in PLOT_ESTS:
        y = [snr_for_target(res[c]["rmse"][est][ib], SNRS, target) for c in counts]
        mfc = "none" if est == "disp" else COL[est]
        ax.plot(counts, y, marker=MARK[est], ms=MSS[est], color=COL[est],
                markerfacecolor=mfc, label=LBL[est])
    y = [snr_for_target(res[c]["crb"][ib], SNRS, target) for c in counts]
    ax.plot(counts, y, "k--", lw=1.6, label="CRB (GLS)")
    ax.set_xlabel("Number of channels")
    ax.set_ylabel("SNR for RMSE = %.3g km (dB), lower = better" % target)
    ax.set_title("Design curve (B = %.3g kHz)" % (BW[ib] / 1e3))
    ax.legend()
    ax.text(0.02, 0.02, "curve absent = target not reached within SNR grid",
            transform=ax.transAxes, fontsize=12, color="0.3")
    f = os.path.join(embed_dir, "F4_db_gain_vs_channels.png")
    fig.savefig(f); plt.close(fig); return f


def fig5_distributions(f5, embed_dir, ib=None, is_=None):
    BW = f5["BW"]; SNRS = f5["SNRS"]; Ztrue = f5["Ztrue"]; est = f5["est"]
    if ib is None:
        ib = len(BW) // 2
    if is_ is None:
        is_ = int(np.argmin(np.abs(SNRS - 50.0)))
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for k in PLOT_ESTS:
        e = np.clip(np.sort(np.abs(np.ravel(est[k][:, ib, is_, :] - Ztrue))),
                    1e-6, None)
        y = np.arange(1, len(e) + 1) / len(e)
        ax.loglog(e, y, color=COL[k], label=LBL[k])
    for frac, ls, lab in [(0.01, ":", "1%"), (0.05, "--", "5%")]:
        ax.axvline(frac * Ztrue, color="k", ls=ls, lw=1.0)
        ax.text(frac * Ztrue * 1.08, 0.4, "%s tolerance" % lab, rotation=90,
                va="center", ha="left", fontsize=12, color="0.3")
    ax.set_xlabel("|error| (km)")
    ax.set_ylabel(r"fraction of estimates $\leq x$")
    ax.set_title("%d ch | ECDF of error (B = %.3g kHz, SNR = %.0f dB)"
                 % (f5["n"], BW[ib] / 1e3, SNRS[is_]))
    ax.legend(loc="lower right")
    f = os.path.join(embed_dir, "F5_error_distributions.png")
    fig.savefig(f); plt.close(fig); return f


def fig6_gap(res, counts, BW, SNRS, embed_dir):
    key = counts[-1]; r = res[key]; ib = len(BW) // 2
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for est in PLOT_ESTS:
        mfc = "none" if est == "disp" else COL[est]
        axes[0].semilogy(SNRS, r["std"][est][ib] / r["crb"][ib],
                         marker=MARK[est], ms=MSS[est], color=COL[est],
                         markerfacecolor=mfc, label=LBL[est])
        axes[1].semilogy(SNRS, np.abs(r["bias"][est][ib]),
                         marker=MARK[est], ms=MSS[est], color=COL[est],
                         markerfacecolor=mfc, label=LBL[est])
    axes[0].axhline(1.0, color="k", ls="--", lw=1.4, label=r"CRB ($=10^0$)")
    axes[0].set_ylabel("std / CRB")
    axes[0].set_title("Variance efficiency (%d ch, B = %.3g kHz)"
                      % (key, BW[ib] / 1e3))
    axes[1].set_ylabel("|bias| (km)")
    axes[1].set_title("Bias (%d ch)" % key)
    for a in axes:
        a.set_xlabel("SNR (dB)")
    axes[0].legend()
    f = os.path.join(embed_dir, "F6_gap_to_crb.png")
    fig.savefig(f); plt.close(fig); return f


# ---------------------------------- main ----------------------------------
def _load_res(run_dir, replot):
    """Return (res, counts, f5). replot=True loads estcomp_*.npz (fast);
    replot=False recomputes everything from the per-pair crb_results_*.npz."""
    res, counts, f5_all = {}, [], {}

    if replot:
        files = sorted(glob.glob(os.path.join(run_dir, "estcomp_*ch.npz")))
        for f in files:
            d = np.load(f, allow_pickle=True)
            n = int(os.path.basename(f).rsplit("_", 1)[1].replace("ch.npz", ""))
            counts.append(n)
            res[n] = dict(
                BW=d["BANDWIDTHS"], SNRS=d["SNRS_dB"],
                rmse={k: d["rmse_" + k] for k in ESTS},
                bias={k: d["bias_" + k] for k in ESTS},
                std={k: d["std_" + k] for k in ESTS},
                mae={k: d["mae_" + k] for k in ESTS},
                success={k: d["success_" + k] for k in ESTS},
                crb=np.nanmean(d["crb_gls"], axis=0))
            f5_all[n] = dict(est={k: d["est_" + k] for k in ESTS},
                             D=d["pair_D_abs"], kbest=int(d["kbest"]),
                             Ztrue=float(d["event_distance_km"]),
                             BW=d["BANDWIDTHS"], SNRS=d["SNRS_dB"], n=n)
        return res, sorted(counts), (f5_all[max(counts)] if counts else None)

    files = sorted(glob.glob(os.path.join(run_dir, "crb_results_*ch.npz")))
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "pair_Z" not in d.files:
            print("skip (no pair_Z):", f); continue
        n = int(d["channel_indices_full"].shape[0]); counts.append(n)
        BW = d["BANDWIDTHS"]; SNRS = d["SNRS_dB"]
        Ztrue = float(d["event_distance_km"])
        D = d["D_abs"]; D2 = D ** 2; kbest = int(np.argmax(D))
        tol = 0.01 * Ztrue
        print("count %2d: computing estimators + GLS bound..." % n)

        est = combine_estimators(d["pair_Z"], D2, kbest)
        surf = {k: metrics(v, Ztrue, tol) for k, v in est.items()}
        crb_gls = gls_bound(d)

        res[n] = dict(BW=BW, SNRS=SNRS, toll=tol,
                      rmse={k: surf[k]["rmse"] for k in est},
                      bias={k: surf[k]["bias"] for k in est},
                      std={k: surf[k]["std"] for k in est},
                      mae={k: surf[k]["mae"] for k in est},
                      success={k: surf[k]["success"] for k in est},
                      crb=np.nanmean(crb_gls, axis=0))
        f5_all[n] = dict(est=est, D=D, kbest=kbest, Ztrue=Ztrue, BW=BW,
                         SNRS=SNRS, n=n)

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
    return res, counts, (f5_all[max(counts)] if counts else None)


def main(run_dir, replot=False):
    embed_dir = os.path.join(run_dir, "figs_estimators")
    os.makedirs(embed_dir, exist_ok=True)
    res, counts, f5 = _load_res(run_dir, replot)
    if not counts:
        print("nothing to plot in", run_dir, "(--replot needs estcomp_*.npz)"); return
    print(("replotting from estcomp files: " if replot else "computed counts: ")
          + str(counts))
    BW = res[counts[0]]["BW"]; SNRS = res[counts[0]]["SNRS"]
    made = [fig1_money(res, counts, BW, SNRS, embed_dir),
            fig2_ratio_maps(res, counts, BW, SNRS, embed_dir),
            fig3_snr_saving(res, counts, BW, SNRS, embed_dir),
            fig4_snr_vs_channels(res, counts, BW, SNRS, embed_dir),
            fig6_gap(res, counts, BW, SNRS, embed_dir)]
    if f5 is not None:
        made.append(fig5_distributions(f5, embed_dir))
    print("figures:")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    _args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(_args[0] if _args else "results_crb_full_pairs",
         replot=("--replot" in sys.argv))
