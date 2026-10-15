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
       "disp": "#009E73", "robust": "#CC79A7",
       "average_pool": "#0072B2", "disp_pool": "#009E73",
       "robust_pool": "#CC79A7"}
LBL = {"single_worst": "single worst pair", "single": "single best pair",
       "average": "pure average", "disp": r"$D^2$-weighted",
       "robust": r"$D^2$ + bisquare (robust)",
       "average_pool": "average (filtered pool)",
       "disp_pool": r"$D^2$-weighted (filtered pool)",
       "robust_pool": r"robust (filtered pool)"}
MARK = {"single_worst": "x", "single": "v", "average": "o", "disp": "s",
        "robust": "D",
        "average_pool": "o", "disp_pool": "s", "robust_pool": "D"}
ESTS = ["single_worst", "single", "average", "disp", "robust"]
POOL_ESTS = ["average_pool", "disp_pool", "robust_pool"]
PLOT_ESTS = ["average_pool", "disp_pool", "robust_pool"]   # filtered pool only
MSS = {"average": 9, "disp": 14, "robust": 8,
       "average_pool": 9, "disp_pool": 14, "robust_pool": 8}
POOL_ALPHA = 0.25   # keep pairs with D >= POOL_ALPHA * max(D)
ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70]


def pool_pair_estimators(Zp, D2, alpha, c=2.5, iqr_factor=1.5):
    """Naive and D^2-weighted estimates over the pool D >= alpha*max(D)."""
    m = np.ones(D2.shape, dtype=bool) if alpha <= 0 else (D2 >= alpha * np.max(D2))
    Zpp = Zp[..., m]
    D2p = D2[m]
    avg = np.nanmean(Zpp, axis=-1)
    disp = (np.nansum(D2p * Zpp, axis=-1)
            / np.nansum(np.broadcast_to(D2p, Zpp.shape), axis=-1))
    return avg, disp

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

    # ---- pool-filtered variants: drop uninformative pairs (D >= a*max D) ----
    m = D2 >= (POOL_ALPHA * np.max(D2))
    Zpp = Zp[..., m]
    D2p = D2[m]
    out["average_pool"] = np.nanmean(Zpp, axis=-1)
    out["disp_pool"] = (np.nansum(D2p * Zpp, axis=-1)
                        / np.nansum(np.broadcast_to(D2p, Zpp.shape), axis=-1))
    med_p = np.nanmedian(Zpp, axis=-1, keepdims=True)
    q1p = np.nanpercentile(Zpp, 25, axis=-1, keepdims=True)
    q3p = np.nanpercentile(Zpp, 75, axis=-1, keepdims=True)
    sp = iqr_factor * (q3p - q1p)
    rp = np.abs(Zpp - med_p)
    tp = rp / (c * np.where(sp > 0, sp, 1.0))
    wbp = np.where(sp > 0, np.where(tp < 1, (1.0 - tp ** 2) ** 2, 0.0), 1.0)
    wp = D2p * wbp
    nump = np.nansum(wp * Zpp, axis=-1)
    denp = np.nansum(wp, axis=-1)
    out["robust_pool"] = np.where(denp > 0, nump / denp, np.nan)
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


def crb_bound(d):
    """Achievable combined CRB via the *star* combination (robust: the GLS
    u-fit is unreliable on wide-D-spread grids, where it can exceed the star
    bound, which is impossible)."""
    cd = d["crb_delay_pairs"]; D = d["D_abs"]
    pairs = list(d["pair_indices"]); nch = int(d["channel_indices_full"].shape[0])
    nf, nbw, nsnr, _ = cd.shape
    out = np.full((nf, nbw, nsnr), np.nan)
    for f in range(nf):
        for ib in range(nbw):
            for is_ in range(nsnr):
                out[f, ib, is_] = crb.star_combined_crb_distance(
                    cd[f, ib, is_], D, pairs, nch)
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
    dB = 20.0 * np.log10(r["rmse"]["average_pool"] / r["rmse"]["robust_pool"])
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


def fig3_snr_saving(res, counts, BW, SNRS, embed_dir, target=TARGET_RMSE,
                    est="disp_pool"):
    use = list(counts)
    cmap = plt.cm.viridis(np.linspace(0.0, 0.85, len(use)))
    mk = ["o", "s", "^", "v", "D", "P", "X", "*"]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for ci, nch in enumerate(use):
        r = res[nch]
        navg = np.array([snr_for_target(r["rmse"]["average_pool"][ib], SNRS, target)
                         for ib in range(len(BW))])
        nn = np.array([snr_for_target(r["rmse"][est][ib], SNRS, target)
                       for ib in range(len(BW))])
        d = navg - nn
        ax.semilogx(BW, d, marker=mk[ci % len(mk)], ms=10, color=cmap[ci],
                    label="%d ch" % nch)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Bandwidth (Hz)")
    ax.set_ylabel("SNR saved over pure averaging (dB)")
    ax.set_title("Weighting benefit: %s vs pure averaging (RMSE = %.3g km)"
                 % (LBL.get(est, est), target))
    ax.legend(title="channels")
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
        axes[0].semilogy(SNRS, r["rmse"][est][ib] / r["crb"][ib],
                         marker=MARK[est], ms=MSS[est], color=COL[est],
                         markerfacecolor=mfc, label=LBL[est] + " (RMSE)")
        axes[0].semilogy(SNRS, r["std"][est][ib] / r["crb"][ib],
                         ls=":", color=COL[est], lw=1.5)
        axes[1].semilogy(SNRS, np.abs(r["bias"][est][ib]),
                         marker=MARK[est], ms=MSS[est], color=COL[est],
                         markerfacecolor=mfc, label=LBL[est])
    axes[0].axhline(1.0, color="k", ls="--", lw=1.4, label="CRB")
    axes[0].set_ylabel("RMSE / CRB (solid);   std / CRB (dotted)")
    axes[0].set_title("Efficiency incl. bias (%d ch, B = %.3g kHz)"
                      % (key, BW[ib] / 1e3))
    axes[1].set_ylabel("|bias| (km)")
    axes[1].set_title("Bias (%d ch)" % key)
    for a in axes:
        a.set_xlabel("SNR (dB)")
    axes[0].legend(fontsize=11)
    f = os.path.join(embed_dir, "F6_gap_to_crb.png")
    fig.savefig(f); plt.close(fig); return f


def fig7_pool(res, counts, BW, SNRS, embed_dir, ib=None, target=TARGET_RMSE):
    key = counts[-1]; r = res[key]
    if ib is None:
        ib = len(BW) // 2
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15.5, 5.4))

    ax0.semilogy(SNRS, r["rmse"]["average"][ib], "o-", color=COL["average"],
                 label="average (all pairs)")
    ax0.semilogy(SNRS, r["rmse"]["average_pool"][ib], "s--", color="#E69F00",
                 label="average (filtered pool)")
    ax0.semilogy(SNRS, r["rmse"]["disp_pool"][ib], "D-", color=COL["disp"],
                 markerfacecolor="none", label="$D^2$ (filtered pool)")
    ax0.semilogy(SNRS, r["rmse"]["robust_pool"][ib], "v--", color=COL["robust"],
                 label="robust (filtered pool)")
    ax0.semilogy(SNRS, r["crb"][ib], "k--", lw=1.6, label="CRB (all pairs)")
    ax0.set_xlabel("SNR (dB)"); ax0.set_ylabel("RMSE (km)")
    ax0.set_title("%d ch, B = %.3g kHz" % (key, BW[ib] / 1e3))
    ax0.legend()

    def sav(rr, base, w):
        nb = np.array([snr_for_target(rr["rmse"][base][b], SNRS, target)
                       for b in range(len(BW))])
        nw = np.array([snr_for_target(rr["rmse"][w][b], SNRS, target)
                       for b in range(len(BW))])
        return np.nanmean(nb - nw)
    s_all = [sav(res[c], "average", "disp") for c in counts]
    s_pool = [sav(res[c], "average_pool", "disp_pool") for c in counts]
    ax1.plot(counts, s_all, "o-", color=COL["disp"], label="all pairs")
    ax1.plot(counts, s_pool, "s--", color="#E69F00", label="filtered pool")
    ax1.set_xlabel("Number of channels")
    ax1.set_ylabel(r"SNR saving of $D^2$ over averaging (dB)")
    ax1.set_title("Residual value of weighting (RMSE = %.3g km)" % target)
    ax1.legend()
    fig.suptitle("Effect of removing uninformative pairs (D >= %.2f max D)"
                 % POOL_ALPHA, y=1.02)
    f = os.path.join(embed_dir, "F7_pool_effect.png")
    fig.savefig(f); plt.close(fig); return f


def fig8_alpha_sweep(res, counts, BW, SNRS, embed_dir, target=TARGET_RMSE):
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    cmap = plt.cm.viridis(np.linspace(0.0, 0.85, len(counts)))
    mk = ["o", "s", "^", "v", "D"]
    for ci, c in enumerate(counts):
        if "alpha_saving" not in res[c]:
            continue
        s = np.nanmean(res[c]["alpha_saving"], axis=1)   # (n_alpha,)
        ax.plot(ALPHAS, s, marker=mk[ci % len(mk)], color=cmap[ci], label="%d ch" % c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"filter threshold $\alpha$   ($D \geq \alpha\,\max D$;  0 = all pairs)")
    ax.set_ylabel(r"SNR saving of $D^2$ over averaging (dB)")
    ax.set_title("Weighting benefit vs pair-selection threshold (RMSE = %.3g km)" % target)
    ax.legend(title="channels")
    f = os.path.join(embed_dir, "F8_alpha_sweep.png")
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
            keys = [k for k in (ESTS + POOL_ESTS) if ("rmse_" + k) in d.files]
            res[n] = dict(
                BW=d["BANDWIDTHS"], SNRS=d["SNRS_dB"],
                rmse={k: d["rmse_" + k] for k in keys},
                bias={k: d["bias_" + k] for k in keys},
                std={k: d["std_" + k] for k in keys},
                mae={k: d["mae_" + k] for k in keys},
                success={k: d["success_" + k] for k in keys},
                crb=np.nanmean(d["crb_star"] if "crb_star" in d.files
                               else d["crb_gls"], axis=0))
            if "alpha_saving" in d.files:
                res[n]["alpha_saving"] = d["alpha_saving"]
            est_keys = [k for k in (ESTS + POOL_ESTS) if ("est_" + k) in d.files]
            f5_all[n] = dict(est={k: d["est_" + k] for k in est_keys},
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
        crb_star = crb_bound(d)

        # ---- alpha sweep: residual weighting gain vs filter threshold ----
        alpha_saving = np.full((len(ALPHAS), len(BW)), np.nan)
        for ai, al in enumerate(ALPHAS):
            avg_a, disp_a = pool_pair_estimators(d["pair_Z"], D2, al)
            rmse_avg = metrics(avg_a, Ztrue, tol)["rmse"]
            rmse_disp = metrics(disp_a, Ztrue, tol)["rmse"]
            for ib in range(len(BW)):
                alpha_saving[ai, ib] = (
                    snr_for_target(rmse_avg[ib], SNRS, TARGET_RMSE)
                    - snr_for_target(rmse_disp[ib], SNRS, TARGET_RMSE))

        res[n] = dict(BW=BW, SNRS=SNRS, toll=tol,
                      rmse={k: surf[k]["rmse"] for k in est},
                      bias={k: surf[k]["bias"] for k in est},
                      std={k: surf[k]["std"] for k in est},
                      mae={k: surf[k]["mae"] for k in est},
                      success={k: surf[k]["success"] for k in est},
                      crb=np.nanmean(crb_star, axis=0),
                      alpha_saving=alpha_saving)
        f5_all[n] = dict(est=est, D=D, kbest=kbest, Ztrue=Ztrue, BW=BW,
                         SNRS=SNRS, n=n)

        np.savez_compressed(
            os.path.join(run_dir, "estcomp_%dch.npz" % n),
            BANDWIDTHS=BW, SNRS_dB=SNRS, event_distance_km=Ztrue, tol=tol,
            pair_D_abs=D, kbest=kbest, alphas=np.array(ALPHAS),
            alpha_saving=alpha_saving,
            **{("est_" + k): v.astype(np.float32) for k, v in est.items()},
            crb_star=crb_star,
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
    if "disp_pool" in res[counts[-1]]["rmse"]:
        made.append(fig7_pool(res, counts, BW, SNRS, embed_dir))
    if "alpha_saving" in res[counts[-1]]:
        made.append(fig8_alpha_sweep(res, counts, BW, SNRS, embed_dir))
    print("figures:")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    _args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(_args[0] if _args else "results_crb_full_pairs",
         replot=("--replot" in sys.argv))
