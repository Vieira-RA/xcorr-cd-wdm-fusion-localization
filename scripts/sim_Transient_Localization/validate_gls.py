"""Validate the joint covariance model against saved mid-run pair estimates."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import crb
from snr_conventions import energy_snr_axis_for_run

for COUNT in (4, 8, 12):
    d = np.load('results_crb_mid/crb_results_%dch.npz' % COUNT, allow_pickle=True)
    SNRS = energy_snr_axis_for_run('results_crb_mid', COUNT, d['SNRS_dB'])
    BW = d['BANDWIDTHS']
    D = d['D_abs']; pairs = list(d['pair_indices']); nf = d['n_fibres']
    Z = d['pair_Z']; cd = d['crb_delay_pairs']
    npairs = len(pairs)

    naive_b = np.full((nf, len(BW), len(SNRS)), np.nan)
    gls_b = np.full_like(naive_b, np.nan)
    star_independent = np.full_like(naive_b, np.nan)
    Wg = np.full((nf, len(BW), len(SNRS), npairs), np.nan)
    for f in range(nf):
        for ib in range(len(BW)):
            for is_ in range(len(SNRS)):
                naive_b[f, ib, is_] = crb.combined_crb_distance(cd[f, ib, is_], D)
                g, w = crb.gls_combined_crb_distance(cd[f, ib, is_], D, pairs, COUNT)
                gls_b[f, ib, is_] = g; Wg[f, ib, is_] = w
                star_independent[f, ib, is_] = crb.star_combined_crb_distance(
                    cd[f, ib, is_], D, pairs, COUNT)

    gls_est = np.full(Z.shape[:4], np.nan)
    for f in range(nf):
        for ib in range(len(BW)):
            for is_ in range(len(SNRS)):
                w = Wg[f, ib, is_]
                if not np.all(np.isfinite(w)) or not np.isfinite(w.sum()) or w.sum() == 0:
                    continue
                gls_est[f, ib, is_] = Z[f, ib, is_] @ w / w.sum()

    def safe_nanmean(a, axis=0):
        a = np.asarray(a, dtype=float)
        count = np.sum(np.isfinite(a), axis=axis)
        total = np.nansum(a, axis=axis)
        return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float),
                         where=count > 0)

    def safe_nanstd(a, axis=-1):
        a = np.asarray(a, dtype=float)
        count = np.sum(np.isfinite(a), axis=axis)
        mean = safe_nanmean(a, axis=axis)
        centered = a - np.expand_dims(mean, axis=axis)
        ss = np.nansum(centered ** 2, axis=axis)
        var = np.divide(ss, count - 1,
                        out=np.full_like(ss, np.nan, dtype=float), where=count > 1)
        return np.sqrt(np.maximum(var, 0.0))

    m = lambda a: safe_nanmean(a, axis=0)
    sim_naive = m(safe_nanstd(d['est_crb'], axis=-1))
    sim_gls = m(safe_nanstd(gls_est, axis=-1))
    b_naive = m(naive_b); b_joint = m(gls_b); b_star = m(star_independent)

    ib = int(len(BW) * 0.8); is_ = len(SNRS) - 1
    print('%2d ch | last cell  sim_naive/sim_GLS = %.2f   ratios sim/reference: '
          'naive %.2f  joint %.2f  legacy-independent-star %.2f' % (
              COUNT, sim_naive[ib, is_] / sim_gls[ib, is_],
              sim_naive[ib, is_] / b_naive[ib, is_],
              sim_gls[ib, is_] / b_joint[ib, is_],
              sim_naive[ib, is_] / b_star[ib, is_]))

    fig, ax = plt.subplots(figsize=(8.4, 6))
    ax.loglog(SNRS, sim_naive[ib], 'o-', ms=4, label='sim: naive (inverse-variance)')
    ax.loglog(SNRS, sim_gls[ib], 's--', ms=4,
              label='sim: joint covariance GLS')
    ax.loglog(SNRS, b_naive[ib], 'k--', lw=1,
              label='independent-pair reference (not a bound)')
    ax.loglog(SNRS, b_joint[ib], 'k-', lw=1.3,
              label='additive timing-model reference')
    ax.loglog(SNRS, b_star[ib], 'k:', lw=1.5,
              label='legacy independent-edge star (not a bound)')
    ax.set_xlabel('Energy SNR (dB)')
    ax.set_ylabel('Std (km)')
    ax.set_title('%d ch | naive vs joint model (nominal B=%.3g Hz)' %
                 (COUNT, BW[ib]))
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig('results_crb_mid/fig_%dch_gls.png' % COUNT)
    plt.close()
print('done')
