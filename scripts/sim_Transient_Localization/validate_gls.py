"""Compare naive vs covariance-aware (GLS) vs star combination on mid data."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import crb

for COUNT in (4, 8, 12):
    d = np.load('results_crb_mid/crb_results_%dch.npz' % COUNT, allow_pickle=True)
    SNRS = d['SNRS_dB']; BW = d['BANDWIDTHS']
    D = d['D_abs']; pairs = list(d['pair_indices']); nf = d['n_fibres']
    Z = d['pair_Z']; cd = d['crb_delay_pairs']
    npairs = len(pairs)

    naive_b = np.full((nf, len(BW), len(SNRS)), np.nan)
    gls_b = np.full_like(naive_b, np.nan)
    star_b = np.full_like(naive_b, np.nan)
    Wg = np.full((nf, len(BW), len(SNRS), npairs), np.nan)
    for f in range(nf):
        for ib in range(len(BW)):
            for is_ in range(len(SNRS)):
                naive_b[f, ib, is_] = crb.combined_crb_distance(cd[f, ib, is_], D)
                g, w = crb.gls_combined_crb_distance(cd[f, ib, is_], D, pairs, COUNT)
                gls_b[f, ib, is_] = g; Wg[f, ib, is_] = w
                star_b[f, ib, is_] = crb.star_combined_crb_distance(
                    cd[f, ib, is_], D, pairs, COUNT)

    gls_est = np.full(Z.shape[:4], np.nan)
    for f in range(nf):
        for ib in range(len(BW)):
            for is_ in range(len(SNRS)):
                w = Wg[f, ib, is_]
                gls_est[f, ib, is_] = Z[f, ib, is_] @ w / w.sum()

    m = lambda a: np.nanmean(a, axis=0)
    sim_naive = m(np.nanstd(d['est_crb'], axis=-1))
    sim_gls = m(np.nanstd(gls_est, axis=-1))
    b_naive = m(naive_b); b_gls = m(gls_b); b_star = m(star_b)

    ib = int(len(BW) * 0.8); is_ = len(SNRS) - 1
    print('%2d ch | last cell  sim_naive/sim_GLS = %.2f   ratios sim/bound: '
          'naive %.2f  GLS %.2f  star %.2f' % (
              COUNT, sim_naive[ib, is_] / sim_gls[ib, is_],
              sim_naive[ib, is_] / b_naive[ib, is_],
              sim_gls[ib, is_] / b_gls[ib, is_],
              sim_naive[ib, is_] / b_star[ib, is_]))

    fig, ax = plt.subplots(figsize=(8.4, 6))
    ax.loglog(SNRS, sim_naive[ib], 'o-', ms=4, label='sim: naive (inverse-variance)')
    ax.loglog(SNRS, sim_gls[ib], 's--', ms=4, label='sim: GLS (covariance)')
    ax.loglog(SNRS, b_naive[ib], 'k--', lw=1, label='bound: naive')
    ax.loglog(SNRS, b_gls[ib], 'k-', lw=1.3, label='bound: GLS')
    ax.loglog(SNRS, b_star[ib], 'k:', lw=1.5, label='bound: star')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Std (km)')
    ax.set_title('%d ch | naive vs GLS vs star (B=%.3g Hz)' % (COUNT, BW[ib]))
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig('results_crb_mid/fig_%dch_gls.png' % COUNT)
    plt.close()
print('done')
