"""
Cramer-Rao lower bound (CRB) definitions for time-delay estimation, following
"CRB for Time-Delay Estimation - Simple Summary".

Unit / convention notes
-----------------------
Signals are discrete real sequences x[n], n = 0..N-1, sampled with spacing dt.

We adopt the doc's two-sided noise PSD N0 and derive its discrete counterpart:

    E   = int s^2 dt          ~  dt * sum_n x[n]^2
    N0  = sigma^2 * dt        (white discrete noise, band-limited to Nyquist)
    SNR = E / N0              =  sum_n x[n]^2 / sigma^2          (dt cancels!)

Therefore, to realise a target energy SNR we set

    sigma^2 = sum_n x[n]^2 / SNR_lin      (per component).

Mean-square (RMS) bandwidth:

    beta^2 = int (ds/dt)^2 dt / int s^2 dt
           = sum_k (2 pi f_k)^2 |X_k|^2 / sum_k |X_k|^2

Deterministic bound (doc section 2, one component):

    CRB_det(tau) = 1 / (SNR * beta^2)

For C independent components with common beta and mean per-component energy
SNR SNR_E, this becomes 1 / (C * SNR_E * beta^2).

Two-sensor effective SNR (doc section 3.2/3.3), generalised to two possibly
different signals via their coherent cross-spectrum:

    S(f)        = P_coh(f) = | sum_c X_i,c(f) conj(X_j,c(f)) |
    rho_i(f)    = S(f) / N_i(f),   N_i(f) = n_comp * N * sigma_i^2 (per-bin)
    SNR_eff(f)  = (S/N1)(S/N2) / (1 + S/N1 + S/N2)
     J_CR         = (T/2pi) * int w^2 SNR_eff(w) dw
                  = a sum over the full two-sided spectrum

    CRB_eff(tau) = 1 / J_CR

For identical signals this reduces exactly to the doc's formula. At high SNR
SNR_eff -> rho/2, so CRB_eff -> 2 * CRB_det (the stochastic two-sensor bound is
a factor 2 looser than the known-template deterministic bound).
"""

import numpy as np
from scipy.optimize import nnls

ENERGY_SNR_DEFINITION = "mean_per_component_energy_over_noise_variance"


def energy_snr_offset_db(n_samples, n_components=2):
    """Convert the legacy sweep axis to mean per-component energy SNR.

    The legacy sweep used ``sigma2 = mean_t(sum_c x_c[t]**2) / SNR``.  The
    resulting mean per-component energy SNR is ``(n_samples/n_components)``
    times that sweep value.  This is for relabelling saved runs; new simulation
    code should set the noise variance from :func:`noise_variance_for_snr`.
    """
    if n_samples <= 0 or n_components <= 0:
        raise ValueError("n_samples and n_components must be positive")
    return 10.0 * np.log10(float(n_samples) / float(n_components))


def _one_sided_weights(n_samples):
    """Weights converting an rFFT sum into a two-sided spectral sum."""
    weights = np.full(n_samples // 2 + 1, 2.0)
    weights[0] = 1.0
    if n_samples % 2 == 0:
        weights[-1] = 1.0
    return weights


# ---------------------------------------------------------------------------
# basic signal quantities
# ---------------------------------------------------------------------------
def _as_2d(sig):
    sig = np.asarray(sig, dtype=float)
    if sig.ndim == 1:
        sig = sig[:, None]
    return sig


def signal_energy(sig):
    """sum_n x[n]^2 per component. Shape (n_comp,) for a (N, n_comp) input."""
    sig = _as_2d(sig)
    return np.sum(sig ** 2, axis=0)


def noise_variance_for_snr(sig, snr_db):
    """
    Per-component noise variance sigma^2 achieving energy SNR = E / N0.

    Uses the average per-component energy so both components share the same
    sigma^2 (average per-component SNR equals snr_db).
    """
    E = signal_energy(sig)
    snr_lin = 10.0 ** (np.asarray(snr_db, dtype=float) / 10.0)
    return np.mean(E) / snr_lin


def per_component_snr(sig, noise_var):
    """Average per-component energy SNR of a signal for a given sigma^2."""
    E = signal_energy(sig)
    return np.mean(E) / noise_var


# ---------------------------------------------------------------------------
# spectra and bandwidth
# ---------------------------------------------------------------------------
def power_spectrum(sig, dt):
    """
    One-sided power spectrum summed over components.

    Returns
    -------
    f : (M,) frequencies [Hz] from np.fft.rfftfreq
    P : (M,) sum_c |X_c(f)|^2  (DC removed before the transform)
    """
    sig = _as_2d(sig)
    sig_ac = sig - sig.mean(axis=0, keepdims=True)
    X = np.fft.rfft(sig_ac, axis=0)
    P = np.sum(np.abs(X) ** 2, axis=1)
    f = np.fft.rfftfreq(sig.shape[0], d=dt)
    return f, P


def coherent_spectrum(sig_i, sig_j, dt):
    """
    Coherent (cross) spectrum magnitude summed over components:

        P_coh(f) = | sum_c X_i,c(f) conj(X_j,c(f)) |

    For identical signals this equals the auto-spectrum of one of them.
    """
    sig_i = _as_2d(sig_i)
    sig_j = _as_2d(sig_j)
    Xi = np.fft.rfft(sig_i - sig_i.mean(axis=0, keepdims=True), axis=0)
    Xj = np.fft.rfft(sig_j - sig_j.mean(axis=0, keepdims=True), axis=0)
    C = np.sum(Xi * np.conj(Xj), axis=1)
    return np.abs(C)


def coherent_bandwidth(sig_i, sig_j, dt):
    """
    Mean-square bandwidth of the coherent (common) signal:

        P_coh(f) = | sum_c X_i,c conj(X_j,c) |
        beta^2   = sum_k w_k^2 P_coh[k] / sum_k P_coh[k]
    """
    sig_i = _as_2d(sig_i)
    f = np.fft.rfftfreq(sig_i.shape[0], d=dt)
    w = 2.0 * np.pi * f
    P = coherent_spectrum(sig_i, sig_j, dt)
    one_sided = _one_sided_weights(sig_i.shape[0])
    return np.sum(one_sided * w ** 2 * P) / np.sum(one_sided * P)


def mean_square_bandwidth(sig, dt):
    """beta^2 = sum_k w_k^2 |X_k|^2 / sum_k |X_k|^2  [rad^2/s^2]."""
    f, P = power_spectrum(sig, dt)
    w = 2.0 * np.pi * f
    one_sided = _one_sided_weights(_as_2d(sig).shape[0])
    return np.sum(one_sided * w ** 2 * P) / np.sum(one_sided * P)


# ---------------------------------------------------------------------------
# deterministic bound (doc section 2)
# ---------------------------------------------------------------------------
def crb_delay_deterministic(sig, dt, noise_var):
    """
    Single-sensor deterministic bound (doc section 2), summed over components:

        J_det = sum_c E_c * beta_c^2 / sigma^2,   CRB = 1 / J_det

    Each independent component contributes its own information.
    """
    return 1.0 / fisher_information_deterministic(sig, dt, noise_var)


def fisher_information_deterministic(sig, dt, noise_var):
    """J_det = sum_c E_c beta_c^2 / sigma^2  [1/s^2]."""
    sig = _as_2d(sig)
    f = np.fft.rfftfreq(sig.shape[0], d=dt)
    w = 2.0 * np.pi * f
    one_sided = _one_sided_weights(sig.shape[0])
    X = np.fft.rfft(sig - sig.mean(axis=0, keepdims=True), axis=0)
    J = 0.0
    for c in range(sig.shape[1]):
        P = np.abs(X[:, c]) ** 2
        x_ac = sig[:, c] - np.mean(sig[:, c])
        E = np.sum(x_ac ** 2)
        beta2 = np.sum(one_sided * w ** 2 * P) / np.sum(one_sided * P)
        J += E * beta2 / noise_var
    return J


def crb_delay_deterministic_pair(sig_i, sig_j, dt, noise_var):
    """
    Deterministic (single-signal, doc section 2) reference bound for a channel
    pair: the mean of the two single-sensor deterministic Fisher informations.
    At high SNR this sits a factor 2 below the two-sensor effective bound
    (the reference-channel noise penalty).
    """
    J = 0.5 * (fisher_information_deterministic(sig_i, dt, noise_var)
               + fisher_information_deterministic(sig_j, dt, noise_var))
    return 1.0 / J


# ---------------------------------------------------------------------------
# two-sensor effective-SNR bound (doc section 3.2/3.3)
# ---------------------------------------------------------------------------
def effective_snr_spectrum(P_coh_c, noise_var, n_samples, n_comp=1):
    """
    Per-bin effective SNR for a single component of two sensors with equal
    per-component noise.

        rho       = P_coh / (N sigma^2)
        SNR_eff   = rho^2 / (1 + 2 rho)

    P_coh_c is the coherent cross-spectrum PSD of ONE component.
    """
    rho = P_coh_c / (n_samples * noise_var)
    return rho ** 2 / (1.0 + 2.0 * rho)


def _component_coherent_spectra(sig_i, sig_j):
    """Return a list of per-component coherent cross-spectra |Xi_c conj(Xj_c)|."""
    sig_i = _as_2d(sig_i)
    sig_j = _as_2d(sig_j)
    Xi = np.fft.rfft(sig_i - sig_i.mean(axis=0, keepdims=True), axis=0)
    Xj = np.fft.rfft(sig_j - sig_j.mean(axis=0, keepdims=True), axis=0)
    return [
        np.abs(Xi[:, c] * np.conj(Xj[:, c])) for c in range(sig_i.shape[1])
    ]


def fisher_information_delay(sig_i, sig_j, dt, noise_var):
    """
    J_CR = sum_c sum_k q_k w_k^2 SNR_eff,c[k]       [1/s^2],
    where q_k=2 for positive-frequency bins paired with a negative bin,
    and q_k=1 at DC and (for even N) Nyquist.

    Each independent signal component contributes its own Fisher information
    (they are independent observations of the same delay), so the per-component
    contributions are summed. The factor 2 folds the negative-frequency half of
    the (T/2pi)*int w^2 SNR_eff dw integral into the one-sided rfft sum.
    """
    sig_i = _as_2d(sig_i)
    f = np.fft.rfftfreq(sig_i.shape[0], d=dt)
    w = 2.0 * np.pi * f
    one_sided = _one_sided_weights(sig_i.shape[0])
    J = 0.0
    for P_coh_c in _component_coherent_spectra(sig_i, sig_j):
        snr_eff = effective_snr_spectrum(P_coh_c, noise_var, sig_i.shape[0])
        J += np.sum(one_sided * w ** 2 * snr_eff)
    return J


def crb_delay_effective(sig_i, sig_j, dt, noise_var):
    """CRB_eff(tau) = 1 / J_CR  [s^2]."""
    J = fisher_information_delay(sig_i, sig_j, dt, noise_var)
    return 1.0 / J


# ---------------------------------------------------------------------------
# distance mapping and pair combination
# ---------------------------------------------------------------------------
def crb_distance(crb_delay, D_abs_ps_per_km):
    """
    Map a delay CRB [s^2] to a distance standard deviation [km]:

        sigma_Z = sqrt(CRB_tau) * 1e12 / |int D dlambda|     (ps -> km)

    D_abs_ps_per_km is |int D dlambda| in ps/km.
    """
    return np.sqrt(crb_delay) * 1e12 / np.abs(D_abs_ps_per_km)


def combined_crb_distance(crb_delay_pairs, D_abs_pairs):
    """
    Inverse-variance combination of pair distance CRBs.

        sigma_Z,comb = [ sum_ij 1 / sigma_Z,ij^2 ]^{-1/2}

    NOTE: assumes the pair estimates are independent. All channel pairs share
    channels, so this is optimistic. Use gls_combined_crb_distance for the
    covariance-aware combination under the additive channel-timing model.
    """
    crb_delay_pairs = np.asarray(crb_delay_pairs, dtype=float)
    D_abs_pairs = np.asarray(D_abs_pairs, dtype=float)
    sigma_Z = crb_distance(crb_delay_pairs, D_abs_pairs)
    return 1.0 / np.sqrt(np.sum(1.0 / sigma_Z ** 2))


def star_combined_crb_distance(crb_delay_pairs, D_abs_pairs, pair_indices,
                               n_channels):
    """
    Legacy independent-edge star approximation (not a joint CRB).

    This sums marginal pair information while ignoring covariance between
    edges that share the reference channel.  It is retained only for
    reproducing old results.  Use :func:`gls_combined_crb_distance` for the
    covariance-aware combination under the additive channel-timing model.

    The result can be optimistic and must not be labelled an achievable CRB.
    """
    crb_delay_pairs = np.asarray(crb_delay_pairs, dtype=float)
    D_abs_pairs = np.asarray(D_abs_pairs, dtype=float)
    best = np.inf
    for r in range(n_channels):
        var = 0.0
        for k, (i, j) in enumerate(pair_indices):
            if (i == r or j == r) and np.isfinite(crb_delay_pairs[k]) \
                    and crb_delay_pairs[k] > 0:
                var += D_abs_pairs[k] ** 2 / (crb_delay_pairs[k] * 1e24)
        if var > 0:
            best = min(best, 1.0 / np.sqrt(var))
    return best


# ---------------------------------------------------------------------------
# covariance-aware (GLS) combination
# ---------------------------------------------------------------------------
def _incidence(pair_indices, n_channels):
    """Signed incidence matrix A with A[k, i]=+1, A[k, j]=-1 for pair (i,j)."""
    A = np.zeros((len(pair_indices), n_channels))
    for k, (i, j) in enumerate(pair_indices):
        A[k, i] = 1.0
        A[k, j] = -1.0
    return A


def _channel_dispersion_coordinates(D_abs_pairs, pair_indices, n_channels):
    """Recover 1-D channel coordinates from complete pairwise |D_ij| values."""
    D_abs_pairs = np.asarray(D_abs_pairs, dtype=float)
    pairs = np.asarray(pair_indices, dtype=int)
    coord = np.full(n_channels, np.nan, dtype=float)
    coord[0] = 0.0
    for k, (i, j) in enumerate(pairs):
        if i == 0:
            coord[j] = D_abs_pairs[k]
        elif j == 0:
            coord[i] = D_abs_pairs[k]
    if not np.all(np.isfinite(coord)):
        raise ValueError("pair set must connect every channel to reference 0")
    predicted = np.abs(coord[pairs[:, 0]] - coord[pairs[:, 1]])
    scale = max(float(np.max(D_abs_pairs)), 1.0)
    if not np.allclose(predicted, D_abs_pairs, rtol=1e-8, atol=1e-10 * scale):
        raise ValueError("D_abs pairs are not consistent with 1-D channel coordinates")
    return coord


def per_channel_variances(crb_delay_pairs, pair_indices, n_channels,
                          nonneg=True):
    """
    Per-channel delay variances ``u_k`` such that ``Var(tau_ij)=u_i+u_j``.
    The non-negative least-squares fit is a model assumption to validate, not
    an identity for arbitrary pairwise CRBs.
    """
    A = np.zeros((len(pair_indices), n_channels))
    for k, (i, j) in enumerate(pair_indices):
        A[k, i] = 1.0
        A[k, j] = 1.0
    y = np.asarray(crb_delay_pairs, dtype=float)
    valid = np.isfinite(y) & (y >= 0.0)
    if not np.any(valid):
        return np.full(n_channels, np.nan)
    # Closed-form ordinary least squares for a complete pair graph.  Most
    # production sweeps have every pair, so this avoids thousands of optimizer
    # calls when rebuilding the full covariance-bound surface.
    if (n_channels > 2 and np.all(valid) and
            len(pair_indices) == n_channels * (n_channels - 1) // 2):
        row_sums = np.zeros(n_channels, dtype=float)
        for k, (i, j) in enumerate(pair_indices):
            row_sums[i] += y[k]
            row_sums[j] += y[k]
        total_u = np.sum(y) / (n_channels - 1)
        u_closed = (row_sums - total_u) / (n_channels - 2)
        if not nonneg or np.all(u_closed >= 0.0):
            return u_closed
    # Scale tiny variances (often around 1e-20 s^2) before the solve.
    scale = float(np.max(y[valid]))
    if scale <= 0.0:
        return np.zeros(n_channels)
    if nonneg:
        u_scaled, _ = nnls(A[valid], y[valid] / scale)
        u = u_scaled * scale
    else:
        u, *_ = np.linalg.lstsq(A[valid], y[valid], rcond=None)
    return u


def gls_combined_crb_distance(crb_delay_pairs, D_abs_pairs, pair_indices,
                              n_channels):
    """Conditional joint-GLS variance under an additive channel-timing model.

    Assumption: pair-delay errors are signed differences of independent
    per-channel timing errors, so ``Sigma_tau = A diag(u) A.T`` and
    ``Var(tau_ij)=u_i+u_j``.  The non-negative ``u`` values are fitted to the
    marginal pair CRBs.  In distance units,
    ``Sigma_Z = diag(1e12/|D|) Sigma_tau diag(1e12/|D|)``.

    A spanning star is sufficient under this model, but its edges are not
    independent.  The common event-time nuisance parameter is eliminated in
    closed form from the channel-level timing model.  The returned full-length
    weight vector is zero off the selected star.
    This is not a universal combined CRB: it is valid only to the extent that
    the additive channel-timing covariance model is adequate. Validate both
    marginal and cross-covariance predictions against Monte Carlo pair errors.

    Returns
    -------
    sigma_Z : float  combined distance std (km)
    weights : (n_pairs,) weights supported on a covariance-aware star
    """
    crb_delay_pairs = np.asarray(crb_delay_pairs, dtype=float)
    D_abs_pairs = np.asarray(D_abs_pairs, dtype=float)
    u = per_channel_variances(crb_delay_pairs, pair_indices, n_channels)
    if not np.all(np.isfinite(u)):
        return np.nan, np.full(len(pair_indices), np.nan)
    coord_s_per_km = _channel_dispersion_coordinates(
        D_abs_pairs, pair_indices, n_channels) * 1e-12
    w = np.zeros(len(pair_indices), dtype=float)
    zero_tol = max(float(np.max(u)), np.finfo(float).tiny) * 1e-12
    exact = np.flatnonzero(u <= zero_tol)
    # Boundary NNLS solutions indicate that the additive covariance model is
    # not identifiable from these marginal CRBs; do not turn them into a
    # spurious zero-noise channel and an overconfident combined reference.
    if len(exact):
        return np.nan, w

    precision = 1.0 / u
    dbar = float(np.sum(precision * coord_s_per_km) / np.sum(precision))
    centered = coord_s_per_km - dbar
    information = float(np.sum(precision * centered ** 2))
    if not np.isfinite(information) or information <= 0.0:
        return np.nan, w
    # The BLUE for channel times has coefficients a_i = J^-1 u_i^-1
    # (d_i - dbar), which sum to zero.  Relative to channel 0 this becomes a
    # star-weighted mean of pair-distance estimates with weights a_i*d_i.
    a = precision * centered / information
    for k, (i, j) in enumerate(pair_indices):
        if i == 0:
            w[k] = a[j] * coord_s_per_km[j]
        elif j == 0:
            w[k] = a[i] * coord_s_per_km[i]
    var = 1.0 / information
    return np.sqrt(var), w


# ---------------------------------------------------------------------------
# CRB-optimal (inverse-variance) distance estimator
# ---------------------------------------------------------------------------
def crb_weighted_mean(distances, D_abs, crb_delay_pairs, inlier_mask=None):
    """
    Distance estimate with per-pair inverse-variance weights.

        w_ij = 1 / sigma_Z,ij^2  =  D_abs,ij^2 / (CRB_tau,ij * 1e24)

    When all pairs share the same beta/SNR this reduces to w ~ D_abs^2
    (the existing 'wavelength-separation' weighting).

    Returns (estimate, weights) where weights is a full-length array.
    """
    Z = np.asarray(distances, dtype=float)
    D_abs = np.asarray(D_abs, dtype=float)
    crb_delay_pairs = np.asarray(crb_delay_pairs, dtype=float)

    if inlier_mask is None:
        inlier_mask = np.ones_like(Z, dtype=bool)
    inlier_mask = np.asarray(inlier_mask, dtype=bool)

    w = np.zeros_like(Z)
    sel = inlier_mask & np.isfinite(Z) & np.isfinite(crb_delay_pairs)
    w[sel] = D_abs[sel] ** 2 / (crb_delay_pairs[sel] * 1e24)
    if not np.any(w > 0):
        return np.nan, w
    return np.sum(w * Z) / np.sum(w), w


# ---------------------------------------------------------------------------
# self-test / reference
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    N, dt = 2048, 1e-6
    sig = rng.randn(N)
    snr_db = 80.0
    sigma2 = noise_variance_for_snr(sig, snr_db)

    snr = per_component_snr(sig, sigma2)
    beta2 = mean_square_bandwidth(sig, dt)
    crb_det = crb_delay_deterministic(sig, dt, sigma2)
    print(f"SNR (dB, realised) : {10*np.log10(snr):.4f}")
    print(f"beta^2             : {beta2:.4e} rad^2/s^2")
    print(f"1/(SNR beta^2)     : {crb_det:.4e} s^2")

    # effective bound for two identical sensors
    crb_eff = crb_delay_effective(sig, sig, dt, sigma2)
    print(f"CRB_eff            : {crb_eff:.4e} s^2")
    print(f"ratio CRB_eff/CRB_det: {crb_eff/crb_det:.4f}  (expect 2 at high SNR)")
