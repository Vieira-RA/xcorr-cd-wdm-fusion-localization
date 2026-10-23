"""
Validate the CRB implementation against the actual parabolic cross-correlation
estimator (toy Gaussian pulse), for both bounds:

  - CRB_det = 1 / (SNR beta^2)                      (doc section 2)
  - CRB_eff = 1 / J_CR, two-sensor effective SNR    (doc section 3.3)

The estimator is the same one used in the paper pipeline: magnitude cross-
correlation peak refined by three-point parabolic interpolation.

Run:  python validate_crb.py
"""

import numpy as np

import crb
from signal_processing import parabolic_corr_delay_1d, parabolic_corr_delay_2d


def validate_rfft_two_sided_sums():
    """Check rFFT endpoint/interior weights against explicit full FFT sums."""
    rng = np.random.RandomState(2468)
    dt = 2.5e-7
    for n in (1024, 1025):
        sig_i = rng.randn(n, 2)
        sig_j = rng.randn(n, 2)
        sig_i += np.array([0.7, -0.3])
        sig_j += np.array([-0.2, 0.4])
        if n % 2 == 0:
            # Ensure the Nyquist endpoint is non-zero and counted once.
            nyq = (-1.0) ** np.arange(n)
            sig_i[:, 0] += 0.5 * nyq
            sig_j[:, 0] -= 0.25 * nyq

        xi = np.fft.fft(sig_i - sig_i.mean(axis=0), axis=0)
        xj = np.fft.fft(sig_j - sig_j.mean(axis=0), axis=0)
        f = np.fft.fftfreq(n, d=dt)
        omega = 2.0 * np.pi * f
        p = np.sum(np.abs(xi) ** 2, axis=1)
        beta2_full = np.sum(omega ** 2 * p) / np.sum(p)
        beta2_rfft = crb.mean_square_bandwidth(sig_i, dt)
        np.testing.assert_allclose(beta2_rfft, beta2_full, rtol=1e-12)

        coherent = np.abs(np.sum(xi * np.conj(xj), axis=1))
        beta2_coherent_full = np.sum(omega ** 2 * coherent) / np.sum(coherent)
        beta2_coherent_rfft = crb.coherent_bandwidth(sig_i, sig_j, dt)
        np.testing.assert_allclose(beta2_coherent_rfft, beta2_coherent_full,
                                   rtol=1e-12)
        np.testing.assert_allclose(coherent, coherent[(-np.arange(n)) % n],
                                   rtol=1e-12, atol=1e-12)

        sigma2 = 0.3
        j_full = 0.0
        for c in range(sig_i.shape[1]):
            rho = np.abs(xi[:, c] * np.conj(xj[:, c])) / (n * sigma2)
            eff = rho ** 2 / (1.0 + 2.0 * rho)
            j_full += np.sum(omega ** 2 * eff)
        j_rfft = crb.fisher_information_delay(sig_i, sig_j, dt, sigma2)
        np.testing.assert_allclose(j_rfft, j_full, rtol=1e-12)
    print("One-sided rFFT moments/Fisher information match full two-sided FFT sums.")


# ---------------------------------------------------------------------------
def fractional_shift(x, dt, delay):
    """Exact fractional delay via the Fourier shift theorem."""
    n = len(x)
    freq = np.fft.fftfreq(n, d=dt)
    return np.fft.ifft(np.fft.fft(x) * np.exp(2j * np.pi * freq * delay)).real


def make_pulse(B, dt, t_end_factor=160.0):
    sigma_t = 0.3748 / B
    t_end = t_end_factor * sigma_t
    n = int(t_end / dt) + 1
    t = np.linspace(0, t_end, n)
    t0 = t_end / 2.0
    return np.exp(-0.5 * ((t - t0) / sigma_t) ** 2)


def run_case(B, snrs_db, n_real, dt, true_delay, dim, seed=12345):
    """Return dict of MC std [ns] and the two bounds [ns]."""
    pulse = make_pulse(B, dt)
    shifted = fractional_shift(pulse, dt, true_delay)

    if dim == 1:
        sig_i, sig_j = pulse, shifted
        est_fun = lambda a, b: parabolic_corr_delay_1d(a, b, dt)
    else:  # 2-D, give the second component a fraction of the pulse
        sig_i = np.column_stack([pulse, 0.3 * pulse])
        sig_j = np.column_stack([shifted, 0.3 * shifted])
        est_fun = lambda a, b: parabolic_corr_delay_2d(a, b, dt)

    std_ns, crb_det_ns, crb_eff_ns, crb_eff_coh_ns = [], [], [], []
    for snr_db in snrs_db:
        sigma2 = crb.noise_variance_for_snr(sig_i, snr_db)
        # bounds
        crb_det = crb.crb_delay_deterministic(sig_i, dt, sigma2)
        crb_eff = crb.crb_delay_effective(sig_i, sig_j, dt, sigma2)
        # coherent-identical special case (channel j = channel i) for reference
        crb_eff_id = crb.crb_delay_effective(sig_i, sig_i, dt, sigma2)

        rng = np.random.RandomState(seed)
        est = np.empty(n_real)
        for r in range(n_real):
            a = sig_i + np.sqrt(sigma2) * rng.randn(*sig_i.shape)
            b = sig_j + np.sqrt(sigma2) * rng.randn(*sig_j.shape)
            est[r] = est_fun(a, b)

        std_ns.append(np.std(est) * 1e9)
        crb_det_ns.append(np.sqrt(crb_det) * 1e9)
        crb_eff_ns.append(np.sqrt(crb_eff) * 1e9)
        crb_eff_coh_ns.append(np.sqrt(crb_eff_id) * 1e9)

    return dict(
        std=np.array(std_ns),
        crb_det=np.array(crb_det_ns),
        crb_eff=np.array(crb_eff_ns),
        crb_eff_coh=np.array(crb_eff_coh_ns),
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    validate_rfft_two_sided_sums()

    # --- 1. deterministic identity sanity check ---------------------------
    rng = np.random.RandomState(0)
    x = rng.randn(4096)
    dt = 1.0 / 3e6
    s2 = crb.noise_variance_for_snr(x, 25.0)
    snr = crb.per_component_snr(x, s2)
    beta2 = crb.mean_square_bandwidth(x, dt)
    J_det = snr * beta2
    print("Deterministic identity:  J = SNR*beta^2")
    print(f"  SNR*beta^2            = {J_det:.6e}")
    print(f"  1/CRB_det             = {1.0/crb.crb_delay_deterministic(x, dt, s2):.6e}")
    print()

    # --- 2. toy MC vs bounds ---------------------------------------------
    B = 1.0e5
    dt = 1.0 / (20.0 * B)
    true_delay = 1.0e-6
    snrs_db = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60])
    n_real = 400

    for dim in (1, 2):
        res = run_case(B, snrs_db, n_real, dt, true_delay, dim)
        print(f"--- toy pulse, {dim}-D estimator, B={B:.0e} Hz, N_real={n_real} ---")
        print("SNR[dB]   MC_std     CRB_det    CRB_eff   MC/CRB_eff")
        for i, s in enumerate(snrs_db):
            ratio = res["std"][i] / res["crb_eff"][i]
            print(
                f"{s:6.1f}   {res['std'][i]:8.4f}  {res['crb_det'][i]:8.4f}"
                f"  {res['crb_eff'][i]:8.4f}   {ratio:6.3f}"
            )
        print()
