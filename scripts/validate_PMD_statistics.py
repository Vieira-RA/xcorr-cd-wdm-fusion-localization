# ===== scripts/validate_pmd_statistics.py =====

"""
Validation of PMD statistical model against the 2004 paper.
"""

import matplotlib
matplotlib.use('Agg')   # <-- HEADLESS MODE: only save, no GUI popup

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from tqdm import tqdm

from pmd_model import (
    generate_pmd_waveplates,
    jones_at_frequency,
    extract_pmd_vector,
)
from fiber_propagation import propagate_unitary


def run_pmd_simulation(
    L: float,
    L_F: float,
    D_pmd: float,
    lambda0: float,
    n_realizations: int,
    freq_grid: np.ndarray,
    delta_freq_step: float,
    seed_base: int = 0,
) -> dict:
    n_freq = len(freq_grid)
    tau_grid = np.zeros((n_realizations, n_freq, 3))
    dgd_grid = np.zeros((n_realizations, n_freq))

    for r in tqdm(range(n_realizations), desc="Simulating fibres", unit="realization"):
        seed = seed_base + r
        z, beta0, beta_prime, L_seg = generate_pmd_waveplates(
            L, L_F, D_pmd, lambda0, seed=seed
        )

        # --- Debug print on first realization ---
        if r == 0:
            sigma_bp = np.std(beta_prime[0, :])  # approximate
            print(f"L_seg = {L_seg:.3f} m")
            print(f"Estimated sigma_beta_prime = {sigma_bp:.3e}")
            expected_sigma = D_pmd / np.sqrt(3.0 * L_seg)
            print(f"Expected sigma = {expected_sigma:.3e}")
            print(f"D_pmd = {D_pmd:.3e} s/√m")
            print(f"L = {L:.1f} m")
            print(f"Theoretical <τ²> = {D_pmd**2 * L:.3e} s² = {D_pmd**2 * L * 1e24:.4f} ps²")

        # Compute Jones matrices for all frequencies
        U_list = []
        for dw in freq_grid:
            U = jones_at_frequency(z, beta0, beta_prime, dw)
            U_list.append(U)

        # Compute PMD vector using central difference
        if len(freq_grid) > 1:
            step = freq_grid[1] - freq_grid[0]
            for i in range(len(freq_grid)):
                if i == 0:
                    U_plus = U_list[i+1]
                    U_minus = U_list[i]
                    delta_omega_eff = step
                elif i == len(freq_grid)-1:
                    U_plus = U_list[i]
                    U_minus = U_list[i-1]
                    delta_omega_eff = step
                else:
                    U_plus = U_list[i+1]
                    U_minus = U_list[i-1]
                    delta_omega_eff = 2 * step
                tau, dgd = extract_pmd_vector(U_minus, U_plus, delta_omega_eff)
                tau_grid[r, i, :] = tau
                dgd_grid[r, i] = dgd

    tau2_all = np.sum(tau_grid**2, axis=2)
    mean_tau2 = np.mean(tau2_all)

    return {
        'tau_grid': tau_grid,
        'dgd_grid': dgd_grid,
        'mean_tau2': mean_tau2,
        'freq_grid': freq_grid,
        'L': L,
        'D_pmd': D_pmd,
    }


def validate_pmd_statistics():
    # ---- Simulation parameters ----
    L_km = 1.0
    L = L_km * 1e3           # metres
    L_F = 20.0               # correlation length (metres)
    D_pmd = 3.162e-15        # 0.1 ps/√km → s/√m  (CHECK THIS VALUE!)
    lambda0 = 1550e-9
    n_realizations = 250     # increase to 500+ for smoother curves

    mean_tau2_theory = (D_pmd**2) * L
    x_grid = np.linspace(0, 10, 200)
    omega_grid = x_grid / np.sqrt(mean_tau2_theory)

    # ---- Run simulation ----
    results = run_pmd_simulation(
        L=L,
        L_F=L_F,
        D_pmd=D_pmd,
        lambda0=lambda0,
        n_realizations=n_realizations,
        freq_grid=omega_grid,
        delta_freq_step=omega_grid[1] - omega_grid[0],
        seed_base=42
    )

    tau_grid = results['tau_grid']
    dgd_grid = results['dgd_grid']
    mean_tau2_sim = results['mean_tau2']
    freq_grid = results['freq_grid']

    # ---- Autocorrelation ----
    n_freq = len(freq_grid)
    max_lag_idx = n_freq // 2
    autocorr = np.zeros(max_lag_idx)
    for lag_idx in range(max_lag_idx):
        dot_products = []
        for i in range(n_freq - lag_idx):
            tau1 = tau_grid[:, i, :]
            tau2 = tau_grid[:, i+lag_idx, :]
            dot = np.sum(tau1 * tau2, axis=1)
            dot_products.append(dot)
        autocorr[lag_idx] = np.mean(dot_products)

    autocorr_norm = autocorr / mean_tau2_sim
    delta_omega = freq_grid[1] - freq_grid[0]
    lags_norm = np.arange(max_lag_idx) * delta_omega * np.sqrt(mean_tau2_sim)

    def theory_acf(x):
        x_safe = np.where(x > 0, x, 1e-12)
        return 3 / x_safe**2 * (1 - np.exp(-x_safe**2 / 3))

    # ---- Plot 1: Autocorrelation ----
    plt.figure(figsize=(10,6))
    plt.plot(lags_norm, autocorr_norm, 'o-', label='Simulation')
    x_theory = np.linspace(0, lags_norm[-1], 200)
    plt.plot(x_theory, theory_acf(x_theory), 'r--', label='Theory (Eq. 17)')
    plt.xlim([-1, 12])
    plt.xlabel('Normalized frequency lag $\\Delta\\omega \\sqrt{\\langle\\tau^2\\rangle}$')
    plt.ylabel('$\\langle \\boldsymbol{\\tau}(\\omega) \\cdot \\boldsymbol{\\tau}(\\omega+\\Delta\\omega) \\rangle / \\langle \\tau^2 \\rangle$')
    plt.legend()
    plt.grid(True)
    plt.title('PMD Vector Autocorrelation')
    plt.savefig('/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/validation/pmd_autocorr.png', dpi=150)
    print("Autocorrelation plot saved as 'pmd_autocorr.png'")
    plt.close()

    # ---- Plot 2: DGD distribution ----
    dgd_all = dgd_grid.flatten()
    a_sim = np.sqrt(mean_tau2_theory / 3.0)

    plt.figure(figsize=(10,6))
    counts, bins, _ = plt.hist(dgd_all, bins=50, density=True, alpha=0.6, label='Simulation')
    x = np.linspace(0, bins[-1], 200)
    pdf_maxwell = maxwell.pdf(x, scale=a_sim)
    plt.plot(x, pdf_maxwell, 'r-', linewidth=2, label='Maxwellian fit')
    plt.xlabel('DGD (s)')
    plt.ylabel('Probability density')
    plt.legend()
    plt.grid(True)
    plt.title('DGD Distribution')
    plt.savefig('/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/validation/dgd_distribution.png', dpi=150)
    print("DGD distribution plot saved as 'dgd_distribution.png'")
    plt.close()

    # ---- Print statistics ----
    mean_dgd_sim = np.mean(dgd_all)
    mean_dgd_theory = np.sqrt(8 / (3 * np.pi)) * np.sqrt(D_pmd**2 * L)
    print(f"\n=== RESULTS ===")
    print(f"Mean DGD (sim):      {mean_dgd_sim*1e12:.3f} ps")
    print(f"Mean DGD (theory):   {mean_dgd_theory*1e12:.3f} ps")
    print(f"<τ²> (sim):          {mean_tau2_sim*1e24:.4f} ps²")
    print(f"<τ²> (theory):       {(D_pmd**2 * L)*1e24:.4f} ps²")
    print(f"Ratio sim/theory:    {mean_tau2_sim / (D_pmd**2 * L):.3f}")


if __name__ == "__main__":
    validate_pmd_statistics()