"""
Validation of PMD statistics using an Ornstein–Uhlenbeck birefringence model.

This script tests the continuous white‑noise scaling:
    σ_β' = sqrt(γ₀² / (3·Δz))
and verifies that <τ²> = γ₀² L without any empirical correction.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from tqdm import tqdm

from birefringence_ou import generate_ou_birefringence
from pmd_model import jones_at_frequency, extract_pmd_vector


def run_pmd_simulation_ou(
    L: float,
    L_F: float,
    D_pmd: float,
    lambda0: float,
    n_realizations: int,
    freq_grid: np.ndarray,
    dz: float = 0.1,
    seed_base: int = 0,
) -> dict:
    n_freq = len(freq_grid)
    tau_grid = np.zeros((n_realizations, n_freq, 3))
    dgd_grid = np.zeros((n_realizations, n_freq))

    for r in tqdm(range(n_realizations), desc="Simulating OU fibres", unit="realization"):
        seed = seed_base + r
        z, beta0, beta_prime = generate_ou_birefringence(
            L, L_F, D_pmd, lambda0, dz=dz, seed=seed
        )

        if r == 0:
            # Print diagnostics for the first realization
            print(f"Grid step dz = {dz:.3f} m")
            print(f"Number of spatial points = {len(z)}")
            print(f"RMS |β₀| = {np.sqrt(np.mean(np.sum(beta0**2, axis=0))):.3f} rad/m")
            print(f"Expected σ_β' = {D_pmd / np.sqrt(3.0 * dz):.3e}")

        # Compute Jones matrices for all frequencies
        U_list = []
        for dw in freq_grid:
            U = jones_at_frequency(z, beta0, beta_prime, dw)
            U_list.append(U)

        # Extract PMD vectors via central difference
        if len(freq_grid) > 1:
            step = freq_grid[1] - freq_grid[0]
            for i in range(len(freq_grid)):
                if i == 0:
                    U_plus = U_list[i + 1]
                    U_minus = U_list[i]
                    delta_omega_eff = step
                elif i == len(freq_grid) - 1:
                    U_plus = U_list[i]
                    U_minus = U_list[i - 1]
                    delta_omega_eff = step
                else:
                    U_plus = U_list[i + 1]
                    U_minus = U_list[i - 1]
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


def validate_pmd_statistics_ou():
    # Simulation parameters (same as before)
    L_km = 1.0
    L = L_km * 1e3
    L_F = 50.0
    D_pmd = 3.162e-15
    lambda0 = 1550e-9
    n_realizations = 100
    dz = 50               # 0.1 m → Ω ≈ 0.086 ≪ 1

    mean_tau2_theory = (D_pmd**2) * L
    x_grid = np.linspace(24.9, 25.1, 6)
    omega_grid = x_grid / np.sqrt(mean_tau2_theory)

    results = run_pmd_simulation_ou(
        L=L, L_F=L_F, D_pmd=D_pmd, lambda0=lambda0,
        n_realizations=n_realizations,
        freq_grid=omega_grid,
        dz=dz,
        seed_base=42
    )

    tau_grid = results['tau_grid']
    dgd_grid = results['dgd_grid']
    mean_tau2_sim = results['mean_tau2']
    freq_grid = results['freq_grid']

    # Autocorrelation
    n_freq = len(freq_grid)
    max_lag_idx = n_freq // 2
    autocorr = np.zeros(max_lag_idx)
    for lag_idx in range(max_lag_idx):
        dot_products = []
        for i in range(n_freq - lag_idx):
            tau1 = tau_grid[:, i, :]
            tau2 = tau_grid[:, i + lag_idx, :]
            dot = np.sum(tau1 * tau2, axis=1)
            dot_products.append(dot)
        autocorr[lag_idx] = np.mean(dot_products)
    autocorr_norm = autocorr / mean_tau2_sim
    delta_omega = freq_grid[1] - freq_grid[0]
    lags_norm = np.arange(max_lag_idx) * delta_omega * np.sqrt(mean_tau2_sim)

    def theory_acf(x):
        x_safe = np.where(x > 0, x, 1e-12)
        return 3 / x_safe**2 * (1 - np.exp(-x_safe**2 / 3))

    plt.figure(figsize=(10, 6))
    plt.plot(lags_norm, autocorr_norm, 'o-', label='Simulation (OU)')
    x_theory = np.linspace(0, lags_norm[-1], 200)
    plt.plot(x_theory, theory_acf(x_theory), 'r--', label='Theory (Eq. 17)')
    plt.xlabel(r'$\Delta\omega \sqrt{\langle\tau^2\rangle}$')
    plt.ylabel(r'$\langle \boldsymbol{\tau}(\omega)\cdot\boldsymbol{\tau}(\omega+\Delta\omega) \rangle / \langle\tau^2\rangle$')
    plt.legend()
    plt.grid(True)
    plt.title('PMD Vector Autocorrelation (OU model)')
    plt.savefig('pmd_autocorr_ou.png', dpi=150)
    plt.close()

    # DGD distribution
    dgd_all = dgd_grid.flatten()
    a_sim = np.sqrt(mean_tau2_sim / 3.0)
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(dgd_all, bins=50, density=True, alpha=0.6, label='Simulation')
    x = np.linspace(0, bins[-1], 200)
    pdf_maxwell = maxwell.pdf(x, scale=a_sim)
    plt.plot(x, pdf_maxwell, 'r-', linewidth=2, label='Maxwellian fit')
    plt.xlabel('DGD (s)')
    plt.ylabel('Probability density')
    plt.legend()
    plt.grid(True)
    plt.title('DGD Distribution (OU model)')
    plt.savefig('dgd_distribution_ou.png', dpi=150)
    plt.close()

    # Statistics
    mean_dgd_sim = np.mean(dgd_all)
    mean_dgd_theory = np.sqrt(8 / (3 * np.pi)) * np.sqrt(mean_tau2_theory)

    print("\n=== OU MODEL RESULTS ===")
    print(f"Mean DGD (sim):      {mean_dgd_sim*1e12:.3f} ps")
    print(f"Mean DGD (theory):   {mean_dgd_theory*1e12:.3f} ps")
    print(f"<τ²> (sim):          {mean_tau2_sim*1e24:.4f} ps²")
    print(f"<τ²> (theory):       {mean_tau2_theory*1e24:.4f} ps²")
    print(f"Ratio sim/theory:    {mean_tau2_sim / mean_tau2_theory:.3f}")
    print("\nExpect ratio ≈ 1.0 when using σ_β' = sqrt(γ₀²/(3·dz))")


if __name__ == "__main__":
    validate_pmd_statistics_ou()