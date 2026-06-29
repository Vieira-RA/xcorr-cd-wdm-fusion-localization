"""
Validation of PMD statistics using physical (non‑normalized) units.

This script:
  - Computes the absolute mean‑square DGD ⟨τ²⟩ and compares it to γ₀²L.
  - Checks the DGD distribution (Maxwellian).
  - Computes the PMD vector autocorrelation in real frequency units [rad/s]
    and compares it with the theoretical expression.
  - Prints the stationarity condition β₀ ≫ β′·Δω_max and the ratio.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import maxwell
from tqdm import tqdm

from pmd_model import (
    generate_pmd_waveplates,
    jones_at_frequency,
    extract_pmd_vector,
)

# ==================== Parameters ====================
L_km = 1.0
L = L_km * 1e3                 # m
L_F = 20.0                     # m
D_pmd = 2.5298e-15              # s/√m   (0.1 ps/√km)
lambda0 = 1550e-9              # m
n_realizations = 1500           # more realizations → smoother curves
n_freq = 2500
dw_max = 2.0 * np.pi * 19e12    # 5 THz max offset (covers C‑band with margin)

# ==================== Derived quantities ====================
c = 299792458.0
omega0 = 2.0 * np.pi * c / lambda0

mean_tau2_theory = D_pmd**2 * L
mean_dgd_theory = np.sqrt(8.0 / (3.0 * np.pi)) * np.sqrt(mean_tau2_theory)

# Birefringence magnitudes
beta0_mag = (omega0 / np.sqrt(L_F)) * D_pmd          # rad/m
sigma_beta_prime = np.sqrt(D_pmd**2 / L_F)           # s/m   (variance = γ₀²/Δz)
#beta_prime_mag_typical = np.sqrt(3) * sigma_beta_prime   # typical |β′|
beta_prime_mag_typical = sigma_beta_prime   # typical |β′|

# Stationarity ratio R = |β′|·Δω_max / |β₀|
R = (beta_prime_mag_typical * dw_max) / beta0_mag

print("================================================")
print(" Physical PMD validation – stationarity check")
print("================================================")
print(f" |β₀|   = {beta0_mag:.3f} rad/m")
print(f" σ_β'   = {sigma_beta_prime:.3e} s/m")
print(f" typ|β′|= {beta_prime_mag_typical:.3e} s/m  (√3·σ_β')")
print(f" Δω_max = {dw_max:.3e} rad/s  ({dw_max/(2*np.pi*1e12):.1f} THz)")
print(f" R = |β′|·Δω_max / |β₀| = {R:.4f}")
if R < 0.1:
    print("(R < 0.1) → R ≪ 1, stationarity condition satisfied.")
else:
    print(" → R is not small; stationarity may break down.")
print("================================================\n")

# ==================== Frequency grid (physical) ====================
freq_grid = np.linspace(0.0, dw_max, n_freq)       # [rad/s]
delta_omega = freq_grid[1] - freq_grid[0]

# ==================== Simulation ====================
def run_simulation():
    n_f = len(freq_grid)
    tau_grid = np.zeros((n_realizations, n_f, 3))
    dgd_grid = np.zeros((n_realizations, n_f))

    for r in tqdm(range(n_realizations), desc="Simulating fibres", unit="real"):
        seed = 42 + r
        z, beta0, beta_prime, L_seg = generate_pmd_waveplates(
            L, L_F, D_pmd, lambda0, seed=seed
        )
        # Compute Jones matrices
        U_list = [jones_at_frequency(z, beta0, beta_prime, dw) for dw in freq_grid]

        # Central‑difference PMD extraction
        for i in range(n_f):
            if i == 0:
                U_plus, U_minus = U_list[1], U_list[0]
                dw_eff = delta_omega
            elif i == n_f - 1:
                U_plus, U_minus = U_list[-1], U_list[-2]
                dw_eff = delta_omega
            else:
                U_plus, U_minus = U_list[i+1], U_list[i-1]
                dw_eff = 2 * delta_omega
            tau, dgd = extract_pmd_vector(U_minus, U_plus, dw_eff)
            tau_grid[r, i, :] = tau
            dgd_grid[r, i] = dgd

    tau2_all = np.sum(tau_grid**2, axis=2)
    mean_tau2 = np.mean(tau2_all)
    return tau_grid, dgd_grid, mean_tau2

tau_grid, dgd_grid, mean_tau2_sim = run_simulation()

print(f"<τ²> (sim)   : {mean_tau2_sim*1e24:.4f} ps²")
print(f"<τ²> (theory): {mean_tau2_theory*1e24:.4f} ps²")
print(f"Ratio sim/theory: {mean_tau2_sim/mean_tau2_theory:.3f}\n")

# ==================== Autocorrelation in physical units ====================
n_f = len(freq_grid)
max_lag_idx = n_f // 2
autocorr = np.zeros(max_lag_idx)
for lag_idx in range(max_lag_idx):
    dot_prods = []
    for i in range(n_f - lag_idx):
        tau1 = tau_grid[:, i, :]
        tau2 = tau_grid[:, i+lag_idx, :]
        dot_prods.append(np.sum(tau1 * tau2, axis=1))
    autocorr[lag_idx] = np.mean(np.concatenate(dot_prods))

autocorr_norm = autocorr / mean_tau2_sim
lags_phys = np.arange(max_lag_idx) * delta_omega   # rad/s

def theory_acf_physical(dw):
    tau2 = mean_tau2_theory
    x2 = dw**2 * tau2 / 3.0
    # normalized ACF = (1 - exp(-x2)) / x2
    return np.where(x2 > 1e-24, (1.0 - np.exp(-x2)) / x2, 1.0)

acf_theory = theory_acf_physical(lags_phys)

# ---- Correlation bandwidth from the paper (normalized value 2.2) ----
x_3dB_paper = 2.2          # from Shtaif & Mecozzi (2004)
dw_3dB_paper = x_3dB_paper / np.sqrt(mean_tau2_theory)   # rad/s
freq_3dB_paper_THz = dw_3dB_paper / (2 * np.pi * 1e12)

# ---- Numerical -3 dB point (where ACF = 0.5) for comparison ----
from scipy.optimize import fsolve
def acf_minus_half(dw):
    return theory_acf_physical(dw) - 0.5
dw_guess = 1.6 / np.sqrt(mean_tau2_theory)
dw_3dB_num = fsolve(acf_minus_half, dw_guess)[0]
freq_3dB_num_THz = dw_3dB_num / (2 * np.pi * 1e12)

print(f"Paper correlation bandwidth (Δω_c √⟨τ²⟩ = 2.2):")
print(f"  Δf_c = {freq_3dB_paper_THz:.3f} THz  (Δω = {dw_3dB_paper:.3e} rad/s)")
print(f"Numerically solved -3 dB point:")
print(f"  Δf   = {freq_3dB_num_THz:.3f} THz  (Δω = {dw_3dB_num:.3e} rad/s)")

# ---- Plot autocorrelation with paper's and numerical -3 dB markers ----
plt.figure(figsize=(10, 6))
plt.plot(lags_phys/(2*np.pi*1e12), autocorr_norm, 'o-', markersize=3, label='Simulation')
plt.plot(lags_phys/(2*np.pi*1e12), acf_theory, 'r--', label='Theory (Eq. 17)')
plt.axhline(0.5, color='grey', linestyle=':', alpha=0.7)
# Paper's bandwidth (2.2)
plt.axvline(freq_3dB_paper_THz, color='blue', linestyle='--',
            label=f'Paper Δω_c = 2.2 ({freq_3dB_paper_THz:.2f} THz)')
# Optional: numerical solution (overlaps, so use dotted to distinguish)
plt.axvline(freq_3dB_num_THz, color='green', linestyle=':', alpha=0.5,
            label=f'Num. –3 dB ({freq_3dB_num_THz:.2f} THz)')
plt.xlabel(r'$\Delta\omega/2\pi$ (THz)')
plt.ylabel(r'$\langle\boldsymbol{\tau}(\omega)\cdot\boldsymbol{\tau}(\omega+\Delta\omega)\rangle / \langle\tau^2\rangle$')
plt.legend()
plt.grid(True)
plt.title('PMD Autocorrelation (physical units)')
plt.savefig('/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/pmd_autocorr_physical.png', dpi=150)
plt.close()
print("Autocorrelation plot saved as 'pmd_autocorr_physical.png'")

# ==================== Extract independent DGD samples ====================
# Correlation bandwidth from paper (Δf_c ≈ 4.38 THz for 0.08 ps/√km, 1 km)
# We use a conservative stride: at least Δf_c between kept samples.
freq_corr_bandwidth_THz = freq_3dB_paper_THz   # 4.38 THz
# Convert to index stride: how many frequency steps equal Δf_c?
delta_f_THz = delta_omega / (2 * np.pi * 1e12)
stride = max(1, int(np.ceil(freq_corr_bandwidth_THz / delta_f_THz)))
print(f"Using every {stride} frequency points to ensure independence (Δf ≥ {freq_corr_bandwidth_THz:.2f} THz)")

# Extract independent DGD values from all realizations and selected frequencies
independent_indices = np.arange(0, n_freq, stride)
dgd_independent = dgd_grid[:, independent_indices].flatten()

# ==================== DGD distribution (independent samples) ====================
a_sim = np.sqrt(mean_tau2_sim / 3.0)

plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(dgd_independent, bins=50, density=True, alpha=0.6, label='Simulation (indep. samples)')
x = np.linspace(0, bins[-1], 200)
plt.plot(x, maxwell.pdf(x, scale=a_sim), 'r-', lw=2, label='Maxwellian fit')
plt.xlabel('DGD (s)')
plt.ylabel('Probability density')
plt.legend()
plt.grid(True)
plt.title('DGD Distribution (independent frequency samples)')
plt.savefig('/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/dgd_distribution_physical.png', dpi=150)
plt.close()
print("DGD plot saved as 'dgd_distribution_physical.png'")

# ==================== Final stats (using independent samples) ====================
mean_dgd_sim = np.mean(dgd_independent)
mean_dgd_theory = np.sqrt(8.0 / (3.0 * np.pi)) * np.sqrt(mean_tau2_theory)
print(f"\nMean DGD (sim, indep.):   {mean_dgd_sim*1e12:.3f} ps")
print(f"Mean DGD (theory):        {mean_dgd_theory*1e12:.3f} ps")
print(f"Stationarity ratio R = {R:.4f}")

# ==================== DGD distribution ====================
dgd_all = dgd_grid.flatten()
a_sim = np.sqrt(mean_tau2_sim / 3.0)

plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(dgd_all, bins=50, density=True, alpha=0.6, label='Simulation')
x = np.linspace(0, bins[-1], 200)
plt.plot(x, maxwell.pdf(x, scale=a_sim), 'r-', lw=2, label='Maxwellian fit')
plt.xlabel('DGD (s)')
plt.ylabel('Probability density')
plt.legend()
plt.grid(True)
plt.title('DGD Distribution (physical units)')
plt.savefig('/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/dgd_distribution_physical.png', dpi=150)
plt.close()
print("DGD plot saved as 'dgd_distribution_physical.png'")

# ==================== Final stats ====================
mean_dgd_sim = np.mean(dgd_all)
print(f"\nMean DGD (sim)   : {mean_dgd_sim*1e12:.3f} ps")
print(f"Mean DGD (theory): {mean_dgd_theory*1e12:.3f} ps")
print(f"Stationarity ratio R = {R:.4f}")