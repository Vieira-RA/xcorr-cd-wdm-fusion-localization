#!/usr/bin/env python3
"""
Validate the random birefringence generator by Monte‑Carlo simulation
of the resulting DGD using the PMD operator (central‑difference on Jones matrices).

The Maxwellian PDF is computed directly from the paper’s expressions,
without Scipy.
"""
import numpy as np
import matplotlib.pyplot as plt

from fiber_propagation import (
    generate_birefringence_sections,
    scale_birefringence_for_frequency,
    propagate_through_sections,
    quaternion_to_jones
)

# ---------- physical constants ----------
c = 299792458.0               # m/s

# ---------- fibre parameters ----------
L = 1_500.0                 # 100 km
L_F = 150.0                   # 100 m correlation length
lambda0 = 1550e-9             # 1550 nm
omega0 = 2.0 * np.pi * c / lambda0
pmd_coeff = 0.1               # 0.1 ps/√km
delta_nu = 1e2                # 1 GHz offset for central difference
delta_omega = 2.0 * np.pi * delta_nu

num_realisations = 2000
seed_base = 12345

# ---------- expected mean DGD ----------
L_km = L / 1000.0
expected_mean_ps = pmd_coeff * np.sqrt(L_km)   # ps
# Mean‑square DGD from the Maxwellian relation: <τ²> = (3π/8) <τ>²
tau_sq_mean = (3.0 * np.pi / 8.0) * expected_mean_ps**2   # ps²
print(f"Expected mean DGD = {expected_mean_ps:.4f} ps")
print(f"Expected mean square DGD = {tau_sq_mean:.4f} ps")


# ---------- custom Maxwellian PDF ----------
def maxwell_pdf(tau: np.ndarray, tau_sq_mean: float) -> np.ndarray:
    """
    Maxwellian probability density for DGD magnitude τ.

    Uses the paper's 3‑D Gaussian (Eq. 28) multiplied by 4πτ²:
        PDF(τ) = 4π τ² * (3/(2π<τ²>))^(3/2) * exp(-3τ²/(2<τ²>)).

    Parameters
    ----------
    tau : array_like
        DGD values (same units as sqrt(tau_sq_mean)).
    tau_sq_mean : float
        Mean‑square DGD ⟨τ²⟩.
    """
    coeff = (3.0 / (2.0 * np.pi * tau_sq_mean)) ** 1.5
    return 4.0 * np.pi * tau**2 * coeff * np.exp(-3.0 * tau**2 / (2.0 * tau_sq_mean))

# ---------- Monte Carlo loop ----------
dgds_ps = []

for i in range(num_realisations):
    seed = seed_base + i

    # base birefringence at ω₀
    beta0 = generate_birefringence_sections(L, L_F, lambda0, pmd_coeff,
                                            seed=seed, magnitude='constant')

    # three frequencies
    beta_minus = scale_birefringence_for_frequency(beta0, omega0, omega0 - delta_omega)
    beta_center = scale_birefringence_for_frequency(beta0, omega0, omega0)
    beta_plus  = scale_birefringence_for_frequency(beta0, omega0, omega0 + delta_omega)

    # propagate
    q_minus  = propagate_through_sections(beta_minus,  L_F)
    q_center = propagate_through_sections(beta_center, L_F)
    q_plus   = propagate_through_sections(beta_plus,  L_F)

    # Jones matrices
    U_minus  = quaternion_to_jones(q_minus)
    U_center = quaternion_to_jones(q_center)
    U_plus   = quaternion_to_jones(q_plus)

    # PMD vector
    dU_domega = (U_plus - U_minus) / (2.0 * delta_omega)
    M = 2j * dU_domega @ U_center.conj().T
    Omega1 = 0.5 * (M[0,1] + M[1,0]).real
    Omega2 = 0.5 * (M[0,1] - M[1,0]).imag
    Omega3 = M[0,0].real
    DGD = np.sqrt(Omega1**2 + Omega2**2 + Omega3**2)
    dgds_ps.append(DGD * 1e12)   # ps

dgds_ps = np.array(dgds_ps)

# ---------- statistics ----------
mean_dgd_ps = np.mean(dgds_ps)
print(f"Simulated mean DGD = {mean_dgd_ps:.4f} ps")
print(f"Relative error       = {(mean_dgd_ps - expected_mean_ps) / expected_mean_ps * 100:.2f}%")
print(f"Simulated mean square DGD = {np.mean(dgds_ps**2):.4f} ps")


# ---------- histogram + Maxwellian fit ----------
plt.hist(dgds_ps, bins=50, density=True, alpha=0.7, label='Simulated')
x = np.linspace(0, np.max(dgds_ps), 200)
plt.plot(x, maxwell_pdf(x, tau_sq_mean), 'r-', label='Maxwellian (theory)')
plt.xlabel('DGD (ps)')
plt.ylabel('Probability density')
plt.legend()
plt.title(f'DGD distribution (L={L_km} km, κ={pmd_coeff} ps/√km)')
plt.savefig('/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/output/dgd_validation.png', dpi=150)
plt.show()