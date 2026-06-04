#!/usr/bin/env python3
"""Monte Carlo validation of PMD statistics using the random wave‑plate model."""
import numpy as np
import matplotlib.pyplot as plt
from pmd_model import generate_multiwavelength_birefringence, extract_pmd_vector
from fiber_propagation import propagate_unitary

# ---------- Physical parameters ----------
L = 3000.0                       # 1 km
L_F = 50.0                       # 50 m correlation length
D_pmd = 3.16e-15                 # 0.1 ps/√km -> s/√m
lambda0 = 1550e-9                # m
delta_lambda = 0.01e-9            # 0.1 nm wavelength offset

c = 299792458.0
omega0 = 2 * np.pi * c / lambda0
lambda1 = lambda0 - delta_lambda / 2
lambda2 = lambda0 + delta_lambda / 2
omega1 = 2 * np.pi * c / lambda1
omega2 = 2 * np.pi * c / lambda2
delta_omega = omega2 - omega1       # positive, ~2π·12.5 GHz

wavelengths = [lambda1, lambda2]

N_MC = 10000
tau_samples = np.empty(N_MC)

# Spatial grid (same for all realisations)
dz = L_F
n_z = int(np.ceil(L / dz)) + 1
z = np.linspace(0, L, n_z)

print(f"Running {N_MC} Monte Carlo trials...")
for i in range(N_MC):
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{N_MC}")

    # Generate fibre at λ₀ and scale to two wavelengths
    _, _, profiles = generate_multiwavelength_birefringence(
        L=L, L_F=L_F, D_pmd=D_pmd, lambda0=lambda0,
        wavelengths=wavelengths, dz=dz, seed=i,  # each trial gets a unique seed
    )
    beta1 = profiles[lambda1]
    beta2 = profiles[lambda2]

    # Propagate Jones matrices
    U1 = propagate_unitary(z, beta1)
    U2 = propagate_unitary(z, beta2)

    # Extract DGD
    _, dgd = extract_pmd_vector(U1, U2, delta_omega)
    tau_samples[i] = dgd

# ---------- Statistics ----------
tau_sq_mean_theory = D_pmd**2 * L           # theoretical mean square DGD
tau_sq_mean_sample = np.mean(tau_samples**2)

print(f"\nTheoretical ⟨τ²⟩ = {tau_sq_mean_theory:.4e} s²")
print(f"Sample     ⟨τ²⟩ = {tau_sq_mean_sample:.4e} s²")
print(f"Relative error    = {abs(tau_sq_mean_sample/tau_sq_mean_theory - 1)*100:.2f}%")

# ---------- Maxwellian PDF ----------
def maxwell_pdf(tau: np.ndarray, tau_sq_mean: float) -> np.ndarray:
    """Maxwellian probability density for DGD magnitude τ (units: 1/s)."""
    coeff = (3.0 / (2.0 * np.pi * tau_sq_mean)) ** 1.5
    return 4.0 * np.pi * tau**2 * coeff * np.exp(-3.0 * tau**2 / (2.0 * tau_sq_mean))

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(8, 5))

# Histogram of samples (x-axis in picoseconds, density per ps)
ax.hist(tau_samples * 1e12, bins=50, density=True, alpha=0.7,
        label=f'MC (N={N_MC})', color='steelblue')

# Theoretical curve: convert to ps⁻¹
tau_vals = np.linspace(0, np.max(tau_samples) * 1.1, 300)
pdf = maxwell_pdf(tau_vals, tau_sq_mean_theory)   # 1/s
tau_vals_ps = tau_vals * 1e12
pdf_ps = pdf / 1e12                                # 1/ps

ax.plot(tau_vals_ps, pdf_ps, 'r-', linewidth=2, label='Maxwellian (theory)')

ax.set_xlabel('DGD [ps]')
ax.set_ylabel('Probability density [ps$^{-1}$]')
ax.set_title('PMD statistics: random wave‑plate model vs. Maxwellian')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('../output/pmd_maxwellian_validation.png', dpi=150)
plt.show()