#!/usr/bin/env python3
"""Frequency correlation of output Stokes vector in the first‑order PMD model.

For one random fibre, the output Stokes vector S(ω) is computed for many
wavelengths across the C‑band. The autocorrelation R(Δν) is computed and
its width gives the polarisation correlation bandwidth.
"""

import numpy as np
import matplotlib.pyplot as plt
from pmd_model import generate_birefringence_profile, scale_birefringence_to_wavelength
from fiber_propagation import propagate_unitary

# ---------- Fibre parameters ----------
L = 1000.0               # 1 km
L_F = 50.0               # correlation length [m]
D_pmd = 3.16e-15         # 0.1 ps/√km
lambda0 = 1550e-9
c = 299792458.0
omega0 = 2 * np.pi * c / lambda0

# ---------- Frequency grid ----------
# Use 1000 points from 1530 nm to 1570 nm (C‑band), symmetric
lam_min, lam_max = 1530e-9, 1570e-9
N_omega = 1000
wavelengths = np.linspace(lam_min, lam_max, N_omega)
omegas = 2 * np.pi * c / wavelengths

# Generate base fibre
z, beta0 = generate_birefringence_profile(
    L=L, L_F=L_F, D_pmd=D_pmd, lambda0=lambda0, dz=L_F, seed=42
)

# Input Stokes vector (horizontal linear polarisation)
S_in = np.array([1.0, 0.0, 0.0])

# Pre‑allocate output Stokes array (3, N_omega)
S_out = np.empty((3, N_omega))

for i, wl in enumerate(wavelengths):
    beta_i = scale_birefringence_to_wavelength(beta0, lambda0, wl)
    U = propagate_unitary(z, beta_i)

    # Jones → Stokes (horizontal input)
    J_out = U @ np.array([1.0, 0.0], dtype=complex)
    S_out[0, i] = np.abs(J_out[0])**2 - np.abs(J_out[1])**2
    S_out[1, i] = 2 * np.real(J_out[0] * np.conj(J_out[1]))
    S_out[2, i] = 2 * np.imag(J_out[0] * np.conj(J_out[1]))

# ---------- Frequency autocorrelation ----------
# Use optical frequency in THz for convenience
nu = omegas / (2 * np.pi) * 1e-12   # THz
# Interpolate to a uniform nu grid (already uniform from linear wavelength spacing, but not exactly equally spaced in nu)
nu_uniform = np.linspace(nu.min(), nu.max(), N_omega)
S_interp = np.array([np.interp(nu_uniform, nu, S_out[k]) for k in range(3)])

# Autocorrelation: average over starting points
max_lag = N_omega // 2
lags_THz = (nu_uniform[1] - nu_uniform[0]) * np.arange(max_lag)   # Δν in THz
R = np.zeros(max_lag)
for lag in range(max_lag):
    if N_omega - lag > 0:
        prod = np.sum(S_interp[:, :N_omega - lag] * S_interp[:, lag:], axis=0)
        R[lag] = np.mean(prod)
# Normalise by power at zero lag
R /= R[0]

# Correlation bandwidth (FWHM)
idx_half = np.argmin(np.abs(R - 0.5))
FWHM_THz = lags_THz[idx_half] * 2   # full width, assuming symmetric
print(f"FWHM of Stokes correlation = {FWHM_THz:.3f} THz")
print(f"Equivalent wavelength bandwidth ≈ {FWHM_THz * (lambda0**2 / c * 1e12):.2f} nm (at 1550 nm)")

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(lags_THz, R, 'b-', linewidth=1.5)
ax.axhline(0.5, color='gray', linestyle='--')
ax.axvline(FWHM_THz/2, color='red', linestyle='--', label=f'FWHM ≈ {FWHM_THz:.2f} THz')
ax.set_xlabel('Frequency lag Δν [THz]')
ax.set_ylabel('Autocorrelation R(Δν)')
ax.set_title('Polarisation correlation bandwidth (first‑order PMD)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('../output/pmd_correlation_bandwidth.png', dpi=150)
plt.show()