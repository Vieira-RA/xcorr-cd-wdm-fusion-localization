#!/usr/bin/env python3
"""Time‑modulated birefringence for 11 C‑band wavelengths (1550 nm central).

The birefringence profile is generated once from the random wave‑plate model.
For each wavelength λ, the static profile β₀(z) is scaled by ω/ω₀.
A common sinusoidal modulation (1 + A·sin(ω_mod t)) is then applied,
so all channels experience the same perturbation.
The rotation angle φ(t) of the output Jones matrix is plotted for every λ.
"""

import numpy as np
import matplotlib.pyplot as plt
from pmd_model import generate_birefringence_profile
from fiber_propagation import propagate_unitary

# ---------- Fibre and modulation parameters ----------
L = 10000.0               # 10 km
L_F = 50.0               # correlation length [m]
D_pmd = 3.16e-15         # 0.1 ps/√km
lambda0 = 1550e-9        # central wavelength [m]
c = 299792458.0
omega0 = 2 * np.pi * c / lambda0

A_mod = 0.00015             # modulation amplitude
f_mod = 2.0              # modulation frequency [Hz]
omega_mod = 2 * np.pi * f_mod

# Time grid (one period)
t = np.linspace(0, 10 / f_mod, 201)

# ---------- C‑band wavelengths (11 total, equally spaced) ----------
# Symmetric around 1550 nm with step 4 nm (covers 1530–1570 nm)
lambda_start = 1520e-9
lambda_stop = 1580e-9
n_lambda = 16
wavelengths = np.linspace(lambda_start, lambda_stop, n_lambda)  # includes 1550 nm

# ---------- Generate base birefringence at λ₀ ----------
dz = L_F / 1
z, beta0 = generate_birefringence_profile(
    L=L, L_F=L_F, D_pmd=D_pmd, lambda0=lambda0, dz=dz, seed=42
)

# ---------- Pre‑compute scaling factors for each λ ----------
scales = np.array([lambda0 / wl for wl in wavelengths])  # = ω/ω₀

# ---------- Time loop ----------
phi_all = np.zeros((n_lambda, len(t)))   # rotation angles [rad]

for idx_t, time in enumerate(t):
    # common modulation factor
    mod = 1.0 + A_mod * np.sin(omega_mod * time)

    for idx_wl, wl in enumerate(wavelengths):
        beta_t = scales[idx_wl] * mod * beta0
        U = propagate_unitary(z, beta_t)
        # rotation angle from |trace|
        phi = 2.0 * np.arccos(np.clip(np.abs(np.trace(U)) / 2.0, -1.0, 1.0))
        phi_all[idx_wl, idx_t] = phi

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, n_lambda))

for idx_wl, wl in enumerate(wavelengths):
    ax.plot(t, phi_all[idx_wl], color=colors[idx_wl],
            label=f'{wl*1e9:.1f} nm' if idx_wl % 2 == 0 else '',
            alpha=0.8)

# Highlight central wavelength
ax.plot(t, phi_all[n_lambda//2], 'k-', linewidth=2, label=f'{lambda0*1e9:.0f} nm (central)')

ax.set_xlabel('Time [s]')
ax.set_ylabel('Rotation angle φ [rad]')
ax.set_title('Output rotation angle vs. time for 11 C‑band wavelengths\n(same sinusoidal perturbation)')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
ax.grid(True)
plt.tight_layout()
plt.savefig('../output/multiwavelength_modulation.png', dpi=150)
plt.show()