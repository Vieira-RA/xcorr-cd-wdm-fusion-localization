#!/usr/bin/env python3
"""Validate multi‑wavelength birefringence scaling and reproducibility."""
import numpy as np
import matplotlib.pyplot as plt
from pmd_model import generate_multiwavelength_birefringence

# Parameters
L = 1000.0                # 1 km
L_F = 10.0                # 10 m
D_pmd = 3.16e-15          # 0.1 ps/√km
lambda0 = 1550e-9
wavelengths = [1530e-9, 1550e-9, 1565e-9]  # C‑band edges and centre

# Generate with a fixed seed
z, beta0, profiles = generate_multiwavelength_birefringence(
    L, L_F, D_pmd, lambda0, wavelengths, seed=42
)

# ---------- Check 1: scaling correctness ----------
# At λ = λ₀ the profile should equal beta0
np.testing.assert_allclose(profiles[1550e-9], beta0, atol=1e-15)
print("Check 1 passed: 1550 nm profile unchanged.")

# At other λ, verify magnitude ratio
for wl in [1530e-9, 1565e-9]:
    expected_scale = lambda0 / wl
    actual_scale = np.linalg.norm(profiles[wl], axis=0) / np.linalg.norm(beta0, axis=0)
    np.testing.assert_allclose(actual_scale, expected_scale * np.ones_like(actual_scale), atol=1e-12)
    print(f"Check 2 ({wl*1e9:.1f} nm): scale factor = {expected_scale:.4f}, OK.")

# ---------- Check 3: reproducibility ----------
z2, beta0_2, profiles2 = generate_multiwavelength_birefringence(
    L, L_F, D_pmd, lambda0, wavelengths, seed=42
)
np.testing.assert_array_equal(beta0_2, beta0)
for wl in wavelengths:
    np.testing.assert_array_equal(profiles2[wl], profiles[wl])
print("Check 3 passed: identical output for same seed.")

# ---------- Check 4: different seed gives different fibre ----------
z3, beta0_3, _ = generate_multiwavelength_birefringence(
    L, L_F, D_pmd, lambda0, wavelengths, seed=123
)
assert not np.allclose(beta0_3, beta0)
print("Check 4 passed: different seed → different fibre.")

# ---------- Visualisation (optional) ----------
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(z, np.linalg.norm(beta0, axis=0), label=f'{lambda0*1e9:.0f} nm', color='black')
for wl in [1530e-9, 1565e-9]:
    ax.plot(z, np.linalg.norm(profiles[wl], axis=0), '--', label=f'{wl*1e9:.0f} nm')
ax.set_xlabel('Position [m]')
ax.set_ylabel('|β| [rad/m]')
ax.set_title('Birefringence magnitude at three C‑band wavelengths')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('../output/multiwavelength_beta.png', dpi=150)
plt.show()