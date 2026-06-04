#!/usr/bin/env python3
"""Demonstrate the difference between continuous and discrete birefringence
autocorrelation functions and their integral correlation lengths."""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lc = 10.0       # correlation length for exponential (integral scale = Lc)
L_seg = 10.0    # segment length in wave‑plate model (integral scale = L_seg/2)
R0 = 1.0        # variance <β²> = 1 for simplicity

# ---------- 1. Continuous exponential correlation ----------
def R_exp(u, Lc):
    return R0 * np.exp(-np.abs(u) / Lc)

# ---------- 2. Discrete piecewise‑constant (triangular) correlation ----------
def R_tri(u, L_seg):
    return R0 * np.maximum(1 - np.abs(u) / L_seg, 0)

# Numerical integration to compute integral scales
u_fine = np.linspace(-200, 200, 200001)
du = u_fine[1] - u_fine[0]

# Integral correlation length = ∫_0^∞ R(u)/R(0) du
# Use symmetry: full integral from -∞ to ∞ divided by 2
L_int_exp = np.trapezoid(R_exp(u_fine, Lc) / R0, u_fine) / 2
L_int_tri = np.trapezoid(R_tri(u_fine, L_seg) / R0, u_fine) / 2

print(f"Exponential process: Lc = {Lc:.1f} m, integral length = {L_int_exp:.1f} m")
print(f"Wave‑plate model:    L_seg = {L_seg:.1f} m, integral length = {L_int_tri:.1f} m")
print(f"Ratio L_int_exp / L_int_tri = {L_int_exp / L_int_tri:.2f} (should be 2 if L_seg = Lc)")

# ---------- Plotting ----------
u_plot = np.linspace(-30, 30, 500)
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(u_plot, R_exp(u_plot, Lc), 'b-', linewidth=2,
        label=f'Exponential: R(u) = exp(-|u|/{Lc:.0f} m)')
ax.plot(u_plot, R_tri(u_plot, L_seg), 'r--', linewidth=2,
        label=f'Piecewise constant (L_seg = {L_seg:.0f} m): R(u) = tri(u/{L_seg:.0f})')

# Indicate integral lengths as areas under the positive half
u_pos = np.linspace(0, 30, 200)
ax.fill_between(u_pos, 0, R_exp(u_pos, Lc), alpha=0.15, color='blue')
ax.fill_between(-u_pos, 0, R_exp(-u_pos, Lc), alpha=0.15, color='blue')
ax.fill_between(u_pos, 0, R_tri(u_pos, L_seg), alpha=0.15, color='red')
ax.fill_between(-u_pos, 0, R_tri(-u_pos, L_seg), alpha=0.15, color='red')

# Annotations
ax.annotate(f'Integral length = {L_int_exp:.1f} m',
            xy=(L_int_exp, 0.6), xytext=(L_int_exp+2, 0.7),
            arrowprops=dict(arrowstyle='->'), fontsize=10, color='blue')
ax.annotate(f'Integral length = {L_int_tri:.1f} m',
            xy=(L_int_tri, 0.5), xytext=(L_int_tri+2, 0.4),
            arrowprops=dict(arrowstyle='->'), fontsize=10, color='red')

ax.set_xlabel('Lag u [m]')
ax.set_ylabel('Autocorrelation R(u) / ⟨β²⟩')
ax.set_title('Autocorrelation functions: continuous vs. discrete birefringence')
ax.legend(loc='upper right')
ax.grid(True)
plt.tight_layout()
plt.savefig('../output/correlation_length_comparison.png', dpi=150)
plt.show()