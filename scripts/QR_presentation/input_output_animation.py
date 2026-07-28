"""
Two simple visualizations with improved aesthetics:
1. Static Poincaré sphere showing just the Input SOP (single point).
2. Animated Poincaré sphere showing only the end point moving in time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
from PIL import Image

from pmd_model import generate_pmd_waveplates, propagate_unitary

# ==================== Aesthetics ====================
SPHERE_ALPHA = .8   # Sphere transparency (0 = invisible, 1 = opaque)
SPHERE_COLOR = '#f56942'

# ==================== Parameters ====================
L_km = 3.0
L = L_km * 1e3  # meters
L_F = 100.0  # correlation length (m)
D_pmd = 1.8e-16  # s/√m
lambda0 = 1550e-9  # m

# Input polarization (horizontal linear)
s0_jones = np.array([1.0, 0.0], dtype=complex)
s0_jones = s0_jones / np.linalg.norm(s0_jones)

# ==================== Generate static fibre ====================
seed = 42
z, beta0, beta_prime, L_seg = generate_pmd_waveplates(
    L, L_F, D_pmd, lambda0, seed=seed
)

print(f"Fiber generated: {len(z)} points, {L_seg:.3f} m segments")

# ==================== Pauli matrices ====================
PAULI = np.array([
    [[1, 0], [0, -1]],
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]]
], dtype=complex)

def jones_to_stokes(jones_vec):
    stokes = np.zeros(3, dtype=float)
    for k in range(3):
        stokes[k] = np.real(np.conj(jones_vec).T @ PAULI[k] @ jones_vec)
    norm = np.linalg.norm(stokes)
    if norm > 0:
        stokes = stokes / norm
    return stokes

def compute_sop_trajectory(beta_profile, z_positions, input_jones):
    n_z = len(z_positions)
    stokes = np.zeros((3, n_z), dtype=float)
    stokes[:, 0] = jones_to_stokes(input_jones)
    for i in range(1, n_z):
        z_up_to = z_positions[:i+1]
        beta_up_to = beta_profile[:, :i+1]
        U_at_z = propagate_unitary(z_up_to, beta_up_to)
        jones_at_z = U_at_z @ input_jones
        stokes[:, i] = jones_to_stokes(jones_at_z)
    return stokes

# ==================== Static SOP ====================
input_stokes = jones_to_stokes(s0_jones)
print(f"Input SOP: {input_stokes}")

# ==================== Dynamic SOP (modulated) ====================
print("\n=== Computing dynamic SOP (modulated birefringence) ===")
f_mod = 50.0  # Hz
omega_mod = 2.0 * np.pi * f_mod
A = 0.075  # amplitude (5%)

n_periods = 2
n_samples_per_period = 40
n_time_steps = n_periods * n_samples_per_period
t_end = n_periods / f_mod
t_grid = np.linspace(0.0, t_end, n_time_steps)

print(f"Time steps: {n_time_steps}, modulation: {f_mod} Hz, amplitude: {A*100:.1f}%")

# Pre-compute end point trajectory
end_point_traj = np.zeros((n_time_steps, 3), dtype=float)

for t_idx, t in enumerate(t_grid):
    modulation = 1.0 + A * np.sin(omega_mod * t)
    beta_t = modulation * beta0
    stokes_at_z = compute_sop_trajectory(beta_t, z, s0_jones)
    end_point_traj[t_idx] = stokes_at_z[:, -1]

# ==================== STATIC SPHERE (Input SOP only) ====================
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')

# Draw Poincaré sphere with transparency setting
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 60)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z_sphere, rstride=6, cstride=6,
                  color=SPHERE_COLOR, alpha=SPHERE_ALPHA, linewidth=0.8)

# Input SOP point
ax.scatter(input_stokes[0], input_stokes[1], input_stokes[2],
           c='limegreen', s=400, marker='o', label='Input SOP',
           edgecolors='darkgreen', linewidth=2.5, zorder=10)

# Label
#ax.text(input_stokes[0]*1.15, input_stokes[1]*1.15, input_stokes[2]*1.15,
#        'Input SOP', fontsize=16, color='darkgreen', weight='bold', ha='center')

ax.set_xlabel('S₁', fontsize=16, labelpad=10)
ax.set_ylabel('S₂', fontsize=16, labelpad=10)
ax.set_zlabel('S₃', fontsize=16, labelpad=10)
ax.tick_params(axis='both', labelsize=14)
ax.set_title('Fixed Input State of Polarization', fontsize=20, weight='bold', pad=20)
ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=25, azim=-60)

plt.tight_layout()
plt.savefig('sop_input_static.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved sop_input_static.png")

# ==================== ANIMATED END POINT ONLY ====================
print("\n=== Creating animation of end point only ===")

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')

# Pre-draw sphere with transparency
ax.plot_wireframe(x, y, z_sphere, rstride=6, cstride=6,
                  color=SPHERE_COLOR, alpha=SPHERE_ALPHA, linewidth=0.8)
ax.set_xlabel('S₁', fontsize=16, labelpad=10)
ax.set_ylabel('S₂', fontsize=16, labelpad=10)
ax.set_zlabel('S₃', fontsize=16, labelpad=10)
ax.tick_params(axis='both', labelsize=14)
ax.set_xlim([-1.1, 1.1]); ax.set_ylim([-1.1, 1.1]); ax.set_zlim([-1.1, 1.1])
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=25, azim=-60)

persistence = 10
frames = []

for t_idx in range(n_time_steps):
    ax.clear()
    # Redraw sphere with same transparency
    ax.plot_wireframe(x, y, z_sphere, rstride=6, cstride=6,
                      color=SPHERE_COLOR, alpha=SPHERE_ALPHA, linewidth=0.8)
    ax.set_xlabel('S₁', fontsize=16, labelpad=10)
    ax.set_ylabel('S₂', fontsize=16, labelpad=10)
    ax.set_zlabel('S₃', fontsize=16, labelpad=10)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim([-1.1, 1.1]); ax.set_ylim([-1.1, 1.1]); ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=25, azim=-60)

    start_idx = max(0, t_idx - persistence + 1)
    end_trace = end_point_traj[start_idx:t_idx+1]

    if len(end_trace) > 1:
        n_trace = len(end_trace)
        for i in range(n_trace - 1):
            alpha_val = 0.15 + 0.85 * (i / n_trace)
            ax.plot([end_trace[i, 0], end_trace[i+1, 0]],
                   [end_trace[i, 1], end_trace[i+1, 1]],
                   [end_trace[i, 2], end_trace[i+1, 2]],
                   color='red', linewidth=3.0, alpha=alpha_val)
        alphas = np.linspace(0.15, 1.0, n_trace)
        ax.scatter(end_trace[:, 0], end_trace[:, 1], end_trace[:, 2],
                  c='red', s=40, alpha=alphas, zorder=5)

    current = end_point_traj[t_idx]
    ax.scatter(current[0], current[1], current[2],
              c='crimson', s=350, marker='*', label='End point',
              edgecolors='darkred', linewidth=2.5, zorder=10)

    ax.scatter(input_stokes[0], input_stokes[1], input_stokes[2],
              c='limegreen', s=120, marker='o', label='Input SOP',
              edgecolors='darkgreen', linewidth=1.5, alpha=0.6, zorder=1)

    t = t_grid[t_idx]
    modulation = 1.0 + A * np.sin(omega_mod * t)
    ax.set_title(f'End Point Motion (z = L)\nt = {t:.3f} s, m = {modulation:.3f}',
                 fontsize=18, weight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=14, framealpha=0.9)

    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    frames.append(buf)

    if (t_idx + 1) % 20 == 0:
        print(f"  Progress: {t_idx + 1}/{n_time_steps} frames")

# Save as GIF
print("Creating GIF...")
images = [Image.open(f) for f in frames]
if images:
    images[0].save('sop_endpoint_dynamic.gif',
                   save_all=True, append_images=images[1:],
                   optimize=True, duration=50, loop=0)
    print("Saved sop_endpoint_dynamic.gif")

for f in frames:
    f.close()

print("\n=== Done! ===")
print("\nGenerated files with improved aesthetics:")
print("  - sop_input_static.png (static input SOP only)")
print("  - sop_endpoint_dynamic.gif (end point moving with trail)")
print(f"\nSphere transparency set to {SPHERE_ALPHA}. Adjust SPHERE_ALPHA at the top to change.")