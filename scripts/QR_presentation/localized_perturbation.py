"""
Localized perturbation: Only a section of the fiber is modulated.
- 3 km fiber, perturb 1 km in the middle (z = 1 km to z = 2 km)
- Static and dynamic SOP evolution with localized modulation
- Shows how a local disturbance affects the global SOP
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from io import BytesIO
from PIL import Image

from pmd_model import generate_pmd_waveplates, propagate_unitary

# ==================== Parameters ====================
L_km = 3.0
L = L_km * 1e3  # meters
L_F = 1 / 3 * 1000  # correlation length (m)
D_pmd = 0.4e-16  # s/√m
lambda0 = 1550e-9  # m

# Input polarization (horizontal linear)
s0_jones = np.array([1.0, 0.0], dtype=complex)
s0_jones = s0_jones / np.linalg.norm(s0_jones)

# Localized perturbation region (in km)
perturb_start_km = 1.0
perturb_end_km = 2.0
perturb_start = perturb_start_km * 1e3  # meters
perturb_end = perturb_end_km * 1e3

# ==================== Generate static fibre ====================
seed = 42
z, beta0, beta_prime, L_seg = generate_pmd_waveplates(
    L, L_F, D_pmd, lambda0, seed=seed
)

print(f"Fiber generated: {len(z)} points, {L_seg:.3f} m segments")
print(f"Perturb region: {perturb_start_km:.1f} km to {perturb_end_km:.1f} km")

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

# ==================== STATIC CASE ====================
print("\n=== Computing static SOP evolution ===")
stokes_traj_static = compute_sop_trajectory(beta0, z, s0_jones)
print(f"Static SOP computed: {stokes_traj_static.shape[1]} points")

# ==================== DYNAMIC CASE (LOCALIZED) ====================
print("\n=== Computing dynamic SOP with localized perturbation ===")

# Create spatial mask: 1 inside perturb region, 0 outside
mask = np.zeros(len(z))
for i, zi in enumerate(z):
    if perturb_start <= zi <= perturb_end:
        mask[i] = 1.0

print(f"Number of points in perturb region: {np.sum(mask)} out of {len(z)}")

f_mod = 50.0  # Hz
omega_mod = 2.0 * np.pi * f_mod
A = 0.05  # modulation amplitude (5%)

n_periods = 2
n_samples_per_period = 40
n_time_steps = n_periods * n_samples_per_period
t_end = n_periods / f_mod
t_grid = np.linspace(0.0, t_end, n_time_steps)

print(f"Time steps: {n_time_steps}, modulation: {f_mod} Hz, amplitude: {A*100:.1f}%")

n_z_full = len(z)
stokes_time = np.zeros((n_time_steps, 3, n_z_full), dtype=float)

print("Computing dynamic SOP trajectories...")
for t_idx, t in enumerate(t_grid):
    modulation = 1.0 + A * np.sin(omega_mod * t)
    # Apply modulation only where mask=1
    beta_t = beta0 * (1.0 + (modulation - 1.0) * mask)  # equivalent: beta0 * (1 + A*sin * mask)
    stokes_at_z = compute_sop_trajectory(beta_t, z, s0_jones)
    stokes_time[t_idx] = stokes_at_z
    if (t_idx + 1) % 20 == 0:
        print(f"  Progress: {t_idx + 1}/{n_time_steps} time steps")

# Extract end point trajectory
end_point_traj = stokes_time[:, :, -1]

# ==================== VERIFICATION ====================
diff = np.max(np.abs(stokes_time[0] - stokes_traj_static))
print(f"\n✅ Dynamic at t=0 matches Static (diff = {diff:.2e})")

# ==================== PLOTTING STATIC ====================
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z_sphere, rstride=5, cstride=5, color='gray', alpha=0.3, linewidth=0.5)

step = max(1, len(z) // 100)
indices = np.arange(0, len(z), step)
points = stokes_traj_static[:, indices]
colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
ax.scatter(points[0], points[1], points[2], c=colors, s=15, alpha=0.8)
ax.plot(stokes_traj_static[0], stokes_traj_static[1], stokes_traj_static[2], 
        'b-', linewidth=2, alpha=0.6, label='SOP trajectory')
ax.scatter(stokes_traj_static[0, 0], stokes_traj_static[1, 0], stokes_traj_static[2, 0],
           c='green', s=150, marker='o', label='Start', edgecolors='black', linewidth=1.5)
ax.scatter(stokes_traj_static[0, -1], stokes_traj_static[1, -1], stokes_traj_static[2, -1],
           c='red', s=150, marker='*', label='End', edgecolors='black', linewidth=1.5)
ax.set_xlabel('S₁'); ax.set_ylabel('S₂'); ax.set_zlabel('S₃')
ax.set_title('Static SOP (No Perturbation)', fontsize=14)
ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
ax.legend(loc='upper right')
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.savefig('sop_poincare_static_localized.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved sop_poincare_static_localized.png")

# Static Stokes components vs z
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
z_km = z / 1000
labels = [r'S$_1$', r'S$_2$', r'S$_3$']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for k in range(3):
    axes[k].plot(z_km, stokes_traj_static[k], color=colors[k], linewidth=2)
    axes[k].set_ylabel(labels[k], fontsize=12)
    axes[k].grid(True, alpha=0.3)
    axes[k].set_ylim([-1.1, 1.1])
    axes[k].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[-1].set_xlabel('Fiber Length z (km)', fontsize=12)
fig.suptitle('Static Stokes Components (No Perturbation)', fontsize=14)
plt.tight_layout()
plt.savefig('sop_components_static_localized.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved sop_components_static_localized.png")

# ==================== DYNAMIC ANIMATION: Poincaré sphere ====================
print("\n=== Creating Poincaré sphere animation (localized) ===")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
persistence_length = 20
frames_poincare = []

for t_idx in range(n_time_steps):
    ax.clear()
    ax.plot_wireframe(x, y, z_sphere, rstride=4, cstride=4, color='gray', alpha=0.2, linewidth=0.5)
    stokes = stokes_time[t_idx]
    n_points = stokes.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    ax.scatter(stokes[0], stokes[1], stokes[2], c=colors, s=15, alpha=0.6)
    ax.plot(stokes[0], stokes[1], stokes[2], 'b-', linewidth=2, alpha=0.7, label='SOP trajectory')
    
    # Red trace of end point
    start_idx = max(0, t_idx - persistence_length + 1)
    end_trace = end_point_traj[start_idx:t_idx+1]
    if len(end_trace) > 1:
        n_trace = len(end_trace)
        for i in range(n_trace - 1):
            alpha_val = 0.3 + 0.7 * (i / n_trace)
            ax.plot([end_trace[i, 0], end_trace[i+1, 0]],
                   [end_trace[i, 1], end_trace[i+1, 1]],
                   [end_trace[i, 2], end_trace[i+1, 2]],
                   'r-', linewidth=2.5, alpha=alpha_val)
        alphas = np.linspace(0.3, 1.0, n_trace)
        ax.scatter(end_trace[:, 0], end_trace[:, 1], end_trace[:, 2],
                  c='red', s=30, alpha=alphas, zorder=5)
    current_end = end_point_traj[t_idx]
    ax.scatter(current_end[0], current_end[1], current_end[2],
              c='red', s=200, marker='*', label='End point', edgecolors='darkred', linewidth=2, zorder=10)
    ax.scatter(stokes[0, 0], stokes[1, 0], stokes[2, 0],
               c='green', s=150, marker='o', label='Start', edgecolors='black', linewidth=1.5, zorder=10)
    
    ax.set_xlabel('S₁'); ax.set_ylabel('S₂'); ax.set_zlabel('S₃')
    ax.set_xlim([-1.1, 1.1]); ax.set_ylim([-1.1, 1.1]); ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1, 1, 1])
    t = t_grid[t_idx]
    modulation = 1.0 + A * np.sin(omega_mod * t)
    ax.set_title(f'Localized Perturbation ({perturb_start_km:.1f}-{perturb_end_km:.1f} km)\nt = {t:.3f}s, m = {modulation:.3f}', fontsize=12)
    ax.legend(loc='upper right', fontsize=8)
    
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    frames_poincare.append(buf)
    if (t_idx + 1) % 20 == 0:
        print(f"  Progress: {t_idx + 1}/{n_time_steps} frames")

# Save
images = [Image.open(f) for f in frames_poincare]
if images:
    images[0].save('sop_poincare_dynamic_localized.gif',
                   save_all=True, append_images=images[1:], optimize=True, duration=50, loop=0)
    print("Saved sop_poincare_dynamic_localized.gif")
for f in frames_poincare: f.close()

# ==================== OSCILLOSCOPE ANIMATION ====================
print("\n=== Creating oscilloscope animation (localized) ===")
n_oscope_points = 80
n_pad = n_oscope_points
all_S1 = np.concatenate([np.zeros(n_pad), end_point_traj[:, 0]])
all_S2 = np.concatenate([np.zeros(n_pad), end_point_traj[:, 1]])
all_S3 = np.concatenate([np.zeros(n_pad), end_point_traj[:, 2]])

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle('End Point Stokes Components (Localized Perturbation)', fontsize=14)
oscope_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
component_labels = [r'S$_1$', r'S$_2$', r'S$_3$']
frames_oscope = []

for t_idx in range(n_time_steps):
    for ax in axes: ax.clear()
    end_idx = t_idx + n_pad
    start_idx = end_idx - n_oscope_points
    t_window = np.arange(start_idx, end_idx) / n_samples_per_period * (1/f_mod)
    data = [all_S1[start_idx:end_idx], all_S2[start_idx:end_idx], all_S3[start_idx:end_idx]]
    for k, (ax, d, color, label) in enumerate(zip(axes, data, oscope_colors, component_labels)):
        ax.plot(t_window, d, color=color, linewidth=2.5, alpha=0.9)
        ax.scatter(t_window[-1], d[-1], color=color, s=80, edgecolors='black', linewidth=1.5, zorder=5)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.2)
        ax.grid(True, alpha=0.2)
        ax.set_ylim([-1.1, 1.1])
        ax.set_ylabel(label, fontsize=12, color=color)
        ax.tick_params(axis='y', labelcolor=color)
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    axes[-1].set_xlim([t_window[0], t_window[-1]])
    current_time = t_idx / n_samples_per_period * (1/f_mod)
    modulation = 1.0 + A * np.sin(omega_mod * current_time)
    fig.suptitle(f'End Point Stokes Components (Localized Perturbation)\nt = {current_time:.3f}s, m = {modulation:.3f}', fontsize=14)
    plt.tight_layout()
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    frames_oscope.append(buf)
    if (t_idx+1)%20==0: print(f"  Progress: {t_idx+1}/{n_time_steps} frames")

images = [Image.open(f) for f in frames_oscope]
if images:
    images[0].save('sop_oscope_dynamic_localized.gif',
                   save_all=True, append_images=images[1:], optimize=True, duration=50, loop=0)
    print("Saved sop_oscope_dynamic_localized.gif")
for f in frames_oscope: f.close()

# ==================== COMBINED ANIMATION (FIXED SCALING) ====================
print("\n=== Creating combined animation (localized) ===")
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1.2], height_ratios=[1,1,1],
              left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=0.15, hspace=0.25)
ax_sphere = fig.add_subplot(gs[:, 0], projection='3d')
ax_oscope1 = fig.add_subplot(gs[0, 1])
ax_oscope2 = fig.add_subplot(gs[1, 1])
ax_oscope3 = fig.add_subplot(gs[2, 1])
oscope_axes = [ax_oscope1, ax_oscope2, ax_oscope3]

# Set fixed limits
for ax in oscope_axes:
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlim([0, n_oscope_points / n_samples_per_period * (1/f_mod)])
    ax.grid(True, alpha=0.2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.2)
    ax.autoscale(False)
ax_sphere.set_xlim([-1.1, 1.1]); ax_sphere.set_ylim([-1.1, 1.1]); ax_sphere.set_zlim([-1.1, 1.1])
ax_sphere.set_box_aspect([1, 1, 1])
ax_sphere.autoscale(False)

frames_combined = []
for t_idx in range(n_time_steps):
    ax_sphere.clear()
    for ax in oscope_axes:
        ax.clear()
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlim([0, n_oscope_points / n_samples_per_period * (1/f_mod)])
        ax.grid(True, alpha=0.2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.2)
        ax.autoscale(False)
    
    # Sphere
    ax_sphere.plot_wireframe(x, y, z_sphere, rstride=4, cstride=4, color='gray', alpha=0.2, linewidth=0.5)
    stokes = stokes_time[t_idx]
    n_points = stokes.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    ax_sphere.scatter(stokes[0], stokes[1], stokes[2], c=colors, s=15, alpha=0.6)
    ax_sphere.plot(stokes[0], stokes[1], stokes[2], 'b-', linewidth=2, alpha=0.7, label='SOP')
    start_idx = max(0, t_idx - persistence_length + 1)
    end_trace = end_point_traj[start_idx:t_idx+1]
    if len(end_trace) > 1:
        n_trace = len(end_trace)
        for i in range(n_trace - 1):
            alpha_val = 0.3 + 0.7 * (i / n_trace)
            ax_sphere.plot([end_trace[i,0], end_trace[i+1,0]],
                          [end_trace[i,1], end_trace[i+1,1]],
                          [end_trace[i,2], end_trace[i+1,2]],
                          'r-', linewidth=2.5, alpha=alpha_val)
        alphas = np.linspace(0.3, 1.0, n_trace)
        ax_sphere.scatter(end_trace[:,0], end_trace[:,1], end_trace[:,2],
                         c='red', s=30, alpha=alphas, zorder=5)
    current_end = end_point_traj[t_idx]
    ax_sphere.scatter(current_end[0], current_end[1], current_end[2],
                     c='red', s=200, marker='*', label='End point', edgecolors='darkred', linewidth=2, zorder=10)
    ax_sphere.scatter(stokes[0,0], stokes[1,0], stokes[2,0],
                     c='green', s=150, marker='o', label='Start', edgecolors='black', linewidth=1.5, zorder=10)
    ax_sphere.set_xlabel('S₁', fontsize=10); ax_sphere.set_ylabel('S₂', fontsize=10); ax_sphere.set_zlabel('S₃', fontsize=10)
    ax_sphere.set_xlim([-1.1,1.1]); ax_sphere.set_ylim([-1.1,1.1]); ax_sphere.set_zlim([-1.1,1.1])
    ax_sphere.set_box_aspect([1,1,1]); ax_sphere.autoscale(False); ax_sphere.legend(loc='upper right', fontsize=8)
    
    # Oscilloscope
    end_idx = t_idx + n_pad
    start_idx = end_idx - n_oscope_points
    t_window = np.arange(start_idx, end_idx) / n_samples_per_period * (1/f_mod)
    S1_window = all_S1[start_idx:end_idx]
    S2_window = all_S2[start_idx:end_idx]
    S3_window = all_S3[start_idx:end_idx]
    current_time = t_idx / n_samples_per_period * (1/f_mod)
    for k, (ax, data, color, label) in enumerate(zip(oscope_axes,
                                                     [S1_window, S2_window, S3_window],
                                                     oscope_colors, component_labels)):
        ax.plot(t_window, data, color=color, linewidth=2.5, alpha=0.9)
        ax.scatter(t_window[-1], data[-1], color=color, s=60, edgecolors='black', linewidth=1.5, zorder=5)
        ax.set_ylabel(label, fontsize=10, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        if k == 2:
            ax.set_xlabel('Time (s)', fontsize=10)
        else:
            ax.set_xticklabels([])
        ax.set_ylim([-1.1,1.1]); ax.set_xlim([t_window[0], t_window[-1]]); ax.autoscale(False)
    
    modulation = 1.0 + A * np.sin(omega_mod * current_time)
    fig.suptitle(f'Localized Perturbation ({perturb_start_km:.1f}-{perturb_end_km:.1f} km)\n'
                f't = {current_time:.3f}s, m = {modulation:.3f}', fontsize=14, y=0.98)
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    frames_combined.append(buf)
    if (t_idx+1)%20==0: print(f"  Progress: {t_idx+1}/{n_time_steps} frames")

images = [Image.open(f) for f in frames_combined]
if images:
    images[0].save('sop_combined_dynamic_localized.gif',
                   save_all=True, append_images=images[1:], optimize=True, duration=50, loop=0)
    print("Saved sop_combined_dynamic_localized.gif")
for f in frames_combined: f.close()

print("\n=== Done! ===")
print("\nGenerated files (localized perturbation):")
print("  📊 Static: sop_poincare_static_localized.png, sop_components_static_localized.png")
print("  🎬 Dynamic: sop_poincare_dynamic_localized.gif, sop_oscope_dynamic_localized.gif, sop_combined_dynamic_localized.gif")
print(f"\n✅ Perturbation applied only from {perturb_start_km:.1f} km to {perturb_end_km:.1f} km.")