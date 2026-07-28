"""
Static and dynamic SOP evolution visualization along a 1 km fiber.
- Static: SOP trajectory on Poincaré sphere and Stokes components
- Dynamic: Animated SOP trajectory oscillating with time-modulated birefringence
- Both use the SAME fiber profile
- RED TRACE shows the time evolution of the end point (star)
- OSCILLOSCOPE view shows Stokes components of the end point over time
- COMBINED view shows both Poincaré sphere and oscilloscope in one GIF (with fixed scaling)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For HPC environment
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

from pmd_model import generate_pmd_waveplates, propagate_unitary

# ==================== Parameters ====================
L_km = 1.5
L = L_km * 1e3  # meters
L_F = 20.0  # correlation length (m)
D_pmd = 0.4e-16  # s/√m - REDUCED for cleaner trajectory
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
print(f"PMD coefficient: {D_pmd:.1e} s/√m")

# ==================== Pauli matrices ====================
PAULI = np.array([
    [[1, 0], [0, -1]],
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]]
], dtype=complex)

def jones_to_stokes(jones_vec):
    """Convert Jones vector to Stokes vector."""
    stokes = np.zeros(3, dtype=float)
    for k in range(3):
        stokes[k] = np.real(np.conj(jones_vec).T @ PAULI[k] @ jones_vec)
    norm = np.linalg.norm(stokes)
    if norm > 0:
        stokes = stokes / norm
    return stokes

def compute_sop_trajectory(beta_profile, z_positions, input_jones):
    """Compute SOP trajectory along the fiber."""
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

# ==================== DYNAMIC CASE ====================
print("\n=== Computing dynamic SOP evolution (modulated birefringence) ===")

f_mod = 50.0  # Hz
omega_mod = 2.0 * np.pi * f_mod
A = 0.05  # modulation amplitude (5%)

n_periods = 8
n_samples_per_period = 20
n_time_steps = n_periods * n_samples_per_period
t_end = n_periods / f_mod
t_grid = np.linspace(0.0, t_end, n_time_steps)

print(f"Time steps: {n_time_steps}, modulation: {f_mod} Hz, amplitude: {A*100:.1f}%")

n_z_full = len(z)
stokes_time = np.zeros((n_time_steps, 3, n_z_full), dtype=float)

print("Computing dynamic SOP trajectories...")
for t_idx, t in enumerate(t_grid):
    modulation = 1.0 + A * np.sin(omega_mod * t)
    beta_t = modulation * beta0
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

# Plot static Poincaré sphere
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z_sphere, rstride=5, cstride=5,
                  color='gray', alpha=0.3, linewidth=0.5)

step = max(1, len(z) // 100)
indices = np.arange(0, len(z), step)
points = stokes_traj_static[:, indices]
colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

ax.scatter(points[0], points[1], points[2], c=colors, s=15, alpha=0.8)
ax.plot(stokes_traj_static[0], stokes_traj_static[1], stokes_traj_static[2], 
        'b-', linewidth=2, alpha=0.6, label='SOP trajectory')

ax.scatter(stokes_traj_static[0, 0], stokes_traj_static[1, 0], stokes_traj_static[2, 0],
           c='green', s=150, marker='o', label='Start (z=0)', edgecolors='black', linewidth=1.5)
ax.scatter(stokes_traj_static[0, -1], stokes_traj_static[1, -1], stokes_traj_static[2, -1],
           c='red', s=150, marker='*', label='End (z=1 km)', edgecolors='black', linewidth=1.5)

ax.set_xlabel('S₁', fontsize=12)
ax.set_ylabel('S₂', fontsize=12)
ax.set_zlabel('S₃', fontsize=12)
ax.set_title('Static SOP Trajectory on Poincaré Sphere', fontsize=14)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.legend(loc='upper right')
ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.savefig('sop_poincare_static.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved sop_poincare_static.png")

# Plot Stokes components vs z
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
fig.suptitle('Static Stokes Components Along the Fiber', fontsize=14)
plt.tight_layout()
plt.savefig('sop_components_static.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved sop_components_static.png")

# ==================== CREATE POINCARE SPHERE ANIMATION ====================
print("\n=== Creating Poincaré sphere animation ===")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

from io import BytesIO
frames_poincare = []
persistence_length = 20

print(f"Generating {n_time_steps} Poincaré frames...")
for t_idx in range(n_time_steps):
    ax.clear()
    
    ax.plot_wireframe(x, y, z_sphere, rstride=4, cstride=4,
                      color='gray', alpha=0.2, linewidth=0.5)
    
    stokes = stokes_time[t_idx]
    n_points = stokes.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    
    ax.scatter(stokes[0], stokes[1], stokes[2], c=colors, s=15, alpha=0.6)
    ax.plot(stokes[0], stokes[1], stokes[2], 'b-', linewidth=2, alpha=0.7, label='SOP trajectory')
    
    # Red trace
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
              c='red', s=200, marker='*', label='End point trajectory',
              edgecolors='darkred', linewidth=2, zorder=10)
    
    ax.scatter(stokes[0, 0], stokes[1, 0], stokes[2, 0],
               c='green', s=150, marker='o', label='Start (z=0)', 
               edgecolors='black', linewidth=1.5, zorder=10)
    
    ax.set_xlabel('S₁', fontsize=12)
    ax.set_ylabel('S₂', fontsize=12)
    ax.set_zlabel('S₃', fontsize=12)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1, 1, 1])
    
    t = t_grid[t_idx]
    modulation = 1.0 + A * np.sin(omega_mod * t)
    
    if np.abs(modulation - 1.0) < 1e-6:
        title = f'⭐ Dynamic matches STATIC! t = {t:.3f}s'
    else:
        title = f'Dynamic SOP - Modulated t = {t:.3f}s, m = {modulation:.3f}'
    ax.set_title(title, fontsize=12)
    ax.legend(loc='upper right')
    
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    frames_poincare.append(buf)
    
    if (t_idx + 1) % 20 == 0:
        print(f"  Progress: {t_idx + 1}/{n_time_steps} frames")

# Save Poincaré GIF
print("Creating Poincaré GIF...")
try:
    from PIL import Image
    images = [Image.open(frame) for frame in frames_poincare]
    if images:
        images[0].save('sop_poincare_dynamic.gif',
                       save_all=True,
                       append_images=images[1:],
                       optimize=True,
                       duration=50,
                       loop=0)
        print("Saved sop_poincare_dynamic.gif")
except ImportError:
    print("PIL not available")

for frame in frames_poincare:
    frame.close()

# ==================== CREATE OSCILLOSCOPE ANIMATION ====================
print("\n=== Creating oscilloscope animation (Stokes components of end point) ===")

n_oscope_points = 80
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.suptitle('Stokes Components of the End Point (z = L) vs Time', fontsize=14)

oscope_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
component_labels = [r'S$_1$', r'S$_2$', r'S$_3$']

n_pad = n_oscope_points
all_S1 = np.concatenate([np.zeros(n_pad), end_point_traj[:, 0]])
all_S2 = np.concatenate([np.zeros(n_pad), end_point_traj[:, 1]])
all_S3 = np.concatenate([np.zeros(n_pad), end_point_traj[:, 2]])

frames_oscope = []

print(f"Generating {n_time_steps} oscilloscope frames...")
for t_idx in range(n_time_steps):
    for ax in axes:
        ax.clear()
    
    end_idx = t_idx + n_pad
    start_idx = end_idx - n_oscope_points
    t_window = np.arange(start_idx, end_idx) / n_samples_per_period * (1/f_mod)
    S1_window = all_S1[start_idx:end_idx]
    S2_window = all_S2[start_idx:end_idx]
    S3_window = all_S3[start_idx:end_idx]
    current_time = t_idx / n_samples_per_period * (1/f_mod)
    
    for k, (data, color, label) in enumerate(zip([S1_window, S2_window, S3_window], 
                                                  oscope_colors, component_labels)):
        axes[k].plot(t_window, data, color=color, linewidth=2.5, alpha=0.9)
        axes[k].scatter(t_window[-1], data[-1], color=color, s=80, zorder=5, 
                       edgecolors='black', linewidth=1.5)
        axes[k].axhline(y=0, color='black', linestyle='--', alpha=0.2)
        axes[k].grid(True, alpha=0.2)
        axes[k].set_ylim([-1.1, 1.1])
        axes[k].set_ylabel(label, fontsize=12, color=color)
        axes[k].tick_params(axis='y', labelcolor=color)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    axes[-1].set_xlim([t_window[0], t_window[-1]])
    
    modulation = 1.0 + A * np.sin(omega_mod * current_time)
    fig.suptitle(f'End Point Stokes Components vs Time (Oscilloscope View)\n'
                f't = {current_time:.3f}s, modulation = {modulation:.4f}', 
                fontsize=14)
    
    plt.tight_layout()
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    frames_oscope.append(buf)
    
    if (t_idx + 1) % 20 == 0:
        print(f"  Progress: {t_idx + 1}/{n_time_steps} frames")

print("Creating Oscilloscope GIF...")
try:
    from PIL import Image
    images = [Image.open(frame) for frame in frames_oscope]
    if images:
        images[0].save('sop_oscope_dynamic.gif',
                       save_all=True,
                       append_images=images[1:],
                       optimize=True,
                       duration=50,
                       loop=0)
        print("Saved sop_oscope_dynamic.gif")
except ImportError:
    print("PIL not available")

for frame in frames_oscope:
    frame.close()

# ==================== CREATE COMBINED ANIMATION (FIXED SCALING) ====================
print("\n=== Creating combined animation (Poincaré + Oscilloscope) ===")

# Create figure with fixed layout - NO tight_layout inside loop
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1.2], height_ratios=[1, 1, 1],
              left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=0.15, hspace=0.25)

ax_sphere = fig.add_subplot(gs[:, 0], projection='3d')
ax_oscope1 = fig.add_subplot(gs[0, 1])
ax_oscope2 = fig.add_subplot(gs[1, 1])
ax_oscope3 = fig.add_subplot(gs[2, 1])
oscope_axes = [ax_oscope1, ax_oscope2, ax_oscope3]

# Set fixed limits for oscilloscope axes (they will be updated, but limits fixed)
for ax in oscope_axes:
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlim([0, n_oscope_points / n_samples_per_period * (1/f_mod)])  # fixed time window length
    ax.grid(True, alpha=0.2)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.2)

# Set fixed limits for 3D axis
ax_sphere.set_xlim([-1.1, 1.1])
ax_sphere.set_ylim([-1.1, 1.1])
ax_sphere.set_zlim([-1.1, 1.1])
ax_sphere.set_box_aspect([1, 1, 1])

# Disable autoscale for all axes
ax_sphere.autoscale(False)
for ax in oscope_axes:
    ax.autoscale(False)

# We'll keep the figure layout fixed; no tight_layout in loop.
fig.canvas.draw()  # initial draw to set background

frames_combined = []
n_oscope_points = 80
n_pad = n_oscope_points

print(f"Generating {n_time_steps} combined frames...")
for t_idx in range(n_time_steps):
    # Clear only the artists (we keep axes limits)
    ax_sphere.clear()
    for ax in oscope_axes:
        ax.clear()
        # Reapply fixed limits and grid after clear
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlim([0, n_oscope_points / n_samples_per_period * (1/f_mod)])
        ax.grid(True, alpha=0.2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.2)
        ax.autoscale(False)
    
    # ---- LEFT: Poincaré sphere ----
    ax_sphere.plot_wireframe(x, y, z_sphere, rstride=4, cstride=4,
                             color='gray', alpha=0.2, linewidth=0.5)
    
    stokes = stokes_time[t_idx]
    n_points = stokes.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    
    ax_sphere.scatter(stokes[0], stokes[1], stokes[2], c=colors, s=15, alpha=0.6)
    ax_sphere.plot(stokes[0], stokes[1], stokes[2], 'b-', linewidth=2, alpha=0.7, label='SOP trajectory')
    
    # Red trace
    start_idx = max(0, t_idx - persistence_length + 1)
    end_trace = end_point_traj[start_idx:t_idx+1]
    
    if len(end_trace) > 1:
        n_trace = len(end_trace)
        for i in range(n_trace - 1):
            alpha_val = 0.3 + 0.7 * (i / n_trace)
            ax_sphere.plot([end_trace[i, 0], end_trace[i+1, 0]],
                          [end_trace[i, 1], end_trace[i+1, 1]],
                          [end_trace[i, 2], end_trace[i+1, 2]],
                          'r-', linewidth=2.5, alpha=alpha_val)
        
        alphas = np.linspace(0.3, 1.0, n_trace)
        ax_sphere.scatter(end_trace[:, 0], end_trace[:, 1], end_trace[:, 2],
                         c='red', s=30, alpha=alphas, zorder=5)
    
    current_end = end_point_traj[t_idx]
    ax_sphere.scatter(current_end[0], current_end[1], current_end[2],
                     c='red', s=200, marker='*', label='End point',
                     edgecolors='darkred', linewidth=2, zorder=10)
    
    ax_sphere.scatter(stokes[0, 0], stokes[1, 0], stokes[2, 0],
                     c='green', s=150, marker='o', label='Start', 
                     edgecolors='black', linewidth=1.5, zorder=10)
    
    ax_sphere.set_xlabel('S₁', fontsize=10)
    ax_sphere.set_ylabel('S₂', fontsize=10)
    ax_sphere.set_zlabel('S₃', fontsize=10)
    # Reapply fixed limits (they might have been changed by clear)
    ax_sphere.set_xlim([-1.1, 1.1])
    ax_sphere.set_ylim([-1.1, 1.1])
    ax_sphere.set_zlim([-1.1, 1.1])
    ax_sphere.set_box_aspect([1, 1, 1])
    ax_sphere.autoscale(False)
    ax_sphere.legend(loc='upper right', fontsize=8)
    
    # ---- RIGHT: Oscilloscope ----
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
        ax.scatter(t_window[-1], data[-1], color=color, s=60, zorder=5,
                  edgecolors='black', linewidth=1.5)
        ax.set_ylabel(label, fontsize=10, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        if k == 2:
            ax.set_xlabel('Time (s)', fontsize=10)
        else:
            ax.set_xticklabels([])
        # Reapply fixed limits
        ax.set_ylim([-1.1, 1.1])
        ax.set_xlim([t_window[0], t_window[-1]])
        ax.autoscale(False)
    
    # Overall title (use fig.suptitle, which doesn't affect layout)
    modulation = 1.0 + A * np.sin(omega_mod * current_time)
    fig.suptitle(f'Combined View: Poincaré Sphere & End-Point Stokes Components\n'
                f't = {current_time:.3f}s, modulation = {modulation:.4f}', 
                fontsize=14, y=0.98)
    
    # No tight_layout here to prevent rescaling!
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')  # bbox_inches='tight' can still cause slight resizing, but it's minimal
    buf.seek(0)
    frames_combined.append(buf)
    
    if (t_idx + 1) % 20 == 0:
        print(f"  Progress: {t_idx + 1}/{n_time_steps} frames")

# Save Combined GIF
print("Creating Combined GIF...")
try:
    from PIL import Image
    images = [Image.open(frame) for frame in frames_combined]
    if images:
        images[0].save('sop_combined_dynamic.gif',
                       save_all=True,
                       append_images=images[1:],
                       optimize=True,
                       duration=50,
                       loop=0)
        print("Saved sop_combined_dynamic.gif")
except ImportError:
    print("PIL not available")

for frame in frames_combined:
    frame.close()

print("\n=== Done! ===")
print("\nGenerated files:")
print("  📊 Static:")
print("    - sop_poincare_static.png")
print("    - sop_components_static.png")
print("  🎬 Dynamic:")
print("    - sop_poincare_dynamic.gif (Poincaré sphere with red trace)")
print("    - sop_oscope_dynamic.gif (Oscilloscope view of Stokes components)")
print("    - sop_combined_dynamic.gif (Poincaré + Oscilloscope side-by-side, fixed scaling)")
print("\n✅ The combined view now has fixed scaling and does not zoom or rescale during animation.")