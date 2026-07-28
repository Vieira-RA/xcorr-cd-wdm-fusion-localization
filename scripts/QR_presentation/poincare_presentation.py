import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# ------------------------------------------------------------------
# 1. GENERATE A BEAUTIFUL TRAJECTORY (for illustration purposes)
# ------------------------------------------------------------------
def generate_trajectory(num_points=2000):
    """
    Generate a beautiful, physically meaningful trajectory on the Poincaré sphere.
    This simulates a SOP rotating under the influence of birefringence.
    """
    t = np.linspace(0, 4*np.pi, num_points)
    
    # A complex trajectory: precession of a vector on the sphere
    # This simulates a SOP rotating around an axis, like a fiber experiencing twist
    theta = np.pi/4 + 0.3 * np.sin(t/2)  # latitude oscillation
    phi = t + 0.2 * np.sin(t/3)           # longitude rotation
    
    # Convert to Stokes components
    S1 = np.cos(phi) * np.sin(theta)
    S2 = np.sin(phi) * np.sin(theta)
    S3 = np.cos(theta)
    
    # Add a slight precession to make it more interesting
    # This simulates polarization evolution in a spun fiber
    S1_original = S1.copy()  # Store original S1 before modification
    S1 = S1 * np.cos(0.05*t) - S3 * np.sin(0.05*t)
    S3 = S3 * np.cos(0.05*t) + S1_original * np.sin(0.05*t)
    
    # Recalculate S2 to maintain DOP = 1
    norm = np.sqrt(S1**2 + S2**2 + S3**2)
    S1, S2, S3 = S1/norm, S2/norm, S3/norm
    
    return S1, S2, S3


# ------------------------------------------------------------------
# 2. BEAUTIFUL POINCARÉ SPHERE WITH TRAJECTORY
# ------------------------------------------------------------------
def plot_beautiful_poincare(S1, S2, S3, filename='poincare_beautiful.png'):
    """
    Plot a publication‑ready Poincaré sphere with a beautiful trajectory.
    """
    # Use a modern, clean style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. TRANSPARENT SPHERE
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 80)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Use a very subtle, semi-transparent sphere
    ax.plot_surface(x, y, z, color='#4682B4', alpha=0.08, edgecolor='none')
    
    # 2. GRID LINES (longitude and latitude)
    # Longitudes (meridians)
    for theta in np.linspace(0, 2*np.pi, 12):
        x_line = np.cos(theta) * np.sin(v)
        y_line = np.sin(theta) * np.sin(v)
        z_line = np.cos(v)
        ax.plot(x_line, y_line, z_line, color='#4682B4', alpha=0.15, linewidth=0.8)
    
    # Latitudes (parallels)
    for phi in np.linspace(0, np.pi, 6):
        x_line = np.cos(u) * np.sin(phi)
        y_line = np.sin(u) * np.sin(phi)
        z_line = np.ones_like(u) * np.cos(phi)
        ax.plot(x_line, y_line, z_line, color='#4682B4', alpha=0.15, linewidth=0.8)
    
    # 3. AXIS LABELS AND EQUATOR
    # Equator (highlighted)
    equator_theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(equator_theta), np.sin(equator_theta), np.zeros_like(equator_theta),
            color='#4682B4', alpha=0.3, linewidth=1.5, linestyle='--')
    
    # Axis arrows and labels
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], color='black', alpha=0.6, linewidth=1.5)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], color='black', alpha=0.6, linewidth=1.5)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], color='black', alpha=0.6, linewidth=1.5)
    
    # Axis labels with offset
    ax.text(1.25, 0, 0, r'$S_1$', fontsize=16, fontweight='bold', ha='center', va='center')
    ax.text(0, 1.25, 0, r'$S_2$', fontsize=16, fontweight='bold', ha='center', va='center')
    ax.text(0, 0, 1.25, r'$S_3$', fontsize=16, fontweight='bold', ha='center', va='center')
    
    # 4. THE TRAJECTORY (with color gradient)
    # Create a colormap: from purple to cyan to yellow
    colors = plt.cm.plasma(np.linspace(0, 1, len(S1)))
    
    # Plot the trajectory with varying line width and color
    for i in range(len(S1) - 1):
        ax.plot([S1[i], S1[i+1]], [S2[i], S2[i+1]], [S3[i], S3[i+1]],
                color=colors[i], linewidth=2.5, alpha=0.8)
    
    # 5. START AND END POINTS (with markers)
    ax.scatter(S1[0], S2[0], S3[0], color='#00CC66', s=150, marker='o', 
               edgecolor='white', linewidth=2, label='Start', zorder=10)
    ax.scatter(S1[-1], S2[-1], S3[-1], color='#FF4444', s=150, marker='*',
               edgecolor='white', linewidth=2, label='End', zorder=10)
    
    # 6. DOP INDICATOR (a small text box showing DOP = 1)
    ax.text(-0.9, -0.9, -1.1, 'DOP = 1', fontsize=12, color='#2C3E50', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 7. SETTINGS
    ax.set_xlim([-1.15, 1.15])
    ax.set_ylim([-1.15, 1.15])
    ax.set_zlim([-1.15, 1.15])
    ax.set_box_aspect([1, 1, 1])
    
    # Remove the background pane
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)
    
    # Set the view angle
    ax.view_init(elev=30, azim=45)
    
    # 8. LEGEND
    ax.legend(loc='upper right', fontsize=12, framealpha=0.8, edgecolor='none')
    
    # 9. TITLE
    ax.set_title('Polarization Evolution on the Poincaré Sphere', 
                 fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Poincaré plot saved to {filename}")
    return fig, ax


# ------------------------------------------------------------------
# 3. STOKES PARAMETER EVOLUTION PLOT
# ------------------------------------------------------------------
def plot_stokes_evolution(S1, S2, S3, filename='stokes_evolution.png'):
    """
    Plot the evolution of Stokes parameters S1, S2, S3 over time.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    t = np.linspace(0, 100, len(S1))
    
    # Plot with thick, transparent lines for a modern look
    ax.plot(t, S1, color='#E74C3C', linewidth=3, alpha=0.8, label=r'$S_1$')
    ax.plot(t, S2, color='#2ECC71', linewidth=3, alpha=0.8, label=r'$S_2$')
    ax.plot(t, S3, color='#3498DB', linewidth=3, alpha=0.8, label=r'$S_3$')
    
    # Fill the area with subtle transparency
    ax.fill_between(t, S1, alpha=0.1, color='#E74C3C')
    ax.fill_between(t, S2, alpha=0.1, color='#2ECC71')
    ax.fill_between(t, S3, alpha=0.1, color='#3498DB')
    
    ax.set_xlabel('Time (a.u.)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Stokes Parameter', fontsize=14, fontweight='bold')
    ax.set_title('Evolution of Stokes Parameters', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.1, 1.1])
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Stokes evolution plot saved to {filename}")
    return fig, ax


# ------------------------------------------------------------------
# 4. COMBINED PLOT (for presentations)
# ------------------------------------------------------------------
def plot_combined_presentation(S1, S2, S3, filename='combined_presentation.png'):
    """
    Create a combined figure with Poincaré sphere and Stokes evolution.
    """
    # Use a modern, clean style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig = plt.figure(figsize=(16, 8))
    
    # Left: Poincaré sphere
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Transparent sphere
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, color='#4682B4', alpha=0.08, edgecolor='none')
    
    # Grid lines (simplified)
    for theta in np.linspace(0, 2*np.pi, 8):
        x_line = np.cos(theta) * np.sin(v)
        y_line = np.sin(theta) * np.sin(v)
        z_line = np.cos(v)
        ax1.plot(x_line, y_line, z_line, color='#4682B4', alpha=0.12, linewidth=0.6)
    
    for phi in np.linspace(0, np.pi, 4):
        x_line = np.cos(u) * np.sin(phi)
        y_line = np.sin(u) * np.sin(phi)
        z_line = np.ones_like(u) * np.cos(phi)
        ax1.plot(x_line, y_line, z_line, color='#4682B4', alpha=0.12, linewidth=0.6)
    
    # Axis labels
    ax1.plot([-1.2, 1.2], [0, 0], [0, 0], color='black', alpha=0.4, linewidth=1)
    ax1.plot([0, 0], [-1.2, 1.2], [0, 0], color='black', alpha=0.4, linewidth=1)
    ax1.plot([0, 0], [0, 0], [-1.2, 1.2], color='black', alpha=0.4, linewidth=1)
    ax1.text(1.25, 0, 0, r'$S_1$', fontsize=14, fontweight='bold')
    ax1.text(0, 1.25, 0, r'$S_2$', fontsize=14, fontweight='bold')
    ax1.text(0, 0, 1.25, r'$S_3$', fontsize=14, fontweight='bold')
    
    # Equator
    equator_theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(equator_theta), np.sin(equator_theta), np.zeros_like(equator_theta),
            color='#4682B4', alpha=0.2, linewidth=1, linestyle='--')
    
    # Trajectory with color gradient
    colors = plt.cm.plasma(np.linspace(0, 1, len(S1)))
    for i in range(len(S1) - 1):
        ax1.plot([S1[i], S1[i+1]], [S2[i], S2[i+1]], [S3[i], S3[i+1]],
                color=colors[i], linewidth=2.5, alpha=0.8)
    
    # Start and end points
    ax1.scatter(S1[0], S2[0], S3[0], color='#00CC66', s=120, marker='o',
               edgecolor='white', linewidth=2, label='Start')
    ax1.scatter(S1[-1], S2[-1], S3[-1], color='#FF4444', s=120, marker='*',
               edgecolor='white', linewidth=2, label='End')
    
    ax1.set_xlim([-1.15, 1.15])
    ax1.set_ylim([-1.15, 1.15])
    ax1.set_zlim([-1.15, 1.15])
    ax1.set_box_aspect([1, 1, 1])
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('none')
    ax1.yaxis.pane.set_edgecolor('none')
    ax1.zaxis.pane.set_edgecolor('none')
    ax1.grid(False)
    ax1.view_init(elev=30, azim=45)
    ax1.set_title('Poincaré Sphere', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.8)
    
    # Right: Stokes evolution
    ax2 = fig.add_subplot(122)
    
    t = np.linspace(0, 100, len(S1))
    ax2.plot(t, S1, color='#E74C3C', linewidth=2.5, alpha=0.8, label=r'$S_1$')
    ax2.plot(t, S2, color='#2ECC71', linewidth=2.5, alpha=0.8, label=r'$S_2$')
    ax2.plot(t, S3, color='#3498DB', linewidth=2.5, alpha=0.8, label=r'$S_3$')
    
    ax2.set_xlabel('Time (a.u.)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Stokes Parameter', fontsize=12, fontweight='bold')
    ax2.set_title('Stokes Evolution', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim([-1.1, 1.1])
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {filename}")
    return fig


# ------------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    
    print("Generating beautiful Poincaré sphere for presentation...")
    
    # Generate a physically realistic trajectory
    S1, S2, S3 = generate_trajectory(3000)
    
    # Plot 1: Beautiful Poincaré sphere
    plot_beautiful_poincare(S1, S2, S3, 'poincare_beautiful.png')
    
    # Plot 2: Stokes evolution
    plot_stokes_evolution(S1, S2, S3, 'stokes_evolution.png')
    
    # Plot 3: Combined figure (for presentations)
    plot_combined_presentation(S1, S2, S3, 'presentation_combined.png')
    
    print("\n✅ All figures generated successfully!")