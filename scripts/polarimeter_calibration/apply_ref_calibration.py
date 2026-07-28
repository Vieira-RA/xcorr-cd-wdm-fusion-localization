import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ------------------------------------------------------------------
# 1. LOADING FUNCTIONS (same as before)
# ------------------------------------------------------------------
def load_oscilloscope_trace(filepath):
    """Load a single voltage trace from a Tektronix/Keysight style CSV."""
    return np.genfromtxt(filepath, delimiter=',', skip_header=6, usecols=-1)

def load_oscilloscope_sop(base_dir, sop_idx, num_channels=4):
    """
    Load all channels for a single SOP and return the mean voltage for each.
    sop_idx is 0-based (e.g., SOP000, SOP001, ...).
    """
    channel_means = []
    for ch in range(1, num_channels + 1):
        fname = os.path.join(base_dir, f"SOP{sop_idx:03d}_Ch{ch}.csv")
        trace = load_oscilloscope_trace(fname)
        channel_means.append(np.mean(trace))
    return np.array(channel_means)  # Shape (4,)


# ------------------------------------------------------------------
# 2. MAIN SCRIPT
# ------------------------------------------------------------------
if __name__ == "__main__":

    # ------------ CONFIGURATION ------------
    data_dir = '/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/data/REFERENCED_CALIBRATION/'  # adjust
    calibration_file = 'calibration_matrix_reference.csv'  # path to your saved 4x4 matrix

    # Number of SOPs (from 0 to 99, so 100 files)
    num_sops = 4

    # ------------ LOAD CALIBRATION MATRIX ------------
    C = np.loadtxt(calibration_file, delimiter=',')
    if C.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {C.shape}")

    print("Loaded calibration matrix C:")
    print(C)

    # ------------ LOAD OSCILLOSCOPE DATA ------------
    D_matrix = []  # rows = SOPs, columns = channels
    print(f"Loading {num_sops} SOPs from {data_dir} ...")
    for idx in range(num_sops):
        D_vec = load_oscilloscope_sop(data_dir, idx+1)
        D_matrix.append(D_vec)
    D_matrix = np.array(D_matrix)  # shape (num_sops, 4)

    # ------------ APPLY CALIBRATION ------------
    # S = C @ D.T  (each column is a Stokes vector)
    S_all = C @ D_matrix.T  # shape (4, num_sops)

    # Extract components
    S0 = S_all[0, :]
    S1 = S_all[1, :]
    S2 = S_all[2, :]
    S3 = S_all[3, :]

    print(S0)

    # ------------ COMPUTE METRICS ------------
    # Degree of Polarization (DOP)
    dop_vals = np.sqrt(S1**2 + S2**2 + S3**2) / (S0 + 1e-12)
    mean_dop = np.mean(dop_vals)
    std_dop = np.std(dop_vals)

    print(f"\nNumber of SOPs: {num_sops}")
    print(f"Mean DOP: {mean_dop:.6f}")
    print(f"Std DOP:  {std_dop:.6f}")

    # Sphere coverage (variance of normalized S1,S2,S3)
    norm = np.sqrt(S1**2 + S2**2 + S3**2) + 1e-12
    s1n = S1 / norm
    s2n = S2 / norm
    s3n = S3 / norm
    sphere_var = np.var(s1n) + np.var(s2n) + np.var(s3n)
    print(f"Sphere variance: {sphere_var:.4f} (theoretical max ~0.333 for uniform)")

    # ------------ PLOT ON POINCARÉ SPHERE ------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Unit sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.15, edgecolor='none')

    # Plot the measured SOPs (normalized S1,S2,S3)
    ax.scatter(s1n, s2n, s3n, c='blue', s=20, alpha=0.7, label='Measured SOPs')

    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('S3')
    ax.set_title(f'Poincaré Sphere: {num_sops} SOPs (Mean DOP = {mean_dop:.4f})')
    ax.legend()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.savefig('poincare_measured_sops.png', dpi=150)
    print("\nPlot saved to 'poincare_measured_sops.png'")
    plt.show()

    # ------------ OPTIONAL: SAVE RECONSTRUCTED STOKES VECTORS ------------
    np.savetxt('reconstructed_stokes_vectors.csv', S_all.T, delimiter=',',
               header='S0,S1,S2,S3', comments='')
    print("Reconstructed Stokes vectors saved to 'reconstructed_stokes_vectors.csv'")