import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ------------------------------------------------------------------
# 1. YOUR PM1000 LOADER (as provided)
# ------------------------------------------------------------------
def load_pm1000(filename):
    """
    Load PM1000 measurement file (text), interpreting the Normalization flag.

    Returns
    -------
    dict:
        'metadata'    : dict of all #‑header parameters.
        'timestamps'  : list of float (ns).
        'power_adc'   : list of float – raw power channel after /32768 (arbitrary units).
        'stokes_norm' : list of [s1, s2, s3] – normalised Stokes components (‑1…1).
        'raw_scaled'  : list of [col1, col2, col3, col4] exactly as MATLAB's "S".
    """
    meta = {}
    lines_data = []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                content = line[1:].strip()
                if content.endswith(';'):
                    content = content[:-1]
                if '=' in content:
                    k, v = content.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    meta[k] = v
            else:
                lines_data.append(line)

    # Parse numerical data
    timestamps = []
    raw_vals = []   # each row: [power_raw, s1_raw, s2_raw, s3_raw]
    for line in lines_data:
        parts = line.split(',')
        if len(parts) == 5:   # with timestamp
            t = float(parts[0])
            vals = [float(p) for p in parts[1:5]]
        elif len(parts) == 4:
            t = None
            vals = [float(p) for p in parts[0:4]]
        else:
            continue
        timestamps.append(t)
        raw_vals.append(vals)

    # Generate timestamps if missing
    if timestamps and timestamps[0] is None:
        period = float(meta.get('SamplePeriod_ns', 1))
        timestamps = [i * period for i in range(len(raw_vals))]
    else:
        timestamps = [float(t) for t in timestamps]

    # Convert to MATLAB‑style scaled values (S)
    S = []
    for row in raw_vals:
        power = row[0] / 32768.0
        s1 = (row[1] - 32768.0) / 32768.0
        s2 = (row[2] - 32768.0) / 32768.0
        s3 = (row[3] - 32768.0) / 32768.0
        S.append([power, s1, s2, s3])

    # Interpret according to Normalization flag
    normalized = meta.get('Normalization', '0') == '1'
    if normalized:
        power_adc = [row[0] for row in S]
        stokes_norm = [row[1:] for row in S]
    else:
        power_adc = [row[0] for row in S]
        stokes_norm = [[row[1]/row[0], row[2]/row[0], row[3]/row[0]] for row in S]

    return {
        'metadata': meta,
        'timestamps': timestamps,
        'raw_scaled': S,
        'power_adc': power_adc,
        'stokes_norm': stokes_norm
    }


# ------------------------------------------------------------------
# 2. OSCILLOSCOPE LOADERS
# ------------------------------------------------------------------
def load_oscilloscope_trace(filepath):
    """Load a single voltage trace from a Tektronix/Keysight style CSV."""
    return np.genfromtxt(filepath, delimiter=',', skip_header=6, usecols=-1)

def load_oscilloscope_sop(base_dir, sop_idx, num_channels=4):
    """Load all channels for a single SOP and return the mean voltage for each."""
    channel_means = []
    for ch in range(1, num_channels + 1):
        fname = os.path.join(base_dir, f"SOP{sop_idx:03d}_Ch{ch}.csv")
        trace = load_oscilloscope_trace(fname)
        channel_means.append(np.mean(trace))
    return np.array(channel_means)  # Shape (4,)


# ------------------------------------------------------------------
# 3. IDEAL TETRAHEDRON POINTS (for reference)
# ------------------------------------------------------------------
ideal_points = {
    1: np.array([1.0, 0.0, 0.0, 1.0]),
    2: np.array([1.0, 2*np.sqrt(2)/3, 0.0, -1/3]),
    3: np.array([1.0, -np.sqrt(2)/3, np.sqrt(6)/3, -1/3]),
    4: np.array([1.0, -np.sqrt(2)/3, -np.sqrt(6)/3, -1/3])
}


# ------------------------------------------------------------------
# 4. PLOTTING FUNCTION
# ------------------------------------------------------------------
def plot_poincare(points, labels, colors, markers, title, filename,
                  show_ideal=False, ideal_points=None, connect_to_ideal=False):
    """
    Generic Poincaré sphere plotter.
    points: list of 4x? arrays (each with S1,S2,S3)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Unit sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.15, edgecolor='none')

    # Plot points
    for pts, label, color, marker in zip(points, labels, colors, markers):
        ax.scatter(pts[0], pts[1], pts[2],
                   color=color, s=80, marker=marker, label=label)

    if show_ideal and ideal_points is not None:
        for i, (key, S_ideal) in enumerate(ideal_points.items()):
            ax.scatter(S_ideal[1], S_ideal[2], S_ideal[3],
                       color='red', s=100, marker='*', label='Ideal' if i==0 else "")
            ax.text(S_ideal[1]*1.1, S_ideal[2]*1.1, S_ideal[3]*1.1,
                    f'P{key}', color='red', fontsize=10)

    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('S3')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")
    plt.show()


# ------------------------------------------------------------------
# 5. MAIN SCRIPT
# ------------------------------------------------------------------
if __name__ == "__main__":

    # --- Directories (adjust to your actual paths) ---
    scope_data_dir = '/users/240404662/PhD/xcorr-cd-wdm-fusion-localization/data/REFERENCED_CALIBRATION/'
    ref_data_dir = '/users/240404662/PhD/xcorr-cd-wdm-fusion-localization/data/REFERENCED_CALIBRATION/'

    # --- Load data ---
    D_matrix = []   # columns = raw detector vectors from scope
    S_measured = [] # columns = averaged Stokes vectors from PM1000

    print("=" * 60)
    print("   LOADING MEASURED DATA")
    print("=" * 60)

    for idx in range(1, 5):
        # Oscilloscope
        D_vec = load_oscilloscope_sop(scope_data_dir, idx)
        D_matrix.append(D_vec)
        print(f"SOP{idx:03d} Oscilloscope (D0..D3): {D_vec}")

        # PM1000 file
        ref_file = os.path.join(ref_data_dir, f"REF_SOP_{idx}.txt")
        pm_data = load_pm1000(ref_file)

        # Compute absolute Stokes vectors per sample
        stokes_samples = []
        for i in range(len(pm_data['raw_scaled'])):
            S0 = pm_data['raw_scaled'][i][0]
            s1, s2, s3 = pm_data['stokes_norm'][i]
            S1 = S0 * s1
            S2 = S0 * s2
            S3 = S0 * s3
            stokes_samples.append([S0, S1, S2, S3])
        S_avg = np.mean(stokes_samples, axis=0)
        S_measured.append(S_avg)
        print(f"SOP{idx:03d} PM1000 (S0..S3):     {S_avg}")
        print()

    # Convert to 4x4 matrices (columns = points)
    D_matrix = np.column_stack(D_matrix)  # (4,4)
    S_measured = np.column_stack(S_measured)  # (4,4)

    # --- Compute referenced calibration ---
    C = S_measured @ np.linalg.inv(D_matrix)

    print("=" * 60)
    print("   CALIBRATION MATRIX C")
    print("=" * 60)
    print(C)

    # --- Reconstruct Stokes vectors from calibration ---
    S_reconstructed = C @ D_matrix

    # --- Verification ---
    print("\n" + "=" * 60)
    print("   VERIFICATION")
    print("=" * 60)
    for i in range(4):
        S_rec = S_reconstructed[:, i]
        S_meas = S_measured[:, i]
        dop_val = np.sqrt(S_rec[1]**2 + S_rec[2]**2 + S_rec[3]**2) / (S_rec[0] + 1e-12)
        print(f"Point {i+1}: DOP = {dop_val:.6f}")

        # Angular deviation between measured and reconstructed
        u = S_meas[1:4] / (np.linalg.norm(S_meas[1:4]) + 1e-12)
        v = S_rec[1:4] / (np.linalg.norm(S_rec[1:4]) + 1e-12)
        cos_theta = np.clip(np.dot(u, v), -1, 1)
        angle_deg = np.arccos(cos_theta) * 180 / np.pi
        print(f"       Angular deviation from PM1000: {angle_deg:.4f}°")
        print()

    # --- Plot 1: Ideal vs Measured ---
    # Extract normalized S1,S2,S3 for measured (ignore S0)
    measured_norm = [S_measured[:, i][1:4] / (np.linalg.norm(S_measured[:, i][1:4]) + 1e-12) for i in range(4)]
    ideal_norm = [np.array(list(ideal_points.values()))[i][1:4] for i in range(4)]

    plot_poincare(
        points=measured_norm + ideal_norm,
        labels=['Measured (PM1000)']*4 + ['Ideal']*4,
        colors=['blue']*4 + ['red']*4,
        markers=['o']*4 + ['*']*4,
        title='Poincaré Sphere: Ideal Tetrahedron vs Measured SOPs',
        filename='poincare_ideal_vs_measured.png',
        show_ideal=False,  # we already include ideal in points
        ideal_points=None
    )

    # --- Plot 2: Measured vs Reconstructed ---
    rec_norm = [S_reconstructed[:, i][1:4] / (np.linalg.norm(S_reconstructed[:, i][1:4]) + 1e-12) for i in range(4)]

    # Combine measured and reconstructed for plotting
    all_points = measured_norm + rec_norm
    labels_plot2 = ['Measured']*4 + ['Reconstructed']*4
    colors_plot2 = ['blue']*4 + ['red']*4
    markers_plot2 = ['o']*4 + ['^']*4

    plot_poincare(
        points=all_points,
        labels=labels_plot2,
        colors=colors_plot2,
        markers=markers_plot2,
        title='Poincaré Sphere: Measured vs Reconstructed SOPs',
        filename='poincare_measured_vs_reconstructed.png',
        show_ideal=False,
        ideal_points=None
    )

    # --- Save calibration matrix ---
    np.savetxt('calibration_matrix_reference.csv', C, delimiter=',')
    print("\nCalibration matrix saved to 'calibration_matrix_reference.csv'")