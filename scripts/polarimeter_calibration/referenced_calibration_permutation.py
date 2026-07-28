import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import itertools

# ------------------------------------------------------------------
# 1. PM1000 LOADER
# ------------------------------------------------------------------
def load_pm1000(filename):
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

    timestamps = []
    raw_vals = []
    for line in lines_data:
        parts = line.split(',')
        if len(parts) == 5:
            t = float(parts[0])
            vals = [float(p) for p in parts[1:5]]
        elif len(parts) == 4:
            t = None
            vals = [float(p) for p in parts[0:4]]
        else:
            continue
        timestamps.append(t)
        raw_vals.append(vals)

    if timestamps and timestamps[0] is None:
        period = float(meta.get('SamplePeriod_ns', 1))
        timestamps = [i * period for i in range(len(raw_vals))]
    else:
        timestamps = [float(t) for t in timestamps]

    S = []
    for row in raw_vals:
        power = row[0] / 32768.0
        s1 = (row[1] - 32768.0) / 32768.0
        s2 = (row[2] - 32768.0) / 32768.0
        s3 = (row[3] - 32768.0) / 32768.0
        S.append([power, s1, s2, s3])

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
    return np.genfromtxt(filepath, delimiter=',', skip_header=6, usecols=-1)


def moving_average(data, window_size=5):
    """Apply a simple moving average filter to a 1D array."""
    if window_size < 2:
        return data
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')


def load_oscilloscope_sop_mean(base_dir, sop_idx, num_channels=4):
    """Load all channels for a single SOP and return the mean voltage for each."""
    channel_means = []
    for ch in range(1, num_channels + 1):
        fname = os.path.join(base_dir, f"SOP{sop_idx:03d}_Ch{ch}.csv")
        trace = load_oscilloscope_trace(fname)
        channel_means.append(np.mean(trace))
    return np.array(channel_means)


def load_oscilloscope_trajectory(base_dir, trajectory_idx, num_channels=4, smooth_window=5):
    """
    Load full traces for a trajectory, apply moving average smoothing,
    and return (num_samples, 4).
    """
    traces = []
    fname_ch1 = os.path.join(base_dir, f"SOP_TRAJECTORY{trajectory_idx:03d}_Ch1.csv")
    ch1_trace = load_oscilloscope_trace(fname_ch1)
    ch1_smooth = moving_average(ch1_trace, smooth_window)
    num_samples = len(ch1_smooth)
    traces.append(ch1_smooth)
    
    for ch in range(2, num_channels + 1):
        fname = os.path.join(base_dir, f"SOP_TRAJECTORY{trajectory_idx:03d}_Ch{ch}.csv")
        trace = load_oscilloscope_trace(fname)
        trace_smooth = moving_average(trace, smooth_window)
        
        if len(trace_smooth) != num_samples:
            min_len = min(num_samples, len(trace_smooth))
            if ch == 1:
                traces[0] = traces[0][:min_len]
            traces.append(trace_smooth[:min_len])
            num_samples = min_len
        else:
            traces.append(trace_smooth)
    
    return np.column_stack(traces)


# ------------------------------------------------------------------
# 3. PLOTTING FUNCTION (with view rotation controls)
# ------------------------------------------------------------------
def plot_trajectory(s1n, s2n, s3n, perm, mean_dop, std_dop, sphere_var, save_dir,
                    view_azimuth=0, view_elevation=20):
    """
    Plot a single trajectory and save it to the specified directory.
    
    Parameters
    ----------
    view_azimuth : float, degrees
        Rotation angle around the S3 axis (S1 x S2 plane).
        Positive = clockwise (when looking down S3 axis).
        Default: 0 (standard view).
    view_elevation : float, degrees
        Elevation angle above the S1-S2 plane.
        Default: 20.
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
    
    # Trajectory with color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(s1n)))
    for i in range(len(s1n) - 1):
        ax.plot([s1n[i], s1n[i+1]], [s2n[i], s2n[i+1]], [s3n[i], s3n[i+1]],
                color=colors[i], linewidth=1.5, alpha=0.8)
    
    # Start/end markers
    ax.scatter(s1n[0], s2n[0], s3n[0], color='green', s=80, marker='o', label='Start')
    ax.scatter(s1n[-1], s2n[-1], s3n[-1], color='red', s=80, marker='*', label='End')
    
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('S3')
    ax.set_title(f'Permutation {perm}\nMean DOP = {mean_dop:.4f}, Std DOP = {std_dop:.4f}, Var = {sphere_var:.4f}')
    ax.legend()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1, 1, 1])
    
    # ---------- VIEW ROTATION CONTROLS ----------
    # Rotate the view: azimuth (around S3 axis) and elevation (above S1-S2 plane)
    ax.view_init(elev=view_elevation, azim=view_azimuth)
    
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Time progression')
    
    plt.tight_layout()
    perm_str = ''.join(str(p) for p in perm)
    filename = os.path.join(save_dir, f'permutation_{perm_str}.png')
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    return filename


# ------------------------------------------------------------------
# 4. MAIN BRUTE‑FORCE SCRIPT
# ------------------------------------------------------------------
if __name__ == "__main__":

    # ------------ CONFIGURATION ------------
    scope_calib_dir = '/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/data/REFERENCED_CALIBRATION/'
    ref_data_dir = '/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/data/REFERENCED_CALIBRATION/'
    trajectory_dir = '/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/data/REFERENCED_CALIBRATION/'
    
    trajectory_idx = 3
    num_sop_points = 4
    max_plot_points = 5000
    
    # Moving average window size (set to 1 to disable)
    smooth_window = 7

    # ---------- VIEW ROTATION CONTROLS ----------
    # Azimuth: rotation around S3 axis (S1 x S2 plane)
    # Positive = clockwise when looking down S3 axis.
    # Try: 0, 45, 90, 180, -45, etc.
    view_azimuth = 35  # degrees
    
    # Elevation: angle above the S1-S2 plane
    # Try: 0 (equator), 20 (default), 90 (top view), -20 (below)
    view_elevation = 20  # degrees

    output_dir = 'permutation_trajectories'
    os.makedirs(output_dir, exist_ok=True)

    # ------------ LOAD REFERENCE DATA ------------
    D_calib = []
    S_calib = []

    print("=" * 60)
    print("   LOADING REFERENCE CALIBRATION DATA")
    print("=" * 60)

    for idx in range(1, num_sop_points + 1):
        D_vec = load_oscilloscope_sop_mean(scope_calib_dir, idx)
        D_calib.append(D_vec)
        print(f"SOP{idx:03d} Oscilloscope (Ch1..Ch4): {D_vec}")

        ref_file = os.path.join(ref_data_dir, f"REF_SOP_{idx}.txt")
        pm_data = load_pm1000(ref_file)
        
        stokes_samples = []
        for i in range(len(pm_data['raw_scaled'])):
            S0 = pm_data['raw_scaled'][i][0]
            s1, s2, s3 = pm_data['stokes_norm'][i]
            stokes_samples.append([S0, S0*s1, S0*s2, S0*s3])
        S_avg = np.mean(stokes_samples, axis=0)
        S_calib.append(S_avg)
        print(f"SOP{idx:03d} PM1000 (S0..S3):     {S_avg}")
        print()

    D_calib = np.column_stack(D_calib)
    S_calib = np.column_stack(S_calib)

    # ------------ LOAD TRAJECTORY DATA (WITH SMOOTHING) ------------
    print("=" * 60)
    print("   LOADING TRAJECTORY DATA")
    print("=" * 60)
    print(f"Applying moving average with window size = {smooth_window}")
    D_traj = load_oscilloscope_trajectory(trajectory_dir, trajectory_idx, smooth_window=smooth_window)
    num_samples = D_traj.shape[0]
    print(f"Trajectory loaded: {num_samples} samples.")

    # ------------ BRUTE‑FORCE ALL PERMUTATIONS ------------
    print("\n" + "=" * 60)
    print("   BRUTE‑FORCING CHANNEL PERMUTATIONS")
    print("=" * 60)

    results = []
    best_score = -np.inf
    best_perm = None
    best_C = None
    best_S_traj = None

    for perm in itertools.permutations(range(4)):
        D_calib_perm = D_calib[perm, :]

        try:
            C = S_calib @ np.linalg.inv(D_calib_perm)
        except np.linalg.LinAlgError:
            continue

        D_traj_perm = D_traj[:, perm]
        S_traj = C @ D_traj_perm.T

        S0 = S_traj[0, :]
        S1 = S_traj[1, :]
        S2 = S_traj[2, :]
        S3 = S_traj[3, :]
        
        dop_vals = np.sqrt(S1**2 + S2**2 + S3**2) / (S0 + 1e-12)
        mean_dop = np.mean(dop_vals)
        std_dop = np.std(dop_vals)
        
        norm = np.sqrt(S1**2 + S2**2 + S3**2) + 1e-12
        s1n = S1 / norm
        s2n = S2 / norm
        s3n = S3 / norm
        sphere_var = np.var(s1n) + np.var(s2n) + np.var(s3n)
        
        score = mean_dop * (1 - std_dop) * (1 + sphere_var)
        
        results.append({
            'perm': perm,
            'mean_dop': mean_dop,
            'std_dop': std_dop,
            'sphere_var': sphere_var,
            'score': score,
            'C': C,
            'S_traj': S_traj,
            's1n': s1n,
            's2n': s2n,
            's3n': s3n
        })
        
        if score > best_score:
            best_score = score
            best_perm = perm
            best_C = C
            best_S_traj = S_traj

    # ------------ PRINT RESULTS ------------
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\nTop 10 permutations:")
    print(" Rank | Permutation    | Mean DOP  | Std DOP  | Sphere Var | Score")
    print("------|----------------|-----------|----------|------------|---------")
    for i, res in enumerate(results[:10]):
        print(f"  {i+1:2}   | {res['perm']}     | {res['mean_dop']:.6f} | {res['std_dop']:.6f} | {res['sphere_var']:.4f} | {res['score']:.4f}")

    print(f"\n✅ Best permutation: {best_perm}")
    print(f"   Meaning: Physical Ch{best_perm[0]+1} -> D0, Ch{best_perm[1]+1} -> D1, ...")

    # ------------ PLOT ALL PERMUTATIONS ------------
    print("\n" + "=" * 60)
    print("   PLOTTING ALL PERMUTATIONS")
    print("=" * 60)
    print(f"View settings: azimuth = {view_azimuth}°, elevation = {view_elevation}°")

    for res in results:
        n_points = len(res['s1n'])
        if n_points > max_plot_points:
            step = n_points // max_plot_points
            indices = np.arange(0, n_points, step)
            s1n_plot = res['s1n'][indices]
            s2n_plot = res['s2n'][indices]
            s3n_plot = res['s3n'][indices]
        else:
            s1n_plot = res['s1n']
            s2n_plot = res['s2n']
            s3n_plot = res['s3n']
        
        filename = plot_trajectory(
            s1n_plot, s2n_plot, s3n_plot,
            res['perm'],
            res['mean_dop'],
            res['std_dop'],
            res['sphere_var'],
            output_dir,
            view_azimuth=view_azimuth,
            view_elevation=view_elevation
        )
        print(f"Saved: {filename}")

    # ------------ SAVE BEST CALIBRATION ------------
    np.savetxt('calibration_matrix_best_permutation.csv', best_C, delimiter=',')
    np.savetxt('channel_permutation_best.txt', [best_perm], fmt='%d')
    print("\nBest calibration matrix saved to 'calibration_matrix_best_permutation.csv'")
    print("Best channel permutation saved to 'channel_permutation_best.txt'")
    print(f"\nAll permutation plots saved to: {output_dir}/")