import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ------------------------------------------------------------------
# 1. LOADING FUNCTIONS
# ------------------------------------------------------------------
def load_oscilloscope_trace(filepath):
    """
    Load a full voltage trace from a Tektronix/Keysight style CSV.
    Returns the entire 1D array (time series), not just the mean.
    """
    return np.genfromtxt(filepath, delimiter=',', skip_header=6, usecols=-1)


def moving_average(data, window_size=5):
    """
    Apply a simple moving average filter to a 1D array.
    Uses 'valid' mode (output length = len(data) - window_size + 1).
    """
    if window_size < 2:
        return data
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')


def load_oscilloscope_trajectory(base_dir, trajectory_idx, num_channels=4, smooth_window=5):
    """
    Load all 4 channels for a single trajectory, apply moving average smoothing,
    and return the time traces.
    
    Args:
        base_dir: str, folder containing the CSV files.
        trajectory_idx: int, e.g., 0 for SOP_TRAJECTORY000_Ch1.csv
        num_channels: int, number of channels (default 4).
        smooth_window: int, moving average window size (default 5).
    
    Returns:
        D_traces: numpy array of shape (num_samples, num_channels)
                  Each column is the smoothed voltage trace for one channel.
    """
    traces = []
    # Read first channel to get length
    fname_ch1 = os.path.join(base_dir, f"SOP_TRAJECTORY{trajectory_idx:03d}_Ch1.csv")
    ch1_trace = load_oscilloscope_trace(fname_ch1)
    
    # Apply smoothing
    ch1_trace_smooth = moving_average(ch1_trace, smooth_window)
    num_samples = len(ch1_trace_smooth)
    traces.append(ch1_trace_smooth)
    
    # Read remaining channels
    for ch in range(2, num_channels + 1):
        fname = os.path.join(base_dir, f"SOP_TRAJECTORY{trajectory_idx:03d}_Ch{ch}.csv")
        trace = load_oscilloscope_trace(fname)
        trace_smooth = moving_average(trace, smooth_window)
        
        if len(trace_smooth) != num_samples:
            print(f"Warning: Channel {ch} has {len(trace_smooth)} samples after smoothing, expected {num_samples}")
            # Trim to the shortest
            min_len = min(num_samples, len(trace_smooth))
            if ch == 1:
                traces[0] = traces[0][:min_len]
            traces.append(trace_smooth[:min_len])
            num_samples = min_len
        else:
            traces.append(trace_smooth)
    
    # Stack columns to form (num_samples, num_channels)
    D_matrix = np.column_stack(traces)  # shape (num_samples, 4)
    return D_matrix


# ------------------------------------------------------------------
# 2. MAIN SCRIPT
# ------------------------------------------------------------------
if __name__ == "__main__":

    # ------------ CONFIGURATION ------------
    data_dir = '/home/240404662/PhD/xcorr-cd-wdm-fusion-localization/data/REFERENCED_CALIBRATION/'  # adjust
    calibration_file = 'calibration_matrix_reference.csv'  # path to your saved 4x4 matrix
    
    # Which trajectory file to load (e.g., 0 for SOP_TRAJECTORY000)
    trajectory_idx = 0
    
    # Smoothing window size (number of points to average)
    smooth_window = 5
    
    # Number of points to plot (optional: downsample for cleaner plot)
    # Set to None to plot all points.
    max_points = 500  # e.g., plot every Nth point to avoid overplotting
    
    # ------------ LOAD CALIBRATION MATRIX ------------
    C = np.loadtxt(calibration_file, delimiter=',')
    if C.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {C.shape}")
    
    print("Loaded calibration matrix C:")
    print(C)
    
    # ------------ LOAD OSCILLOSCOPE TRAJECTORY WITH SMOOTHING ------------
    print(f"Loading trajectory SOP_TRAJECTORY{trajectory_idx:03d} from {data_dir} ...")
    print(f"Applying moving average with window size = {smooth_window}")
    D_traces = load_oscilloscope_trajectory(data_dir, trajectory_idx, smooth_window=smooth_window)
    num_samples = D_traces.shape[0]
    print(f"Loaded {num_samples} smoothed time samples.")
    
    # ------------ APPLY CALIBRATION ------------
    # S = C @ D.T  (each column is a Stokes vector)
    S_all = C @ D_traces.T  # shape (4, num_samples)
    
    # Extract components
    S0 = S_all[0, :]
    S1 = S_all[1, :]
    S2 = S_all[2, :]
    S3 = S_all[3, :]
    
    # Normalize polarization components to unit sphere (ignore S0 variations)
    norm = np.sqrt(S1**2 + S2**2 + S3**2) + 1e-12
    s1n = S1 / norm
    s2n = S2 / norm
    s3n = S3 / norm
    
    # ------------ DOWNSAMPLE (if requested) ------------
    if max_points is not None and num_samples > max_points:
        step = num_samples // max_points
        indices = np.arange(0, num_samples, step)
        s1n = s1n[indices]
        s2n = s2n[indices]
        s3n = s3n[indices]
        S0 = S0[indices]
        print(f"Downsampled to {len(s1n)} points for plotting.")
    
    # ------------ COMPUTE METRICS ------------
    dop_vals = np.sqrt(S1**2 + S2**2 + S3**2) / (S0 + 1e-12)
    mean_dop = np.mean(dop_vals)
    std_dop = np.std(dop_vals)
    print(f"Mean DOP: {mean_dop:.6f}")
    print(f"Std DOP:  {std_dop:.6f}")
    
    # ------------ PLOT TRAJECTORY ON POINCARÉ SPHERE ------------
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Unit sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.15, edgecolor='none')
    
    # Plot the trajectory as a line with color gradient indicating time
    # Create a color array based on index (time progression)
    colors = plt.cm.viridis(np.linspace(0, 1, len(s1n)))
    
    # Plot as a line with color gradient
    for i in range(len(s1n) - 1):
        ax.plot([s1n[i], s1n[i+1]], [s2n[i], s2n[i+1]], [s3n[i], s3n[i+1]],
                color=colors[i], linewidth=1.5, alpha=0.8)
    
    # Plot start and end points with markers
    ax.scatter(s1n[0], s2n[0], s3n[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(s1n[-1], s2n[-1], s3n[-1], color='red', s=100, marker='*', label='End')
    
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('S3')
    ax.set_title(f'Poincaré Sphere Trajectory (Window={smooth_window}, Mean DOP = {mean_dop:.4f})')
    ax.legend()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1, 1, 1])
    
    # Add a colorbar to show time progression
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Time progression')
    
    plt.tight_layout()
    plt.savefig('poincare_trajectory_smooth.png', dpi=150)
    print("\nPlot saved to 'poincare_trajectory_smooth.png'")
    plt.show()
    
    # ------------ OPTIONAL: SAVE RECONSTRUCTED STOKES VECTORS ------------
    np.savetxt('trajectory_stokes_vectors_smooth.csv', 
               np.column_stack([S0, S1, S2, S3]), 
               delimiter=',', header='S0,S1,S2,S3', comments='')
    print("Smoothed trajectory Stokes vectors saved to 'trajectory_stokes_vectors_smooth.csv'")