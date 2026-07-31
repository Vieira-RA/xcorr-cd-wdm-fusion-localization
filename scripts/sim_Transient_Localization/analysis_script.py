"""
Load sweep_results.npz and produce all figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plotting import (
    plot_std_heatmap,
    plot_bias_heatmap,
    plot_success_rate_heatmap,
)

# ========================= Load data ============================
data = np.load("sweep_results.npz", allow_pickle=True)
BANDWIDTHS = data['BANDWIDTHS']
SNRS = data['SNRS']
event_distance_km = data['event_distance_km'].item()

phase_est = data['phase_est']
phat_est  = data['phat_est']
int_est   = data['int_est']
par_est   = data['par_est']

demo_bw = data['demo_bw'].item()
demo_snr = data['demo_snr'].item()
demo_phase_pairs = data['demo_phase_pairs']
demo_phat_pairs = data['demo_phat_pairs']
demo_int_pairs = data['demo_int_pairs']
demo_par_pairs = data['demo_par_pairs']

phase_inlier_all = data['phase_inlier_pairs']
phat_inlier_all  = data['phat_inlier_pairs']
int_inlier_all   = data['int_inlier_pairs']
par_inlier_all   = data['par_inlier_pairs']

bw_times = data['bw_times']
pair_indices = data['pair_indices']
wavelengths_nm = data['wavelengths_nm']

true_dist = event_distance_km

# ========================= Compute metrics =====================
def compute_metrics(estimates):
    std = np.nanstd(estimates, axis=2)
    bias = np.nanmean(estimates, axis=2) - true_dist
    success_rel = np.nanmean(np.abs(estimates - true_dist) < 0.1*true_dist, axis=2)
    success_abs = np.nanmean(np.abs(estimates - true_dist) < 5.0, axis=2)
    return std, bias, success_rel, success_abs

phase_std, phase_bias, phase_success, phase_success_abs = compute_metrics(phase_est)
phat_std,  phat_bias,  phat_success,  phat_success_abs  = compute_metrics(phat_est)
int_std,   int_bias,   int_success,   int_success_abs   = compute_metrics(int_est)
par_std,   par_bias,   par_success,   par_success_abs   = compute_metrics(par_est)

# ========================= Figure 1: Schematic =================
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([0,1,2,3], [0,0.5,0,0], 'ko-', markersize=4)
ax.text(0,0.8,'Laser\ncomb', ha='center')
ax.text(1,0.8,'Sensing\nfibre', ha='center')
ax.text(2,0.8,'CD delay\nτ(λ)', ha='center')
ax.text(3,0.8,'Polarimeter\n→ S₁,S₂', ha='center')
ax.set_title('Simplified sensing principle')
ax.axis('off')
plt.tight_layout()
plt.savefig('fig01_schematic.png', dpi=150)
plt.close()

# ========================= Figure 3: Std heatmaps ==============
methods_std = {
    'Phase‑slope': phase_std,
    'GCC‑PHAT': phat_std,
    'Integer peak': int_std,
    'Parabolic peak': par_std,
}
for name, std in methods_std.items():
    plot_std_heatmap(std, SNRS, BANDWIDTHS,
                     f'{name} std (km)', f'fig03_{name.replace(" ","_").lower()}_std.png',
                     vmax=100)

# ========================= Figure 4: Bias & success ============
methods_bias = {
    'Phase‑slope': phase_bias,
    'GCC‑PHAT': phat_bias,
    'Integer peak': int_bias,
    'Parabolic peak': par_bias,
}
for name, bias in methods_bias.items():
    plot_bias_heatmap(bias, SNRS, BANDWIDTHS,
                      f'{name} bias (km)', f'fig04_{name.replace(" ","_").lower()}_bias.png',
                      vmin=-50, vmax=50)

methods_success_rel = {
    'Phase‑slope': phase_success,
    'GCC‑PHAT': phat_success,
    'Integer peak': int_success,
    'Parabolic peak': par_success,
}
for name, succ in methods_success_rel.items():
    plot_success_rate_heatmap(succ, SNRS, BANDWIDTHS,
                              f'{name} success rate (10%)',
                              f'fig04_{name.replace(" ","_").lower()}_success_rel.png')

methods_success_abs = {
    'Phase‑slope': phase_success_abs,
    'GCC‑PHAT': phat_success_abs,
    'Integer peak': int_success_abs,
    'Parabolic peak': par_success_abs,
}
for name, succ in methods_success_abs.items():
    plot_success_rate_heatmap(succ, SNRS, BANDWIDTHS,
                              f'{name} success rate (±5 km)',
                              f'fig04_{name.replace(" ","_").lower()}_success_abs.png')

# ========================= Figure 6: Pairwise scatter ==========
if len(demo_phase_pairs) > 0:
    delta_lambda = np.array([np.abs(wavelengths_nm[i] - wavelengths_nm[j]) for i, j in pair_indices])
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(delta_lambda, demo_phase_pairs, s=15, alpha=0.6, label='Phase‑slope')
    ax.scatter(delta_lambda, demo_phat_pairs, s=15, alpha=0.6, marker='s', label='GCC‑PHAT')
    ax.scatter(delta_lambda, demo_int_pairs, s=15, alpha=0.6, marker='D', label='Integer peak')
    ax.scatter(delta_lambda, demo_par_pairs, s=15, alpha=0.6, marker='^', label='Parabolic peak')
    ax.axhline(true_dist, color='k', ls='--')
    ax.set_xlabel('Wavelength separation (nm)')
    ax.set_ylabel('Estimated distance (km)')
    ax.set_title(f'Pairwise estimates, B={demo_bw*1e-3:.0f} kHz, SNR={demo_snr} dB')
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig06_pairwise_scatter.png', dpi=150)
    plt.close()
else:
    print("Demo setting not found – pairwise scatter skipped.")

# ========================= Figure 7: Timing ====================
fig, ax = plt.subplots(figsize=(7,4))
ax.loglog(BANDWIDTHS, bw_times, 'o-')
ax.set_xlabel('Bandwidth (Hz)')
ax.set_ylabel('Computation time per bandwidth (s)')
ax.set_title('Execution time vs bandwidth (all SNRs, all realisations)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig07_timing.png', dpi=150)
plt.close()

# ========================= Figure 8: Global pair distribution =====
all_phase_in = phase_inlier_all
all_phat_in  = phat_inlier_all
all_int_in   = int_inlier_all
all_par_in   = par_inlier_all

fig, ax = plt.subplots(figsize=(12, 6))
lens = [len(arr) for arr in (all_phase_in, all_phat_in, all_int_in, all_par_in) if len(arr) > 0]
if lens:
    sample_size = min(5000, min(lens))
    idx_phase = np.random.choice(len(all_phase_in), sample_size, replace=False) if len(all_phase_in) > 0 else []
    idx_phat  = np.random.choice(len(all_phat_in), sample_size, replace=False) if len(all_phat_in) > 0 else []
    idx_int   = np.random.choice(len(all_int_in), sample_size, replace=False) if len(all_int_in) > 0 else []
    idx_par   = np.random.choice(len(all_par_in), sample_size, replace=False) if len(all_par_in) > 0 else []

    if len(idx_phase): ax.scatter(np.arange(sample_size), all_phase_in[idx_phase], s=2, alpha=0.4, label='Phase‑slope')
    if len(idx_phat):  ax.scatter(np.arange(sample_size), all_phat_in[idx_phat],   s=2, alpha=0.4, marker='s', label='GCC‑PHAT')
    if len(idx_int):   ax.scatter(np.arange(sample_size), all_int_in[idx_int],     s=2, alpha=0.4, marker='D', label='Integer peak')
    if len(idx_par):   ax.scatter(np.arange(sample_size), all_par_in[idx_par],     s=2, alpha=0.4, marker='^', label='Parabolic peak')

    if len(all_phase_in): ax.axhline(np.mean(all_phase_in), color='C0', linestyle='-', linewidth=2, label=f'Phase‑slope mean: {np.mean(all_phase_in):.1f} km')
    if len(all_phat_in):  ax.axhline(np.mean(all_phat_in),  color='C1', linestyle='-', linewidth=2, label=f'GCC‑PHAT mean: {np.mean(all_phat_in):.1f} km')
    if len(all_int_in):   ax.axhline(np.mean(all_int_in),   color='C2', linestyle='-', linewidth=2, label=f'Integer peak mean: {np.mean(all_int_in):.1f} km')
    if len(all_par_in):   ax.axhline(np.mean(all_par_in),   color='C3', linestyle='-', linewidth=2, label=f'Parabolic peak mean: {np.mean(all_par_in):.1f} km')
else:
    ax.text(0.5, 0.5, 'No inliers to plot', transform=ax.transAxes, ha='center', va='center')

ax.axhline(true_dist, color='k', linestyle='--', linewidth=1.5, label=f'True distance: {true_dist:.1f} km')
ax.set_xlabel('Arbitrary sample index')
ax.set_ylabel('Estimated distance (km)')
ax.set_title('IQR‑filtered individual pair distance estimates across the entire sweep (all methods)')
ax.set_ylim([300,700])
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig('fig08_global_pair_distribution.png', dpi=150)
plt.close()

print("All figures saved from sweep_results.npz")