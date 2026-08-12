#!/usr/bin/env python3
"""
Load full 4‑D sweep results and produce heatmaps for various fibre counts.
Output files are saved in a dedicated subfolder inside the result directory.
"""
import numpy as np
import os
from plotting import plot_std_heatmap, plot_bias_heatmap

# ======================= Configuration =======================
# Path to the folder that contains full_4d_arrays_*ch.npz
result_dir = "/users/240404662/PhD/xcorr-cd-wdm-fusion-localization/scripts/sim_Transient_Localization/results/baseline_20260808_103842"

channel_counts = [4, 6, 8, 10, 12]
fibre_counts = [1, 2, 4, 6, 8, 10]

# ---- create a subfolder for these fibre‑count heatmaps ----
output_subdir = "fibre_count_analysis"
output_dir = os.path.join(result_dir, output_subdir)
os.makedirs(output_dir, exist_ok=True)
print(f"Output folder: {output_dir}")

# ======================= Main loop =======================
for ch in channel_counts:
    fname = os.path.join(result_dir, f"full_4d_arrays_{ch}ch.npz")
    if not os.path.exists(fname):
        print(f"WARNING: {fname} not found – skipping {ch}ch.")
        continue

    data = np.load(fname, allow_pickle=True)
    band = data['BANDWIDTHS']
    snrs = data['SNRS']
    true_km = data['event_distance_km'].item()

    methods_2d = ['phase_est_2d', 'phat_est_2d', 'int_est_2d', 'par_est_2d']
    methods_rv = ['phase_est_rv', 'phat_est_rv', 'int_est_rv', 'par_est_rv']

    for nf in fibre_counts:
        for method_name in methods_2d + methods_rv:
            arr_4d = data[method_name]               # shape (n_fibres, nBW, nSNR, nReal)
            if nf > arr_4d.shape[0]:
                continue
            sub = arr_4d[:nf, ...]                   # first nf fibres
            avg = np.nanmean(sub, axis=0)             # average over fibres
            std = np.nanstd(avg, axis=2)              # across realisations
            bias = np.nanmean(avg, axis=2) - true_km

            # Build descriptive label and filename
            label = method_name.replace('_est_', ' ').replace('_', ' ').title()
            safe = f"{method_name}_{ch}ch_{nf}fibres"
            fig_std_path = os.path.join(output_dir, f'fig_std_{safe}.png')
            fig_bias_path = os.path.join(output_dir, f'fig_bias_{safe}.png')

            plot_std_heatmap(std, snrs, band,
                             f'{label} std (km) – {ch}ch, {nf} fibres',
                             fig_std_path, vmax=100)
            plot_bias_heatmap(bias, snrs, band,
                              f'{label} bias (km) – {ch}ch, {nf} fibres',
                              fig_bias_path, vmin=-50, vmax=50)
    print(f"Finished channel_count = {ch}")

print(f"All fibre‑count heatmaps saved in {output_dir}")