#!/usr/bin/env python3
"""
Script 1 – Jones‑matrix generator for CD localisation sweep.
Reads `sweep_config.yaml` and, for each bandwidth, computes the clean
Jones matrices U[ch, t] for all channels after applying chromatic‑dispersion
delays.  Results are saved as compressed .npz files in the output directory.
"""
import os
import time
import yaml
import numpy as np
from tqdm import tqdm

from fiber_propagation import propagate_unitary
from pmd_model import generate_pmd_waveplates
from chromatic_dispersion import (
    frequency_to_wavelength,
    relative_channel_delays,
    delay_jones_sequence,
    integrated_dispersion,
)

def generate_alternating_grid(f_min, f_max, n_channels, spacing):
    """
    Generate frequency grid with alternating placement:
    f_max, f_min, f_max-spacing, f_min+spacing, f_max-2*spacing, ...
    Result is sorted in increasing frequency.
    """
    f_low = []
    f_high = []
    f_low.append(f_min)
    f_high.append(f_max)
    k = 1
    while len(f_low) + len(f_high) < n_channels:
        f_high.append(f_max - k * spacing)
        if len(f_low) + len(f_high) < n_channels:
            f_low.append(f_min + k * spacing)
        k += 1
    all_freqs = f_low + f_high
    # Actually we need ascending order for correct index‑frequency mapping: channel 0 = lowest frequency.
    return np.sort(all_freqs)

# ====================== load configuration ======================
with open("sweep_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# unpack configuration (explicit float/int conversions for safety)
fib = cfg["fibre"]
L_km = float(fib["L_km"])
L = L_km * 1e3
L_F = float(fib["L_F"])
D_pmd = float(fib["D_pmd"])
lambda0_nm = float(fib["lambda0_nm"])
lambda0 = lambda0_nm * 1e-9
c = 299792458.0
omega0 = 2 * np.pi * c / lambda0
fibre_seed = int(fib["seed"])

A_pulse = float(cfg["transient"]["A_pulse"])

fs_factor = float(cfg["time"]["fs_factor"])
t_end_factor = float(cfg["time"]["t_end_factor"])

# unpack WDM config
wdm = cfg["wdm"]
f_min = float(wdm["f_min_Hz"])
f_max = float(wdm["f_max_Hz"])

generation = wdm.get("generation", "custom").lower()
if generation == "full_grid":
    spacing = float(wdm.get("full_grid_spacing_Hz", 50e9))
    freq_channels = np.arange(f_min, f_max + spacing/2, spacing)
    if freq_channels[-1] < f_max:
        freq_channels = np.append(freq_channels, f_max)
    freq_channels = np.sort(freq_channels)
else:
    n_channels = int(wdm["n_channels"])
    grid_type = wdm.get("channel_grid", "uniform").lower()
    if grid_type == "alternating":
        spacing = float(wdm.get("channel_spacing_Hz", 50e9))
        freq_channels = generate_alternating_grid(f_min, f_max, n_channels, spacing)
    else:
        freq_channels = np.linspace(f_min, f_max, n_channels)

# update n_channels and derived quantities
n_channels = len(freq_channels)
omega_channels = 2 * np.pi * freq_channels
wavelengths_nm = frequency_to_wavelength(freq_channels)

event_distance_km = float(cfg["event"]["distance_km"])

sweep_cfg = cfg["sweep"]["bandwidths"]
BANDWIDTHS = np.logspace(np.log10(float(sweep_cfg["start_Hz"])),
                         np.log10(float(sweep_cfg["stop_Hz"])),
                         int(sweep_cfg["num"]))

output_dir = cfg["output"]["jones_dir"]

# ====================== pre‑compute fibre profile ======================
z, beta0, beta_prime, L_seg = generate_pmd_waveplates(
    L, L_F, D_pmd, lambda0, seed=fibre_seed
)

# channel delays (wavelength dependent, same for all bandwidths)
channel_delays = relative_channel_delays(wavelengths_nm, event_distance_km)

# create output directory
os.makedirs(output_dir, exist_ok=True)

# ====================== save metadata ======================
import itertools
pair_indices = list(itertools.combinations(range(n_channels), 2))
integrals_abs = np.array([
    abs(integrated_dispersion(wavelengths_nm[i], wavelengths_nm[j]))
    for i, j in pair_indices
])

np.savez_compressed(os.path.join(output_dir, "metadata.npz"),
                    wavelengths_nm=wavelengths_nm,
                    freq_channels=freq_channels,
                    channel_delays=channel_delays,
                    event_distance_km=event_distance_km,
                    pair_indices=np.array(pair_indices),
                    integrals_abs=integrals_abs,
                    BANDWIDTHS=BANDWIDTHS,
                    cfg=cfg)   # store the full config for reproducibility

# ====================== main sweep over bandwidths ======================
total_bw = len(BANDWIDTHS)
bw_times = []
overall_start = time.time()

for i_bw, bw in enumerate(tqdm(BANDWIDTHS, desc="Generating Jones matrices")):
    bw_start = time.time()
    sigma_t = 0.3748 / bw
    fs = fs_factor * bw
    dt = 1.0 / fs
    t_end = t_end_factor * sigma_t
    n_samples = int(t_end / dt) + 1
    t_grid = np.linspace(0, t_end, n_samples)
    t0 = t_end / 2

    # Gaussian modulation envelope (no noise)
    g = np.exp(-((t_grid - t0) ** 2) / (2 * sigma_t ** 2))
    s_env = 1.0 + A_pulse * g

    # Allocate Jones matrices for all channels × time steps
    U_all = np.zeros((n_channels, n_samples, 2, 2), dtype=complex)

    for ch_idx, omega_ch in enumerate(omega_channels):
        delta_omega = omega_ch - omega0
        beta_base = beta0 + delta_omega * beta_prime
        for t_idx in range(n_samples):
            beta_t = s_env[t_idx] * beta_base
            U_all[ch_idx, t_idx] = propagate_unitary(z, beta_t)

    # Apply chromatic‑dispersion delays (uses dt)
    for ch in range(n_channels):
        U_all[ch] = delay_jones_sequence(U_all[ch], dt, channel_delays[ch])

    # Save this bandwidth’s Jones matrices
    filename = os.path.join(output_dir, f"jones_bw_{bw:.0f}Hz.npz")
    np.savez_compressed(filename,
                        U_all=U_all,
                        dt=dt,
                        bw=bw,
                        n_samples=n_samples)

    # timing / progress
    bw_times.append(time.time() - bw_start)
    avg_per_bw = np.mean(bw_times)
    remaining = avg_per_bw * (total_bw - i_bw - 1)
    print(f"B = {bw:.0f} Hz done. Est. remaining: {remaining/60:.1f} min")

print(f"All Jones matrices saved in '{output_dir}/'")
print("Script 1 completed.")