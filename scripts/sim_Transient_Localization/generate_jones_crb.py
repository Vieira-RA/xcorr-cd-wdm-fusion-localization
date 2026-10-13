#!/usr/bin/env python3
"""
CRB-study Jones-matrix generator.

Like generate_jones_data_MF.py, but only integrates the union of WDM channels
that the analysis will actually use (much cheaper than the full 241-channel
grid).  Saves one .npz per (fibre, bandwidth) plus a metadata file.

Run:  python generate_jones_crb.py sweep_config_crb_test.yaml
"""
import os
import sys
import time
import numpy as np
import yaml
from tqdm import tqdm

from fiber_propagation import PAULI, _tree_product
from pmd_model import generate_pmd_waveplates
from chromatic_dispersion import (
    frequency_to_wavelength,
    relative_channel_delays,
    delay_jones_sequence,
)
from channel_selection import union_of_selections


def propagate_channels(z, beta0_all, delta):
    """
    Batched Jones propagation for many channels at once.

    Solves dU/dz = -i/2 [ beta(z,t) . sigma ] U with

        beta(z, t) = beta0_all[ch](z) * (1 + delta(z, t))

    for every channel and time snapshot simultaneously, using the closed-form
    SU(2) segment matrices and a single binary-tree product of *batched* (BLAS)
    2x2 matrix multiplications.

    Parameters
    ----------
    z : (N,) position grid.
    beta0_all : (n_ch, 3, N) channel-dependent static birefringence.
    delta : (N, nt) or (N,) or scalar, relative magnitude modulation
        (same across channels), sampled on the z grid.

    Returns
    -------
    U : (n_ch, nt, 2, 2) complex
    """
    z = np.asarray(z, dtype=float)
    beta0_all = np.asarray(beta0_all, dtype=float)
    n_ch = beta0_all.shape[0]

    beta_seg = beta0_all[:, :, :-1]                 # (n_ch, 3, M)
    M = beta_seg.shape[2]
    dz = np.diff(z)                                 # (M,)
    mag = np.linalg.norm(beta_seg, axis=1)          # (n_ch, M)
    theta0 = mag * dz                               # (n_ch, M)

    safe = mag > 0
    denom = np.where(safe, mag, 1.0)
    n_hat = beta_seg / denom[:, None, :]
    n_hat *= safe[:, None, :]

    # S[ch, m] = n_hat . sigma  -> (n_ch, M, 2, 2)
    S = np.einsum("cjk,jab->ckab", n_hat, PAULI)

    delta = np.asarray(delta, dtype=float)
    if delta.ndim == 0:
        delta = np.full((z.shape[0], 1), float(delta))
    elif delta.ndim == 1:
        delta = delta[:, None]
    if delta.shape[0] != z.shape[0]:
        raise ValueError("delta first axis must match z")
    delta = delta[:-1]                              # (M, nt)
    nt = delta.shape[1]

    theta = theta0[:, :, None] * (1.0 + delta[None, :, :])   # (n_ch, M, nt)
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    I2 = np.eye(2, dtype=complex)

    # segment matrices -> (n_ch, M, nt, 2, 2)
    mats = c[..., None, None] * I2 - 1j * s[..., None, None] * S[:, :, None, :, :]
    # fold (channel, time) into one batch, tree-product over the M axis
    mats = mats.transpose(1, 0, 2, 3, 4).reshape(M, n_ch * nt, 2, 2)
    U = _tree_product(mats)                          # (n_ch*nt, 2, 2)
    return U.reshape(n_ch, nt, 2, 2)


def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- fibre ----
    fib = cfg["fibre"]
    L = float(fib["L_km"]) * 1e3
    L_F = float(fib["L_F"])
    D_pmd = float(fib["D_pmd"])
    lambda0 = float(fib["lambda0_nm"]) * 1e-9
    base_seed = int(fib["seed"])

    # ---- transient / time grid ----
    A_pulse = float(cfg["transient"]["A_pulse"])
    fs_factor = float(cfg["time"]["fs_factor"])
    t_end_factor = float(cfg["time"]["t_end_factor"])

    # ---- WDM full grid ----
    wdm = cfg["wdm"]
    f_min = float(wdm["f_min_Hz"])
    f_max = float(wdm["f_max_Hz"])
    spacing = float(wdm.get("full_grid_spacing_Hz", 50e9))
    freq_channels = np.arange(f_min, f_max + spacing / 2, spacing)
    if freq_channels[-1] < f_max:
        freq_channels = np.append(freq_channels, f_max)
    freq_channels = np.sort(freq_channels)
    wavelengths_nm = frequency_to_wavelength(freq_channels)
    N_full = len(freq_channels)

    event_distance_km = float(cfg["event"]["distance_km"])
    channel_delays = relative_channel_delays(wavelengths_nm, event_distance_km)

    # ---- sweep grids ----
    sw = cfg["sweep"]
    BANDWIDTHS = np.logspace(np.log10(float(sw["bandwidths"]["start_Hz"])),
                             np.log10(float(sw["bandwidths"]["stop_Hz"])),
                             int(sw["bandwidths"]["num"]))
    n_fibres = int(sw["n_fibres"])

    # ---- channels to actually generate ----
    analysis = cfg.get("analysis", {})
    counts = analysis.get("channel_counts", [N_full])
    scheme = analysis.get("channel_selection", "alternating")
    union_indices = np.array(union_of_selections(N_full, counts, scheme))
    n_union = len(union_indices)

    output_dir = cfg["output"]["jones_dir"]
    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(output_dir, "metadata.npz"),
        wavelengths_nm=wavelengths_nm,
        freq_channels=freq_channels,
        channel_delays=channel_delays,
        event_distance_km=event_distance_km,
        union_indices=union_indices,
        channel_counts=np.array(counts),
        BANDWIDTHS=BANDWIDTHS,
        cfg=cfg,
    )

    print(f"Full grid: {N_full} channels. Generating {n_union} "
          f"({scheme}) for counts {counts}.")
    print(f"Bandwidths: {len(BANDWIDTHS)} | Fibres: {n_fibres}")

    omega0 = 2 * np.pi * 299792458.0 / lambda0
    overall = time.time()

    for i_fibre in range(n_fibres):
        fibre_seed = base_seed + i_fibre
        print(f"\n===== Fibre {i_fibre} (seed={fibre_seed}) =====")
        z, beta0, beta_prime, L_seg = generate_pmd_waveplates(
            L, L_F, D_pmd, lambda0, seed=fibre_seed
        )
        fibre_dir = os.path.join(output_dir, f"fibre_{i_fibre}")
        os.makedirs(fibre_dir, exist_ok=True)

        for i_bw, bw in enumerate(tqdm(BANDWIDTHS, desc=f"Fibre {i_fibre} BW")):
            sigma_t = 0.3748 / bw
            fs = fs_factor * bw
            dt = 1.0 / fs
            t_end = t_end_factor * sigma_t
            n_samples = int(t_end / dt) + 1
            t_grid = np.linspace(0, t_end, n_samples)
            t0 = t_end / 2.0
            s_env = 1.0 + A_pulse * np.exp(-((t_grid - t0) ** 2) / (2 * sigma_t ** 2))

            # Channel birefringence profiles: beta(z,t) = beta_base(z) * s_env(t)
            beta_all = np.empty((n_union, 3, z.shape[0]))
            for local_ch, full_idx in enumerate(union_indices):
                delta_omega = 2 * np.pi * freq_channels[full_idx] - omega0
                beta_all[local_ch] = beta0 + delta_omega * beta_prime

            delta_z_t = np.broadcast_to((s_env - 1.0)[None, :],
                                        (z.shape[0], n_samples))
            U_all = propagate_channels(z, beta_all, delta_z_t)

            for local_ch, full_idx in enumerate(union_indices):
                U_all[local_ch] = delay_jones_sequence(
                    U_all[local_ch], dt, channel_delays[full_idx]
                )

            np.savez_compressed(
                os.path.join(fibre_dir, f"jones_bw_{bw:.0f}Hz.npz"),
                U_all=U_all,
                dt=dt,
                bw=bw,
                n_samples=n_samples,
            )

    print(f"\nDone in {(time.time() - overall) / 60:.1f} min -> {output_dir}/")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "sweep_config_crb_test.yaml"
    main(config)
