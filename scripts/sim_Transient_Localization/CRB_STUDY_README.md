# CRB-constrained CD localisation study

This folder contains a self-contained study of the **Cramér–Rao bound (CRB)**
for transient CD localisation, using the **2-D SOP** signal with parabolic
interpolation after rotating the centroid to the north pole.

It reproduces exactly the definitions in
`Cramér–Rao Bound (CRB) for Time-Delay Estimation — Simple Summary.md`:
energy SNR `E/N0`, the mean-square bandwidth `beta^2`, and the two-sensor
**effective SNR**.

## 1. Definitions implemented (`crb.py`)

Discrete signals `x[n]`, `n = 0..N-1`, spacing `dt`, per-component noise
variance `sigma^2`.

| Quantity | Formula | Code |
|---|---|---|
| Energy SNR | `SNR = sum_n x[n]^2 / sigma^2`  (= `E/N0` with `N0 = sigma^2 dt`) | `per_component_snr` |
| Noise for a target SNR | `sigma^2 = sum_n x[n]^2 / SNR_lin` | `noise_variance_for_snr` |
| Mean-square bandwidth | `beta^2 = sum_k (2 pi f_k)^2 |X_k|^2 / sum_k |X_k|^2` | `mean_square_bandwidth` |
| Deterministic bound (sec. 2) | `CRB = 1 / sum_c E_c beta_c^2 / sigma^2` | `crb_delay_deterministic(_pair)` |
| Two-sensor effective SNR (sec. 3.2/3.3) | `SNR_eff(f) = rho^2/(1+2 rho)`, `rho = P_coh/(N sigma^2)` | `effective_snr_spectrum` |
| Fisher information | `J = sum_c 2 sum_{k>0} w_k^2 SNR_eff,c[k]` | `fisher_information_delay` |
| Distance mapping | `sigma_Z = sqrt(CRB_tau) * 1e12 / |int D dlambda|` | `crb_distance` |
| Pair combination | inverse-variance, and correlation-aware **star** combination | `combined_crb_distance`, `star_combined_crb_distance` |

Notes:
* `P_coh(f) = | sum_c X_i,c(f) conj(X_j,c(f)) |` is the **coherent
  cross-spectrum** used as the common signal (choice (a)); it reduces to the
  doc's single-`S` formula for identical channels.
* Each independent signal component contributes its own Fisher information, so
  the per-component contributions are summed.
* The effective bound is validated against Monte-Carlo: the parabolic
  correlator saturates it at high SNR and `CRB_eff ~ 2 * CRB_det` asymptotically
  (the reference-channel noise penalty).

## 2. Pipeline / files

| File | Role |
|---|---|
| `crb.py` | CRB definitions (pure functions). |
| `channel_selection.py` | WDM subset schemes (alternating / extremes / uniform / first_n). |
| `generate_jones_crb.py` | Jones generator — integrates only the union of analysed channels, using a **batched** closed-form SU(2) tree-product (`propagate_channels`) that solves all channels × time snapshots in one set of BLAS matmuls (~30× faster than the per-sample `propagate_unitary` loop). |
| `process_crb_sweep.py` | 2-D SOP, energy-SNR noise, parabolic estimator, CRB-weighted pairing; saves raw + CRB arrays. |
| `analyse_crb_sweep.py` | std/bias heatmaps with CRB overlay, std-vs-SNR, multi-fibre combining. |
| `validate_crb.py` | Toy + identity validation of the bounds. |

Configs: `sweep_config_crb_test.yaml` (2 fibres, 24x24, counts {4,8}),
`..._test_bandpass.yaml`, and full-scale `sweep_config_crb.yaml`.

## 3. How to run

```bash
python validate_crb.py                              # bound validation
python generate_jones_crb.py sweep_config_crb_test.yaml
python process_crb_sweep.py  sweep_config_crb_test.yaml
python analyse_crb_sweep.py  sweep_config_crb_test.yaml
```

## 4. Raw data saved (per channel count, `crb_results_<n>ch.npz`)

`BANDWIDTHS`, `SNRS_dB`, `SNRS_lin`, `channel_indices_full/local`,
`wavelengths_nm`, `pair_indices`, `D_abs`, `event_distance_km`;
per (fibre, bw, snr, real): `est_crb`, `est_lambda`, `est_crb_iqr`;
per (fibre, bw, snr): `crb_delay_pairs`, `crb_dist_pairs`,
`crb_dist_combined_eff`, `crb_dist_combined_det`, `sigma2`;
per (fibre, bw, channel): `beta2`, `energy`; and optionally `pair_Z`
(all per-pair distances, when `output.save_pairs: "all"`).

## 5. Key findings (test scale)

* Bias -> 500 km (true event distance) at high SNR; large at low SNR
  (threshold region).
* Std tracks the **star-combined CRB** at high SNR (ratio ~0.9–1.2 across
  bandwidths); a direct per-pair Monte-Carlo on real signals reaches
  `MC/CRB_eff ~ 1.07`.
* The naive all-pairs inverse-variance combination is ~2x optimistic (pairs
  share channels); the `n_ch - 1` **star** combination is the achievable bound.
* Multi-fibre averaging follows `1/sqrt(N_fibres)`.
* **Filtering does not move the CRB** (S/N is invariant under linear
  filtering). Empirically at the test bandwidth: no-filter ~= low-pass (3B),
  while band-pass (0.5B–3B) is *worse* because it discards the low-frequency
  transient energy. All variants converge to the same CRB at high SNR.
* The three pair-weightings (`1/sigma_Z^2` CRB, `D^2`, and CRB+IQR) give
  essentially identical std for this channel set.

## 6. HPC run (final study)

The pipeline is self-contained and checkpointed, so it is safe to resubmit.

```bash
./run_crb_hpc.sh sweep_config_crb.yaml
```

Requirements: `numpy scipy matplotlib pyyaml tqdm joblib` and the
`sop-simulation-lib` importable (editable install or `PYTHONPATH` to its `src`).
Python 3.8+ is fine (the library now uses `from __future__ import annotations`).

Key config knobs (`sweep_config_crb.yaml`):
* `run.n_jobs` — parallel workers for the Monte-Carlo stage (set to
  `$SLURM_CPUS_PER_TASK`).
* `sweep.n_fibres`, `sweep.n_realisations`, grid `num`s, `analysis.channel_counts`.
* `output.save_pairs` — `"all"` keeps every per-pair distance (large);
  `"none"` keeps only the aggregated + CRB arrays.
* `filter.mode` — `"none"` for the canonical bound study.

Stage costs (single node): generation is serial ODE integration
(~1 min per fibre·bandwidth·12 channels here); the Monte-Carlo stage dominates
and scales with `n_fibres × n_bandwidths × n_snrs × n_realisations × pairs`
and parallelises over realisations / cores. A `ckpt_<n>ch.npz` checkpoint is
written after each fibre and removed on completion; if the job dies, rerun the
same command to resume.

SLURM example:

```bash
#!/bin/bash
#SBATCH --job-name=crb
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=24:00:00
module load python
export PYTHONPATH=/path/to/sop-simulation-lib/src:$PYTHONPATH
python -c "import yaml,json; p='sweep_config_crb.yaml'; c=yaml.safe_load(open(p)); c['run']['n_jobs']=32; yaml.safe_dump(c,open(p,'w'))"
./run_crb_hpc.sh sweep_config_crb.yaml
```

## 7. Notes

* Noise seeds are **common across bandwidths and SNRs** (seed depends on fibre
  and realisation only), matching the existing pipeline. This smooths the
  heatmaps but produces correlated vertical stripes at low SNR. For the final
  HPC run, more realisations wash these out; alternatively make the seed
  depend on `(fibre, bandwidth, SNR, realisation)` for independent cells.
* The star combination uses the best reference channel; the `n_ch - 1`
  independent delays avoid the pair-redundancy inflation of the naive combine.
