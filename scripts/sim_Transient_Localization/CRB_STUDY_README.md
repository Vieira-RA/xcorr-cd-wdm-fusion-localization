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
| Pair combination | inverse-variance (naive), correlation-aware **star**, and covariance-aware **GLS** | `combined_crb_distance`, `star_combined_crb_distance`, `gls_combined_crb_distance` |

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
* The naive all-pairs inverse-variance combination is optimistic (pairs share
  channels/noise, so they are correlated and only `n_ch - 1` delays are
  independent). The covariance-aware **GLS** combination `w = Sigma_Z^+ 1`
  (structured `Sigma_tau = A diag(u) A^T`) is tight: sim/bound ratio
  **1.0-1.1** vs **1.5 (4ch) / 2.0 (8ch) / 2.6 (12ch)** for the naive bound,
  matching the star combination. The GLS and naive estimators themselves
  perform nearly identically (`est_gls` ~= `est_crb`) — the covariance model
  fixes the *bound*, not the estimator.
* Multi-fibre averaging follows `1/sqrt(N_fibres)`.
* **Filtering does not move the CRB** (S/N is invariant under linear
  filtering). Empirically at the test bandwidth: no-filter ~= low-pass (3B),
  while band-pass (0.5B–3B) is *worse* because it discards the low-frequency
  transient energy. All variants converge to the same CRB at high SNR.
* The three pair-weightings (`1/sigma_Z^2` CRB, `D^2`, and CRB+IQR) give
  essentially identical std for this channel set.

## 6. HPC run (final study)

`run_crb_hpc.sh` is a **SLURM batch script** (16 cores, 32 GB, 24 h), matching
the cluster setup of `run_sweep.sh` (conda env `phd_env`, repo at
`~/PhD/xcorr-cd-wdm-fusion-localization`). Edit the config, then submit:

```bash
sbatch run_crb_hpc.sh
tail -f crb_sweep_<jobid>.log
```

It runs three stages in order, each guarded by a flag at the top of the script
(`RUN_GENERATOR`, `RUN_PROCESSOR`, `RUN_ANALYSIS`) and error-checked:
**generate → process → analyse**.

* Stage 1 (generation) uses multithreaded BLAS for the batched tree-product.
* Stage 2 (Monte-Carlo) runs one BLAS thread per joblib worker; the worker count
  is `CRB_N_JOBS` (set to `$SLURM_CPUS_PER_TASK` in the script), which overrides
  `run.n_jobs` in the config.
* Stage 2 is **checkpointed per fibre** (`ckpt_<n>ch.npz`, removed on
  completion), so a resubmission resumes where it stopped.

There is also a built-in import sanity check — if `sop-simulation-lib` isn't
installed in `phd_env` the job aborts immediately with a clear message.
Requirements: `numpy scipy matplotlib pyyaml tqdm joblib` + `sop-simulation-lib`
editable install (or `PYTHONPATH` to its `src`).

Key config knobs (`sweep_config_crb.yaml`):
* `sweep.n_fibres`, `sweep.n_realisations`, grid `num`s, `analysis.channel_counts`.
* `output.save_pairs` — `"none"` keeps the per-realisation estimates + CRB
  arrays only; `"all"` also stores every per-pair distance (much larger).
* `filter.mode` — `"none"` for the canonical bound study.

Non-SLURM (interactive) use: `bash run_crb_hpc.sh` still works; it is a normal
shell script.

## 7. Estimator comparison (SNR-free fusion)

All fusion rules below need **no SNR knowledge** — only the dispersion integral
`D = |∫D dλ|` (known). They are compared on per-pair distances.

| estimator | weights | notes |
|---|---|---|
| `single` | one pair (max `D`) | best single measurement, baseline |
| `single (worst)` | one pair (min `D`) | worst single measurement, baseline |
| `average` | equal | naive fusion |
| `disp` | `w ∝ D²` | inverse-variance from dispersion (BLUE if CRB common) |
| `robust` | `w ∝ D² · Tukey-bisquare` | smooth down-weighting of outliers (prior-study scheme) |

Reference: `crb_gls` = covariance-aware (GLS) combined CRB.

To run, keep **all per-pair distances** (they are discarded otherwise):

```bash
# 1) re-run only Stage 2 on the existing Jones data (no regeneration)
sbatch run_crb_hpc.sh sweep_config_crb_pairs.yaml   # RUN_GENERATOR=no inside
# 2) pull results_crb_full_pairs/ locally, then:
python estimator_comparison.py results_crb_full_pairs
```

Outputs: `figs_estimators/F1..F6_*.png` and `estcomp_<n>ch.npz` (per-realisation
estimates + RMSE/bias/std/MAE/success surfaces + GLS bound) so the figures can
be rebuilt without recomputation.

## 8. Notes
* Noise seeds are **common across bandwidths and SNRs** (seed depends on fibre
  and realisation only), matching the existing pipeline. This smooths the
  heatmaps but produces correlated vertical stripes at low SNR. For the final
  HPC run, more realisations wash these out; alternatively make the seed
  depend on `(fibre, bandwidth, SNR, realisation)` for independent cells.
* The star combination uses the best reference channel; the `n_ch - 1`
  independent delays avoid the pair-redundancy inflation of the naive combine.
