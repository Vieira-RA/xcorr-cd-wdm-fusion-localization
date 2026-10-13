# CRB-constrained CD localisation study

Self-contained study of the **Cramér–Rao bound (CRB)** for transient
chromatic-dispersion (CD) localisation over WDM links, using the **2-D SOP**
signal with parabolic interpolation after rotating the centroid to the north
pole.

It reproduces exactly the definitions in
`Cramér–Rao Bound (CRB) for Time-Delay Estimation — Simple Summary.md`:
energy SNR `E/N0`, mean-square bandwidth `beta^2`, and the two-sensor
**effective SNR**.

---

## 1. Definitions implemented (`crb.py`)

Discrete signals `x[n]`, `n = 0..N-1`, spacing `dt`, per-component noise
variance `sigma^2`.

| Quantity | Formula | Code |
|---|---|---|
| Energy SNR | `SNR = sum_n x[n]^2 / sigma^2`  (= `E/N0`, `N0 = sigma^2 dt`) | `per_component_snr` |
| Noise for target SNR | `sigma^2 = sum_n x[n]^2 / SNR_lin` | `noise_variance_for_snr` |
| Mean-square bandwidth | `beta^2 = sum_k (2 pi f_k)^2 |X_k|^2 / sum_k |X_k|^2` | `mean_square_bandwidth` |
| Deterministic bound | `CRB = 1 / sum_c E_c beta_c^2 / sigma^2` | `crb_delay_deterministic(_pair)` |
| Two-sensor effective SNR | `SNR_eff(f) = rho^2/(1+2 rho)`, `rho = P_coh/(N sigma^2)` | `effective_snr_spectrum` |
| Fisher information | `J = sum_c 2 sum_{k>0} w_k^2 SNR_eff,c[k]` | `fisher_information_delay` |
| Distance mapping | `sigma_Z = sqrt(CRB_tau) * 1e12 / |int D dlambda|` | `crb_distance` |
| Pair combination | inverse-variance (naive), correlation-aware **star**, covariance-aware **GLS** | `combined_crb_distance`, `star_combined_crb_distance`, `gls_combined_crb_distance` |

Notes:
* `P_coh(f) = | sum_c X_i,c(f) conj(X_j,c(f)) |` is the **coherent
  cross-spectrum** used as the common signal; it reduces to the doc's single-`S`
  formula for identical channels.
* Each independent signal component contributes its own Fisher information, so
  per-component contributions are summed.
* Validated against Monte-Carlo: the parabolic correlator saturates the
  effective bound at high SNR (real pair `MC/CRB_eff ~ 1.07`) and
  `CRB_eff ~ 2 * CRB_det` asymptotically (reference-noise penalty).

---

## 2. Pipeline / files

| File | Role |
|---|---|
| `crb.py` | CRB definitions (pure functions). |
| `channel_selection.py` | WDM subset schemes (alternating / extremes / uniform / first_n). |
| `generate_jones_crb.py` | Jones generator — integrates only the union of analysed channels via a **batched** closed-form SU(2) tree-product (`propagate_channels`), solving all channels × time snapshots in one set of BLAS matmuls (~30× faster than the per-sample loop). |
| `process_crb_sweep.py` | 2-D SOP, energy-SNR noise, parabolic estimator, CRB-weighted pairing; stores per-realisation estimates + CRB arrays (+ per-pair distances if requested). **Checkpointed per fibre.** |
| `analyse_crb_sweep.py` | std/bias heatmaps with **GLS** CRB overlay, std-vs-SNR, estimator comparison, multi-fibre and cross-count plots. |
| `estimator_comparison.py` | SNR-free fusion comparison (worst/best single pair, average, `D²`, robust) → publication figures F1–F6 + saved derived arrays. |
| `validate_crb.py` | Toy + identity validation of the bounds. |
| `validate_gls.py` | Naive vs GLS vs star combination validation (needs per-pair data). |
| `compare_filters.py` | no-filter vs low-pass vs band-pass comparison. |

Configs:
* `sweep_config_crb_test.yaml` (+ `_test_lowpass`, `_test_bandpass`) — small tests.
* `sweep_config_crb_mid.yaml` — 5 fibres, 30×30, counts {4,8,12}.
* `sweep_config_crb.yaml` — **production** (10 fibres, 40×40, counts {4,6,8,10,12}, `save_pairs: none`).
* `sweep_config_crb_pairs.yaml` — same grids, `save_pairs: all`, for the estimator study (reuses the production Jones data).

---

## 3. How to run

```bash
python validate_crb.py                                  # bound validation
python generate_jones_crb.py  <config>                  # stage 1 (skip if Jones exists)
python process_crb_sweep.py   <config>                  # stage 2 (checkpointed)
python analyse_crb_sweep.py   <config>                  # stage 3 (heatmaps)
python estimator_comparison.py results_crb_full_pairs   # SNR-free fusion study
```

---

## 4. Raw data saved (`crb_results_<n>ch.npz`)

Metadata: `BANDWIDTHS`, `SNRS_dB`, `SNRS_lin`, `channel_indices_full/local`,
`wavelengths_nm`, `pair_indices`, `D_abs`, `event_distance_km`,
`filter_mode`, `filter_params`, `cfg`.

Per (fibre, bw, snr, real): `est_crb`, `est_gls`, `est_lambda`, `est_crb_iqr`.
Per (fibre, bw, snr): `crb_delay_pairs`, `crb_dist_pairs`,
`crb_dist_combined_eff`, `crb_dist_combined_det`, `crb_dist_combined_gls`,
`sigma2`.
Per (fibre, bw, channel): `beta2`, `energy`.
Optional: `pair_Z` (all per-pair distances) when `output.save_pairs: "all"`.

`estimator_comparison.py` additionally writes `estcomp_<n>ch.npz`
(per-realisation estimates for all fusion rules + RMSE/bias/std/MAE/success
surfaces + GLS bound) so figures can be rebuilt without recomputation.

---

## 5. Key findings

* Bias → ~500 km (true distance) at high SNR; large in the threshold region.
* Std tracks the CRB at high SNR; the parabolic correlator saturates the
  effective bound (real-pair `MC/CRB_eff ~ 1.07`).
* **Naive all-pairs inverse-variance combination is optimistic** (pairs share
  channels/noise → correlated; only `n_ch - 1` delays are independent). The
  correlation-aware **star** and covariance-aware **GLS** combinations are tight:
  sim/bound **1.0–1.1**, vs **1.5 (4ch) / 2.0 (8ch) / 2.6 (12ch)** for the naive bound.
  GLS and naive *estimators* perform nearly identically — the covariance fixes the
  *bound*, not the estimator.
* **SNR-free fusion ranking** (production, 10 fibres):
  **worst single pair ≫ pure average > best single pair > `D²` ≈ `D²`+bisquare ≈ CRB.**
  Equal-weight averaging is *beaten by a single well-chosen pair*; dispersion
  weighting is required to reach the CRB.
* **Multi-fibre averaging** follows `1/sqrt(N_fibres)`, rule-independent.
* **Channel count**: more channels lower the RMSE through `sum D_ij^2`; the SNR
  saving over a single pair grows with channel count.
* **Filtering does not move the CRB** (S/N invariant under linear filtering);
  no-filter ≈ low-pass, while band-pass (0.5B–3B) is worse (discards low-frequency
  transient energy). All converge to the same CRB at high SNR.

---

## 6. HPC run (SLURM)

`run_crb_hpc.sh` is a SLURM batch script (16 cores, 32 GB, 24 h) matching the
cluster setup of `run_sweep.sh` (conda env `phd_env`, repo at
`~/PhD/xcorr-cd-wdm-fusion-localization`). It takes the config as an argument:

```bash
sbatch run_crb_hpc.sh sweep_config_crb.yaml          # production
sbatch run_crb_hpc.sh sweep_config_crb_pairs.yaml    # set RUN_GENERATOR=no to reuse Jones
tail -f crb_sweep_<jobid>.log
```

Stages are guarded by flags at the top (`RUN_GENERATOR`, `RUN_PROCESSOR`,
`RUN_ANALYSIS`) and error-checked: **generate → process → analyse**.
* Stage 1 uses multithreaded BLAS for the batched tree-product.
* Stage 2 runs one BLAS thread per joblib worker; the count is `CRB_N_JOBS`
  (defaults to `$SLURM_CPUS_PER_TASK`, overrides `run.n_jobs`).
* Stage 2 is **checkpointed per fibre** (`crb_ckpt_<n>ch.npz`, removed on
  completion); resubmit to resume.
* Built-in import check: aborts if `sop-simulation-lib` isn't installed.

Requirements: `numpy scipy matplotlib pyyaml tqdm joblib` + `sop-simulation-lib`
editable install (or `PYTHONPATH` to its `src`).

---

## 7. Estimator comparison (SNR-free fusion)

All rules need **no SNR knowledge** — only `D = |∫D dλ|` (known).

| estimator | weights | notes |
|---|---|---|
| `single` | one pair (max `D`) | best single measurement |
| `single (worst)` | one pair (min `D`) | worst single measurement |
| `average` | equal | naive fusion |
| `disp` | `w ∝ D²` | inverse-variance from dispersion (BLUE if CRB common) |
| `robust` | `w ∝ D² · Tukey-bisquare` | smooth down-weighting of outliers (prior-study scheme) |

Reference: `crb_gls` = covariance-aware combined CRB.

```bash
# one Stage-2 re-run on the existing Jones data (keeps per-pair distances):
sbatch run_crb_hpc.sh sweep_config_crb_pairs.yaml    # RUN_GENERATOR=no
# then locally:
python estimator_comparison.py results_crb_full_pairs
```

Outputs: `figs_estimators/F1..F6_*.png` (publication style, 300 dpi) and
`estcomp_<n>ch.npz`.

---

## 8. Notes

* Noise seeds are **common across bandwidths and SNRs** (seed depends on fibre
  and realisation only), matching the existing pipeline. This smooths heatmaps
  but produces correlated vertical stripes at low SNR; more realisations wash
  them out, or make the seed depend on `(fibre, bw, snr, real)` for independent
  cells.
* The star combination uses the best reference channel; the `n_ch - 1`
  independent delays avoid the pair-redundancy inflation of the naive combine.
* The library uses `from __future__ import annotations` for Python 3.8
  compatibility (no effect on ≥3.9).

---

## 9. Paper

`../../paper/` holds the conference draft:
* `CRB_transient_localization_optica.tex` — Optica/OFC style (`opticameet3`),
  with an equation-reference appendix page.
* `README.md` — build notes and figure swap-in instructions.
