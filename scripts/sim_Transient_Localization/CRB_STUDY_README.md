# CRB-constrained CD localisation study

A self-contained study of the **Cramér–Rao bound (CRB)** for transient
chromatic-dispersion (CD) localisation over WDM links, using the **2-D SOP**
signal with parabolic interpolation after rotating the centroid to the north
pole.

It reproduces the definitions in
`Cramér–Rao Bound (CRB) for Time-Delay Estimation — Simple Summary.md`:
energy SNR `E/N0`, mean-square bandwidth `beta^2`, and the two-sensor
**effective SNR**.

> **Status:** two production datasets exist — the **band-edge** run
> (`results_crb_full_pairs`) and the **equispaced** run
> (`results_crb_equispaced`). The equispaced run is the current focus. The
> `paper/` draft is frozen for now (do not edit its narrative).

---

## 1. Definitions (`crb.py`)

| Quantity | Formula | Code |
|---|---|---|
| Energy SNR | `SNR = sum_n x[n]^2 / sigma^2`  (= `E/N0`, `N0 = sigma^2 dt`) | `per_component_snr` |
| Noise for target SNR | `sigma^2 = sum_n x[n]^2 / SNR_lin` | `noise_variance_for_snr` |
| Mean-square bandwidth | `beta^2 = sum_k (2 pi f_k)^2 |X_k|^2 / sum_k |X_k|^2` | `mean_square_bandwidth` |
| Deterministic bound | `CRB = 1 / sum_c E_c beta_c^2 / sigma^2` | `crb_delay_deterministic(_pair)` |
| Effective SNR | `SNR_eff(f) = rho^2/(1+2 rho)`, `rho = P_coh/(N sigma^2)` | `effective_snr_spectrum` |
| Fisher information | `J = sum_c 2 sum_{k>0} w_k^2 SNR_eff,c[k]` | `fisher_information_delay` |
| Distance mapping | `sigma_Z = sqrt(CRB_tau) * 1e12 / |int D dlambda|` | `crb_distance` |
| Pair combination | inverse-variance (naive), **star**, covariance-aware **GLS** | `combined_crb_distance`, `star_combined_crb_distance`, `gls_combined_crb_distance` |

* `P_coh(f) = | sum_c X_i,c(f) conj(X_j,c(f)) |` is the **coherent
  cross-spectrum** (common-signal estimate).
* **Per-pair CRB validated**: empirical pair std / theoretical CRB ≈ **1.08–1.11**
  (equispaced) and **1.02** (production). The pair-level model is correct.
* **Combined bound — use the STAR bound, not GLS.** The GLS fits a per-channel
  additive variance model (`Var(τ_ij)=u_i+u_j`); on **wide-`D`-spread grids**
  (equispaced) the fit degrades and GLS can exceed the star bound (0.80 vs
  0.30) — impossible. `estimator_comparison.py` now uses
  `crb.star_combined_crb_distance` and stores `crb_star`.

---

## 2. Pipeline / files

| File | Role |
|---|---|
| `crb.py` | CRB definitions (pure functions). |
| `channel_selection.py` | Subset schemes: alternating / uniform / extremes / first_n. |
| `generate_jones_crb.py` | Batched Jones generator (~30× faster). Supports **full ITU grid and custom uniformly-spaced grids**. |
| `process_crb_sweep.py` | 2-D SOP, energy-SNR noise, parabolic estimator, stores per-realisation estimates (all-pairs + pool) + CRB arrays (+ `pair_Z`). Checkpointed per fibre. `CRB_N_JOBS` overrides `run.n_jobs`. |
| `analyse_crb_sweep.py` | std/bias heatmaps + CRB overlay, std-vs-SNR, cross-count plots. |
| `estimator_comparison.py` | SNR-free fusion study → figures **F1–F8** + `estcomp_<n>ch.npz`. `--replot` rebuilds figures in seconds from `estcomp`. |
| `estimator_variants.py` | **Controlled estimator test** (parabolic / phase-slope / GCC-PHAT / signed) — reports bias ± std per SNR. |
| `estimator_variants.slurm` | SLURM driver for the above (`$1`=config, `$2`=count). |
| `prototype_peak_bias.py` | Toy test of interpolation bias (symmetric Gaussian). |
| `validate_crb.py`, `validate_gls.py`, `compare_filters.py` | Bounds, GLS/star, filter comparisons. |
| `run_crb_hpc.sh` | SLURM driver (config as `$1`): generate → process → analyse. |
| `estimator_comparison.slurm` | SLURM driver for the figure job (`$1` = run dir). |

Configs: `sweep_config_crb_test*.yaml` (tests), `sweep_config_crb_mid.yaml`,
`sweep_config_crb.yaml` (production, full grid), `sweep_config_crb_pairs.yaml`
(production + `save_pairs: all`), **`sweep_config_crb_equispaced.yaml`** (current:
custom 30-ch equispaced, uniform {5,10,15}, 10 fibres, 100 real, 50×50,
`save_pairs: all`).

---

## 3. How to run

```bash
python validate_crb.py
python generate_jones_crb.py  <config>
python process_crb_sweep.py   <config>          # checkpointed per fibre
python analyse_crb_sweep.py   <config>
python estimator_comparison.py <run_dir> [--replot]
python estimator_variants.py  <config> <count>  # via estimator_variants.slurm
```

HPC (SLURM, conda `phd_env`):
```bash
sbatch run_crb_hpc.sh <config>                         # simulate
sbatch --dependency=afterok:<sim> estimator_comparison.slurm <run_dir>   # figures
sbatch estimator_variants.slurm <config> <count>       # estimator test
```

---

## 4. Raw data (`crb_results_<n>ch.npz`)

Metadata (`BANDWIDTHS`, `SNRS_dB`, `channel_indices_*`, `wavelengths_nm`,
`pair_indices`, `D_abs`, `event_distance_km`, `filter_mode`, `cfg`), per-cell
estimates (`est_crb`, `est_gls`, `est_lambda`, `est_crb_iqr`), CRB arrays
(`crb_delay_pairs`, `crb_dist_pairs`, `crb_dist_combined_*`), `sigma2`,
`beta2`, `energy`, and (with `save_pairs: all`) `pair_Z`.

`estcomp_<n>ch.npz`: per-realisation estimates (all-pairs + **pool** +
`--replot` surfaces), `crb_star`, and the `alpha_saving` sweep.

---

## 5. Findings

**Bounds**
* Per-pair CRB is accurate (empirical/theory ≈ 1.1).
* Use the **star** combined bound; GLS is unreliable for wide-`D` grids.

**Band-edge (extremes-first) grid — production, 12 ch**
* Naive all-pairs averaging is `10–10^3×` above the CRB (small-`D` pairs
  dominate). On a **filtered pool** (`D ≥ 0.25 max D`), equal / `D²` / robust
  **coincide** on the CRB → *weighting ≡ selection* there.

**Equispaced grid — current**
* **Robust (bisquare) is the best combiner.** High-SNR RMSE (averaged over BW):

  | count | average | `D²` | **robust** | CRB(star) |
  |---|---|---|---|---|
  | 5 ch | 10.38 | 9.74 | **9.35** | 4.73 |
  | 10 ch | 8.42 | 7.97 | **6.29** | 3.01 |
  | 15 ch | 6.60 | 6.25 | **5.48** | 2.39 |

  Robust wins the RMSE in **510–551 / 600** cells; ~10–25 % lower RMSE than
  plain averaging at mid SNR; comparable bias. **Robust separates from the
  others here** (unlike the band-edge grid) because the pool retains a range of
  precision and outliers.
* **Weighting benefit vs filter threshold (F8):** at α = 0 (all pairs) ≈
  **2.9 dB (10 ch), 3.6 dB (5 ch), 4.8 dB (15 ch)** — nonzero, **grows with
  channel count**, but modest (the closest equispaced pairs are 414 GHz apart,
  `D ≈ 70 ps/km`, vs 8 ps/km for adjacent ITU channels). Filtering (α ≳ 0.3)
  removes most of it.
* **Variance** is ~2–2.5× the star CRB at high BW/SNR.
* **Bias is bandwidth-dependent**: negligible at high `B` (~−0.04 km at 2 MHz),
  ~−2 km at ~22 kHz. The high-SNR floor is a **low-bandwidth** effect
  (sub-sample interpolation on coarse `dt`).
* Multi-fibre averaging follows `1/sqrt(N_fibres)`.
* Linear filtering does not move the CRB.

---

## 6. Open items / handoff for the next session

**In flight**
* `estimator_variants.slurm` (parabolic / phase-slope / GCC-PHAT / **signed**)
  is being run to decide the estimator. Goal: find one with **low bias *and*
  low std**. Prior knowledge: **phase-slope is unbiased but high-variance**.

**Decisions pending (after `variants_*` tables)**
1. If a variant (likely **signed**: coherent `corr_x+corr_y` + parabola) has low
   bias and low std → adopt it, rerun the equispaced grid, and the high-SNR
   floor drops.
2. If only phase-slope is unbiased → either apply a **multiplicative bias
   correction** (~1/(1−0.0046) calibrated per bandwidth) or present the
   **bias↔variance trade-off as a limitation** (already agreed acceptable).
3. Make **robust the headline combiner** for the equispaced study, and add a
   **bias-vs-bandwidth** panel.

**Not to touch yet**
* `paper/` narrative is frozen; update only after the estimator decision.

**Recompute needed to refresh figures with the star bound / RMSE panel**
```bash
sbatch estimator_comparison.slurm results_crb_equispaced
sbatch estimator_comparison.slurm results_crb_full_pairs   # optional
```

---

## 7. Figure catalogue (`results_<run>/figs_estimators/`)

| fig | shows |
|---|---|
| F1 `rmse_vs_snr` | RMSE vs SNR, 3 bandwidths (RMSE, i.e. bias included) |
| F2 `improvement_maps` | improvement over robust (dB) + high-SNR diagnostic |
| F3 `isoprecision` | SNR saving vs bandwidth, per channel count |
| F4 `db_gain_vs_channels` | SNR for RMSE = 5 km vs channel count (design curve) |
| F5 `error_distributions` | ECDF of \|error\| (1 % / 5 % tolerance lines) |
| F6 `gap_to_crb` | **RMSE/CRB (solid) + std/CRB (dotted)** and \|bias\| |
| F7 `pool_effect` | all-pairs vs filtered pool (selection ≡ weighting) |
| F8 `alpha_sweep` | weighting benefit vs filter threshold α (equispaced) |

Rebuild without recompute: `python estimator_comparison.py <run_dir> --replot`.

---

## 8. Notes / gotchas

* **HPC monitoring:** jobs survive SSH drops; re-login and `tail -f
  crb_sweep_<jobid>.log` / `crb_figs_<jobid>.log` / `crb_variants_<jobid>.log`.
  Use the **literal** job id (`${JOBID}` only if set in the same shell). Never
  run heavy Python on the login node — submit via SLURM.
* Noise seeds are common across bandwidths/SNRs (correlated stripes at low SNR).
* Library uses `from __future__ import annotations` for Python 3.8.
* `run_crb_hpc.sh` and `estimator_comparison.slurm` take the config / run dir as
  `$1`; `estimator_variants.slurm` takes config + count.
