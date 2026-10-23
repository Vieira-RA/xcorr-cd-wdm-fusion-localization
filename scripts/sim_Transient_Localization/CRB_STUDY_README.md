# Per-pair CRB and practical CD localisation study

A self-contained study of the **Cramér–Rao bound (CRB)** for transient
chromatic-dispersion (CD) localisation over WDM links, using the **2-D SOP**
signal with parabolic interpolation after rotating the centroid to the north
pole.

The per-pair delay bound uses mean per-component energy SNR `E/N0`, mean-square
bandwidth `beta^2`, and the two-sensor **effective SNR**. Pairwise fusion is
evaluated empirically because pair estimates reuse noisy channels and are
correlated; the independent-edge star sum is not a combined CRB.

> **Status:** production datasets exist for the **band-edge** run
> (`results_crb_full_pairs`) and the equispaced **parabolic** and **GCC-PHAT**
> runs (`results_crb_equispaced`, `results_crb_equispaced_gcc`). Both estimator
> runs are used in the current paper draft.

---

## 1. Definitions (`crb.py`)

| Quantity | Formula | Code |
|---|---|---|
| Energy SNR | `mean_c(sum_n x[n,c]^2) / sigma^2` (= per-component `E/N0`) | `per_component_snr` |
| Noise for target SNR | `sigma^2 = mean_c(sum_n x[n,c]^2) / SNR_E,lin` | `noise_variance_for_snr` |
| Mean-square bandwidth | two-sided `sum_k (2 pi f_k)^2 |X_k|^2 / sum_k |X_k|^2` | `mean_square_bandwidth` |
| Deterministic bound | `CRB = 1 / sum_c E_c beta_c^2 / sigma^2` | `crb_delay_deterministic(_pair)` |
| Effective SNR | `SNR_eff(f) = rho^2/(1+2 rho)`, `rho = P_coh/(N sigma^2)` | `effective_snr_spectrum` |
| Fisher information | full two-sided sum; rFFT weights interior bins by 2, DC/Nyquist by 1 | `fisher_information_delay` |
| Distance mapping | `sigma_Z = sqrt(CRB_tau) * 1e12 / |int D dlambda|` | `crb_distance` |
| Pair combination | equal, `D^2`, robust; held-out covariance GLS benchmark | `estimator_comparison.py`, `empirical_covariance_gls.py` |

* `P_coh(f) = | sum_c X_i,c(f) conj(X_j,c(f)) |` is the **coherent
  cross-spectrum** (common-signal estimate).
* `validate_crb.py` checks the effective-SNR pair bound on controlled
  Gaussian-pulse Monte Carlo. Production pair std/CRB ratios vary by pair and
  regime; report the tested cell and distribution rather than one global ratio.
* Pair errors share channel traces. The independent-edge star sum ignores this
  covariance and is retained only for reproducing old results, not as a CRB.
* The additive channel-timing model `Var(τ_ij)=u_i+u_j` is a diagnostic, not a
  universal covariance law. `validate_joint_crb.py` checks its fit and held-out
  behavior. `empirical_covariance_gls.py` supplies a split-sample GLS benchmark;
  it is explicitly empirical, not a CRB.
* Example diagnostic (15 ch, nominal 22 kHz, energy SNR 127.8 dB): the additive
  marginal-variance fit has median absolute relative error 13% (90th percentile
  36%), is usable for 7/10 fibre fits, and its held-out model-weighted std is
  1.35x the fitted reference. On the same split, empirical-covariance GLS has
  RMSE 2.65 km, std 0.089 km, and bias -2.65 km. This is why the empirical
  benchmark, not the fitted covariance reference, is used in the paper.

**SNR correction:** legacy saved sweeps used mean total two-component sample
power divided by per-component noise variance. For `N=1200`, conversion to mean
per-component energy SNR adds `10 log10(N/2)=27.7815 dB`. The plotting helpers
convert the old axis; new configs use the energy-SNR convention and reproduce
the same noise variances without regenerating Jones traces.

**Bandwidth convention:** `sigma_t=0.3748/B_nom` defines a nominal Gaussian
envelope parameter. `B_nom` is the Fourier-amplitude spectrum FWHM; the
Fourier-power spectrum's full 3-dB width is `0.7071 B_nom`. `B_nom` is used on
the figures and the CRB uses the measured SOP spectrum.

---

## 2. Pipeline / files

| File | Role |
|---|---|
| `crb.py` | CRB definitions (pure functions). |
| `channel_selection.py` | Subset schemes: alternating / uniform / extremes / first_n. |
| `generate_jones_crb.py` | Batched Jones generator (~30× faster). Supports **full ITU grid and custom uniformly-spaced grids**. |
| `process_crb_sweep.py` | 2-D SOP, mean per-component energy-SNR noise, delay estimators, pair CRBs and per-realisation estimates (optionally `pair_Z`). Checkpointed per fibre. |
| `analyse_crb_sweep.py` | std/bias heatmaps and an explicitly conditional additive-timing-model reference. |
| `estimator_comparison.py` | Empirical fusion study → figures **F1–F8** + `estcomp_<n>ch.npz`. `--replot` rebuilds from saved estimator metrics. |
| `estimator_variants.py` | **Controlled estimator test** (parabolic / phase-slope / GCC-PHAT / signed) — reports bias ± std per SNR. |
| `estimator_variants.slurm` | SLURM driver for the above (`$1`=config, `$2`=count). |
| `prototype_peak_bias.py` | Toy test of interpolation bias (symmetric Gaussian). |
| `validate_crb.py`, `validate_gls.py`, `validate_joint_crb.py`, `compare_filters.py` | Pair-bound and covariance-model diagnostics. |
| `empirical_covariance_gls.py` | Held-out empirical covariance-GLS fusion benchmark (not a CRB). |
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
python empirical_covariance_gls.py <run_dir> 15 --bandwidth 22000
```

HPC (SLURM, conda `phd_env`):
```bash
sbatch run_crb_hpc.sh <config>                         # simulate
sbatch --dependency=afterok:<sim> estimator_comparison.slurm <run_dir>   # figures
sbatch estimator_variants.slurm <config> <count>       # estimator test
```

---

## 4. Raw data (`crb_results_<n>ch.npz`)

Metadata (`BANDWIDTHS`, energy `SNRS_dB`, `snr_definition`, `n_samples`,
`channel_indices_*`, `wavelengths_nm`, `pair_indices`, `D_abs`,
`event_distance_km`, `filter_mode`, `cfg`), per-cell
estimates (`est_crb`, `est_gls`, `est_lambda`, `est_crb_iqr`), CRB arrays
(`crb_delay_pairs`, `crb_dist_pairs`, `crb_dist_combined_*`), `sigma2`,
`beta2`, per-channel mean 2-D power in `energy`, and (with `save_pairs: all`)
`pair_Z`. The old production files lack `snr_definition` and store the legacy
power-normalized axis; plotting helpers convert those axes to energy SNR.
The `crb_dist_combined_eff`/`*_det` fields assume independent pairs, while
`crb_dist_combined_gls` is conditional on the additive channel-timing model;
neither is treated as a universal combined CRB in the paper.

`estcomp_<n>ch.npz`: per-realisation estimates (all-pairs + **pool** +
`--replot` surfaces) and the `alpha_saving` sweep. Legacy `crb_star` fields are
not used as combined CRBs. `--replot` also reads the matching
`crb_results_<n>ch.npz` metadata to convert legacy SNR axes.

---

## 5. Findings

**Bounds and covariance**
* The two-sensor effective-SNR CRB is a per-pair reference; production
  variance ratios depend on channel pair, bandwidth, and SNR.
* Do not interpret the independent-edge star sum as an achievable combined
  bound. The fitted additive channel-timing model is not adequate everywhere;
  use it only as a diagnostic and use held-out empirical covariance GLS when a
  covariance-aware fusion benchmark is needed.

**Band-edge (extremes-first) grid — production, 12 ch**
* Equal averaging over all pairs is strongly affected by low-dispersion pairs.
  At target RMSE 5 km, filtering saves roughly 28–35 dB for 4–12 channels.
  After filtering, the residual gain from `D²` weighting is near zero in this
  grid. This is an empirical result, not combined-CRB attainment.

**Equispaced grid — current**
* **Robust (bisquare) usually gives the lowest RMSE.** At energy SNR 116.3 dB
  (log-interpolated from the saved grid), bandwidth-averaged RMSE is:

  | count | average | `D²` | **robust** |
  |---|---|---|---|---|
  | 5 ch | 10.29 | 9.66 | **9.27** |
  | 10 ch | 8.34 | 7.89 | **6.22** |
  | 15 ch | 6.52 | 6.17 | **5.40** |

  For 15 channels robust wins against `D²` in **2451/2500** cells and against
  equal averaging in **2500/2500** cells; it is not guaranteed to win every
  cell for all counts. The covariance-model reference is not included in this
  table because its fit is not uniformly valid.
* **Weighting benefit vs filter threshold (F8):** with the threshold defined on
  `D²`, at α = 0 (all pairs) ≈
  **2.9 dB (10 ch), 3.6 dB (5 ch), 4.8 dB (15 ch)** — nonzero, **grows with
  channel count**, but modest (the closest equispaced pairs are 414 GHz apart,
  `D ≈ 70 ps/km`, vs 8 ps/km for adjacent ITU channels). Filtering (α ≳ 0.3)
  removes most of it; the retained pairs satisfy `|D|/max(|D|) >= sqrt(alpha)`.
* **Bias is bandwidth-dependent**: parabolic bias is about −2.63 km at nominal
  22 kHz and −0.035 km at 2 MHz (15 channels, 100 dB on the legacy axis).
  GCC-PHAT reduces the 22-kHz bias to about −0.68 km and is nearly unbiased at
  2 MHz. These values are at 127.8 dB energy SNR per SOP component. The
  high-SNR floor at low nominal bandwidth is an interpolation effect
  (sub-sample interpolation on coarse `dt`).
* Multi-fibre averaging follows `1/sqrt(N_fibres)`.
* In the filter-comparison utility the dashed reference is computed from the
  unfiltered clean traces. A lossy filter can reduce information; the utility
  does not compute a filtered-observation CRB, so do not interpret that curve
  as the exact bound after filtering.

---

## 6. Handoff

**See [`../../HANDOFF.md`](../../HANDOFF.md) for the full handoff.** Summary:

* **Estimator decision:** parabolic and **GCC-PHAT** are
  **regime-dependent** — parabolic has lower variance in tested cases and often
  wins at low nominal bandwidth; GCC-PHAT reduces bias and can win at high
  energy SNR / high bandwidth. Phase-slope was near-unbiased but high-variance;
  "signed" gave no improvement. Both
  runs exist: `results_crb_equispaced` (parabolic) and
  `results_crb_equispaced_gcc` (GCC-PHAT).
* Robust fusion wins in most, but not every, equispaced-grid cell.
* The paper distinguishes the per-pair CRB from empirical multichannel fusion;
  the additive covariance-model GLS is not treated as a universal CRB.

**Next steps:** venue-specific editing and author details; compile only when a
LaTeX installation/template is available; optionally prototype a
bias-corrected parabolic estimator. Commands and gotchas: `HANDOFF.md`.

---

## 7. Figure catalogue (`results_<run>/figs_estimators/`)

| fig | shows |
|---|---|
| F1 `rmse_vs_snr` | RMSE vs SNR, 3 bandwidths (RMSE, i.e. bias included) |
| F2 `improvement_maps` | improvement over robust (dB) + high-SNR diagnostic |
| F3 `isoprecision` | SNR saving vs bandwidth, per channel count |
| F4 `db_gain_vs_channels` | SNR for RMSE = 5 km vs channel count (design curve) |
| F5 `error_distributions` | ECDF of \|error\| (1 % / 5 % tolerance lines) |
| F6 `bias_variance` | RMSE and std (km), plus absolute bias; GCC-PHAT run |
| F7 `pool_effect` | equispaced GCC-PHAT: empirical pair filtering and weighting gains |
| F8 `alpha_sweep` | weighting benefit vs the `D²`-fraction threshold α (parabolic) |
| F9 `empirical_covariance_gls` | held-out covariance-GLS benchmark (not a CRB) |
| F10 `equispaced_fusion_strategy` | best single pair vs all-pair and filtered fusion (parabolic) |
| F11 `pair_weight_distribution` | normalized dispersion and cumulative `D²` weight for equispaced/band-edge subsets |
| Band-edge pair filtering | `paper/figures/bandedge_pair_filtering.png` |

Rebuild without estimator recomputation (needs `estcomp_*ch.npz` and matching
`crb_results_*ch.npz` metadata): `python estimator_comparison.py <run_dir> --replot`.
Held-out covariance diagnostic: `python empirical_covariance_gls.py <run_dir> <count> --bandwidth <Hz>`.
Paper fusion figures, from saved metrics/dispersion arrays only:
`python plot_fusion_narrative.py` (no simulation rerun).

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
