# CRB-constrained CD localisation study

A self-contained study of the **Cramér–Rao bound (CRB)** for transient
chromatic-dispersion (CD) localisation over WDM links, using the **2-D SOP**
signal with parabolic interpolation after rotating the centroid to the north
pole.

It reproduces exactly the definitions in
`Cramér–Rao Bound (CRB) for Time-Delay Estimation — Simple Summary.md`:
energy SNR `E/N0`, mean-square bandwidth `beta^2`, and the two-sensor
**effective SNR**.

---

## 1. Definitions implemented (`crb.py`)

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

Notes:
* `P_coh(f) = | sum_c X_i,c(f) conj(X_j,c(f)) |` is the **coherent
  cross-spectrum** used as the common signal.
* Each component contributes its own Fisher information (summed).
* Validated against Monte-Carlo: the parabolic correlator saturates the
  effective bound at high SNR (real pair `MC/CRB_eff ~ 1.07`).

---

## 2. Pipeline / files

| File | Role |
|---|---|
| `crb.py` | CRB definitions (pure functions). |
| `channel_selection.py` | WDM subset schemes: alternating / uniform / extremes / first_n. |
| `generate_jones_crb.py` | Jones generator — **batched** closed-form SU(2) tree-product (`propagate_channels`), ~30× faster than the per-sample loop. Supports full ITU grid **and custom uniformly-spaced** grids. |
| `process_crb_sweep.py` | 2-D SOP, energy-SNR noise, parabolic estimator, stores per-realisation estimates (all-pairs **and** pool variants) + CRB arrays (+ per-pair distances). **Checkpointed per fibre.** `CRB_N_JOBS` env overrides `run.n_jobs`. |
| `analyse_crb_sweep.py` | std/bias heatmaps with GLS CRB overlay, std-vs-SNR, estimator comparison, multi-fibre and cross-count plots. |
| `estimator_comparison.py` | SNR-free fusion study → publication figures **F1–F8**, plus `estcomp_<n>ch.npz`. Supports `--replot` (rebuild figures from `estcomp` in seconds, no recompute). |
| `validate_crb.py` | Toy + identity validation of the bounds. |
| `validate_gls.py` | Naive vs GLS vs star combination validation (needs per-pair data). |
| `compare_filters.py` | no-filter vs low-pass vs band-pass comparison. |
| `run_crb_hpc.sh` | SLURM driver (config as `$1`): generate → process → analyse. |
| `estimator_comparison.slurm` | SLURM driver for the figure job (`$1` = run dir). |

Configs:
* `sweep_config_crb_test.yaml` (+ `_test_lowpass`, `_test_bandpass`) — small tests.
* `sweep_config_crb_mid.yaml` — 5 fibres, 30×30, counts {4,8,12}.
* `sweep_config_crb.yaml` — **production** (full ITU grid, 40×40, counts {4,6,8,10,12}, `save_pairs: none`).
* `sweep_config_crb_pairs.yaml` — production grids, `save_pairs: all` (estimator study).
* `sweep_config_crb_equispaced.yaml` — **current** study: custom 30-channel equispaced grid, uniform selection, counts {5,10,15}, 10 fibres, 100 real, 50×50, `save_pairs: all`.

---

## 3. How to run

```bash
python validate_crb.py                                  # bound validation
python generate_jones_crb.py  <config>                  # stage 1 (skip if Jones exists)
python process_crb_sweep.py   <config>                  # stage 2 (checkpointed)
python analyse_crb_sweep.py   <config>                  # stage 3 (heatmaps)
python estimator_comparison.py <run_dir> [--replot]     # F1–F8
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

`estimator_comparison.py` writes `estcomp_<n>ch.npz`: per-realisation estimates
for the all-pairs and pool estimators, RMSE/bias/std/MAE/success surfaces, the
GLS bound, and the `alpha_saving` sweep — so figures rebuild without recompute.

---

## 5. Findings

* **CRB attained.** The parabolic correlator saturates the effective bound at
  high SNR (`MC/CRB_eff ~ 1.07`) in all-pairs fusion.
* **Naive all-pairs fusion fails.** Over *all* channel pairs, equal-weight
  averaging sits `10`–`10^3`× above the CRB: since `sigma_Z ∝ 1/D`, pairs with
  small dispersion separation `D` carry almost no information yet dominate an
  equal-weight mean.
* **Pair selection ≡ weighting.** On a *filtered* pool (`D ≥ 0.25 max D`) the
  retained pairs have near-equal precision, so **equal averaging, `D^2`-weighting
  and robust coincide on the CRB**; the residual benefit of weighting is ≈0 dB
  (Fig. F7). Weighting is the **soft-filter equivalent** of an explicit
  threshold. → *the essential step is pair selection, not the combining rule.*
* **Governing quantity.** The weighted-vs-naive gap is set by the **spread of
  the per-pair variances** in whatever pool is fed to the combiner.
* **Multi-fibre** averaging follows `1/sqrt(N_fibres)`, rule-independent.
* **Filtering** (linear) does not move the CRB (S/N invariant); low-pass ≈ none,
  band-pass worse (discards low-frequency transient energy).
* Naive all-pairs inverse-variance over-counts pair information by
  `1.5× (4ch) / 2.0× (8ch) / 2.6× (12ch)`; star/GLS are tight (`1.0–1.1`).

---

## 6. Open hypothesis — equally-spaced-grid study (current)

**Motivation.** The "weighting is irrelevant once filtered" result was obtained
with the **extremes-first** selection, which makes small channel sets
**bimodal**: a few cross-band pairs (`D ≈ 2000 ps/km`) and a few adjacent
edge pairs (`D ≈ 10 ps/km`). Filtering (`D ≥ 0.25 max D`) then keeps only the
near-full-band pairs, whose `D` are almost equal — hence weighting adds nothing.

**Hypothesis.** On an **equally-spaced** grid the pair separations span a broad
range (`1Δ, 2Δ, …, (N−1)Δ`), so:
1. **All-pairs fusion:** weighting should beat naive averaging by a large,
   channel-count-dependent margin (the "weighting matters" result);
2. **Filtered pool:** a *residual* weighting gain should remain (≈ few dB),
   because the retained separations still span a factor ~4, unlike the
   band-edge case where the spread collapses to ~0.

If confirmed, the equispaced grid becomes the **definitive** result. If not,
the "selection ≡ weighting" conclusion is robust across channel-set geometries.

**Design (`sweep_config_crb_equispaced.yaml`).** Custom **30-channel equispaced**
master over 184–196 THz; **uniform** selection ⇒ equispaced subsets of
**{5, 10, 15}** channels; 10 fibres; 100 realisations; 50×50 (SNR × bandwidth);
`save_pairs: all`.

**New figure F8 — α-sweep.** Weighting benefit vs filter threshold
`α` (`D ≥ α·max D`; `α = 0` ⇒ all pairs), per channel count. Expectation: on the
equispaced grid F8 does **not** collapse to ~0 at `α = 0.1–0.3`, and the gain is
larger for more channels.

---

## 7. HPC runs

Production / pairs / equispaced (SLURM; conda env `phd_env`, repo in `~/PhD/…`):

```bash
# simulate (generate -> process -> analyse); RUN_GENERATOR=yes for new Jones
sbatch run_crb_hpc.sh sweep_config_crb_equispaced.yaml

# figure job (needs pair_Z); chain it after the sim job:
sbatch --dependency=afterok:<simjob> estimator_comparison.slurm results_crb_equispaced
```

Stage 2 is **checkpointed per fibre** (resubmit to resume). `CRB_N_JOBS` (default
`$SLURM_CPUS_PER_TASK`) sets the joblib worker count.

---

## 8. Figure catalogue (`results_<run>/figs_estimators/`)

| fig | shows | purpose |
|---|---|---|
| **F1** `rmse_vs_snr` | RMSE vs SNR, 3 bandwidths (12 ch) | estimator comparison; pool curves coincide on CRB |
| **F2** `improvement_maps` | improvement over robust (dB) map + high-SNR diagnostic | where/how much; shows the mid-band dip is a bandwidth effect |
| **F3** `isoprecision` | SNR saving vs bandwidth, per channel count | value of weighting (all-pairs baseline) |
| **F4** `db_gain_vs_channels` | SNR for RMSE = 5 km vs channel count | design curve (lower = better) |
| **F5** `error_distributions` | ECDF of \|error\| | tail behaviour; success at 1 %/5 % tolerance |
| **F6** `gap_to_crb` | std/CRB and \|bias\| vs SNR | variance efficiency and bias floor |
| **F7** `pool_effect` | all-pairs vs filtered pool (RMSE; residual saving) | **the "selection vs weighting" result** |
| **F8** `alpha_sweep` | weighting benefit vs filter threshold α | **current study** — how much of the gain a hard filter recovers |

Regenerate from saved data in seconds (needs `estcomp_*.npz`):
```bash
python estimator_comparison.py <run_dir> --replot
```

---

## 9. Paper

`paper/` holds the conference draft (Optica/OFC style, `opticameet3.sty`).

| file | role |
|---|---|
| `paper/CRB_transient_localization_optica.tex` | primary draft; title/abstract/intro/results/conclusion reframed to **"pair selection reaches the CRB"**; Fig. 1 (TikZ concept), Fig. 2 = F1, Fig. 3 = F7; equation-reference appendix |
| `paper/crb_transient_localization_conf.tex` | earlier generic-`article` draft (reference) |
| `paper/README.md` | build steps and figure provenance |

Narrative: **uninformative (small-`D`) pairs destroy naive fusion; selecting
well-separated pairs attains the CRB, and weighting is the soft-filter
equivalent.** Design rules (SNR↔bandwidth, channel scaling, `1/√N` fibres) given
in the text.

---

## 10. Notes

* Noise seeds are **common across bandwidths and SNRs** (seed depends on fibre
  and realisation), which smooths heatmaps but produces correlated vertical
  stripes at low SNR.
* Library uses `from __future__ import annotations` for Python 3.8 compatibility.
* The **equispaced study may add the all-pairs weighting result** to the paper
  if F8 confirms the hypothesis (currently the paper leads with the
  selection-is-key result).
