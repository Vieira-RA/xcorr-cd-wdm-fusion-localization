# HANDOFF — CRB transient localisation study

**Read this first.** It is the entry point for a fresh session; repository
changes and generated outputs may be uncommitted.

---

## 1. What this project is
Localise a transient event along a WDM fibre from its **2-D SOP** signature and
the **inter-channel delay caused by chromatic dispersion**. The event source may
be mechanical, lightning-associated, or otherwise, provided it creates a
detectable shared SOP signature. The current simulation models one generic
localized perturbation, not event-specific lightning physics. The CRB is a
validated **per-pair** reference; multichannel fusion is compared using
Monte-Carlo RMSE, bias, and variance plus a held-out empirical covariance-GLS
benchmark. Main levers: **pair selection**, **robust fusion**, and
**delay-estimator choice**.

## 2. Status (as of this handoff)
- **Done:** two-sensor effective-SNR per-pair CRB derived and checked; band-edge and
  equispaced production runs; parabolic **and** GCC-PHAT estimator runs; full
  estimator-variant test; figures F1–F11 plus covariance diagnostics; full paper draft.
- **Done:** a revised, compiled three-page summary is ready for review. It has the
  final summary title and 35-word abstract, author/affiliation details, the
  application-specific references, and three vector-PDF figures. The earlier
  summary files are preserved separately.
- **Key results:**
  1. Pair filtering strongly improves equal averaging on the band-edge grid;
     the amount is grid-dependent. Threshold `alpha` is defined on `D²`:
     `D² >= alpha * max(D²)`.
  2. On the equispaced grid, all-pair `D²` weighting is close to filtered equal
     averaging; robust all-pair fusion gives modest additional gains. The
     band-edge pair-weight distribution is strongly separated, so `D²`
     weighting nearly acts like a hard cutoff there.
  3. Robust (bisquare) fusion wins in most, but not every, equispaced-grid cell.
  4. Parabolic interpolation has lower variance in the tested regimes but a
     low-bandwidth bias; GCC-PHAT reduces that bias and can win at high energy
     SNR/high bandwidth, but is not unbiased throughout the sweep.
  5. Scaling: `σ_Z ∝ 1/(√SNR_E·B_nom)`, monotone in channel count,
     `1/√N_fib` in the tested regime.
  6. On the 12-channel alternating band-edge grid at `B_nom=22.51 kHz`, the
     5-km RMSE target requires 119.84 dB for equal all-pair averaging, 90.02 dB
     for all-pair `D²` or robust fusion, and 90.01 dB for filtered equal
     averaging: a 29.8-dB reduction for dispersion-aware fusion versus equal
     all-pair averaging.

## 3. Where things live
| Path | What |
|---|---|
| `scripts/sim_Transient_Localization/CRB_STUDY_README.md` | **Full study doc**: definitions, files, findings, figure catalogue, notes. |
| `paper/CRB_transient_localization_optica.tex` | **Full paper draft** (per-pair CRB, empirical fusion; figures and equation appendix). |
| `paper/CRB_transient_localization_summary_revised.tex` | **Current concise summary** (three pages; current title/authors/citations). |
| `paper/CRB_transient_localization_summary_revised.pdf` | Compiled summary for review. |
| `paper/CRB_transient_localization_summary.tex` and `.pdf` | Earlier summary version, preserved; not the current review copy. |
| `paper/figures/` | Paper figures (concept is TikZ in the `.tex`). |
| `paper/figures/F10_equispaced_fusion_strategy_vector.pdf`, `F11_pair_weight_distribution_vector.pdf`, `bandedge_target_snr_vector.pdf` | Vector figures used by the revised three-page summary. |
| `results_crb_full_pairs/` | Band-edge run (save_pairs all). |
| `results_crb_equispaced/` | Equispaced run, **parabolic** estimator. |
| `results_crb_equispaced_gcc/` | Equispaced run, **GCC-PHAT** estimator. |

Each `results_*` holds `crb_results_<n>ch.npz`, `estcomp_<n>ch.npz`,
`summary_*.npz`, and figures in `figs_estimators/`.

## 4. Key decisions (and why)
- The per-pair effective-SNR CRB is the theoretical reference. The independent-
  edge star sum is not a combined CRB because star edges share reference-channel
  noise. The additive channel-timing GLS is conditional and not valid in every
  sweep cell; use the held-out empirical-covariance GLS benchmark for fusion
  comparisons and do not label it a CRB.
- **Estimator:** keep **both** parabolic and GCC-PHAT; the choice is
  regime-dependent. `analysis.estimator: "parabolic" | "gcc_phat"` in the config.
- **SNR:** saved legacy production axes are converted by `10 log10(N/2)` to mean
  per-component energy SNR (`N=1200`, offset 27.7815 dB). Configs now set noise
  using the per-component energy convention; no simulation rerun is needed for
  the saved results.
- **Bandwidth:** `B_nom` is a nominal Gaussian-envelope parameter; the input
  envelope's full power 3-dB width is `0.7071 B_nom`.
- **Event model:** the PMD model covers the 0.5-km disturbed section centred
  500 km from the receiver; the remaining path is represented by deterministic
  CD delays. This is the intended simplification, not a simulated 500-km PMD
  cascade.
- **Three-page summary:** title is *Broadband Transient Localization over WDM
  Fibre: Empirical Fusion Guided by Pairwise Bounds*. It uses a 35-word abstract
  whose headline is the 29.8-dB band-edge SNR reduction. The reduction is
  attributed to **dispersion-aware fusion**, not to robust fusion alone: `D²`,
  robust all-pair, and filtered-equal methods all reach about 90 dB in that
  setting. The estimator crossover remains in text; Fig. 3 now directly shows
  the headline target-SNR comparison.
- **Summary authors:** Rafael Vieira, Pedro Freire, and Sergei Turitsyn; Aston
  University, United Kingdom; `randr24@aston.ac.uk`.
- **Approved summary abstract (35 words):** “Transient localization from
  wavelength-dependent polarization signatures in WDM fibres. Chromatic
  dispersion maps distance to inter-channel delay. At 5-km RMSE, dispersion-aware
  fusion cuts SNR by 30 dB versus equal all-pair averaging on a 12-channel
  band-edge layout.”
- **Summary-only citation update:** Mecozzi et al. (2021, 2025) support the SOP
  sensing context; Vaskinn et al. (2025) and Tang et al. (2024) support related
  inter-channel/CD sensing; Gordon--Kogelnik and Shtaif--Mecozzi support PMD
  modeling; Corning PI1445/PI1440 support the fibre parameters. The broad He--Liu
  DAS review was removed from the revised summary. The detailed manuscript was
  not updated in this summary pass and still has author placeholders.

## 5. Next steps (the current summary task is complete)
1. Review `paper/CRB_transient_localization_summary_revised.pdf` with the
   coauthors; confirm the final title, author order, affiliation, email, and
   application framing.
2. Choose a target venue and adapt the summary to its actual template/limits.
   The revised summary is a generic three-page article, not yet an Optica-format
   submission.
3. If the long technical manuscript is to be submitted, carry over the approved
   title, authors, citations, terminology, and updated figures deliberately;
   it was not modified as part of the summary-only work.
4. Optional: paired uncertainty intervals for small equispaced fusion gains,
   using saved `pair_Z` data (no simulation rerun).
5. *(Longer term)* Experimental validation and/or a bias-corrected parabolic
   estimator.

## 6. How to run (HPC: SLURM, conda `phd_env`, library `sop-simulation-lib`)
```bash
# simulate (generate -> process -> analyse). RUN_GENERATOR=yes for new Jones.
sbatch run_crb_hpc.sh <config>
# figures (needs pair_Z); chain after the sim job:
sbatch --dependency=afterok:<simjob> estimator_comparison.slurm <run_dir>
# estimator test:
sbatch estimator_variants.slurm <config> <count>
```
Local, no recompute (needs `estcomp_*.npz`):
```bash
python estimator_comparison.py <run_dir> --replot
python estimator_compare_fig.py results_crb_equispaced results_crb_equispaced_gcc 15
python empirical_covariance_gls.py results_crb_equispaced 15 --bandwidth 22000
python plot_fusion_narrative.py
```
The current summary PDF is already compiled. To rebuild it, open
`paper/CRB_transient_localization_summary_revised.tex` in TeXstudio and run
PDFLaTeX twice for citations; its vector figure PDFs are in `paper/figures/`.

## 7. Gotchas (learned the hard way)
- **Never** run heavy Python on the login node — submit via SLURM.
- `tail -f crb_<...>_<jobid>.log` with the **literal** job id (`${JOBID}` only if
  set in the same shell). Jobs survive SSH drops.
- `git pull` on the HPC can abort on **untracked** files that exist upstream
  (`mv` them aside, then pull).
- `--replot` needs `estcomp_*.npz` and matching `crb_results_*.npz` metadata for
  SNR-axis conversion. Legacy independent-edge star overlays are not combined
  CRBs; fusion figures report empirical RMSE/bias/std.
- The library needs `from __future__ import annotations` for Python 3.8.
- Noise seeds are common across bandwidths/SNRs (correlated stripes at low SNR).

## 8. Script map (`scripts/sim_Transient_Localization/`)
`crb.py` (per-pair bound and conditional covariance model) · `channel_selection.py` · `generate_jones_crb.py` (batched;
full + custom grids) · `process_crb_sweep.py` (checkpointed; `analysis.estimator`)
· `analyse_crb_sweep.py` · `estimator_comparison.py` (F1–F8; `--replot`) ·
`plot_fusion_narrative.py` (paper F10/F11) ·
`estimator_compare_fig.py` (overlay) · `estimator_variants.py` ·
`prototype_peak_bias.py` · `validate_crb.py` / `validate_gls.py` /
`validate_joint_crb.py` / `empirical_covariance_gls.py` /
`compare_filters.py` · SLURM drivers `run_crb_hpc.sh`,
`estimator_comparison.slurm`, `estimator_variants.slurm`.
