# Conference paper drafts

Two versions:

| file | template | notes |
|---|---|---|
| `CRB_transient_localization_optica.tex` | **Optica** (`opticameet3.sty`, OFC style) | **primary detailed draft** — includes derivations, model assumptions, and interpretation notes |
| `crb_transient_localization_conf.tex` | generic `article` | earlier draft, kept for reference |

## Build (Optica version)

The Optica style file `opticameet3.sty` must be in this folder (download it
with the OPTICAmeetings template from the conference website). Then:

```bash
pdflatex CRB_transient_localization_optica.tex
pdflatex CRB_transient_localization_optica.tex   # twice for references
```

The current primary file is intentionally a long technical manuscript for
understanding the model, equations, assumptions, and saved-data results. Its
appendices contain derivations and implementation conventions, not venue-trimmed
submission text. Condense it only after the scientific narrative is settled.
No page or figure-count limit is being applied to this working version.

## Figures

`figures/` contains the current full-run figures used by the primary manuscript.
Several unreferenced PNGs are retained as older alternatives; the TeX file is
the source of truth for which files are included.

```bash
# from scripts/sim_Transient_Localization/
python estimator_comparison.py results_crb_equispaced --replot
python estimator_comparison.py results_crb_equispaced_gcc --replot
python estimator_comparison.py results_crb_full_pairs --replot
python estimator_compare_fig.py results_crb_equispaced results_crb_equispaced_gcc 15
python empirical_covariance_gls.py results_crb_equispaced 15 --bandwidth 22000
python plot_fusion_narrative.py
```

| figure | source |
|---|---|
| Concept | drawn in TikZ (self-contained) |
| Estimator comparison | `F1_estimators_compare.png` (parabolic and GCC-PHAT) |
| Fusion / selection | `F7_pool_effect.png`, `F8_alpha_sweep.png`, `F10_equispaced_fusion_strategy.png` |
| Pair-weight mechanism | `F11_pair_weight_distribution.png` |
| Bias and design | `F6_bias_variance.png`, `F4_db_gain_vs_channels.png`, `F5_error_distributions.png` |
| Held-out covariance benchmark | `F9_empirical_covariance_gls.png` |
| Band-edge selection | `bandedge_pair_filtering.png` |

The CRB drawn in older figures has been removed from the current fusion figures:
the star sum treated correlated star edges as independent. The current paper
uses the per-pair CRB and labels the held-out covariance-GLS result as an
empirical benchmark, not a CRB.
`F6_gap_to_crb_legacy.png` is retained only for audit and is not referenced by
the manuscript.
F10 and F11 are built from saved estimator metrics and pair-dispersion metadata;
no Jones/noise simulation rerun is needed. The equispaced grid is the main
soft-weighting comparison, with the alternating band-edge result as contrast.

## Three-page summary for review

The current concise review copy is separate from the detailed manuscript:

| file | purpose |
|---|---|
| `CRB_transient_localization_summary_revised.tex` | Current three-page summary; includes the approved title, authors, 35-word abstract, and expanded context/method references. |
| `CRB_transient_localization_summary_revised.pdf` | Compiled review copy. |
| `CRB_transient_localization_summary.tex` / `.pdf` | Earlier version, retained unchanged. |

The revised summary uses vector figures `F10_equispaced_fusion_strategy_vector.pdf`,
`F11_pair_weight_distribution_vector.pdf`, and `bandedge_target_snr_vector.pdf`.
The last figure highlights the 29.8-dB SNR reduction for a 5-km RMSE target on
the 12-channel band-edge grid. Its estimator comparison remains in the text.
Rebuild from this folder by opening the revised `.tex` in TeXstudio and running
PDFLaTeX twice so citations and figure references resolve. The full technical
manuscript above was not changed in this summary-only update.

## Before submitting
- The revised summary has author and affiliation details; the detailed manuscript may still have placeholders.
- Verify reference metadata and the Corning product-sheet URLs.
- Confirm the author order and that the generic three-page layout matches the chosen venue.
- Keep SNR labels on the per-component energy-SNR convention used by the CRB.
- Treat `B` as the nominal Gaussian-envelope parameter; the Fourier-power
  spectrum's full 3-dB width is `0.707 B`.
- Refresh quoted figures/numbers from the current saved arrays before submission.
