# Conference paper draft (3 pages)

`crb_transient_localization_conf.tex` — proof-of-concept draft:
*Reaching the Cramér–Rao Bound in Multi-Channel Transient Localisation over WDM
Links: Why Dispersion-Weighted Fusion Wins.*

## Build

```bash
pdflatex crb_transient_localization_conf.tex
pdflatex crb_transient_localization_conf.tex   # twice for references
```
Requires a standard TeX distribution with `tikz` (shapes.geometric,
arrows.meta, positioning).

## Figures

`figures/` currently holds **placeholders copied from the 5-fibre mid run**.
Replace them with the full-run versions before submission:

```bash
# from scripts/sim_Transient_Localization/results_crb_full_pairs/figs_estimators/
cp F1_rmse_vs_snr.png        ../../../paper/figures/
cp F3_isoprecision.png       ../../../paper/figures/
cp F4_db_gain_vs_channels.png ../../../paper/figures/
```

| figure in draft | source file |
|---|---|
| Fig. 1 (concept) | drawn in TikZ (self-contained) |
| Fig. 2 (money plot) | `F1_rmse_vs_snr.png` |
| Fig. 3 (scaling) | `F3_isoprecision.png` + `F4_db_gain_vs_channels.png` |

The improvement maps (`F2_improvement_maps.png`) and error distributions
(`F5_*`) are natural **supplementary** figures for a longer/journal version.

## Before submitting
- Fill in **authors / affiliations**.
- Verify every bibliography entry (they are plausible placeholders).
- Numbers to refresh once the full pairs run is in: the RMSE/CRB ratio
  (`\approx 1.0`–`1.1`), the improvement factors, and the channel-count dB gain.
