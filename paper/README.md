# Conference paper drafts

Two versions:

| file | template | notes |
|---|---|---|
| `CRB_transient_localization_optica.tex` | **Optica** (`opticameet3.sty`, OFC style) | **primary** — includes an equation-reference appendix page |
| `crb_transient_localization_conf.tex` | generic `article` | earlier draft, kept for reference |

## Build (Optica version)

The Optica style file `opticameet3.sty` must be in this folder (download it
with the OPTICAmeetings template from the conference website). Then:

```bash
pdflatex CRB_transient_localization_optica.tex
pdflatex CRB_transient_localization_optica.tex   # twice for references
```

The Optica template asks for a **35-word abstract** and **≤ 3 figures** — the
draft complies. The final **“Equation reference”** page is a working reference
and is **not intended for submission** (trim it, or move it to supplementary,
for the camera-ready).

## Figures

`figures/` currently holds **placeholders from the 5-fibre mid run**. Replace
with the full-run versions before submission:

```bash
# from scripts/sim_Transient_Localization/results_crb_full_pairs/figs_estimators/
cp F1_rmse_vs_snr.png          ../../../paper/figures/
cp F3_isoprecision.png         ../../../paper/figures/
cp F4_db_gain_vs_channels.png  ../../../paper/figures/
```

| figure | source |
|---|---|
| Fig. 1 (concept) | drawn in TikZ (self-contained) |
| Fig. 2 (RMSE vs SNR) | `F1_rmse_vs_snr.png` |
| Fig. 3 (scaling) | `F3_isoprecision.png` + `F4_db_gain_vs_channels.png` |

`F2_improvement_maps.png` and the error-distribution figures are natural
**supplementary** material.

## Before submitting
- Fill in authors / affiliations / e-mail.
- Verify every reference (they are plausible placeholders).
- Refresh the quoted numbers (RMSE/CRB ≈ 1.0–1.1, improvement factors, channel
  dB gain) once the full pairs run lands.
