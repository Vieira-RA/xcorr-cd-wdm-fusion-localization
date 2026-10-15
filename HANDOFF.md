# HANDOFF — CRB transient localisation study

**Read this first.** It is the entry point for a fresh session. Everything is
committed/pushed on `main`.

---

## 1. What this project is
Localise a transient mechanical perturbation along a WDM fibre by estimating the
**inter-channel delay caused by chromatic dispersion** from the **2-D SOP**
signal, and study how close a practical estimator gets to the **Cramér–Rao
bound (CRB)**. Narrative: *"how to reach the CRB in practice"* via three levers —
**pair selection**, **robust fusion**, **delay-estimator choice**.

## 2. Status (as of this handoff)
- **Done:** CRB derived + validated (per-pair std/theory ≈ 1.1); band-edge and
  equispaced production runs; parabolic **and** GCC-PHAT estimator runs; full
  estimator-variant test; figures F1–F8; full paper draft.
- **Key results:**
  1. **Pair selection ≡ weighting**: over all pairs, equal averaging is
     `10–10^3×` above the CRB; on a filtered pool (`D ≥ α·max D`) equal / `D²` /
     robust **coincide** on the CRB.
  2. **Robust (bisquare) fusion is best on the equispaced grid** (RMSE wins
     >90 % of cells; ~10–25 % below plain averaging).
  3. **Estimator bias/variance trade-off**: parabolic is variance-optimal but
     **biased** (−2.6 km @22 kHz), dominating at high SNR; **GCC-PHAT is
     unbiased** but ~2–3× higher variance → wins at high SNR / high bandwidth,
     parabolic wins at low SNR / low bandwidth.
  4. Scaling: `σ_Z ∝ 1/(√SNR·B)`, monotone in channel count, `1/√N_fib`.

## 3. Where things live
| Path | What |
|---|---|
| `scripts/sim_Transient_Localization/CRB_STUDY_README.md` | **Full study doc**: definitions, files, findings, figure catalogue, notes. |
| `paper/CRB_transient_localization_optica.tex` | **Full paper draft** (Optica style, 7 figs, equation appendix). |
| `paper/figures/` | Paper figures (concept is TikZ in the `.tex`). |
| `results_crb_full_pairs/` | Band-edge run (save_pairs all). |
| `results_crb_equispaced/` | Equispaced run, **parabolic** estimator. |
| `results_crb_equispaced_gcc/` | Equispaced run, **GCC-PHAT** estimator. |

Each `results_*` holds `crb_results_<n>ch.npz`, `estcomp_<n>ch.npz`,
`summary_*.npz`, and `figs_estimators/F1..F8`.

## 4. Key decisions (and why)
- **CRB overlay = STAR bound**, not GLS. The GLS per-channel-variance fit
  degrades on wide-`D` grids (it exceeded the star bound → impossible). Code
  stores `crb_star`.
- **Estimator:** keep **both** parabolic and GCC-PHAT; the choice is
  regime-dependent. `analysis.estimator: "parabolic" | "gcc_phat"` in the config.
- **Narrative:** "how to reach the CRB" (selection, robust fusion, estimator
  bias), not "weighting wins".

## 5. Immediate next steps (prioritised)
1. **Compile the paper** (`pdflatex` twice) — no LaTeX on the dev machine, so
   build elsewhere; check the TikZ concept figure and float placement.
2. **Trim for submission**: abstract → ~35 words; move the equation appendix to
   supplementary; choose the final 3-figure set (concept + Fig. 2 overlay + one
   of F7/F6).
3. **Fill authors/affiliations**; verify the 6 references.
4. *(Optional)* Prototype a **bias-corrected parabolic** estimator (low variance
   + no bias) — would beat both.
5. *(Optional)* Experimental validation.

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
```

## 7. Gotchas (learned the hard way)
- **Never** run heavy Python on the login node — submit via SLURM.
- `tail -f crb_<...>_<jobid>.log` with the **literal** job id (`${JOBID}` only if
  set in the same shell). Jobs survive SSH drops.
- `git pull` on the HPC can abort on **untracked** files that exist upstream
  (`mv` them aside, then pull).
- `--replot` needs `estcomp_*.npz`; the CRB overlay needs `crb_star` (recompute
  once if absent).
- The library needs `from __future__ import annotations` for Python 3.8.
- Noise seeds are common across bandwidths/SNRs (correlated stripes at low SNR).

## 8. Script map (`scripts/sim_Transient_Localization/`)
`crb.py` (bounds) · `channel_selection.py` · `generate_jones_crb.py` (batched;
full + custom grids) · `process_crb_sweep.py` (checkpointed; `analysis.estimator`)
· `analyse_crb_sweep.py` · `estimator_comparison.py` (F1–F8; `--replot`) ·
`estimator_compare_fig.py` (overlay) · `estimator_variants.py` ·
`prototype_peak_bias.py` · `validate_crb.py` / `validate_gls.py` /
`compare_filters.py` · SLURM drivers `run_crb_hpc.sh`,
`estimator_comparison.slurm`, `estimator_variants.slurm`.
