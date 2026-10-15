# Paper 1: Disperse, cross-correlate and fuse. Paper 1: quick transient localization.
Simulation, experimental data and analysis scripts for the first journal paper.

## Contents

- `scripts/sim_Transient_Localization/` — transient CD localisation over WDM.
  The **CRB-constrained study** (definitions, batched Jones generator,
  checkpointed Monte-Carlo sweep, SNR-free fusion estimators, analysis) is
  documented in
  **[`scripts/sim_Transient_Localization/CRB_STUDY_README.md`](scripts/sim_Transient_Localization/CRB_STUDY_README.md)**.
  **Current focus:** the equispaced-grid study
  (`sweep_config_crb_equispaced.yaml`, `results_crb_equispaced`) — findings and
  a next-session handoff are in §5–§6 of that README.
- `paper/` — conference draft (Optica/OFC style); see
  [`paper/README.md`](paper/README.md). *Narrative frozen pending the estimator
  decision.*

The simulation code depends on the shared library
[`sop-simulation-lib`](https://github.com/Vieira-RA/sop-simulation-lib)
(`pip install -e .` or `PYTHONPATH` to its `src`).
