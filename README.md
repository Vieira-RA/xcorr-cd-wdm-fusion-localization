# Paper 1: Disperse, cross-correlate and fuse. Paper 1: quick transient localization.
Simulation, experimental data and analysis scripts for the first journal paper.

## Contents

- `scripts/sim_Transient_Localization/` — transient CD localisation over WDM.
  The **CRB-constrained study** (definitions, batched Jones generator,
  checkpointed Monte-Carlo sweep, SNR-free fusion estimators, analysis) is
  documented in
  **[`scripts/sim_Transient_Localization/CRB_STUDY_README.md`](scripts/sim_Transient_Localization/CRB_STUDY_README.md)**.
- `paper/` — conference draft (Optica/OFC style); see
  [`paper/README.md`](paper/README.md).

The simulation code depends on the shared library
[`sop-simulation-lib`](https://github.com/Vieira-RA/sop-simulation-lib)
(`pip install -e .` or `PYTHONPATH` to its `src`).
