# Polarimeter Calibration — Progress & Next Steps

## 1. Done so far

### 1.1 `measure_trajectory.py` — trajectory rotation alignment
- Added the **Kabsch** (Kabsch–Umeyama) algorithm (SVD optimal rotation, `det = +1`) plus **ICP** (nearest-neighbour + Kabsch) to rotate the PM1000 trajectory onto the scope trajectory.
- Handles both the static rotation between the two reference frames and the fact that the trajectories are **not time-synchronised** (correspondence via nearest neighbour).
- Robust to large rotations via PCA principal-axis + random-restart initialisation.
- `plot_poincare(...)` now shows a two-panel figure (raw | aligned).

### 1.2 Documentation rewrite
- `In-Line High-Speed All-Fiber-Polarimeter.md` — added the practical referenced-calibration workflow, kept the physics/paper summary, corrected the "Degeneracy and Absolute Reference" section.
- `PM1000_UG.md` — documented the `Python_USB` MATLAB bridge (primary) and the serial/LAN `pm1000.py` driver, corrected the Stokes fractional decode (`/65536`), and added ATE / trigger / EPS1000 registers and device descriptors.

### 1.3 Calibration workflow (already in the repo, now documented)
1. `fused_acquisition_controlled.py N` → acquires detector voltages `D` (scope CH1–4) + reference Stokes `S` (PM1000) at N random **stable** SOPs (EPS1000 scrambler); writes `detector_readings_<ts>.csv` + `stokes_<ts>.csv`.
2. `referenced_calibration_lsq.py` → `C = S · pinv(D)`, prints verification (residual, DOP/SD-DOP, angular deviation), saves `calibration_matrix_reference.csv`.
3. Apply with `measure_trajectory.py`, `apply_calibration_trajectory.py`, `apply_ref_calibration.py`.

## 2. Current task (next step)

**Multi-wavelength calibration** of the in-line polarimeter (DUT): determine the 4×4 matrix **C(λ)** over a grid of wavelengths (proposed step 1 nm), because the tilted-grating coupling efficiency and the detector responsivity are wavelength-dependent.

## 3. Proposed architecture

1. `tunable_laser.py` — thin `requests` wrapper for the laser's web interface:
   - `set_wavelength(nm)`, `get_wavelength()` (read-back + mode-hop check), `wait_until_ready()`.
2. Refactor (low-risk): extract `acquire(n_sops, tag)` from `fused_acquisition_controlled.py` and `calibrate(det_csv, stk_csv) -> (C, metrics)` from `referenced_calibration_lsq.py`.
3. `sweep_calibration.py` — orchestrator:
   ```python
   for wl in wavelengths:
       laser.set_wavelength(wl); laser.wait_until_ready()
       pm.set_wavelength(wl)                 # only if the PM1000 needs it
       det_csv, stk_csv = acquire(n_sops, tag=f"{wl}nm")
       C, metrics = calibrate(det_csv, stk_csv)
       save C(wl); append metrics; plot verification
   ```
4. Output per run: `calibration_<wl>nm.csv`, `summary.csv` (λ, SD-DOP, angular deviation, residual, cond(C), S0), `all_calibration_matrices.npz` (N_wl × 4 × 4), `verification.png`.

Notes:
- The **EPS1000 scrambler needs no per-λ reconfiguration** — it only produces random stable SOPs and the PM1000 measures the actual SOP.
- Optional refinement: after the sweep, fit/smooth C(λ) with a low-order polynomial/spline and interpolate between measured points.

## 4. Information needed (input from the user)

1. **Laser** make/model and its web interface: IP/URL, GET/POST endpoints, request/response format, settle time.
2. **PM1000 wavelength mechanism**: is there a register to write λ (which one), or does it auto-handle λ? (Not found in the repo — register map/manual needed.)
3. **Wavelength range & step**: confirm C+L band and 1 nm (≈95 points @ 1 nm over 1525–1625 nm, ≈35 for C-band).
4. **N SOPs per λ**: suggest 20–40 (paper: convergence after ~20 random SOPs).
5. **Absolute S0 accuracy or normalized SOP only?** If only SOP matters, the PM1000 λ setting may be optional (its power responsivity is λ-dependent; normalized S1/S2/S3 are unaffected).
6. **Scope scales**: confirm 20 mV/div won't saturate at the brightest λ or fall below noise at the band edges (optionally auto-adjust `CH_SCALES`).

## 5. Open design decisions

| Decision | Recommendation |
|---|---|
| Wavelength step | 1 nm (finer if the band edges show rapid variation) |
| SOPs per λ | 20–40 |
| Interpolation | per-λ matrices now; add spline/polynomial fit later if needed |
| PM1000 λ setting | required only if absolute S0 matters; otherwise optional |
| Failure handling | retry + log warnings per λ (mode hop / low power / poor SD-DOP) |
