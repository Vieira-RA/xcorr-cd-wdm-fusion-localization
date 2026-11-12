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

## 2. Current task: multi-wavelength calibration

Determine the 4×4 matrix **C(λ)** over the intended wavelength range because the tilted-grating coupling efficiency and detector responsivity are wavelength-dependent. A first software scaffold is in [`multiwavelength_calibration/`](multiwavelength_calibration/): it reuses `fused_acquisition_controlled.py`, fits one matrix per wavelength, and reports held-out validation. Its example config uses manual laser setting until the laser API is confirmed.

**Resolved: the target is the normalized SOP.** `S1..S3/S0` and DOP are calibrated; `S0` is fitted and recorded as a power/DOP readout but is not an absolute-power calibration. This means the PM1000 wavelength correction is **not** required for the calibration to be valid (its λ-dependent responsivity cancels in the normalized Stokes vector). It also means matrix quality must be judged on the polarization path, not on the raw condition number of `C` — see below.

**Added: conditioning gate + report.** `conditioning.py` reports three numbers per matrix — `condition` (raw σmax/σmin of the 4×4), `condition_row_scaled` (scale-invariant, ideal 1.0) and `condition_polarization` (the 3×4 `D → S1,S2,S3` map that governs normalized-SOP error). The gate (`max_condition`, default 3.0) is applied to the worse of the two scale-invariant numbers and the status (`excellent/ok/warn/poor`) is written to `summary.csv`, `metrics_<wl>nm.json` and to `conditioning_report.{csv,json}`. Reference values: the existing single-wavelength `calibration_matrix_reference.csv` has `condition = 11.46` (S0-row scale artifact) but `condition_row_scaled = 1.36` / `condition_polarization = 1.52`, i.e. well conditioned.

**Added: offline rehearsal (`--simulate`).** `synthetic.py` generates physically-shaped detector/Stokes CSVs from a known smooth `C(λ)` (tetrahedral basis + Stokes-frame rotation + per-row gain ripple), so the whole sweep → fit → conditioning → summary → `.npz` → report path can be exercised without instruments. Rehearsal writes `truth_calibration_<wl>nm.csv` and records `fit_to_truth_relative_error` (measured 0.2–0.4% on 30 SOPs with 0.2% detector noise).

**Added: CoBrite laser control.** `cobrite.py` implements the documented SCPI dialect (TCP 2000, from `cobrite/Manual_CoBrite.pdf`) with the standard library only: `WAV`/`WAV?`/`WAV:LIM?`, `POW`/`POW?`/`APOW?`, `STAT`, `*OPC?` and `BUSY?`. `laser_client.py` exposes it as `laser.mode = "cobrite"` and the sweep now tunes, waits for `BUSY?` to clear, verifies the `WAV?` read-back, optionally cross-checks measured against set power, and aborts with exit code 2 *before* any acquisition if the instrument is unreachable. `python cobrite.py --host <ip>` is a standalone connectivity probe. Covered by fake-instrument tests (protocol, settling, error replies, and a full sweep run against a stub acquisition script).

Before the lab sweep, confirm the usable wavelength range, optical routing (one wavelength per polarimeter), scope scales, and the laser's actual read-back. The sample wavelength grid is illustrative and must be checked against the instruments.

**First hardware results (pilot sweep, 5 x 30 SOPs, 1530-1570 nm @ 10 dBm).**
All five wavelengths completed with `status = ok`; held-out angular error 1.8-2.7 deg
(max 4.9 deg), DOP MAE 0.02-0.04, row-scaled conditioning 1.34-1.42 and polarization
conditioning 1.42-1.58 (gate 3.0), all `excellent`/`ok`. Detector levels are flat across
the band (means 16.5-17.2 mV, S0 342-366 uW), so the data quality is uniform.

**Cross-wavelength usability** (`cross_wavelength_validation.py`, applying C(lambda_i) to
data taken at lambda_j): in-sample 1.75 deg, but **10 nm away gives 8.9 deg mean (18 deg
worst)**, 20 nm 14.5 deg, 40 nm 26.8 deg. A single matrix therefore does **not** stretch
across a 10 nm step.

**Why (corrected after the 1 nm sweep):** the difference between C(lambda) values is a pure
**Stokes rotation** (residual after rotation+gain 0.5-6 %), and measuring it at 1 nm resolution
shows it is *not* a slow drift: adjacent wavelengths differ by **71.9 +/- 11.9 deg per nm**
about a fixed axis **(-0.930, +0.260, +0.258)** (axis scatter 1.2 deg). That is a phase that
wraps every **5.0 nm**, i.e. a **differential group delay of ~1.6 ps** in the fibre between the
DUT taps and the PM1000 (roughly 1 m of non-PM fibre). The earlier 10 nm pilot showed an
apparently smooth 38 deg drift simply because 10 nm x 72 deg/nm aliases to ~0 mod 360.
Input-side birefringence is common-mode and cancels; only the DUT->PM1000 path contributes.

**Consequences:**
- A per-wavelength calibration is valid at *its own* wavelength only (1 nm away is already
  72 deg off). A 1 nm lookup table is therefore the safe approach.
- **5 nm coarse spacing is exactly one wrap period**, so consecutive matrices look
  deceptively similar (7 deg) - that is aliasing, not stability. Coarse grids cannot be used
  to justify interpolation.
- If interpolation is ever wanted, interpolate the *phase*: C(lambda) = R(theta(nu)) C_ref with
  theta linear in optical frequency (DGD and axis fitted once), not the matrix entries.
- Hardware lever: shortening / PM-ising the fibre between the FBG and the PM1000 directly
  widens the usable single-calibration bandwidth (DGD is what sets it here, not the gratings).

**Recommended next step:** refine the grid where the rotation moves (1-2 nm around
1525-1545) and keep 5 nm on the red side; and/or fit the rotation-only model per
wavelength (a handful of SOPs suffice) so C(lambda) becomes a smooth 4-parameter curve
that can be interpolated and validated at unseen wavelengths.

## 3. Proposed architecture

1. `cobrite.py` (implemented) — SCPI client for the ID Photonics CoBrite:
   - `set_wavelength(nm)`, `wavelength()` (read-back), `wait_until_settled()`
     (polls `BUSY?`), `actual_power()` / `output_state()` sanity checks;
   - `probe()` + `python cobrite.py --host <ip>` for a pre-flight connectivity
     and range check.
   (`laser_client.py` also keeps the configurable HTTP adapter for other
   instruments.)
2. Refactor (low-risk): extract `acquire(n_sops, tag)` from `fused_acquisition_controlled.py` and `calibrate(det_csv, stk_csv) -> (C, metrics)` from `referenced_calibration_lsq.py`.
3. `sweep_calibration.py` — orchestrator (implemented):
   ```python
   for wl in wavelengths:
       laser.set_wavelength(wl); laser.wait_until_ready()   # or --simulate
       det_csv, stk_csv = acquire(n_sops, tag=f"{wl}nm")
       C, metrics = calibrate(det_csv, stk_csv)             # incl. conditioning
       save C(wl); append metrics; checkpoint summary.csv
   ```
4. Output per run: `calibration_<wl>nm.csv`, `summary.csv` (λ, SD-DOP, angular deviation, residual, cond row-scaled, polarization cond, conditioning status), `all_calibration_matrices.npz` (N_wl × 4 × 4), `conditioning_report.{csv,json}`, `conditioning_vs_wavelength.png`.

Notes:
- The **EPS1000 scrambler needs no per-λ reconfiguration** — it only produces random stable SOPs and the PM1000 measures the actual SOP.
- Optional refinement: after the sweep, fit/smooth C(λ) with a low-order polynomial/spline and interpolate between measured points.

## 4. Information needed (input from the user)

1. *Resolved:* laser is an **ID Photonics CoBrite**, controlled by SCPI over TCP
   port 2000 (`cobrite/Manual_CoBrite.pdf`); implemented in `cobrite.py`. Only
   the lab IP/hostname and port address still need to be confirmed at run time
   (`--laser-host`, `--laser-address`).
2. **Wavelength range & step**: confirm C+L band and 1 nm (≈95 points @ 1 nm over 1525–1625 nm, ≈35 for C-band). The sweep now reads `WAV:LIM?` from the laser and flags out-of-range requests.
3. **N SOPs per λ**: suggest 20–40 (paper: convergence after ~20 random SOPs).
4. **Scope scales**: confirm 20 mV/div won't saturate at the brightest λ or fall below noise at the band edges (optionally auto-adjust `CH_SCALES`).
5. *Resolved:* PM1000 λ mechanism is **not needed** for the normalized-SOP target; revisit only if absolute S0 calibration is ever required.
6. *Resolved:* target quantity is the normalized SOP (S0 retained as a power/DOP diagnostic).

## 5. Open design decisions

| Decision | Recommendation |
|---|---|
| Wavelength step | 1 nm (finer if the band edges show rapid variation) |
| SOPs per λ | 20–40 |
| Interpolation | per-λ matrices now; add spline/polynomial fit later if needed |
| PM1000 λ setting | **not needed** for normalized SOP (resolved) |
| Failure handling | retry + log warnings per λ (mode hop / low power / poor SD-DOP) |
| Conditioning gate | `max_condition = 3.0` on the worse of row-scaled and polarization cond; require `rank = 4` |
| Poor matrix | record + flag by default; `fail_on_poor_conditioning: true` to abort the λ |
| Laser control | `cobrite` SCPI mode by default; `manual` available via `--laser-mode manual` |
| Laser settle criterion | `BUSY?` == 0 plus verified `WAV?` read-back (`*OPC?` alone is not enough) |
