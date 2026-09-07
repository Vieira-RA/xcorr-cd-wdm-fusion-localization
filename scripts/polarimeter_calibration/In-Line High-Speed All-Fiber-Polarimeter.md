# In-Line High-Speed All-Fiber Polarimeter – Key Points Summary

## Device Overview

- **Type**: Compact, in-line, all-fiber polarimeter based on tilted fiber Bragg gratings.
- **Key Specifications**:
  - RF bandwidth: 500 MHz.
  - Single-calibration optical bandwidth: >30 nm (32 nm demonstrated).
  - Insertion loss: <0.5 dB.
  - PDL: <0.1 dB.
  - DGD: <450 fs (can be reduced to <10 fs).
  - Peak sampling rate: 250 MS/s.

## Physical Principle of Operation

### Tilted Fiber Gratings
- Four 45° tilted fiber Bragg gratings written in high-birefringence (HiBi) fiber.
- Each grating acts as a polarization-sensitive optical tap.
- Scattered light is proportional to a projection of the signal polarization onto the grating's defined polarization state.
- Grating period: ~1.07 μm (phase-matched for 1.55 μm).
- Grating length: 300 μm with ~1% strength.
- Broadband operation: >200 nm bandwidth.

### Tetrahedral Configuration
- **Grating 1 (on-axis)**: Aligned to the HiBi axis → scatters linearly vertically polarised light.
- **Gratings 2–4 (off-axis)**: Rotated by 53° relative to the HiBi axis.
- **Separation between off-axis gratings**: 1/3 of the fibre beat length (4.8 mm / 3).
  - Results in 120° angular rotation in Stokes space about the HiBi axis.
- **Result**: Four non-coplanar projection states forming a tetrahedron in Stokes space.
  - Provides optimal noise performance and low PDL.

### Graphical Design Approach
- Stokes vectors of the four gratings form a tetrahedron.
- First grating: on-axis (projects onto S₁).
- Off-axis gratings: 53° rotation → 106° angle in Stokes space from on-axis.
- 1/3 beat-length separation → 120° rotations in Stokes space.

## Calibration Framework

### General Principle
- The Stokes vector **S** is obtained by multiplying a 4×4 calibration matrix **C** by a 4×1 detector vector **D**:

  `S = C · D`

- **D** holds the four measured scattered powers (detector voltages) from the four gratings.

### Practical calibration in this repository (referenced, with PM1000)

This is the procedure the scripts in this repository actually implement. It is a
**referenced** calibration: the Novoptel **PM1000** provides the true Stokes vector,
and the Novoptel **EPS1000** scrambler generates random but **stable** input SOPs.

**Hardware setup**

- **Device under test (DUT)** = the in-line polarimeter. Its four grating detectors feed oscilloscope channels CH1–CH4.
- **Reference** = PM1000 polarimeter (reads the same SOP, `S0..S3` in µW).
- **Scrambler** = EPS1000 (random stable SOPs).
- **Scope** = Tektronix DPO7254 at `198.192.1.1`.

**Step 1 — Acquire data** (`fused_acquisition_controlled.py`)

```bash
python fused_acquisition_controlled.py 50     # 50 random SOPs (any M ≥ 1)
```

For each measurement the script:
1. Moves the EPS1000 to a new random, **stable** SOP.
2. Freezes the scope (`ACQUIRE:STATE STOP`).
3. Reads the PM1000 reference Stokes `S` and the four detector means `D` (volts) at that same SOP.
4. Resumes the scope and writes two index-aligned CSVs sharing one timestamp:
   - `detector_readings_<ts>.csv`  → columns `index, ch1, ch2, ch3, ch4`
   - `stokes_<ts>.csv`             → columns `index, S0_uW, S1_uW, S2_uW, S3_uW, DOP`

The scope window and the PM1000 averaging exponent (ATE) are matched to the same integration time.

**Step 2 — Solve the calibration matrix** (`referenced_calibration_lsq.py`)

```bash
python referenced_calibration_lsq.py            # auto-picks the latest CSV pair
# or explicitly:
python referenced_calibration_lsq.py detector_readings_<ts>.csv stokes_<ts>.csv
```

It builds the over-determined system (M SOPs) and solves the least-squares problem

`C = S · pinv(D)`   so that   `S ≈ C · D`

where `D` is 4×M (detector volts) and `S` is 4×M (reference Stokes, µW). The result is written to **`calibration_matrix_reference.csv`**.

**Step 3 — Verification** (printed by `referenced_calibration_lsq.py`)
- Relative residual `||S − C·D|| / ||S||`.
- DOP of measured vs reconstructed Stokes (mean and std; the std is the **SD-DOP**).
- Angular deviation (mean / max / std) between measured and reconstructed Stokes.

**Step 4 — Apply the matrix**

Any detector vector `D` is converted to Stokes with `S = C @ D`:
- `measure_trajectory.py` — live/periodic trajectory on the Poincaré sphere (also rotates the PM1000 trajectory onto the scope trajectory to compare reference frames).
- `apply_calibration_trajectory.py` — full time-resolved Stokes trajectory from scope traces.
- `apply_ref_calibration.py` — apply to a batch of static SOPs and report DOP / sphere coverage.

### Referenced calibration (minimal 4-point, as in the paper)

The paper (Section III, Eq. 2) describes the minimal version: launch **exactly four known, non-degenerate SOPs**, form the 4×4 matrices **S** (known Stokes rows) and **D** (detector rows), and invert:

`C = S · D⁻¹`

This is simple but noise-sensitive unless the four SOPs are chosen well. The least-squares version above (M > 4 points) is preferred and is what `referenced_calibration_lsq.py` implements.

### Reference-Free Self-Calibration (paper's alternative)

The key innovation of the paper is calibration **without** a reference polarimeter or known SOPs. This is implemented in `sop-simulation-lib/src/calibration.py` (Mikhailov et al., 2014).

**Constraints required:**
1. Input signals must have DOP ≈ 1 (fully polarised).
2. Signal power must be constant **or** measured independently.

**Two-step calibration process:**

#### Step 1: Power Calibration (First Row of C)
- Fit the top row (C₀ᵢ) to the measured power data using linear least squares.

#### Step 2: DOP Calibration (Lower 12 Elements)
- Adjust the remaining 12 elements to minimise DOP deviation (non-linear least-squares), using the ideal tetrahedral calibration matrix as the initial guess:

  `C_guess = 1/4 · [ 4C₀₀ 4C₀₁ 4C₀₂ 4C₀₃ ; 3η -η -η -η ; 0 2η√2 -η√2 -η√2 ; 0 0 η√6 -η√6 ]`

  where η is the average scale factor from the first row.

### Degeneracy and Absolute Reference
- The reference-free calibration is undefined up to a Stokes rotation: the DOP
  constraint is invariant under rotations of S1–S3, so it needs an absolute
  reference to fix the orientation.
- For an absolute reference: Grating 1 is on-axis (aligned with the HiBi axis) →
  represents a projection onto S₁.
- Alternative: launch known linear SOPs and align the calibration matrix accordingly.
- The referenced (PM1000) calibration has no such *mathematical* degeneracy — given
  the reference, C is unique — but its result is expressed in the PM1000's reference
  frame. Since the PM1000 and the DUT's intrinsic axes (grating 1 / HiBi axis) are
  generally not physically aligned, and this alignment is difficult in practice, the
  referenced calibration also differs from the DUT's intrinsic frame by a static
  Stokes rotation. In that sense both methods are "correct up to a Stokes rotation";
  the difference is whether that rotation is a free fit parameter (reference-free) or
  an inherited physical misalignment (referenced).

## Calibration Performance Metrics

### Internal Metric
- Standard deviation of DOP (using 1 as the maximum) across all calibration points.
- Correlates well with the absolute metric (comparison to a reference polarimeter).
- Calibration converges after ~20 random SOPs.
- Total calibration time: <1 second.

### Single-Calibration Bandwidth
- Defined as the wavelength range where measured SD-DOP < 1%.
- Achieved: 32 nm centred at 1550 nm.
- Self-calibration outperforms 4-point reference calibration within ±10 nm of the calibration wavelength.

### Performance Results
- SD-DOP: 0.02% (HiBi) vs 0.15% (reference benchtop).
- Maximum angular deviation from reference: 0.5°.
- Validated over 999 random SOPs (DOP ~100%).

## Remote Calibration Capability
- Reference-free calibration works over long fibre distances (demonstrated with 30 km).
- Random polarisation rotation during propagation does **not** affect calibration.
- Constraints (DOP ≈ 1, power measurement) are preserved over the fibre link.
- No increase in required calibration points for remote operation.

## High-Speed Acquisition System

### Dual Polarimeter Architecture
- **Slow polarimeter**: 10–40 kS/s, continuously monitors for events.
- **Fast polarimeter**: 1–250 MS/s, data stored in an FPGA cyclic buffer.
- **Trigger mechanism**: when the slow polarimeter detects a rapid polarisation change, fast data is downloaded from the buffer.
- Enables long-duration monitoring (days/weeks) with high-speed capture of intermittent events.

## Applications
- Security and safety monitoring in distributed fibre networks.
- Characterisation of fast polarisation dynamics (e.g., vector soliton molecules).
- Telecommunications and optical sensors.
- General lightwave test and measurement.

## Key Advantages
1. Compact, in-line design (no moving parts).
2. High speed (500 MHz electrical bandwidth).
3. Broad single-calibration bandwidth (32 nm).
4. Reference-free self-calibration.
5. Low insertion loss and PDL.
6. Remote calibration capability.
7. True real-time acquisition with long-term monitoring.
