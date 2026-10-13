# CD Localisation via SOP Transient Analysis in WDM Systems

## 1. Research Context

**Objective:**  
Localise a transient mechanical perturbation (e.g., vibration, strain) along a long-haul optical fibre using state‑of‑polarisation (SOP) measurements from multiple WDM channels.

**Physical principle:**  
A perturbation induces a local change in the fibre birefringence, modulating the SOP evolution.  
Chromatic dispersion (CD) causes different wavelengths to travel at slightly different speeds, creating a wavelength‑dependent time delay between the SOP signals of different WDM channels.  
By estimating these delays for many channel pairs, the distance to the perturbation can be computed via the integrated fibre dispersion.

**Key fibre effects modelled:**  
- Polarisation‑mode dispersion (PMD) – random birefringence along the fibre, modelled with a wave‑plate model with first order birefringence relation with frequency. This could generate the proper frequency autocorrelation function in frequency.
- Chromatic dispersion (CD) – deterministic, modelled via wavelength‑dependent group delay.  
- Fibre length: from 100 km to 10000 km, event at an arbitrary distance.

**WDM configuration:**  
- C+L band: 184–196 THz, channels spaced by 50 GHz (ITU‑T grid).  
- Full grid of ~241 channels available; analysis uses selected subsets (e.g., 4, 6, 8, 10, 12 channels) chosen by an alternating scheme (extremes first).

---

## 2. Signal Processing Pipeline

### 2.1. Jones Matrix Generation
- PMD fibre profile generated for each independent fibre realisation (seed‑based).  
- Gaussian modulation envelope `s_env(t) = 1 + A_pulse * exp(-t²/(2σ²))` where σ = 0.3748 / bandwidth.  
- Jones matrices `U[ch, t]` computed for all channels and time steps via ODE solver.  
- CD delays applied to the Jones sequences using frequency‑domain fractional delay (phase shift).  
- All clean Jones matrices saved for later post‑processing.

### 2.2. Signal Representations
From the clean Jones matrices three parallel signal types are derived:

1. **2‑D Stokes vectors (S1, S2)** – obtained via quaternion‑based rotation of a fixed input Stokes vector, followed by centroid rotation to the North Pole.  
2. **Rotation‑vector magnitude** `|φ(t)|` – quaternion → rotation vector `φ`, then Euclidean norm. DC removed.  
3. **3‑D quaternion vector part (q1, q2, q3)** – unit quaternion from Jones, sign‑regularised, DC removed, only the vector components used.

### 2.3. Noise Model
- Noise added directly to the clean signal representations.  
- SNR defined using peak power of the AC (DC‑free) signal: `noise_var = peak_power / (dim * snr_linear)`.  
- Stokes: 2 components share noise budget → factor 2 in denominator.  
- Quaternion: 4 components share noise budget → factor 4.  
- Rotation‑vector: scalar → factor 1.  
- Optional zero‑phase low‑pass Butterworth filter (cutoff = 3×bandwidth) applied before delay estimation.

### 2.4. Delay Estimation Methods
For each channel pair, the relative time delay is estimated using:

1. **Weighted Phase‑Slope** – magnitude‑weighted linear fit to the unwrapped cross‑spectral phase.  
2. **GCC‑PHAT** – phase‑transform generalised cross‑correlation with bandwidth mask.  
3. **Integer Peak** – sample‑index maximum of the cross‑correlation.  
4. **Parabolic Peak** – three‑point parabolic interpolation of the cross‑correlation peak, bounded by a physically plausible lag window `[0, τ_max + 5σ]`.

All methods operate on both 2‑D Stokes (using 2‑component dot‑product correlation) and 1‑D signals (rotation‑vector magnitude or quaternion vector correlation).

### 2.5. Distance Conversion
For a pair of channels `(i, j)` with wavelengths `λ_i, λ_j`, the distance is computed as:
Z = (estimated_delay × 10¹² ps/s) / |∫ D(λ) dλ|
where the integral of the dispersion curve D(λ) = D₀ + S₀(λ − λ₀) is evaluated analytically.

### 2.6. Aggregation & Outlier Rejection
- **IQR pre‑filter**: factor 1.5 removes gross outliers.  
- **Adaptive Weighted Mean**: each pair’s distance is weighted by:
  - `w_λ = D_int²` (precision ∝ 1/var ∝ D_int²).
  - `w_robust` (Tukey bisquare weight) down‑weights inliers far from the median.
- The final distance estimate is the weighted mean over all pairs.

### 2.7. Multi‑Fibre Averaging
- Independent fibre realisations (different PMD seeds) are simulated.  
- Distance estimates are first aggregated per fibre, then averaged across fibres.  
- Standard deviation of the final estimate scales as `σ / √(n_fibres)`.

---

## 3. Simulation Architecture

**Repositories:**
- `sop-simulation-lib` – shared Python library containing physics and DSP modules.  
  Installed in editable mode (`pip install -e .`).  
  Key modules: `signal_processing`, `quaternion`, `rotations`, `estimation`, `plotting`, `chromatic_dispersion`, `pmd_model`, `fiber_propagation`.

- `xcorr-cd-wdm-fusion-localization` – paper‑specific scripts and results.  
  Structure: `scripts/sim_Transient_Localization/` contains generator, post‑processor, YAML config, analysis scripts.

**Workflow:**
1. `sweep_config.yaml` defines all parameters (fibre, WDM grid, bandwidth sweep, SNR grid, number of fibres, etc.).
2. `generate_jones_data.py` – produces Jones matrices for all bandwidths and fibres; saved to `jones_data/`.
3. `process_sweep.py` (or `_MF.py`) – loads Jones matrices, derives all signal types, runs Monte‑Carlo over noise realisations, aggregates, and generates heatmaps (std, bias, success rate). Supports parallelisation via `joblib`.
4. Analysis scripts e.g. `full4Darrays_analysis.py` slice the saved 4‑D data to compare different fibre counts and channel counts.

---

## 4. Library Functions (relevant to this work)

- **`signal_processing`**:
  - `cross_correlation_2d_fft` / `_1d_fft`
  - `phase_slope_delay_2d` / `_1d`
  - `gcc_phat_2d` / `_1d`
  - `integer_corr_delay_2d` / `_1d`
  - `parabolic_corr_delay_2d` / `_1d`
  - `parabolic_fit` (sub‑sample refinement)

- **`quaternion`**:
  - `jones_to_quaternion`
  - `regularize_signs`
  - `quaternion_to_rotation_vector`
  - `quaternion_rotate_stokes`

- **`estimation`**:
  - `iqr_filter`
  - `adaptive_weighted_mean` (with configurable scale method, bisquare constant)

- **`plotting`**:
  - `plot_std_heatmap`, `plot_bias_heatmap`, `plot_success_rate_heatmap` (publication‑ready, with `TwoSlopeNorm` for symmetric bias maps)

---

## 5. Key Findings & Comparisons

- **2‑D Stokes methods consistently performs a very little bit better the rotation‑vector magnitude PROBABLY due to:**
  1. Noise averaging across two independent components (√2 advantage in correlation peak SNR).
- **Multi‑fibre averaging** apparently provides a √N improvement in standard deviation, independent of the estimation method.
- **The bandwidth of the perturbation is the fundamental limiting factor** alongside with signal SNR (Cramér–Rao bound): no DSP can overcome a slow transient.

---

## 6. Parameter Tuning (IQR & Robust Weights)

- Not yet defined

---

## 7. Current Status (August 2026)

- Full Monte‑Carlo sweep with 60 bandwidths × 60 SNRs × 50 realisations × 10 fibres has been completed.
- Results for channel counts 4,6,8,10,12 and fibre counts 1,2,4,6,8,10 have been extracted and plotted as standard‑deviation and bias heatmaps for the four different techniques.