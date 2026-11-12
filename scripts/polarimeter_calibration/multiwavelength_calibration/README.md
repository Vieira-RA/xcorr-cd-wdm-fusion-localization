# Multiwavelength calibration of the tilted-FBG polarimeter

This subfolder adds a wavelength sweep around the existing **referenced,
single-wavelength** calibration workflow. At each wavelength it acquires stable
SOPs with the EPS1000, records the four FBG-polarimeter detector voltages on
the oscilloscope and the reference Stokes vector on the PM1000, then estimates
one 4x4 matrix `C(lambda)`:

```text
S_uW = C(lambda) @ D_volts
C(lambda) = S @ pinv(D)
```

The PM1000 and FBG polarimeter should receive the **same single wavelength**
during calibration. This workflow does not combine wavelengths or control a
wavelength demultiplexer. It reuses the already-tested parent script
`fused_acquisition_controlled.py` for the PM1000/EPS1000/scope acquisition.

## Primary target: normalized SOP (S0 kept as a diagnostic)

The calibration target is the **normalized** Stokes vector, i.e. the SOP on the
Poincare sphere (`S1..S3 / S0`) and the DOP. `S0` is still fitted and recorded
because it is a useful optical-power and DOP readout, but it is **not** treated
as a calibrated absolute power measurement.

Consequences for this workflow:

- The PM1000 wavelength correction is **not** required: its wavelength-dependent
  responsivity scales `S0` (and all of `S1..S3` equally), which cancels in the
  normalized SOP. Only absolute power calibration would need it.
- Ranking matrix quality must be done on the polarization path, not on the raw
  singular-value spread of `C` (see the conditioning section below).
- `S0`-related metrics (`relative_residual`, `dop_mae`) are still reported, but
  the go/no-go numbers for the SOP target are the held-out **angular deviation**
  and the **polarization condition number**.

## Before the first lab run

1. Confirm the laser, PM1000, and FBG polarimeter wavelength ranges. The example
   grid in `config.example.json` (`1540, 1550, 1560 nm`) is illustrative and
   must be checked against the actual instruments before running. In `cobrite`
   mode the sweep reads the laser's own `WAV:LIM?` range and flags any requested
   wavelength outside it.
2. Establish how the PM1000 handles wavelength-dependent responsivity. The
   existing notes do not identify a wavelength-setting register. Set it
   manually if required, or decide whether the calibration target is normalized
   SOP only rather than absolute Stokes power across wavelengths.
3. Connect the PM1000 and FBG polarimeter so they observe the same SOP at each
   calibration wavelength. Keep the optical branches and scope gain settings
   fixed throughout the sweep.
4. Verify scope scales at the shortest and longest wavelengths (no clipping;
   signal above the noise floor).
5. Use a small pilot grid first. With `laser.mode = "cobrite"` (the shipped
   default) the sweep tunes the laser itself and verifies the read-back; with
   `"manual"` it prompts for each wavelength and the displayed read-back.
6. Rehearse the planned config once with `--simulate` (see below) so file layout,
   thresholds and report expectations are confirmed before the hardware run.

The acquisition script currently contains the existing lab settings for the
scope IP, PM1000/EPS1000 descriptors, detector channels and scope scales. Check
those settings before the first run. The PM1000 wavelength is **not** changed by
this new code because its wavelength mechanism has not yet been established.

## No admin rights? The two admin-free paths

Setting a static IP, adding a route or enabling a disabled NIC all require local
administrator rights. Neither instrument actually needs that, because both
expose a **USB** path that uses no IP configuration at all.

### Laser: SCPI over the USB virtual COM port

The CoBrite publishes three USB devices: a storage volume, an RNDIS virtual
Ethernet (the ``169.254.x.x`` link) and a **virtual COM port**. The COM port is
an independent interface, so it keeps working even when the RNDIS adapter's IPv4
configuration is broken. Use it with ``transport: "serial"``:

```json
"cobrite": { "transport": "serial", "com_port": "COM4", "baudrate": 115200,
             "address": "1,1,1", "power_dbm": 10.0, "max_power_dbm": 10.0 }
```

The unit is a CDC-ACM gadget: it answers at any baud rate, and replies arrive as
``;<CR><LF><value>;<CR><LF>`` (acknowledgement glued to the value), which the
client already handles. Requires ``pyserial``. Find the port with
``discover_laser.py`` or ``python -c "import serial.tools.list_ports as p; print([x.device for x in p.comports()])"``.

### Scope: VXI-11 or PyVISA, same code

`scope_io.py` gives every acquisition script one interface and chooses the
backend from the environment, so no script edit is needed:

| Variable | Meaning |
|---|---|
| `PM_SCOPE_TRANSPORT` | `vxi11` (TCP 111), `pyvisa` (NI-VISA), or `auto` (default) |
| `PM_SCOPE_IP` | address for VXI-11 (default `198.192.1.1`) |
| `PM_SCOPE_RESOURCE` | explicit VISA resource, e.g. `USB0::0x0699::0x0364::C000001::INSTR` |

Two admin-free options:

1. **USB-TMC** — plug the scope's USB device port in and set
   `PM_SCOPE_RESOURCE=USB0::0x0699::...::INSTR`. No IP, no NIC, no privileges
   (the USBTMC driver is already installed on this host).
2. **LAN, but re-address the *scope*.** The host already holds a link-local
   address (`169.254.x.x/16`) on its wired NIC, so put the instrument in that
   same `/16` from its own front panel — e.g. `169.254.102.200 / 255.255.0.0`,
   gateway blank — then `set PM_SCOPE_IP=169.254.102.200`. This changes nothing
   on the PC and needs no rights.

Timeouts are always in **seconds**; the PyVISA backend converts to the
milliseconds NI-VISA expects, and clears the read terminator around binary block
reads so a `0x0A` byte inside a waveform cannot truncate it.

## Pre-flight checks (do these in order)

Three standalone tools, none of which touches the calibration data:

```bash
python discover_laser.py          # where does the laser answer? (USB/RNDIS or RJ45)
python cobrite.py --host <ip>     # identity, wavelength range, power, output, alarms
python scope_probe.py             # detector clipping / headroom at the chosen ch_scales
```

`discover_laser.py` inspects the host adapters, resolves `cobrite.local`, tries
the known addresses and optionally scans the local /24 on port 2000, then lists
serial ports. It only sends `*IDN?`.

`scope_probe.py` mirrors the acquisition settings, freezes the scope, and reports
per channel: vertical scale, full scale, peak, fraction of full scale, mean,
SNR of the record mean and whether the ADC rails were hit. It prints a
ready-to-paste `--ch-scales` / `"ch_scales"` suggestion. **Full scale is
+/-4 divisions around `CH:POSITION 0`**, so 20 mV/div allows +/-80 mV before
clipping; the acquisition script accepts an override without being edited:

```bash
python ../fused_acquisition_controlled.py 30 --ch-scales 0.02,0.05,0.02,0.02
set PM_CH_SCALES=0.02,0.05,0.02,0.02     # Windows; same effect via environment
```

The sweep forwards `ch_scales` from the config the same way, so the scales used
for a run are recorded in `summary.csv`-adjacent metadata and the plan output.

**Laser power rule:** the CoBrite in this lab is only ever driven at **10 dBm**.
The limit is enforced in three places — `cobrite.py` refuses any `POW` above
`max_power_dbm` (`ABSOLUTE_MAX_POWER_DBM = 10.0`), the sweep config validator
rejects `power_dbm > 10`, and `check_power` compares the measured `APOW?` with
the setpoint. Note the instrument's own `LIM?` reports a **10.00 dBm minimum**,
so 10 dBm is also the lowest power it can emit: if a detector ever saturates,
it cannot be fixed optically at the laser — use the scope scale (or in-line
attenuation).

## Run the pilot sweep

From this folder:

```bash
python sweep_calibration.py --config config.example.json --dry-run
python sweep_calibration.py --config config.example.json
```

The first command only validates and prints the plan. The second command is a
real hardware run. It creates a timestamped directory under `runs/`, then for
each wavelength:

1. tunes the laser to the wavelength and waits until `BUSY?` reports settled
   (CoBrite mode), or prompts for the setting/read-back (manual mode);
2. runs `../fused_acquisition_controlled.py N`, which holds the EPS1000 at
   random stable SOPs and collects index-aligned detector and PM1000 CSVs;
3. reserves a seeded random subset for held-out validation;
4. fits the final `C(lambda)` using all collected SOPs, reports the conditioning
   and saves metrics.

Do **not** use `--dry-run` as evidence that hardware communication works; it
never connects to instruments. The inherited acquisition script requires the
lab Python environment and vendor dependencies (`vxi11`, `Python_USB`, MATLAB
Runtime integration, and `matlab`).

## Rehearse the whole sweep without hardware

`--simulate` replaces the hardware acquisition with synthetic data and runs the
identical fit, conditioning, summary, `.npz` and report path:

```bash
python sweep_calibration.py --simulate --output-root _sim_runs
```

This writes `runs`-style output under `_sim_runs/mwcal_sim_<timestamp>/` and, for
every wavelength, a `truth_calibration_<wl>nm.csv` holding the synthetic matrix
that was used to generate the detector voltages. The run metadata then contains
`fit_to_truth_relative_error`, so the rehearsal checks the pipeline *and* the
estimator: a correctly wired sweep recovers the truth matrix to well under 1%
and reproduces the conditioning of the synthetic truth.

Use it to verify a config before the lab session, to sanity-check the expected
`summary.csv`/report layout, and to test threshold changes. Synthetic settings
live in the `synthetic` block of the config (`detector_noise_rel`,
`stokes_noise_rel`, `power_uw`, `power_spread`, `matrix_scale`, `s0_scale`);
unknown keys are rejected so typos cannot silently change a rehearsal.

## Conditioning: are the matrices well conditioned?

The reconstruction is `S = C @ D`, so detector noise is amplified by the
singular-value spread of `C`. An ideal tilted-FBG matrix is tetrahedral (four
mutually orthogonal projections of equal strength), so the ideal value of every
condition number below is **1.0**.

Three numbers are reported per wavelength, because the raw condition number
alone is misleading for this instrument:

| Metric | Meaning | Typical |
|---|---|---|
| `condition` | `sigma_max / sigma_min` of the full 4x4 `C` | ~4-12 on the real device |
| `condition_row_scaled` | same after dividing each row by its norm; **scale invariant** | ~1.0-1.5 |
| `condition_polarization` | `D -> (S1, S2, S3)` (3x4); governs the normalized-SOP error | ~1.0-1.6 |

The raw number is inflated by the `S0`-row versus `S1..S3`-row unit mismatch (the
measured single-wavelength matrix `../calibration_matrix_reference.csv` has
`condition = 11.46` but `condition_row_scaled = 1.36`). Therefore the **gate is
applied to the worse of the two scale-invariant numbers**, controlled by
`max_condition` in the config (default `3.0`):

| Status | Rule | Action |
|---|---|---|
| `excellent` | <= 1.5 (or the gate, if stricter) | none |
| `ok` | <= `max_condition` | none |
| `warn` | <= 10 | inspect, keep the matrix |
| `poor` | > 10 or rank < 4 | refit or re-acquire this wavelength |

A `warn`/`poor` status is written to `summary.csv` (`conditioning_status`) and
`metrics_<wl>nm.json`, printed per wavelength, and listed in the report. Set
`fail_on_poor_conditioning: true` to abort a wavelength whose matrix is `poor`
instead of recording it.

The sweep prints the conditioning per wavelength and, at the end, writes
`conditioning_report.csv` / `conditioning_report.json` plus a
`conditioning_vs_wavelength.png` plot in the run directory. Re-run the report on
existing data at any time:

```bash
python conditioning_report.py                                  # newest runs/mwcal_*
python conditioning_report.py --run-dir runs/mwcal_<ts> --plot
python conditioning_report.py --npz <run>/all_calibration_matrices.npz
python conditioning_report.py --scan ..                        # every matrix ever produced
python conditioning_report.py ../calibration_matrix_reference.csv --wavelength 1550
```

The report exits non-zero when any matrix is `warn`/`poor`, so it works as a
go/no-go check in scripts. Because the gate is applied to the scale-invariant
numbers, the verdict is unaffected by scope gain, PM1000 power offsets or a
re-cabled detector order (the last one changes the matrix itself, not its
conditioning).

`--scan <path>` walks a whole tree, so every matrix from every past run plus the
single-wavelength `calibration_matrix_reference.csv` can be checked in one call;
matrices whose filename carries no wavelength are listed as `n/a`.

## Outputs

For each wavelength directory (`lambda_...nm/`):

- `detector_readings_<timestamp>.csv` and `stokes_<timestamp>.csv` — raw paired
  measurements from the inherited acquisition script;
- `calibration_<wavelength>nm.csv` — headerless numeric 4x4 matrix, compatible
  with existing `np.loadtxt(..., delimiter=',')` consumers;
- `metrics_<wavelength>nm.json` — training/held-out validation metrics,
  the conditioning report, the active gate and acquisition file references;
- `validation_<wavelength>nm.csv` — reference and reconstructed Stokes values
  for held-out SOPs;
- `run_metadata.json` — requested/read-back wavelength, run details and
  conditioning; in `--simulate` runs it also holds the synthetic truth matrix
  and `fit_to_truth_relative_error`;
- `truth_calibration_<wavelength>nm.csv` — only in `--simulate` runs: the
  synthetic matrix the data was generated from.

At the sweep root:

- `sweep_metadata.json` — simulated flag, laser `*IDN?`/limits/power/output
  state, wavelengths, seeds, conditioning gate;
- `summary.csv` — one row per wavelength, checkpointed after each acquisition,
  including `conditioning_status`, `calibration_matrix_condition_row_scaled`,
  `polarization_condition` and a `simulated` flag;
- `all_calibration_matrices.npz` — `wavelengths_nm` and matrices shaped
  `(N_wavelengths, 4, 4)` for successful wavelengths;
- `conditioning_report.csv` / `conditioning_report.json` — per-wavelength
  conditioning table (written automatically at the end of a sweep);
- `conditioning_vs_wavelength.png` — condition numbers versus wavelength with the
  gate overlaid (written automatically at the end of a sweep).

The fitted matrices use Stokes order `[S0, S1, S2, S3]` and detector order
`[CH1, CH2, CH3, CH4]`. Scope channel scaling, detector cabling, optical routing,
and reference-frame alignment are part of this end-to-end calibration; changing
them can require recalibration.

## Offline calibration of existing data

The numerical fit/validation tool can be run without connecting to hardware:

```bash
python calibrate_reference.py \
  --detector path/to/detector_readings_<timestamp>.csv \
  --stokes path/to/stokes_<timestamp>.csv \
  --wavelength-nm 1550 \
  --output-dir path/to/output
```

At least five SOPs are needed for a held-out split, and the detector matrix in
the training subset must have rank four. The default split is 75% training and
25% validation. The returned calibration matrix is refit using all rows after
the held-out performance is measured. Metrics include relative Stokes residual,
normalized-Stokes RMSE, DOP error, angular deviation, and matrix conditioning.

Unit-level numerical tests (no instruments required):

```bash
python -m unittest test_calibration_math test_conditioning
```

## How far does one calibration stretch? (cross-wavelength validation)

```bash
python cross_wavelength_validation.py                  # newest run
python cross_wavelength_validation.py --run-dir runs/mwcal_<ts>
```

It applies `C(lambda_i)` to the detector data acquired at `lambda_j` and prints matrices
of mean/worst angular error and normalized-Stokes RMSE, plus a summary by wavelength
distance. The diagonal is the in-sample fit; the off-diagonal entries decide the sweep
step (and whether interpolation is worth adding).

## Laser control (ID Photonics CoBrite, SCPI)

`laser.mode` selects how the wavelength is set:

| Mode | Behaviour |
|---|---|
| `cobrite` | Native SCPI control over TCP 2000 (default in `config.example.json`) |
| `manual` | Operator sets the instrument and types the read-back; no transport opened |
| `http` | Configuration-driven HTTP adapter (no vendor URL assumed; needs `requests`) |

CoBrite mode is implemented in `cobrite.py` from the vendor manual in
`../cobrite/Manual_CoBrite.pdf`, using only the standard library. For each
wavelength it:

1. checks the requested value against the instrument's own `WAV:LIM?` range;
2. optionally sets `POW` and switches the output on (`STAT`);
3. sets `WAV <address>,<nm>` and confirms command execution with `*OPC?`;
4. polls `BUSY?` until the port reports *settled* — note the manual is explicit
   that `*OPC?`/`WAV?` do **not** prove the laser has finished tuning, and
   `WAV?` already reports the new target while still tuning;
5. verifies `WAV?` against the request within `tolerance_nm`;
6. optionally compares the measured `APOW?` with the `POW?` setpoint, so a dead
   output, closed shutter or unplugged patch cable fails loudly instead of
   silently producing 30 SOPs of detector noise;
7. waits `settle_s` for the power ramp, then returns the metadata that ends up
   in `summary.csv` and `run_metadata.json`.

```json
"laser": {
  "mode": "cobrite",
  "tolerance_nm": 0.05,
  "settle_s": 0.5,
  "readback_timeout_s": 120.0,
  "poll_interval_s": 0.25,
  "cobrite": {
    "host": "cobrite.local", "port": 2000, "address": "1,1,1",
    "password": "IDP", "timeout_s": 10.0, "min_settle_s": 1.0,
    "power_dbm": null, "enable_output": false,
    "check_power": true, "power_tolerance_db": 3.0
  }
}
```

`address` is the port identifier `<chassis>,<slot>,<device>` (`1,1,1` for a
single-port chassis). `power_dbm: null` leaves the power setpoint untouched;
`enable_output` is off by default so the sweep never switches the laser on by
itself. Endpoint overrides need no config edit:

```bash
python sweep_calibration.py --laser-host 192.168.1.50 --laser-address 1,1,1
python sweep_calibration.py --laser-mode manual      # back to the prompt
```

If the instrument cannot be reached, the sweep stops **before** any acquisition
and returns exit code 2 with the reason. When it can, `sweep_metadata.json` at
the run root records the `*IDN?` string, the port's wavelength limits and the
power/output state observed at the start, and any configured wavelength outside
those limits is reported.

Connectivity check without running a sweep:

```bash
python cobrite.py --host cobrite.local
python cobrite.py --host cobrite.local --set-nm 1550 --power-dbm 10
```

HTTP mode remains available for instruments whose web API is known: set
`laser.mode` to `"http"` and fill in `set_request`, `read_request` and
`response_wavelength_path`. It supports GET/POST/PUT/PATCH and verifies the
read-back; it intentionally contains no guessed vendor URL.

No matrix interpolation is performed. First inspect `summary.csv` and the
per-wavelength matrices; only add interpolation after validating it at
wavelengths not used to fit the interpolation.
