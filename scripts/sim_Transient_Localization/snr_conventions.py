"""SNR-axis conversions for CRB sweep data.

The current convention is the mean energy SNR per real signal component,
``mean_c(sum_n x[n,c]**2) / sigma2``.  Older production files stored the
legacy ratio ``mean_t(sum_c x[t,c]**2) / sigma2`` instead.  Their conversion
is ``N / n_components`` and is an axis-only change: the saved noise variance,
estimates, and CRBs remain unchanged.
"""

import numpy as np


ENERGY_SNR_DEFINITION = "mean_per_component_energy_over_noise_variance"
LEGACY_SNR_DEFINITION = "mean_total_sample_power_over_component_noise_variance"


def samples_per_record_from_config(cfg):
    """Return the generated sample count for the scale-invariant time grid."""
    time_cfg = cfg["time"]
    fs_factor = float(time_cfg["fs_factor"])
    t_end_factor = float(time_cfg["t_end_factor"])
    # sigma_t = 0.3748 / B and fs = fs_factor * B, so B cancels.
    return int(t_end_factor * 0.3748 * fs_factor) + 1


def energy_snr_axis(saved_snr_db, cfg, snr_definition=None, n_samples=None,
                    n_components=2):
    """Convert a stored SNR axis to mean per-component energy SNR in dB.

    Missing convention metadata denotes the legacy production convention.
    New data explicitly tagged with ``ENERGY_SNR_DEFINITION`` are unchanged.
    """
    if snr_definition is None:
        snr_definition = cfg.get("noise", {}).get("snr_definition")
    values = np.asarray(saved_snr_db, dtype=float)
    if snr_definition == ENERGY_SNR_DEFINITION:
        return values
    if snr_definition not in (None, LEGACY_SNR_DEFINITION):
        raise ValueError("Unsupported saved SNR definition: %s" % snr_definition)
    if n_samples is None:
        n_samples = samples_per_record_from_config(cfg)
    offset_db = 10.0 * np.log10(float(n_samples) / float(n_components))
    return values + offset_db


def energy_snr_axis_from_result(result, saved_snr_db, cfg_override=None):
    """Convert the SNR axis using metadata in an open ``np.load`` result.

    ``cfg_override`` avoids unpickling the old ``cfg`` object stored in some
    production NPZ files (those files were written with a newer NumPy pickle
    module path than the Python 3.8 analysis environment provides).
    """
    files = result.files
    cfg = cfg_override or {}
    if not cfg and "cfg" in files:
        try:
            cfg = result["cfg"].item()
        except (ModuleNotFoundError, AttributeError, ValueError):
            cfg = {}
    # Old production files predate the top-level convention marker and were
    # generated with the mean-total-sample-power sweep.  Do not infer their
    # convention from a YAML file that may since have been updated.
    definition = (str(result["snr_definition"].item())
                  if "snr_definition" in files else LEGACY_SNR_DEFINITION)
    n_samples = None
    if "n_samples" in files:
        n_samples = int(np.nanmedian(result["n_samples"]))
    return energy_snr_axis(saved_snr_db, cfg, definition, n_samples)


def energy_snr_axis_for_run(run_dir, count, saved_snr_db):
    """Convert an estimator-comparison axis using its raw CRB result metadata."""
    import os
    import glob
    import yaml

    path = os.path.join(run_dir, "crb_results_%dch.npz" % int(count))
    cfg_override = {}
    config_dir = os.path.dirname(os.path.abspath(__file__))
    target = os.path.normcase(os.path.basename(os.path.normpath(run_dir)))
    for config_path in glob.glob(os.path.join(config_dir, "sweep_config_crb*.yaml")):
        with open(config_path, "r") as stream:
            candidate = yaml.safe_load(stream)
        configured_run = candidate.get("output", {}).get("run_dir", "")
        if os.path.normcase(os.path.basename(os.path.normpath(configured_run))) == target:
            cfg_override = candidate
            break
    with np.load(path, allow_pickle=True) as result:
        return energy_snr_axis_from_result(result, saved_snr_db, cfg_override)
