#!/usr/bin/env python3
"""
Compare several runs (same Jones data / noise seeds), e.g. no-filter vs
low-pass vs band-pass.

Shows that a linear filter does not move the CRB but changes the threshold SNR
at which the estimator reaches it.

Run:  python compare_filters.py cfg_none.yaml cfg_lowpass.yaml cfg_bandpass.yaml
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from crb import star_combined_crb_distance

plt.rcParams.update({"font.size": 13, "figure.dpi": 150, "savefig.dpi": 150,
                     "savefig.bbox": "tight"})


def load_run(run_dir, count):
    return np.load(os.path.join(run_dir, "crb_results_%dch.npz" % count),
                   allow_pickle=True)


def main(configs, count=8, bw_frac=0.8):
    fig, ax = plt.subplots(figsize=(8.4, 6))
    d0 = None
    for cfg_path in configs:
        cfg = yaml.safe_load(open(cfg_path))
        run_dir = cfg["output"]["run_dir"]
        mode = cfg.get("filter", {}).get("mode", "?")
        d = load_run(run_dir, count)
        d0 = d
        BW = d["BANDWIDTHS"]
        ib = int(len(BW) * bw_frac)
        std = np.nanmean(np.nanstd(d["est_crb"], axis=-1), axis=0)[ib]
        ax.loglog(d["SNRS_dB"], std, "o-", ms=4, label="sim: %s" % mode)

    # CRB (star) from the first run
    cd, D, pairs = d0["crb_delay_pairs"], d0["D_abs"], d0["pair_indices"]
    nf = cd.shape[0]
    star = np.array([
        np.nanmean([star_combined_crb_distance(cd[f, ib, s], D, pairs, count)
                    for f in range(nf)])
        for s in range(len(d0["SNRS_dB"]))])
    ax.loglog(d0["SNRS_dB"], star, "k--", label="CRB (star-combined)")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Std (km)")
    ax.set_title("%d ch | filter comparison, B=%.3g Hz" % (count, BW[ib]))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = os.path.join(yaml.safe_load(open(configs[0]))["output"]["run_dir"],
                       "fig_%dch_filter_compare.png" % count)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    cfgs = sys.argv[1:] if len(sys.argv) > 1 else [
        "sweep_config_crb_test.yaml",
        "sweep_config_crb_test_lowpass.yaml",
        "sweep_config_crb_test_bandpass.yaml",
    ]
    main(cfgs)
