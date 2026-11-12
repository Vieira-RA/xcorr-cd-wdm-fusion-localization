#!/usr/bin/env python3
"""Tiny laser helper: set one wavelength, wait until settled, exit.

Used by ``sweep_trajectory.py`` because the laser needs ``pyserial`` (present in
the phd_env interpreter) while the scope and PM1000 need PyVISA / Python_USB
(present in pm1000_env). One interpreter per job keeps both environments
untouched.

    python laser_set.py 1535.25
"""
from __future__ import annotations

import json
import sys

from cobrite import CoBriteClient


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: laser_set.py <wavelength_nm> [com_port]", file=sys.stderr)
        return 2
    wavelength = float(argv[0])
    com_port = argv[1] if len(argv) > 1 else "COM4"
    client = CoBriteClient(
        {
            "transport": "serial",
            "com_port": com_port,
            "timeout_s": 10.0,
            "min_settle_s": 0.3,
            "poll_interval_s": 0.2,
            "settle_s": 0.0,
        }
    )
    try:
        client.connect()
        client.set_wavelength(wavelength)
        settled = client.wait_until_settled(
            wavelength, tolerance_nm=0.005, timeout_s=90.0, min_settle_s=0.3
        )
        print(json.dumps({
            "wavelength_nm": settled["readback_nm"],
            "settle_s": round(settled["settle_s"], 2),
            "power_dbm": client.actual_power(),
        }))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
