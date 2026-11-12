#!/usr/bin/env python3
"""Long-lived laser agent: wavelengths in on stdin, confirmations out on stdout.

A fine wavelength sweep spends most of its time spawning an interpreter and
re-opening the serial port for every point. This process keeps one SCPI session
open and answers one JSON line per request, so a point costs only the retune.

    python laser_agent.py COM4
    1535.00
    {"wavelength_nm": 1535.0, "settle_s": 1.8, "power_dbm": 10.01}
    1535.02
    ...

Runs in the interpreter that has ``pyserial`` (phd_env); the sweep driver runs in
the interpreter that has PyVISA + Python_USB (pm1000_env).
"""
from __future__ import annotations

import json
import sys

from cobrite import CoBriteClient


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    com_port = argv[0] if argv else "COM4"
    client = CoBriteClient(
        {
            "transport": "serial",
            "com_port": com_port,
            "timeout_s": 10.0,
            "min_settle_s": 0.15,
            "poll_interval_s": 0.1,
            "settle_s": 0.0,
        }
    )
    try:
        client.connect()
        print(json.dumps({"ready": True, "idn": client.idn()}), flush=True)
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            if text.lower() in ("q", "quit", "exit"):
                break
            try:
                wavelength = float(text)
                client.set_wavelength(wavelength)
                settled = client.wait_until_settled(
                    wavelength, tolerance_nm=0.005, timeout_s=90.0, min_settle_s=0.15
                )
                print(json.dumps({
                    "wavelength_nm": settled["readback_nm"],
                    "settle_s": round(settled["settle_s"], 2),
                    "power_dbm": client.actual_power(),
                }), flush=True)
            except Exception as exc:                       # keep the agent alive
                print(json.dumps({"error": "{}: {}".format(type(exc).__name__, exc)}),
                      flush=True)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
