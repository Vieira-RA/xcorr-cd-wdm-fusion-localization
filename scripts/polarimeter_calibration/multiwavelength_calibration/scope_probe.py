#!/usr/bin/env python3
"""Check the detector channels for clipping and pick a sane CH_SCALES set.

This is a read-only scope check that mirrors the acquisition settings in
``../fused_acquisition_controlled.py``: freeze the acquisition, read each
detector channel, and report the peak level against the +/-4-division headroom
implied by the per-channel vertical scale.

It answers three questions before a calibration run:

1. Does any channel clip at the current scale?  (clipping destroys information
   that the least-squares fit cannot recover)
2. Is the signal far enough above the digitizer floor?  The fit uses the mean of
   a 10 000-point record, so the relevant figure is the standard error of that
   mean, not a single sample.
3. Which vertical scales would fit the observed peaks best?

Usage
-----
    python scope_probe.py                                  # current 20 mV/div
    python scope_probe.py --ch-scales 0.02,0.05,0.02,0.02
    python scope_probe.py --scope-ip 198.192.1.1 --json

Requires the transport set up in ``scope_io.py`` (VXI-11 or PyVISA) and a host
NIC on the scope's subnet, or a USB-TMC resource.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import os
from pathlib import Path
import json
import math
import socket
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

SCOPE_IP = "198.192.1.1"
CHANNELS = (1, 2, 3, 4)
DEFAULT_SCALES = {1: 0.02, 2: 0.02, 3: 0.02, 4: 0.02}   # matches the parent script
HORIZONTAL_SCALE = 10e-6      # 100 us window over 10 divisions
RECORD_LENGTH = 10000         # 10 ns/sample, matched to the PM1000
SCOPE_TIMEOUT = 30.0
DIVISIONS = 8.0               # DPO7254 vertical divisions, position 0 => +/-4
TARGET_HEADROOM = 0.7         # aim for peak <= 70% of full scale
E_SERIES = (1.0, 2.0, 5.0)

#: Ports that identify a LAN instrument: 111 = VXI-11/rpcbind (what vxi11 uses),
#: 5025 = SCPI raw socket, 4000 = VXI-11 core channel on some Tektronix units.
INSTRUMENT_PORTS = (111, 5025, 4000)

# The scope access layer lives next to the acquisition script in the parent folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scope_io import open_scope, ScopeError  # noqa: E402


def probe_host(host: str, ports: Tuple[int, ...] = INSTRUMENT_PORTS,
               timeout: float = 0.25) -> List[int]:
    """Return the subset of ``ports`` that accept a TCP connection on ``host``."""
    open_ports = []
    for port in ports:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                open_ports.append(port)
        except OSError:
            continue
    return open_ports


def discover(subnet: str, timeout: float = 0.25, workers: int = 64) -> List[Dict[str, Any]]:
    """Scan a subnet for LAN instruments and report their identity if possible."""
    network = ipaddress.IPv4Network(subnet, strict=False)
    hosts = [str(host) for host in network.hosts()]
    if not hosts:
        return []
    print("Scanning {} hosts on ports {} ...".format(len(hosts), ", ".join(map(str, INSTRUMENT_PORTS))))

    findings: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(probe_host, host, INSTRUMENT_PORTS, timeout): host for host in hosts}
        for future in concurrent.futures.as_completed(futures):
            host = futures[future]
            try:
                open_ports = future.result()
            except Exception:
                continue
            if open_ports:
                findings.append({"host": host, "open_ports": open_ports, "idn": None})
                print("  {}: open ports {}".format(host, open_ports))

    try:
        import vxi11
    except ImportError:
        vxi11 = None

    if vxi11 is not None:
        for finding in findings:
            if 111 not in finding["open_ports"]:
                continue
            try:
                instrument = vxi11.Instrument(finding["host"])
                instrument.timeout = 5.0
                identity = instrument.ask("*IDN?")
                instrument.close()
                finding["idn"] = identity.strip()
                print("  {} -> {}".format(finding["host"], finding["idn"]))
            except Exception as exc:
                print("  {} -> VXI-11 present but *IDN? failed ({})".format(finding["host"], exc))
    return findings


def _read_curve(scope) -> bytes:
    """Read the current channel's CURVE? block and return raw int8 samples."""
    return scope.read_block("CURVE?")


def _read_channel(scope, channel: int) -> Dict[str, float]:
    import numpy as np

    scope.write("DATA:SOURCE CH{}".format(channel))
    y_mult = float(scope.ask("WFMOUTPRE:YMULT?"))
    y_zero = float(scope.ask("WFMOUTPRE:YZERO?"))
    y_off = float(scope.ask("WFMOUTPRE:YOFF?"))
    raw = _read_curve(scope)
    codes = np.frombuffer(raw, dtype=np.int8).astype(float)
    volts = (codes - y_off) * y_mult + y_zero
    samples = volts.size
    mean = float(np.mean(volts))
    std = float(np.std(volts))
    standard_error = std / np.sqrt(samples) if samples else float("nan")
    return {
        "samples": int(samples),
        "mean_v": mean,
        "std_v": std,
        "min_v": float(np.min(volts)),
        "max_v": float(np.max(volts)),
        "peak_v": float(np.max(np.abs(volts))),
        "standard_error_v": standard_error,
        "rail_low": int(np.sum(codes <= -127)),
        "rail_high": int(np.sum(codes >= 127)),
    }


def _nice_scale(minimum: float) -> float:
    """Smallest 1-2-5 value >= ``minimum`` (the DPO series)."""
    if minimum <= 0:
        return 0.001
    decade = 10.0 ** math.floor(math.log10(minimum))
    for factor in E_SERIES:
        candidate = factor * decade
        if candidate >= minimum - 1e-12:
            return round(min(candidate, 10.0), 6)
    return round(min(10.0 * decade, 10.0), 6)


def analyse(measurements: Dict[int, Dict[str, float]], scales: Dict[int, float]) -> Dict[str, Any]:
    rows = []
    suggested = {}
    for channel in CHANNELS:
        data = measurements[channel]
        scale = float(scales[channel])
        full_scale = (DIVISIONS / 2.0) * scale          # +/- 4 divisions
        peak = data["peak_v"]
        usage = peak / full_scale if full_scale else float("inf")
        clipped = bool(data["rail_low"] or data["rail_high"]) or usage >= 0.98
        snr = (
            abs(data["mean_v"]) / data["standard_error_v"]
            if data["standard_error_v"] > 0
            else float("inf")
        )
        # Scale that puts the observed peak at TARGET_HEADROOM of full scale.
        wanted = peak / (TARGET_HEADROOM * (DIVISIONS / 2.0)) if peak > 0 else 0.0
        suggested[channel] = min(max(_nice_scale(wanted), 0.001), 10.0)
        rows.append(
            {
                "channel": channel,
                "scale_v_per_div": scale,
                "full_scale_v": full_scale,
                "peak_v": peak,
                "usage_fraction": usage,
                "clipped": clipped,
                "mean_v": data["mean_v"],
                "std_v": data["std_v"],
                "standard_error_v": data["standard_error_v"],
                "snr_of_mean": snr,
                "samples": data["samples"],
                "suggested_scale_v_per_div": suggested[channel],
            }
        )
    worst_usage = max(row["usage_fraction"] for row in rows)
    return {
        "rows": rows,
        "worst_usage_fraction": worst_usage,
        "any_clipped": any(row["clipped"] for row in rows),
        "suggested_scales": suggested,
        "scale_ok": worst_usage <= TARGET_HEADROOM * 1.15,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope-ip", default=os.environ.get("PM_SCOPE_IP", SCOPE_IP))
    parser.add_argument("--ch-scales", default=None,
                        help="comma-separated V/div for CH1..CH4 (default 0.02 each)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--keep-running", action="store_true",
                        help="leave the acquisition stopped instead of resuming it")
    parser.add_argument("--discover", nargs="?", const="", default=None, metavar="SUBNET",
                        help="scan a subnet (e.g. 198.192.1.0/24) for the scope instead of "
                             "probing --scope-ip; needs no vxi11 for the port scan")
    return parser


def _parse_scales(text: Optional[str]) -> Dict[int, float]:
    if not text:
        return dict(DEFAULT_SCALES)
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) == 1:
        parts = parts * len(CHANNELS)
    if len(parts) != len(CHANNELS):
        raise SystemExit("--ch-scales needs 1 or 4 comma-separated V/div values.")
    return dict(zip(CHANNELS, (float(part) for part in parts)))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.discover is not None:
        subnet = args.discover or "198.192.1.0/24"
        findings = discover(subnet, timeout=0.25)
        if args.json:
            print(json.dumps(findings, indent=2, sort_keys=True))
        elif not findings:
            print("\nNothing answered on {}.".format(subnet))
            print("Checks: is the cable in the scope's LAN port and the host NIC enabled?")
            print("Is the host NIC on the same subnet (198.192.1.x/24)?")
            print("Does LinkSpeed change when you unplug the cable? That identifies the right NIC.")
        return 0 if findings else 1

    scales = _parse_scales(args.ch_scales)
    try:
        scope = open_scope(args.scope_ip, timeout=SCOPE_TIMEOUT)
        print(scope.ask("*IDN?").strip())
    except ScopeError as exc:
        print("Cannot open the oscilloscope: {}".format(exc), file=sys.stderr)
        print("Hint: without admin rights use the PyVISA path, e.g.", file=sys.stderr)
        print("  set PM_SCOPE_TRANSPORT=pyvisa", file=sys.stderr)
        print("  set PM_SCOPE_IP=169.254.102.200", file=sys.stderr)
        print("or a USB-TMC resource:", file=sys.stderr)
        print("  set PM_SCOPE_RESOURCE=USB0::0x0699::...::INSTR", file=sys.stderr)
        return 2

    try:
        for channel in CHANNELS:
            scope.write("CH{}:STATE ON".format(channel))
            scope.write("CH{}:SCALE {}".format(channel, scales[channel]))
            scope.write("CH{}:POSITION 0.0".format(channel))
        scope.write("HORIZONTAL:SCALE {}".format(HORIZONTAL_SCALE))
        scope.write("HORIZONTAL:RECORDLENGTH {}".format(RECORD_LENGTH))
        scope.write("DATA:ENC SRI")
        scope.write("DATA:WIDTH 1")
        scope.write("WFMOUTPRE:BYT_NR 1")
        scope.write("DATA:START 1")
        scope.write("DATA:STOP 1e10")
        points = int(scope.ask("WFMOUTPRE:NR_PT?"))
        scope.write("DATA:STOP {}".format(points))
        scope.write("TRIGGER:A:MODE AUTO")
        scope.write("ACQUIRE:STOPAFTER RUNSTOP")
        scope.write("ACQUIRE:STATE RUN")
        time.sleep(1.0)

        scope.write("ACQUIRE:STATE STOP")
        measurements = {channel: _read_channel(scope, channel) for channel in CHANNELS}
        if not args.keep_running:
            scope.write("ACQUIRE:STATE RUN")
    finally:
        scope.close()

    result = analyse(measurements, scales)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not result["any_clipped"] else 1

    print("\nDetector channel headroom at the configured vertical scales")
    print("=" * 96)
    print("{:>3s} {:>12s} {:>11s} {:>11s} {:>9s} {:>11s} {:>9s} {:>6s}".format(
        "ch", "V/div", "full scale", "peak", "usage", "mean", "SNR(mean)", "clip"
    ))
    for row in result["rows"]:
        print("{:>3d} {:>12.1f} {:>11.1f} {:>11.1f} {:>8.1f}% {:>11.2f} {:>9.1f} {:>6s}".format(
            row["channel"],
            row["scale_v_per_div"] * 1e3,
            row["full_scale_v"] * 1e3,
            row["peak_v"] * 1e3,
            row["usage_fraction"] * 100.0,
            row["mean_v"] * 1e3,
            row["snr_of_mean"],
            "YES" if row["clipped"] else "no",
        ))
    print("=" * 96)
    print("(mV, except SNR; full scale = +/-(8/2) divisions around CH:POSITION 0)")

    if result["any_clipped"]:
        print("\nCLIPPING DETECTED - increase the scale of the affected channel(s).")
    elif result["scale_ok"]:
        print("\nHeadroom is fine: worst channel uses {:.0f}% of full scale.".format(
            result["worst_usage_fraction"] * 100.0
        ))
    else:
        print("\nNo clipping, but the worst channel uses {:.0f}% of full scale;".format(
            result["worst_usage_fraction"] * 100.0
        ))
        print("consider a larger scale for more margin (see the suggestion below).")

    suggestion = result["suggested_scales"]
    print("\nSuggested scales (peak at ~{:.0f}% of full scale):".format(TARGET_HEADROOM * 100.0))
    print("  --ch-scales {}".format(",".join("{:g}".format(suggestion[c]) for c in CHANNELS)))
    print("  \"ch_scales\": [{}]".format(
        ", ".join("{:g}".format(suggestion[c]) for c in CHANNELS)
    ))
    if abs(result["worst_usage_fraction"] - TARGET_HEADROOM) < 0.35:
        print("\nThe current scales are already reasonable - no change needed.")
    return 0 if not result["any_clipped"] else 1


if __name__ == "__main__":
    sys.exit(main())
