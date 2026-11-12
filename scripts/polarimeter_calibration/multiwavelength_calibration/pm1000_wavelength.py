#!/usr/bin/env python3
"""Find (and then set) the PM1000 wavelength register.

The vendor ``Python_USB`` MATLAB API is a raw ``readpm(addr)`` / ``writepm(addr,
value)`` wrapper: the support files contain no register map and no
wavelength-specific function, and the project notes never identified one. Rather
than guess a register and write to it, this tool finds the register the safe way:

1. take a **snapshot** of the readable register space,
2. change the wavelength **on the PM1000 itself** (its GUI / front panel),
3. take a second snapshot and **diff** them.

Only the register(s) that actually hold the wavelength will change (plus a few
live measurement registers, which the diff flags as noise).

Once the offset is known, ``--set`` writes and verifies it, and the acquisition
script can be given the same offset so the wavelength follows the laser.

Usage
-----
    python pm1000_wavelength.py --dump before.json
    # ... change the wavelength on the PM1000 ...
    python pm1000_wavelength.py --dump after.json
    python pm1000_wavelength.py --diff before.json after.json

    python pm1000_wavelength.py --read 40
    python pm1000_wavelength.py --set 1560 --register 40 --scale 0.01

Register addresses here are **offsets** relative to the PM1000's 512 base, i.e.
what the manual calls the register offset (``readpm(512 + offset)``).

Requires the lab environment (``Python_USB`` + MATLAB Runtime R2021a).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

BASE = 512
DEFAULT_DESCRIPTOR = "PM1000-100M-XL-FA-N20-D 82"
#: Offsets the acquisition code already owns; never written by --set.
RESERVED = set(range(0, 18)) | {45, 92, 226}


def _open_pm():
    import Python_USB
    import matlab

    return Python_USB, matlab


def _init(descriptor: str, usb_type: int):
    Python_USB, matlab = _open_pm()
    pm = Python_USB.initialize()
    ok, diagnosis, handle, in_descr, out_descr = pm.initpm(
        matlab.uint16([usb_type]), descriptor, nargout=5
    )
    if not ok:
        pm.terminate()
        raise SystemExit("initpm failed: {}".format(diagnosis))
    return pm, matlab


def _read_all(pm, matlab, first: int, last: int) -> Dict[int, int]:
    values: Dict[int, int] = {}
    for offset in range(first, last + 1):
        value, ok = pm.readpm(matlab.double([BASE + offset], size=(1, 1)), nargout=2)
        if ok:
            values[offset] = int(value)
    return values


def encode_wavelength(wavelength_nm: float, encoding: str) -> int:
    """Map a wavelength in nm to the register value for the given encoding."""
    wavelength_nm = float(wavelength_nm)
    if wavelength_nm <= 0:
        raise ValueError("wavelength must be positive")
    if encoding == "thz0.01":
        return int(round(299792.458 / wavelength_nm * 100.0))
    if encoding == "thz0.001":
        return int(round(299792.458 / wavelength_nm * 1000.0))
    if encoding == "ghz":
        return int(round(299792.458 / wavelength_nm * 1000.0))
    if encoding == "nm":
        return int(round(wavelength_nm))
    if encoding == "nm0.1":
        return int(round(wavelength_nm * 10.0))
    if encoding == "nm0.01":
        return int(round(wavelength_nm * 100.0))
    raise ValueError("unknown encoding {!r}".format(encoding))


def interpret(offset: int, value: int) -> str:
    """Human hint for a register value that looks like a wavelength/frequency."""
    hints: List[str] = []
    if 1200 <= value <= 1700:
        hints.append("nm (int)")
    if 12000 <= value <= 17000:
        hints.append("nm x10")
    if 120000 <= value <= 170000:
        hints.append("nm x100")
    if 1200000 <= value <= 1700000:
        hints.append("nm x1000")
    if 17600 <= value <= 25000:
        hints.append("THz x100 = {:.2f} THz".format(value / 100.0))
    if 176000 <= value <= 250000:
        hints.append("GHz/10 = {:.1f} GHz".format(value / 10.0 * 0.1))
    if 1760000 <= value <= 2500000:
        hints.append("GHz = {:.1f}".format(value))
    return ", ".join(hints)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--descriptor", default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--usb-type", type=int, default=3, choices=(2, 3))
    parser.add_argument("--first", type=int, default=0, help="first offset (default 0)")
    parser.add_argument("--last", type=int, default=299, help="last offset (default 299)")
    parser.add_argument("--dump", metavar="FILE", help="snapshot all readable registers to JSON")
    parser.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="compare two snapshots")
    parser.add_argument("--read", type=int, metavar="OFFSET", help="read one register offset")
    parser.add_argument("--set-nm", type=float, metavar="NM",
                        help="write a wavelength in nm (encoded per --encoding)")
    parser.add_argument("--set-raw", type=int, metavar="VALUE", help="write a raw 16-bit value")
    parser.add_argument("--register", type=int, help="offset to write")
    parser.add_argument("--encoding", default="thz0.01",
                        choices=("thz0.01", "thz0.001", "ghz", "nm", "nm0.1", "nm0.01"),
                        help="how a wavelength maps to the register (default thz0.01)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.diff:
        try:
            before = json.load(open(args.diff[0]))
            after = json.load(open(args.diff[1]))
        except Exception as exc:
            print("Cannot read snapshots: {}".format(exc), file=sys.stderr)
            return 2
        shared = sorted(set(before) & set(after), key=lambda key: int(key))
        changed = [key for key in shared if before[key] != after[key]]
        print("Compared {} registers, {} changed:".format(len(shared), len(changed)))
        if not changed:
            print("  nothing changed - did the wavelength really change on the PM1000?")
        for key in changed:
            hint = interpret(int(key), int(after[key]))
            print("  offset %-4s %6s -> %-6s %s" % (
                key, before[key], after[key], hint
            ))
        return 0

    pm, matlab = _init(args.descriptor, args.usb_type)
    try:
        if args.dump:
            values = _read_all(pm, matlab, args.first, args.last)
            with open(args.dump, "w", encoding="utf-8") as handle:
                json.dump({str(k): v for k, v in sorted(values.items())}, handle, indent=1)
            print("Snapshot: {} registers -> {}".format(len(values), args.dump))
            return 0

        if args.read is not None:
            value, ok = pm.readpm(
                matlab.double([BASE + args.read], size=(1, 1)), nargout=2
            )
            print("offset {} = {} (ok={}) {}".format(
                args.read, int(value), bool(ok), interpret(args.read, int(value))
            ))
            return 0

        if args.set_nm is not None or args.set_raw is not None:
            if args.register is None:
                print("--set-nm/--set-raw need --register (find it with --dump/--diff).",
                      file=sys.stderr)
                return 2
            if args.register in RESERVED:
                print("Refusing to write offset {}: it belongs to ATE/Stokes/trigger."
                      .format(args.register), file=sys.stderr)
                return 2
            raw = (int(args.set_raw) if args.set_raw is not None
                   else encode_wavelength(args.set_nm, args.encoding))
            if not 0 <= raw <= 0xFFFF:
                print("Value {} does not fit 16 bits; check --encoding.".format(raw),
                      file=sys.stderr)
                return 2
            # The PM1000 ignores a uint16 payload on this register and accepts a
            # double one, so always write the value as matlab.double.
            ok = pm.writepm(
                matlab.double([BASE + args.register], size=(1, 1)),
                matlab.double([raw], size=(1, 1)),
            )
            value, good = pm.readpm(
                matlab.double([BASE + args.register], size=(1, 1)), nargout=2
            )
            print("wrote {} to offset {} -> readback {} (write_ok={}, read_ok={}){}".format(
                raw, args.register, int(value), bool(ok), bool(good),
                "" if int(value) == raw else "   <== DID NOT STICK"
            ))
            return 0
    finally:
        try:
            pm.closepm()
        except Exception:
            pass
        try:
            pm.terminate()
        except Exception:
            pass

    print("Nothing to do: use --dump/--diff, --read or --set.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
