#!/usr/bin/env python3
"""Find where the CoBrite laser answers, over USB (RNDIS/COM) or Ethernet.

Why this exists
---------------
The CoBrite speaks SCPI over TCP port 2000, but "which IP?" depends on how the
box is plugged in (manual, section 3 and 4.3.1):

* **Physical Ethernet** — factory default is a fixed **192.168.0.1**, DHCP off.
  The host must be on 192.168.0.x.
* **USB** — Windows installs three devices: a storage volume, a *virtual
  Ethernet interface (RNDIS)* and a *virtual COM port*.  The USB virtual
  Ethernet has a **different IP** than the RJ45 port; read it with
  ``USBIPADDR?`` over the virtual COM port, or from the touch panel
  (Device Config -> Network Config shows both addresses).
* ``cobrite.local`` (mDNS) works only if the host resolves that name — the
  manual explicitly warns it often does not.

So this script inspects the host's adapters, tries the known names/addresses,
optionally scans the local /24s on port 2000, and lists serial ports.  It never
changes anything on the instrument: every connection is ``*IDN?`` and close.

Usage
-----
    python discover_laser.py                 # adapters + names + scan + COM ports
    python discover_laser.py --no-scan       # skip the port scan (fast)
    python discover_laser.py --subnet 192.168.137.0/24 --subnet 169.254.1.0/24
    python discover_laser.py --json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import json
import os
import platform
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

SCPI_PORT = 2000
IDN_COMMAND = b"*IDN?;"
KNOWN_HOSTNAMES = ("cobrite.local",)
#: 192.168.0.1 = factory-default RJ45 address; 169.254.9.2 = this lab's unit on
#: the USB RNDIS interface (link-local, no DHCP); 169.254.1.1 = common RNDIS default.
KNOWN_ADDRESSES = ("192.168.0.1", "169.254.9.2", "169.254.1.1")
USB_HINTS = ("rndis", "usb", "remote ndis", "id photonics", "cobrite")


# --------------------------------------------------------------------- host info
def _run(command: List[str]) -> str:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=15, errors="replace"
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout or ""


def list_adapters() -> List[Dict[str, Any]]:
    """Return ``[{name, ip, netmask, kind}]`` for the host's IPv4 interfaces."""
    adapters: List[Dict[str, Any]] = []
    system = platform.system().lower()
    if system == "windows":
        output = _run(["ipconfig"])
        current: Optional[Dict[str, Any]] = None
        for raw_line in output.splitlines():
            line = raw_line.rstrip()
            if not line:
                continue
            if not line.startswith(" ") and line.endswith(":"):
                if current:
                    adapters.append(current)
                current = {"name": line[:-1].strip(), "ip": None, "netmask": None}
                continue
            if current is None or ":" not in line:
                continue
            label, _, value = line.partition(":")
            label = label.strip().lower()
            value = value.strip()
            if "ipv4" in label:
                current["ip"] = value.split("(")[0].strip()
            elif "subnet mask" in label or "subnetmask" in label:
                current["netmask"] = value
        if current:
            adapters.append(current)
    else:
        output = _run(["ip", "-4", "-o", "addr", "show"])
        for line in output.splitlines():
            fields = line.split()
            if len(fields) < 4 or fields[2] != "inet":
                continue
            interface = fields[1]
            address = fields[3]
            ip_text, _, prefix = address.partition("/")
            try:
                netmask = str(ipaddress.IPv4Network("0.0.0.0/{}".format(prefix)).netmask)
            except ValueError:
                netmask = None
            adapters.append({"name": interface, "ip": ip_text, "netmask": netmask})

    for adapter in adapters:
        name = (adapter.get("name") or "").lower()
        adapter["kind"] = "usb/rndis" if any(hint in name for hint in USB_HINTS) else "other"
    return [adapter for adapter in adapters if adapter.get("ip")]


def candidate_subnets(adapters: List[Dict[str, Any]], extra: List[str]) -> List[ipaddress.IPv4Network]:
    networks: List[ipaddress.IPv4Network] = []
    for text in extra:
        try:
            networks.append(ipaddress.IPv4Network(text, strict=False))
        except ValueError:
            print("Ignoring invalid subnet {!r}.".format(text), file=sys.stderr)
    for adapter in adapters:
        try:
            interface = ipaddress.IPv4Interface(
                "{}/{}".format(adapter["ip"], adapter.get("netmask") or "255.255.255.0")
            )
        except ValueError:
            continue
        # Keep the probe cheap: never scan wider than a /24.
        if interface.network.prefixlen < 24:
            networks.append(ipaddress.IPv4Network(
                "{}/24".format(interface.ip), strict=False
            ))
        else:
            networks.append(interface.network)
    # De-duplicate, keep order.
    seen = set()
    unique = []
    for network in networks:
        if network not in seen:
            seen.add(network)
            unique.append(network)
    return unique


# ------------------------------------------------------------------ SCPI probing
def probe_tcp(host: str, port: int = SCPI_PORT, timeout: float = 0.4) -> Optional[str]:
    """Return the ``*IDN?`` string if a SCPI instrument answers, else ``None``."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            connection.settimeout(max(timeout, 1.0))
            connection.sendall(IDN_COMMAND)
            payload = b""
            while not payload.strip().endswith(b";"):
                chunk = connection.recv(512)
                if not chunk:
                    break
                payload += chunk
            text = payload.decode("ascii", errors="replace").strip().rstrip(";").strip()
            return text or "(empty response)"
    except OSError:
        return None


def list_serial_ports() -> List[str]:
    ports: List[str] = []
    if platform.system().lower() == "windows":
        try:
            import winreg  # type: ignore

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM"
            )
            index = 0
            while True:
                try:
                    _, value, _ = winreg.EnumValue(key, index)
                except OSError:
                    break
                ports.append(str(value))
                index += 1
        except Exception:
            pass
    else:
        import glob

        ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    return ports


def probe_serial(port: str, timeout: float = 1.5) -> Optional[str]:
    """Try ``*IDN?`` over a serial SCPI session (needs pyserial)."""
    try:
        import serial  # type: ignore
    except ImportError:
        return None
    for baudrate in (115200, 9600, 230400):
        try:
            with serial.Serial(port, baudrate=baudrate, timeout=timeout) as handle:
                handle.write(IDN_COMMAND)
                text = handle.read(512).decode("ascii", errors="replace")
                text = text.strip().rstrip(";").strip()
                if text:
                    return "{} @ {} baud".format(text, baudrate)
        except Exception:
            continue
    return None


def scan(networks: List[ipaddress.IPv4Network], timeout: float, workers: int) -> List[Tuple[str, str]]:
    hosts: List[str] = []
    for network in networks:
        hosts.extend(str(host) for host in network.hosts())
    if not hosts:
        return []
    print("Scanning {} addresses on port {} ...".format(len(hosts), SCPI_PORT))
    found: List[Tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(probe_tcp, host, SCPI_PORT, timeout): host for host in hosts}
        for future in concurrent.futures.as_completed(futures):
            host = futures[future]
            try:
                identity = future.result()
            except Exception:
                identity = None
            if identity:
                found.append((host, identity))
                print("  found {}: {}".format(host, identity))
    return sorted(found)


# ------------------------------------------------------------------------ main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Locate a CoBrite laser over USB (RNDIS/COM) or Ethernet."
    )
    parser.add_argument("--subnet", action="append", default=[],
                        help="extra IPv4 subnet to scan, e.g. 192.168.137.0/24 (repeatable)")
    parser.add_argument("--no-scan", action="store_true", help="skip the port scan")
    parser.add_argument("--timeout", type=float, default=0.4, help="per-host TCP timeout")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {"adapters": [], "names": {}, "known": {}, "found": [], "serial": {}}

    adapters = list_adapters()
    report["adapters"] = adapters
    if not args.json:
        print("Host IPv4 interfaces")
        if not adapters:
            print("  (none detected)")
        for adapter in adapters:
            marker = " <-- USB/RNDIS" if adapter["kind"] == "usb/rndis" else ""
            print("  {:<38s} {:<15s} {}".format(
                adapter["name"][:38], adapter["ip"], adapter.get("netmask") or ""
            ) + marker)

    for hostname in KNOWN_HOSTNAMES:
        try:
            address = socket.gethostbyname(hostname)
            report["names"][hostname] = address
            if not args.json:
                print("\n{} resolves to {}".format(hostname, address))
        except OSError:
            report["names"][hostname] = None
            if not args.json:
                print("\n{} does not resolve on this host".format(hostname))

    candidates = list(KNOWN_ADDRESSES)
    candidates += [address for address in report["names"].values() if address]
    for candidate in candidates:
        identity = probe_tcp(candidate, SCPI_PORT, max(args.timeout, 0.6))
        report["known"][candidate] = identity
        if identity and not args.json:
            print("  SCPI at {}: {}".format(candidate, identity))

    if not args.no_scan:
        networks = candidate_subnets(adapters, args.subnet)
        if not args.json and networks:
            print("\nCandidate subnets: {}".format(", ".join(str(n) for n in networks)))
        report["found"] = [list(item) for item in scan(networks, args.timeout, args.workers)]

    ports = list_serial_ports()
    report["serial"]["ports"] = ports
    report["serial"]["probes"] = {}
    if ports and not args.json:
        print("\nSerial ports: {}".format(", ".join(ports)))
    for port in ports:
        identity = probe_serial(port)
        report["serial"]["probes"][port] = identity
        if identity and not args.json:
            print("  SCPI on {}: {}".format(port, identity))

    working = [host for host, identity in report["known"].items() if identity]
    working += [host for host, _ in report["found"]]
    if not args.json:
        print("\n" + "=" * 68)
        if working:
            host = working[0]
            identity = report["known"].get(host) or next(
                (identity for candidate, identity in report["found"] if candidate == host), ""
            )
            if identity:
                print("Use: {}".format(identity))
            print("Laser endpoint: {}:{}\n".format(host, SCPI_PORT))
            print("Next step:\n  python cobrite.py --host {}\n".format(host))
            print("Config snippet:\n  \"cobrite\": {{\"host\": \"{}\", \"port\": 2000, "
                  "\"address\": \"1,1,1\"}}".format(host))
        elif report["serial"]["probes"]:
            print("No SCPI port answered, but a serial port looks like an instrument.")
            print("Tell me and we will add serial transport (pyserial) to cobrite.py.")
        else:
            print("Nothing found.")
            print("Checks: is the USB cable in the 'USB IN' socket? Does Device Manager")
            print("show the RNDIS adapter and the COM port? Is the RJ45 side on")
            print("192.168.0.x (laser default 192.168.0.1)? Try --subnet <your subnet>.")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if working else 1


if __name__ == "__main__":
    sys.exit(main())
