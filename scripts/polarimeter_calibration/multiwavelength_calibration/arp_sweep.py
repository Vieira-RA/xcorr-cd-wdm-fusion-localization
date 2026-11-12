#!/usr/bin/env python3
"""Find whatever is on a directly-connected Ethernet segment, without its IP.

Why this is needed
------------------
If you do not know an instrument's IP address, port scanning only helps when you
guess the right subnet *and* the instrument's firewall lets the probe through.
Ping is even weaker: a Windows-based scope routinely drops ICMP while its SCPI
service is perfectly alive.

This tool works one layer lower. It sends a single UDP datagram to every address
in a subnet, which forces the host to emit an ARP request for each address. Any
device on the wire answers ARP **for its own address only**, so the ARP cache
afterwards lists every live neighbour with its IP and MAC — no matter what IP it
was configured with, and no matter what its firewall does with IP traffic.

It touches only the requested wired subnet; nothing is broadcast or routed.

Usage
-----
    python arp_sweep.py                                  # 198.192.1.0/24 (default)
    python arp_sweep.py --subnet 192.168.0.0/24
    python arp_sweep.py --iface "Ethernet 15" --probe-ports

The optional ``--probe-ports`` step tests 111 (VXI-11/rpcbind), 5025 (SCPI raw)
and 4000 on each neighbour that was discovered.
"""
from __future__ import annotations

import argparse
import ipaddress
import re
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_SUBNET = "198.192.1.0/24"
INSTRUMENT_PORTS = (111, 5025, 4000)
_ARP_LINE = re.compile(
    r"^\s*(?P<ip>\d{1,3}(?:\.\d{1,3}){3})\s+"
    r"(?P<mac>(?:[0-9a-fA-F]{2}[-:]){5}[0-9a-fA-F]{2})\s+"
    r"(?P<kind>\w+)",
    re.MULTILINE,
)
_INTERFACE_LINE = re.compile(r"Interface:\s*(?P<ip>\d{1,3}(?:\.\d{1,3}){3})", re.MULTILINE)
_ARP_LINE_POSIX = re.compile(
    r"^\s*\S+\s+\((?P<ip>\d{1,3}(?:\.\d{1,3}){3})\)\s+at\s+"
    r"(?P<mac>(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2})",
    re.MULTILINE,
)


def parse_arp_table(text: str, subnet: Optional[ipaddress.IPv4Network] = None) -> List[Dict[str, Any]]:
    """Parse the output of ``arp -a`` into ``[{ip, mac, kind, interface_ip}]``.

    Broadcast/multicast placeholder rows and the interface's own addresses are
    dropped, so the result is the set of real neighbours.
    """
    entries: List[Dict[str, Any]] = []

    # Walk the text in order so each entry is attributed to its interface block.
    current_interface: Optional[str] = None
    for raw_line in text.splitlines():
        interface_match = _INTERFACE_LINE.search(raw_line)
        if interface_match:
            current_interface = interface_match.group("ip")
            continue
        match = _ARP_LINE.match(raw_line)
        if not match:
            posix = _ARP_LINE_POSIX.match(raw_line)
            if not posix:
                continue
            match = posix
            kind = "on-link"
        else:
            kind = match.group("kind")
        address = match.group("ip")
        mac = match.group("mac").lower().replace(":", "-")
        if mac in ("ff-ff-ff-ff-ff-ff", "00-00-00-00-00-00"):
            continue
        try:
            parsed = ipaddress.IPv4Address(address)
        except ValueError:
            continue
        if parsed.is_multicast or parsed.is_unspecified:
            continue
        if address.endswith(".255"):
            continue
        if subnet is not None and parsed not in subnet:
            continue
        if current_interface and address == current_interface:
            continue
        entries.append(
            {
                "ip": address,
                "mac": mac,
                "kind": kind,
                "interface_ip": current_interface,
            }
        )
    # De-duplicate by (ip, mac), keeping the first occurrence.
    seen = set()
    unique = []
    for entry in entries:
        key = (entry["ip"], entry["mac"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def read_arp_table() -> str:
    """Return the local ARP/neighbour table as text (Windows or POSIX)."""
    try:
        completed = subprocess.run(
            ["arp", "-a"], capture_output=True, text=True, timeout=20, errors="replace"
        )
        return completed.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _local_addresses() -> Iterable[str]:
    """Best-effort list of this host's IPv4 addresses (to filter our own rows)."""
    addresses = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError:
        pass
    return addresses


def arp_sweep(subnet: ipaddress.IPv4Network, udp_port: int = 9, wait_s: float = 2.0) -> List[Dict[str, Any]]:
    """Force an ARP lookup for every address in ``subnet`` and read the results."""
    hosts = [str(host) for host in subnet.hosts()]
    if not hosts:
        return []
    print("ARP-sweeping {} addresses in {} (UDP port {}, then reading the ARP table) ...".format(
        len(hosts), subnet, udp_port
    ))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for host in hosts:
            try:
                sock.sendto(b"", (host, udp_port))
            except OSError:
                # No route / unreachable is expected for absent hosts.
                continue
    finally:
        sock.close()
    time.sleep(max(0.2, float(wait_s)))

    own = set(_local_addresses())
    neighbours = [
        entry for entry in parse_arp_table(read_arp_table(), subnet) if entry["ip"] not in own
    ]
    return neighbours


def probe_ports(host: str, ports: Tuple[int, ...] = INSTRUMENT_PORTS, timeout: float = 0.4) -> List[int]:
    """Return which of ``ports`` accept a TCP connection on ``host``."""
    open_ports = []
    for port in ports:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                open_ports.append(port)
        except OSError:
            continue
    return open_ports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subnet", default=DEFAULT_SUBNET,
                        help="wired subnet to sweep (default {})".format(DEFAULT_SUBNET))
    parser.add_argument("--udp-port", type=int, default=9,
                        help="UDP port used only to trigger the ARP lookup (default 9)")
    parser.add_argument("--wait", type=float, default=2.0, help="seconds to wait before reading ARP")
    parser.add_argument("--probe-ports", action="store_true",
                        help="also test VXI-11/SCPI ports on every neighbour found")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        subnet = ipaddress.IPv4Network(args.subnet, strict=False)
    except ValueError as exc:
        print("Invalid subnet {!r}: {}".format(args.subnet, exc), file=sys.stderr)
        return 2

    neighbours = arp_sweep(subnet, udp_port=args.udp_port, wait_s=args.wait)

    if args.probe_ports:
        for neighbour in neighbours:
            neighbour["open_ports"] = probe_ports(neighbour["ip"])

    if args.json:
        import json

        print(json.dumps(neighbours, indent=2, sort_keys=True))
    elif not neighbours:
        print("\nNo neighbours answered ARP in {}.".format(subnet))
        print("That means the segment is electrically empty from this host: check that the")
        print("cable is in the instrument, that the adapter shows 'Up' in Get-NetAdapter,")
        print("and that this subnet matches the adapter's IP/mask.")
    else:
        print("\nLive neighbours on {} ({} found):".format(subnet, len(neighbours)))
        print("{:<16} {:<18} {:<9} {}".format("ip", "mac", "kind", "open ports"))
        for neighbour in neighbours:
            ports = neighbour.get("open_ports")
            print("{:<16} {:<18} {:<9} {}".format(
                neighbour["ip"],
                neighbour["mac"],
                neighbour["kind"],
                " ".join(map(str, ports)) if ports else "",
            ))
        print("\nAn instrument usually shows 111 (VXI-11) and/or 5025. If a host answers")
        print("neither, compare its MAC with the one printed by the instrument's own")
        print("network page before assuming it is the scope.")
    return 0 if neighbours else 1


if __name__ == "__main__":
    sys.exit(main())
