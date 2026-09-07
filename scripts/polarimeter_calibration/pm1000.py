#!/usr/bin/env python3
"""
Minimal live-readout driver for the Novoptel PM1000 polarimeter.

Supported interfaces:
  - USB 2.0 virtual COM port (ASCII protocol, 230400 8N1)
  - LAN/TCP (binary protocol, port 5025)

Usage examples
--------------
USB:
    python pm1000.py --interface usb --port COM3 --n 20 --interval 0.2

LAN:
    python pm1000.py --interface lan --ip 192.168.1.100 --n 20 --interval 0.2

The script reads the live Stokes registers and prints:
    S0       : power (integer PM1000 unit, uW)
    S1,S2,S3 : signed Stokes parameters (16-bit signed units)
    DOP      : degree of polarisation from the normalised Stokes parameters
"""

import argparse
import re
import time
from typing import Optional, Tuple

try:
    import serial  # pyserial
except ImportError:  # pragma: no cover
    serial = None


# ----------------------------------------------------------------------
# PM1000 register addresses.
#
# The PM1000 manual defines a base offset of 512.  The addresses used
# below are therefore already the *communication* addresses:
#     communication_address = 512 + manual_offset
# ----------------------------------------------------------------------
REG_S0_INT = 512 + 10   # S0, integer part (power, µW)
REG_S0_FRAC = 512 + 11  # S0, fractional part (power, µW)
REG_S1_INT = 512 + 12   # S1, integer part (offset-binary, offset=2^15)
REG_S1_FRAC = 512 + 13  # S1, fractional part
REG_S2_INT = 512 + 14   # S2, integer part (offset-binary, offset=2^15)
REG_S2_FRAC = 512 + 15  # S2, fractional part
REG_S3_INT = 512 + 16   # S3, integer part (offset-binary, offset=2^15)
REG_S3_FRAC = 512 + 17  # S3, fractional part

STOKES_REGISTERS = (
    REG_S0_INT, REG_S0_FRAC,
    REG_S1_INT, REG_S1_FRAC,
    REG_S2_INT, REG_S2_FRAC,
    REG_S3_INT, REG_S3_FRAC,
)

# Offset used by the PM1000 for signed Stokes parameters (S1, S2, S3).
# The integer part is in offset-binary form: 0x8000 (32768) corresponds to 0.
SIGNED_OFFSET = 0x8000

# The fractional part is stored as an unsigned 16-bit number representing
# the fraction of one µW (i.e., raw_frac / 65536).
FRACTIONAL_SCALE = 65536.0


class PM1000:
    """Small driver for the Novoptel PM1000 polarimeter."""

    def __init__(
        self,
        interface: str = "lan",
        port: Optional[str] = None,
        ip: Optional[str] = None,
        timeout: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        interface : {'usb', 'lan'}
        port : serial port name, required for USB (e.g. 'COM3').
        ip : IP address, required for LAN.
        timeout : communication timeout in seconds.
        """
        self.interface = interface.lower()
        self.conn = None

        if self.interface == "usb":
            if serial is None:
                raise ImportError(
                    "pyserial is required for USB communication. "
                    "Install it with: pip install pyserial"
                )
            if not port:
                raise ValueError("USB interface requires 'port'.")
            self.conn = serial.Serial(
                port=port,
                baudrate=230400,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout,
            )
        elif self.interface == "lan":
            if not ip:
                raise ValueError("LAN interface requires 'ip'.")
            import socket

            self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.conn.settimeout(timeout)
            self.conn.connect((ip, 5025))
        else:
            raise ValueError("interface must be 'usb' or 'lan'")

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def __enter__(self) -> "PM1000":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _write_usb(self, cmd: str) -> None:
        """Write an ASCII command terminated by CR over USB."""
        self.conn.write((cmd + "\r").encode("ascii"))

    def _read_usb_value(self) -> int:
        """Read a 16-bit hex reply from the USB interface.

        The instrument normally replies with four hex digits followed by CR.
        """
        raw = self.conn.read_until(b"\r").decode("ascii", errors="ignore")
        match = re.search(r"[0-9A-Fa-f]{4}", raw)
        if not match:
            raise IOError(f"Could not parse USB register reply: {raw!r}")
        return int(match.group(0), 16)

    def _write_lan(self, data: bytes) -> None:
        self.conn.sendall(data)

    def _read_lan(self, length: int) -> bytes:
        data = b""
        while len(data) < length:
            chunk = self.conn.recv(length - len(data))
            if not chunk:
                raise IOError("LAN connection closed while reading.")
            data += chunk
        return data

    # ------------------------------------------------------------------
    # Register access
    # ------------------------------------------------------------------
    def write_register(self, addr: int, value: int) -> None:
        """Write a 16-bit value to communication register *addr*."""
        value16 = int(value) & 0xFFFF

        if self.interface == "usb":
            cmd = f"W{addr:03X}{value16:04X}"
            self._write_usb(cmd)
        else:  # LAN
            packet = bytes(
                [
                    0x57,  # 'W'
                    (addr >> 8) & 0xFF,
                    addr & 0xFF,
                    (value16 >> 8) & 0xFF,
                    value16 & 0xFF,
                ]
            )
            self._write_lan(packet)

    def read_register(self, addr: int) -> int:
        """Read a 16-bit value from communication register *addr*."""
        if self.interface == "usb":
            cmd = f"R{addr:03X}0000"
            self._write_usb(cmd)
            return self._read_usb_value()
        else:  # LAN
            packet = bytes(
                [
                    0x52,  # 'R'
                    (addr >> 8) & 0xFF,
                    addr & 0xFF,
                ]
            )
            self._write_lan(packet)
            data = self._read_lan(2)
            return (data[0] << 8) | data[1]

    # ------------------------------------------------------------------
    # Stokes-vector access
    # ------------------------------------------------------------------
    def _combine_int_frac(self, raw_int: int, raw_frac: int, signed: bool) -> float:
        """Combine an integer register and a fractional register into µW.

        For signed Stokes parameters (S1..S3), the integer part is in
        offset-binary form with offset 2^15.
        """
        if signed:
            value = raw_int - SIGNED_OFFSET
        else:
            value = raw_int
        return value + raw_frac / FRACTIONAL_SCALE

    def read_stokes_uw(self) -> Tuple[float, float, float, float]:
        """Read live Stokes parameters as floating-point micro-watt values.

        Returns
        -------
        (S0, S1, S2, S3) where each element is in µW, with fractional precision.
        S1..S3 are normalised to 1 µW (i.e., the same µW units after removing
        the offset).
        """
        s0_int = self.read_register(REG_S0_INT)
        s0_frac = self.read_register(REG_S0_FRAC)
        s1_int = self.read_register(REG_S1_INT)
        s1_frac = self.read_register(REG_S1_FRAC)
        s2_int = self.read_register(REG_S2_INT)
        s2_frac = self.read_register(REG_S2_FRAC)
        s3_int = self.read_register(REG_S3_INT)
        s3_frac = self.read_register(REG_S3_FRAC)

        s0 = self._combine_int_frac(s0_int, s0_frac, signed=False)
        s1 = self._combine_int_frac(s1_int, s1_frac, signed=True)
        s2 = self._combine_int_frac(s2_int, s2_frac, signed=True)
        s3 = self._combine_int_frac(s3_int, s3_frac, signed=True)
        return s0, s1, s2, s3

    def read_stokes_raw(self) -> Tuple[int, int, int, int]:
        """Read live S0, S1, S2, S3 integer parts only.

        (Kept for compatibility; use :meth:`read_stokes_uw` for full precision.)

        Returns
        -------
        tuple of four ints:
            S0 : unsigned 16-bit power integer
            S1 : signed integer (offset 0x8000 removed)
            S2 : signed integer (offset 0x8000 removed)
            S3 : signed integer (offset 0x8000 removed)
        """
        s0 = self.read_register(REG_S0_INT)
        s1_raw = self.read_register(REG_S1_INT)
        s2_raw = self.read_register(REG_S2_INT)
        s3_raw = self.read_register(REG_S3_INT)

        s1 = s1_raw - SIGNED_OFFSET
        s2 = s2_raw - SIGNED_OFFSET
        s3 = s3_raw - SIGNED_OFFSET
        return s0, s1, s2, s3

    def read_stokes_normalized(self) -> Tuple[float, float, float, float]:
        """Read live Stokes parameters and normalise S1..S3 by S0.

        Returns
        -------
        (S0, s1, s2, s3) where S0 is in µW and s1..s3 are normalised (-1..1).
        """
        s0, s1, s2, s3 = self.read_stokes_uw()
        denom = s0 if s0 != 0 else 1.0
        return s0, s1 / denom, s2 / denom, s3 / denom


# ----------------------------------------------------------------------
# Command-line test / live readout
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live Stokes readout from a Novoptel PM1000 polarimeter."
    )
    parser.add_argument(
        "--interface",
        choices=["usb", "lan"],
        default="lan",
        help="Communication interface (default: lan).",
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Serial port for USB (e.g. COM3 or /dev/ttyUSB0).",
    )
    parser.add_argument(
        "--ip",
        default="192.168.1.100",
        help="IP address for LAN (default: 192.168.1.100).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of live readings to acquire (default: 10).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Delay between readings in seconds (default: 0.2).",
    )
    args = parser.parse_args()

    print(f"Connecting to PM1000 via {args.interface}...")
    if args.interface == "usb":
        device = PM1000(interface="usb", port=args.port)
    else:
        device = PM1000(interface="lan", ip=args.ip)

    print("Connected.")
    print(
        f"{'index':>5}  {'S0(µW)':>12}  {'s1':>10}  {'s2':>10}  {'s3':>10}  {'DOP':>7}"
    )

    try:
        for i in range(args.n):
            s0, s1, s2, s3 = device.read_stokes_normalized()
            dop = 0.0
            if s0 > 0:
                dop = (s1 * s1 + s2 * s2 + s3 * s3) ** 0.5
            print(
                f"{i:5d}  {s0:12.3f}  {s1:10.6f}  {s2:10.6f}  {s3:10.6f}  {dop:7.4f}"
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        device.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()