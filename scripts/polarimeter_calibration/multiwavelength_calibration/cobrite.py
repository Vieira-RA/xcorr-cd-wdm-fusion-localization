#!/usr/bin/env python3
"""SCPI client for ID Photonics CoBrite tunable laser instruments.

Source of truth: ``cobrite/Manual_CoBrite.pdf`` (V1.16) plus the vendor example
``cobrite/cobrite-demo.py``. The instrument speaks ASCII SCPI over a raw TCP
("telnet") session on port 2000, commands terminated by ``;`` (or CR), and every
command — including a plain set — is acknowledged with ``;``. Queries answer
``<value>;``. Errors answer ``ERR <code>,<text>;`` (see the manual's error table).

Commands used here (short forms, all case insensitive)::

    *IDN?                 identify the unit
    ECHO 0                disable command echo for this session
    PASS <password>       raise the user level (default password "IDP")
    WAV <C>,<S>,<D>,<nm>  set wavelength          WAV? <C>,<S>,<D>   read it
    WAV:LIM? <C>,<S>,<D>  wavelength limits -> "<min>,<max>"
    BUSY? <C>,<S>,<D>     1 = tuning, 0 = settled
    STAT <C>,<S>,<D>,<0|1> / STAT?     laser output on/off
    POW <dBm> / POW?      power setpoint          APOW?  measured power
    *OPC?                 1 = previous commands executed (does NOT mean settled)
    :SYST:ERR?            next error in the queue

The port address is ``<chassis>,<slot>,<device>``; ``1,1,1`` is the default and
the manual's three-level identifier for a single-port chassis.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


class CoBriteError(RuntimeError):
    """Raised for transport, protocol or instrument-reported errors."""


#: Hard safety ceiling for the optical output power in THIS lab. The instrument
#: itself accepts more (see ``LIM?``); this limit is a user-set rule that no code
#: path may exceed, so a typo in a config or on a command line cannot overdrive
#: the laser or the detector chain.
ABSOLUTE_MAX_POWER_DBM = 10.0


class _TcpIO:
    """Byte transport over a TCP socket (raw SCPI, port 2000)."""

    def __init__(self, sock: socket.socket):
        self._sock = sock

    @classmethod
    def open(cls, host: str, port: int, timeout_s: float) -> "_TcpIO":
        try:
            sock = socket.create_connection((host, port), timeout=timeout_s)
        except OSError as exc:
            raise CoBriteError(
                "Cannot open SCPI session to {}:{} ({}). Check the address/port 2000, "
                "or use transport 'serial' with com_port (e.g. COM4).".format(host, port, exc)
            )
        sock.settimeout(timeout_s)
        return cls(sock)

    def write(self, data: bytes) -> None:
        self._sock.sendall(data)

    def read_some(self, timeout_s: float) -> Optional[bytes]:
        self._sock.settimeout(timeout_s)
        try:
            chunk = self._sock.recv(4096)
        except socket.timeout:
            return None
        if not chunk:
            raise CoBriteError("The instrument closed the TCP connection.")
        return chunk

    def close(self) -> None:
        self._sock.close()


class _SerialIO:
    """Byte transport over a serial port (the CoBrite USB CDC-ACM interface)."""

    def __init__(self, handle: Any):
        self._handle = handle

    @classmethod
    def open(cls, com_port: str, baudrate: int, timeout_s: float) -> "_SerialIO":
        if not com_port:
            raise CoBriteError(
                "transport 'serial' needs com_port (e.g. \"COM4\"). List the ports with "
                "discover_laser.py or serial.tools.list_ports."
            )
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise CoBriteError(
                "pyserial is not installed in this interpreter (pip install pyserial)."
            ) from exc
        try:
            handle = serial.Serial(
                com_port, baudrate=baudrate, timeout=timeout_s, write_timeout=timeout_s
            )
        except Exception as exc:
            raise CoBriteError("Cannot open serial port {}: {}".format(com_port, exc))
        return cls(handle)

    def write(self, data: bytes) -> None:
        self._handle.write(data)
        try:
            self._handle.flush()
        except Exception:
            pass

    def read_some(self, timeout_s: float) -> Optional[bytes]:
        self._handle.timeout = max(0.01, float(timeout_s))
        data = self._handle.read(4096)
        return data or None

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass


class CoBriteClient:
    """Minimal, dependency-free CoBrite SCPI client.

    Parameters (all optional, taken from a ``cobrite`` config dict)
    --------------------------------------------------------------
    transport
        ``"tcp"`` (default) or ``"serial"``. The USB link exposes BOTH a
        virtual Ethernet interface (RNDIS, TCP port 2000) and a virtual COM
        port; serial is the fallback when the host cannot be given an address
        on the USB network (i.e. without admin rights).
    host, port
        TCP endpoint of the instrument (default ``cobrite.local:2000``).
    com_port, baudrate
        Serial endpoint (e.g. ``"COM4"``); the CoBrite is a CDC-ACM gadget that
        answers at any baud rate, 115200 is a safe default.
    address
        Laser port ``"<chassis>,<slot>,<device>"`` (default ``"1,1,1"``).
    timeout_s
        Per-command timeout.
    password
        Sent with ``PASS`` on connect to raise the user level (``None`` = skip).
    echo
        Whether to leave the unit's command echo enabled (default ``False``).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = dict(config or {})
        self.transport = str(config.get("transport", "tcp")).lower()
        if self.transport not in ("tcp", "serial"):
            raise CoBriteError("transport must be 'tcp' or 'serial'.")
        self.host = str(config.get("host", "cobrite.local"))
        self.port = int(config.get("port", 2000))
        self.com_port = str(config.get("com_port") or config.get("serial_port") or "")
        self.baudrate = int(config.get("baudrate", 115200))
        self.address = str(config.get("address", "1,1,1")).replace(" ", "")
        self.timeout_s = float(config.get("timeout_s", 10.0))
        self.password = config.get("password", "IDP")
        self.echo = bool(config.get("echo", False))
        self.min_settle_s = float(config.get("min_settle_s", 1.0))
        self.poll_interval_s = float(config.get("poll_interval_s", 0.25))
        requested_limit = config.get("max_power_dbm", ABSOLUTE_MAX_POWER_DBM)
        self.max_power_dbm = min(float(requested_limit), ABSOLUTE_MAX_POWER_DBM)

        self._io: Optional[Any] = None
        self._idn: Optional[str] = None
        self._limits: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------ transport
    def connect(self) -> "CoBriteClient":
        if self._io is not None:
            return self
        if self.transport == "serial":
            self._io = _SerialIO.open(self.com_port, self.baudrate, self.timeout_s)
        else:
            self._io = _TcpIO.open(self.host, self.port, self.timeout_s)
        # Session-scoped setup: no echo, and the documented user level password.
        try:
            self._send("ECHO {}".format(1 if self.echo else 0))
            if self.password:
                self._send("PASS {}".format(self.password))
            self._idn = self._query("*IDN?")
        except CoBriteError:
            self.close()
            raise
        return self

    def close(self) -> None:
        if self._io is not None:
            try:
                self._io.close()
            finally:
                self._io = None

    def __enter__(self) -> "CoBriteClient":
        return self.connect()

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _read_response(self, command: str) -> str:
        """Collect one reply.

        A TCP query answers ``<value>;`` in one packet, whereas a serial query
        answers ``;<CR><LF><value>;<CR><LF>`` (the acknowledgement arrives glued
        to the value).  A set command answers with the bare acknowledgement.
        So: for a query, keep reading until the reply holds something besides
        the acknowledgement; for a set, the first terminator is enough.
        """
        expects_value = "?" in command
        deadline = time.monotonic() + self.timeout_s
        buffer = ""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                if buffer.strip() or not expects_value:
                    return buffer
                raise CoBriteError(
                    "Timed out waiting for a response to {!r} (received {!r}).".format(
                        command, buffer
                    )
                )
            chunk = self._io.read_some(min(remaining, 1.0))
            if chunk is None:
                if buffer.strip() and (self._clean(buffer) or not expects_value):
                    return buffer
                continue
            buffer += chunk.decode("ascii", errors="replace").replace("\r", "")
            stripped = buffer.strip()
            if not (stripped.endswith(";") or stripped.endswith("\n")):
                continue
            if self.echo:
                # The unit replays the command first, so an early ';' may be
                # an inner terminator: wait briefly for the rest.
                extra = self._io.read_some(0.05)
                if extra:
                    buffer += extra.decode("ascii", errors="replace")
                    continue
            if not expects_value or self._clean(buffer):
                return buffer
            # Query that has only produced the acknowledgement so far: give the
            # value a brief chance to arrive before giving up on it.
            extra = self._io.read_some(min(0.2, max(remaining, 0.01)))
            if not extra:
                return buffer
            buffer += extra.decode("ascii", errors="replace")

    @staticmethod
    def _clean(text: str) -> str:
        """Keep the last non-empty segment (handles echo and multi-command replies)."""
        segments = [segment.strip() for segment in text.split(";")]
        remaining = [segment for segment in segments if segment]
        return remaining[-1] if remaining else ""

    def _send(self, command: str) -> str:
        """Send one command and consume its acknowledgement or response."""
        if self._io is None:
            self.connect()
        try:
            self._io.write((command + ";").encode("ascii"))
        except OSError as exc:
            raise CoBriteError("Transport error while sending {!r}: {}".format(command, exc))
        text = self._clean(self._read_response(command))
        if text.upper().startswith("ERR"):
            raise CoBriteError(
                "Instrument rejected {!r}: {}".format(command, text)
            )
        return text

    def _query(self, command: str) -> str:
        text = self._send(command)
        if text == "":
            raise CoBriteError(
                "Empty response to {!r}; is the laser session healthy?".format(command)
            )
        return text

    def connection_label(self) -> str:
        """Short description of the active link, for logs and metadata."""
        if self.transport == "serial":
            return "serial {} @ {} baud".format(self.com_port or "?", self.baudrate)
        return "tcp {}:{}".format(self.host, self.port)

    # -------------------------------------------------------------------- identity
    def idn(self) -> str:
        if self._idn is None:
            self._idn = self._query("*IDN?")
        return self._idn

    def last_error(self) -> str:
        """Read and clear one entry of the instrument error queue.

        Some firmwares do not implement ``:SYST:ERR?`` (they answer
        ``ERR 100, unknown command``); that is returned as a string instead of
        raised, because this is a diagnostic helper only.
        """
        try:
            return self._send(":SYST:ERR?")
        except CoBriteError as exc:
            return str(exc)

    # ------------------------------------------------------------- wavelength control
    def wavelength_limits(self) -> Tuple[float, float]:
        if self._limits is None:
            text = self._query("WAV:LIM? {}".format(self.address))
            parts = [part for part in text.replace("\n", "").split(",") if part.strip()]
            if len(parts) < 2:
                raise CoBriteError(
                    "Unexpected WAV:LIM? response {!r}; expected '<min>,<max>'.".format(text)
                )
            low, high = float(parts[-2]), float(parts[-1])
            if not low < high:
                raise CoBriteError(
                    "Nonsensical wavelength limits from the instrument: {}.".format((low, high))
                )
            self._limits = (low, high)
        return self._limits

    def wavelength(self) -> float:
        text = self._query("WAV? {}".format(self.address))
        return _first_float(text, "WAV?")

    def set_wavelength(self, wavelength_nm: float) -> None:
        low, high = self.wavelength_limits()
        if not low <= float(wavelength_nm) <= high:
            raise CoBriteError(
                "Requested {:.4f} nm is outside the instrument range {:.4f}-{:.4f} nm.".format(
                    float(wavelength_nm), low, high
                )
            )
        self._send("WAV {},{:.4f}".format(self.address, float(wavelength_nm)))

    def frequency(self) -> float:
        return _first_float(self._query("FREQ? {}".format(self.address)), "FREQ?")

    # ------------------------------------------------------------------- power/state
    def power(self) -> float:
        return _first_float(self._query("POW? {}".format(self.address)), "POW?")

    def actual_power(self) -> float:
        return _first_float(self._query("APOW? {}".format(self.address)), "APOW?")

    def set_power(self, power_dbm: float) -> None:
        value = float(power_dbm)
        if value > self.max_power_dbm + 1e-9:
            raise CoBriteError(
                "Refusing to set {:.2f} dBm: the configured limit is {:.1f} dBm "
                "(lab safety cap, not the instrument maximum).".format(
                    value, self.max_power_dbm
                )
            )
        self._send("POW {},{:.3f}".format(self.address, value))

    def output_state(self) -> int:
        return int(round(_first_float(self._query("STAT? {}".format(self.address)), "STAT?")))

    def set_output(self, enabled: bool) -> None:
        self._send("STAT {},{}".format(self.address, 1 if enabled else 0))

    def busy(self) -> bool:
        """``True`` while the laser port is still tuning."""
        text = self._send("BUSY? {}".format(self.address))
        try:
            return int(round(float(text.replace("\n", "").split(",")[-1]))) == 1
        except (TypeError, ValueError):
            raise CoBriteError("Unexpected BUSY? response {!r}.".format(text))

    def operation_complete(self) -> None:
        """Block until the unit reports that pending commands were executed.

        Note: this is *not* the same as the laser having settled; poll ``BUSY?``
        for that (the manual is explicit about it).
        """
        self._send("*OPC?")

    def wait_until_settled(
        self,
        target_nm: float,
        tolerance_nm: float = 0.1,
        timeout_s: float = 120.0,
        poll_interval_s: Optional[float] = None,
        min_settle_s: Optional[float] = None,
        min_polls: int = 2,
    ) -> Dict[str, Any]:
        """Poll ``BUSY?`` until the port settles, then verify the read-back."""
        poll_interval_s = self.poll_interval_s if poll_interval_s is None else poll_interval_s
        min_settle_s = self.min_settle_s if min_settle_s is None else min_settle_s
        start = time.monotonic()
        deadline = start + float(timeout_s)
        polls = 0
        saw_busy = False
        last_readback: Optional[float] = None

        if min_settle_s > 0:
            time.sleep(min_settle_s)

        while time.monotonic() <= deadline:
            polls += 1
            if self.busy():
                saw_busy = True
            elif polls >= int(min_polls):
                last_readback = self.wavelength()
                if abs(last_readback - float(target_nm)) <= float(tolerance_nm):
                    return {
                        "settle_s": time.monotonic() - start,
                        "busy_polls": polls,
                        "saw_busy": saw_busy,
                        "readback_nm": last_readback,
                    }
            time.sleep(max(0.05, float(poll_interval_s)))

        raise CoBriteError(
            "Laser did not settle on {:.4f} nm within {:.1f} s "
            "(polls={}, last read-back={}, tuning seen={}).".format(
                float(target_nm), float(timeout_s), polls, last_readback, saw_busy
            )
        )


def _first_float(text: str, command: str) -> float:
    """Take the last numeric field of a (possibly multi-port) SCPI response."""
    cleaned = text.replace("\n", ",")
    fields = [field.strip() for field in cleaned.split(",") if field.strip()]
    for field in reversed(fields):
        try:
            return float(field)
        except ValueError:
            continue
    raise CoBriteError("Could not parse a number from {!r} response {!r}.".format(command, text))


def probe(config: Dict[str, Any]) -> Dict[str, Any]:
    """Connect once and read identity, limits, setpoint and measured power."""
    client = CoBriteClient(config).connect()
    try:
        info: Dict[str, Any] = {"idn": client.idn(), "address": client.address,
                               "transport": client.connection_label()}
        info["limits_nm"] = list(client.wavelength_limits())
        info["wavelength_nm"] = client.wavelength()
        info["frequency_thz"] = client.frequency()
        info["power_setpoint_dbm"] = client.power()
        info["output_state"] = client.output_state()
        try:
            info["actual_power_dbm"] = client.actual_power()
        except CoBriteError as exc:
            info["actual_power_dbm"] = None
            info["actual_power_error"] = str(exc)
        info["busy"] = client.busy()
        info["max_power_dbm"] = client.max_power_dbm
        if info.get("actual_power_dbm") is not None:
            info["power_within_limit"] = bool(
                info["actual_power_dbm"] <= client.max_power_dbm + 0.5
            )
        return info
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CoBrite SCPI connectivity probe and manual wavelength setter."
    )
    parser.add_argument("--transport", choices=("tcp", "serial"), default="tcp",
                        help="tcp (default) or serial for the USB COM port")
    parser.add_argument("--host", default="cobrite.local", help="instrument IP or hostname (tcp)")
    parser.add_argument("--port", type=int, default=2000, help="SCPI TCP port (default 2000)")
    parser.add_argument("--com-port", default=None, help="serial port for --transport serial, e.g. COM4")
    parser.add_argument("--baudrate", type=int, default=115200, help="serial baud rate (CDC-ACM ignores it)")
    parser.add_argument("--address", default="1,1,1", help="laser port <chassis>,<slot>,<device>")
    parser.add_argument("--password", default="IDP", help="user-level password ('' to skip)")
    parser.add_argument("--timeout", type=float, default=10.0, help="per-command socket timeout")
    parser.add_argument("--set-nm", type=float, default=None, help="set this wavelength, then wait")
    parser.add_argument("--power-dbm", type=float, default=None, help="optionally set output power (<= 10 dBm)")
    parser.add_argument("--enable-output", action="store_true", help="switch the laser output on")
    parser.add_argument("--settle-timeout", type=float, default=120.0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = {
        "transport": args.transport,
        "com_port": args.com_port,
        "baudrate": args.baudrate,
        "host": args.host,
        "port": args.port,
        "address": args.address,
        "password": args.password or None,
        "timeout_s": args.timeout,
    }
    try:
        info = probe(config)
        print("Instrument : {}".format(info["idn"]))
        print("Link       : {}".format(info.get("transport")))
        print("Port       : {}".format(info["address"]))
        print("Range      : {:.3f} - {:.3f} nm".format(*info["limits_nm"]))
        print("Wavelength : {:.4f} nm ({:.4f} THz)".format(
            info["wavelength_nm"], info["frequency_thz"]
        ))
        print("Power      : setpoint {:.2f} dBm, actual {}  (limit {:.1f} dBm)".format(
            info["power_setpoint_dbm"],
            "n/a" if info["actual_power_dbm"] is None else "{:.2f} dBm".format(info["actual_power_dbm"]),
            info.get("max_power_dbm", 10.0),
        ))
        if info.get("power_within_limit") is False:
            print("WARNING: measured power is above the configured limit;", 
                  "reduce POW before running a calibration.", file=sys.stderr)
        print("Output     : {} (busy={})".format(
            "ON" if info["output_state"] == 1 else "OFF", info["busy"]
        ))

        if args.set_nm is not None or args.power_dbm is not None or args.enable_output:
            client = CoBriteClient(config).connect()
            try:
                if args.power_dbm is not None:
                    client.set_power(args.power_dbm)
                    print("Power setpoint -> {:.2f} dBm".format(args.power_dbm))
                if args.enable_output and client.output_state() != 1:
                    client.set_output(True)
                    print("Output enabled")
                if args.set_nm is not None:
                    client.set_wavelength(args.set_nm)
                    result = client.wait_until_settled(
                        args.set_nm, timeout_s=args.settle_timeout
                    )
                    print("Settled on {:.4f} nm after {:.2f} s ({} busy polls)".format(
                        result["readback_nm"], result["settle_s"], result["busy_polls"]
                    ))
            finally:
                client.close()
        return 0
    except CoBriteError as exc:
        print("CoBrite error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
