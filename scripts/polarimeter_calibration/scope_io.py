#!/usr/bin/env python3
"""Transport-neutral SCPI access to the oscilloscope.

Why this exists
---------------
The DPO7254 can be reached in two ways, and on a locked-down Windows host only
one of them may be usable:

* **VXI-11 over Ethernet** (``vxi11`` package, TCP port 111) - what the original
  scripts used. Requires an IP address/route to the instrument, which normally
  means admin rights to configure a NIC.
* **USBTMC or VXI-11 through NI-VISA / PyVISA** - ``USB0::0x0699::...::INSTR``.
  Needs **no IP address and no network privileges**, which makes it the natural
  bypass when the host cannot be re-addressed.

``open_scope()`` hands back an object exposing exactly the interface the
acquisition code already uses (``write``, ``ask``, ``read_raw``, ``timeout`` in
*seconds*, ``close``), so the scripts need no rewrite. The backend is chosen from
the environment:

    PM_SCOPE_TRANSPORT   "vxi11" | "pyvisa" | "auto"     (default "auto")
    PM_SCOPE_IP          address for VXI-11              (default 198.192.1.1)
    PM_SCOPE_RESOURCE    explicit PyVISA resource string (e.g. USB0::0x0699::...)

Auto order: explicit resource string -> vxi11 (if installed) -> PyVISA discovery
(Tektronix USB first, then ``TCPIP0::<ip>::INSTR``).

Examples
--------
    set PM_SCOPE_RESOURCE=USB0::0x0699::0x0364::C000001::INSTR
    set PM_SCOPE_IP=169.254.102.200
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

DEFAULT_IP = "198.192.1.1"
TEKTRONIX_VENDOR_ID = "0x0699"


class ScopeError(RuntimeError):
    """Any failure to reach or talk to the oscilloscope."""


class ScopeTimeout(ScopeError):
    """A read timed out (the code uses this to skip optional bytes)."""


class _Vxi11Scope:
    """Backend for the pure-Python VXI-11 client (TCP port 111)."""

    name = "vxi11"

    def __init__(self, instrument: Any, ip: str):
        self._instrument = instrument
        self.address = ip

    @property
    def timeout(self) -> float:
        return float(self._instrument.timeout)

    @timeout.setter
    def timeout(self, seconds: float) -> None:
        self._instrument.timeout = float(seconds)

    def write(self, command: str) -> Any:
        try:
            return self._instrument.write(command)
        except Exception as exc:
            raise ScopeError("vxi11 write failed for {!r}: {}".format(command, exc))

    def ask(self, command: str) -> str:
        try:
            return self._instrument.ask(command)
        except Exception as exc:
            raise ScopeError("vxi11 query failed for {!r}: {}".format(command, exc))

    def read_raw(self, size: Optional[int] = None) -> bytes:
        try:
            return self._instrument.read_raw() if size is None else self._instrument.read_raw(size)
        except Exception as exc:
            if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                raise ScopeTimeout(str(exc))
            raise ScopeError("vxi11 read failed: {}".format(exc))

    def read_block(self, command: str = "CURVE?") -> bytes:
        """Send ``command`` and consume one SCPI definite-length block."""
        self.write(command)
        header = self.read_raw(2)
        if header[:1] != b"#":
            raise ScopeError("Unexpected block header: {!r}".format(header))
        digits = int(header[1:2])
        byte_count = int(self.read_raw(digits).decode("ascii"))
        payload = self.read_raw(byte_count)
        saved = self.timeout
        self.timeout = 0.5
        try:
            self.read_raw(1)          # consume the optional trailing newline
        except ScopeTimeout:
            pass
        finally:
            self.timeout = saved
        return payload

    def close(self) -> None:
        try:
            self._instrument.close()
        except Exception:
            pass


class _PyVisaScope:
    """Backend for NI-VISA/PyVISA resources (USB TMC or TCPIP)."""

    name = "pyvisa"

    def __init__(self, resource: Any, address: str):
        self._resource = resource
        self.address = address
        # Text queries are newline-terminated; binary block reads clear this
        # temporarily so a 0x0A byte inside the waveform cannot truncate them.
        try:
            self._resource.write_termination = "\n"
            self._resource.read_termination = "\n"
        except Exception:
            pass

    @property
    def timeout(self) -> float:
        try:
            return float(self._resource.timeout) / 1000.0
        except Exception:
            return 0.0

    @timeout.setter
    def timeout(self, seconds: float) -> None:
        try:
            self._resource.timeout = max(1, int(round(float(seconds) * 1000.0)))
        except Exception:
            pass

    def write(self, command: str) -> Any:
        try:
            return self._resource.write(command)
        except Exception as exc:
            raise ScopeError("VISA write failed for {!r}: {}".format(command, exc))

    def ask(self, command: str) -> str:
        try:
            return self._resource.query(command)
        except Exception as exc:
            raise ScopeError("VISA query failed for {!r}: {}".format(command, exc))

    def read_raw(self, size: Optional[int] = None) -> bytes:
        saved = getattr(self._resource, "read_termination", None)
        try:
            self._resource.read_termination = None
        except Exception:
            pass
        try:
            return self._resource.read_raw() if size is None else self._resource.read_raw(size)
        except Exception as exc:
            if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                raise ScopeTimeout(str(exc))
            raise ScopeError("VISA read failed: {}".format(exc))
        finally:
            try:
                self._resource.read_termination = saved
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._resource.close()
        except Exception:
            pass

    def read_block(self, command: str = "CURVE?") -> bytes:
        """Send ``command`` and decode one SCPI definite-length block.

        PyVISA cannot do this with a fixed-size raw read when no read terminator
        is set (it would wait for the timeout), so ``read_binary_values`` decodes
        the ``#<n><len><payload>`` header itself.
        """
        saved = getattr(self._resource, "read_termination", None)
        try:
            self._resource.read_termination = None
        except Exception:
            pass
        try:
            self._resource.write(command)
            values = self._resource.read_binary_values(
                datatype="b", is_big_endian=False, header_fmt="ieee", container=list
            )
            return bytes((int(value) & 0xFF) for value in values)
        except Exception as exc:
            if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                raise ScopeTimeout(str(exc))
            raise ScopeError("VISA block read failed for {!r}: {}".format(command, exc))
        finally:
            try:
                self._resource.read_termination = saved
            except Exception:
                pass


def _candidate_resources(manager: Any, ip: str) -> List[str]:
    """Rank VISA resources: Tektronix USB first, then the requested TCPIP, then any USB.

    NI-VISA's ``list_resources`` only reports instruments it has discovered, so a
    directly-addressed LAN instrument may not appear there at all. The canonical
    VXI-11 resource strings for ``ip`` are therefore always appended as
    fallbacks (``::inst0::INSTR`` is what Tektronix firmware answers).
    """
    try:
        available = list(manager.list_resources("?*"))
    except Exception:
        available = []

    ranked: List[str] = []
    ranked += [r for r in available if r.upper().startswith("USB") and TEKTRONIX_VENDOR_ID in r.lower()]
    # The directly-addressed VXI-11 form is the reliable one for Tektronix
    # firmware; try it before whatever VISA happened to discover.
    if ip:
        ranked += [
            "TCPIP0::{}::inst0::INSTR".format(ip),
            "TCPIP0::{}::INSTR".format(ip),
        ]
    ranked += [r for r in available if r.upper().startswith("TCPIP") and ip in r]
    ranked += [r for r in available if r.upper().startswith("USB")]
    ranked += [r for r in available if r.upper().startswith("TCPIP")]
    seen = set()
    unique = []
    for resource in ranked:
        # Normalise the abbreviated "TCPIP::host" form VISA sometimes lists.
        if resource.upper().startswith("TCPIP::"):
            resource = "TCPIP0" + resource[len("TCPIP"):]
        if resource in seen:
            continue
        seen.add(resource)
        unique.append(resource)
    return unique


def _open_pyvisa(ip: str, resource: Optional[str]) -> _PyVisaScope:
    try:
        import pyvisa
    except ImportError as exc:
        raise ScopeError(
            "PyVISA is not installed in this interpreter (pip install pyvisa pyvisa-py)."
        ) from exc

    # NI-VISA occasionally wedges its VXI-11 layer after a failed session
    # (VI_ERROR_IO even though the instrument's core channel is alive), so the
    # pure-Python backend is tried first. Override with PM_VISA_LIBRARY.
    requested = os.environ.get("PM_VISA_LIBRARY")
    libraries: List[Optional[str]] = [requested] if requested else ["@py", None]

    errors: List[str] = []
    for library in libraries:
        try:
            manager = pyvisa.ResourceManager() if library is None else pyvisa.ResourceManager(library)
        except Exception as exc:
            errors.append("backend {}: {}".format(library or "system", exc))
            continue

        if resource:
            candidates = [resource]
        else:
            candidates = _candidate_resources(manager, ip)
            if not candidates:
                errors.append(
                    "backend {}: no instruments listed and no address to try".format(
                        library or "system"
                    )
                )
                continue

        failures = []
        for candidate in candidates:
            opened = None
            try:
                opened = manager.open_resource(candidate)
                scope = _PyVisaScope(opened, candidate)
                identity = scope.ask("*IDN?")
                if not identity or "*IDN" in identity.upper():
                    raise ScopeError("no *IDN? answer from {}".format(candidate))
                return scope
            except Exception as exc:
                failures.append("{}: {}".format(candidate, exc))
                # Release the session, otherwise the instrument refuses the next
                # attempt with a miscellaneous VISA error.
                if opened is not None:
                    try:
                        opened.close()
                    except Exception:
                        pass
        errors.append(
            "backend {}: {}".format(library or "system", "; ".join(failures))
        )

    raise ScopeError("No VISA backend could reach the instrument: {}".format(" | ".join(errors)))


def open_scope(
    ip: Optional[str] = None,
    transport: Optional[str] = None,
    resource: Optional[str] = None,
    timeout: float = 30.0,
    verbose: bool = True,
) -> Any:
    """Open the oscilloscope over VXI-11 or PyVISA.

    ``timeout`` is in seconds (the convention the acquisition scripts use).
    """
    ip = ip or os.environ.get("PM_SCOPE_IP") or DEFAULT_IP
    transport = (transport or os.environ.get("PM_SCOPE_TRANSPORT") or "auto").lower()
    resource = resource or os.environ.get("PM_SCOPE_RESOURCE") or None

    errors: List[str] = []

    if transport in ("auto", "pyvisa") and (resource or transport == "pyvisa"):
        try:
            scope = _open_pyvisa(ip, resource)
            scope.timeout = timeout
            if verbose:
                print("Scope transport: PyVISA ({})".format(scope.address))
            return scope
        except ScopeError as exc:
            errors.append("pyvisa: {}".format(exc))
            if transport == "pyvisa":
                raise

    if transport in ("auto", "vxi11"):
        try:
            import vxi11  # type: ignore

            instrument = vxi11.Instrument(ip)
            scope = _Vxi11Scope(instrument, ip)
            scope.timeout = timeout
            if verbose:
                print("Scope transport: VXI-11 ({})".format(ip))
            return scope
        except ImportError:
            errors.append("vxi11: package not installed")
        except Exception as exc:
            errors.append("vxi11({}): {}".format(ip, exc))
            if transport == "vxi11":
                raise ScopeError(
                    "Cannot open VXI-11 at {}: {}. Try PM_SCOPE_TRANSPORT=pyvisa with "
                    "PM_SCOPE_RESOURCE=USB0::0x0699::...::INSTR.".format(ip, exc)
                ) from exc

    if transport in ("auto", "pyvisa") and not resource:
        try:
            scope = _open_pyvisa(ip, None)
            scope.timeout = timeout
            if verbose:
                print("Scope transport: PyVISA ({})".format(scope.address))
            return scope
        except ScopeError as exc:
            errors.append("pyvisa: {}".format(exc))

    raise ScopeError(
        "Cannot reach the oscilloscope. Tried: {}. Set PM_SCOPE_RESOURCE to a PyVISA "
        "resource (USB0::0x0699::...::INSTR) for a USB connection, or PM_SCOPE_IP for "
        "VXI-11.".format("; ".join(errors) if errors else "nothing")
    )


if __name__ == "__main__":
    import sys

    try:
        scope = open_scope()
        print("IDN:", scope.ask("*IDN?").strip())
        scope.close()
    except ScopeError as exc:
        print("Scope error: {}".format(exc), file=sys.stderr)
        sys.exit(1)
