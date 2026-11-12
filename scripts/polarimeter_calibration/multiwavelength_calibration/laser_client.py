#!/usr/bin/env python3
"""Manual, HTTP or CoBrite-SCPI control of a tunable laser.

Three modes, selected by ``laser.mode`` in the sweep config:

``manual`` (default)
    The operator sets the wavelength on the instrument and types the read-back.
    No transport is opened, so nothing can be misconfigured by accident.
``cobrite``
    Native ID Photonics CoBrite control over SCPI (TCP 2000): sets the
    wavelength, polls ``BUSY?`` until the port has settled, then verifies
    ``WAV?`` against the request and checks the measured power. See
    ``cobrite.py`` and ``cobrite/Manual_CoBrite.pdf``.
``http``
    Deliberately configuration-driven HTTP adapter with no vendor URL assumed.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

try:  # works as a module inside the package and as a flat script import
    from .cobrite import CoBriteClient, CoBriteError
    from .cobrite import probe as cobrite_probe
    from .cobrite import _first_float  # noqa: F401  (kept for compatibility)
except ImportError:  # pragma: no cover - script-style execution
    from cobrite import CoBriteClient, CoBriteError
    from cobrite import probe as cobrite_probe
    from cobrite import _first_float  # noqa: F401  (kept for compatibility)


class LaserControlError(RuntimeError):
    """Raised when wavelength setting or read-back verification fails."""


def _render(value: Any, context: Dict[str, Any]) -> Any:
    """Recursively substitute ``{wavelength_nm}`` and ``{output}`` in config."""
    if isinstance(value, dict):
        return {key: _render(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_render(item, context) for item in value]
    if not isinstance(value, str):
        return value
    if value in context:
        return context[value]
    try:
        return value.format(**context)
    except (KeyError, ValueError):
        return value


def _nested_value(payload: Any, dotted_path: str) -> Any:
    value = payload
    if not dotted_path:
        return value
    for key in dotted_path.split("."):
        if isinstance(value, dict):
            if key not in value:
                raise LaserControlError(
                    "Wavelength response has no key {!r} (path {!r}).".format(key, dotted_path)
                )
            value = value[key]
        elif isinstance(value, (list, tuple)):
            try:
                value = value[int(key)]
            except (ValueError, IndexError):
                raise LaserControlError(
                    "Cannot follow response path {!r} through list index {!r}.".format(
                        dotted_path, key
                    )
                )
        else:
            raise LaserControlError(
                "Cannot follow response path {!r}; encountered {!r}.".format(dotted_path, value)
            )
    return value


class LaserController:
    """Set a wavelength manually or through a user-configured HTTP API.

    HTTP configuration format::

        {
          "mode": "http",
          "output": 1,
          "tolerance_nm": 0.1,
          "settle_s": 2.0,
          "timeout_s": 5.0,
          "readback_timeout_s": 30.0,
          "poll_interval_s": 0.5,
          "response_wavelength_path": "wavelength_nm",
          "headers": {},
          "set_request": {
            "method": "POST", "url": "http://.../set",
            "json": {"wavelength_nm": "{wavelength_nm}", "output": "{output}"}
          },
          "read_request": {
            "method": "GET", "url": "http://.../status",
            "params": {"output": "{output}"}
          }
        }

    The exact payload/response keys must be adapted to the laser's manual/API.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config or {})
        self.mode = str(self.config.get("mode", "manual")).lower()
        if self.mode not in ("manual", "http", "cobrite", "scpi"):
            raise LaserControlError(
                "laser.mode must be 'manual', 'cobrite' or 'http' (got {!r}).".format(self.mode)
            )
        if self.mode == "scpi":
            self.mode = "cobrite"
        self._session: Optional[Any] = None
        self._cobrite: Optional[CoBriteClient] = None

    # ------------------------------------------------------------------ CoBrite SCPI
    def _cobrite_options(self) -> Dict[str, Any]:
        block = self.config.get("cobrite")
        if block is None:
            return {}
        if not isinstance(block, dict):
            raise LaserControlError("laser.cobrite must be a JSON object.")
        return dict(block)

    def _cobrite_client(self) -> CoBriteClient:
        if self._cobrite is None:
            settings = self._cobrite_options()
            # Allow transport keys directly under ``laser`` as a shorthand.
            for key in ("transport", "com_port", "baudrate", "host", "port",
                        "address", "timeout_s", "password", "echo"):
                if key in self.config and key not in settings:
                    settings[key] = self.config[key]
            try:
                self._cobrite = CoBriteClient(settings)
            except CoBriteError as exc:
                raise LaserControlError(str(exc))
        return self._cobrite

    def describe(self) -> Dict[str, Any]:
        """Read the instrument identity/state (CoBrite mode only; connects)."""
        if self.mode != "cobrite":
            return {"mode": self.mode}
        settings = self._cobrite_options()
        for key in ("transport", "com_port", "baudrate", "host", "port",
                    "address", "timeout_s", "password"):
            if key in self.config and key not in settings:
                settings[key] = self.config[key]
        try:
            return cobrite_probe(settings)
        except CoBriteError as exc:
            raise LaserControlError(str(exc))

    def _cobrite_set_and_wait(
        self, wavelength_nm: float, output: Any, tolerance: float
    ) -> Dict[str, Any]:
        options = self._cobrite_options()
        client = self._cobrite_client()

        # The CoBrite port address is <chassis>,<slot>,<device>. ``output`` is
        # only used as a fallback device index when no explicit address is set.
        if output not in (None, "") and not options.get("address"):
            client.address = "1,1,{}".format(output)

        power_dbm = options.get("power_dbm")
        enable_output = bool(options.get("enable_output", False))
        check_power = bool(options.get("check_power", True))
        power_tolerance_db = float(options.get("power_tolerance_db", 3.0))
        settle_s = float(options.get("settle_s", self.config.get("settle_s", 0.5)))
        settle_timeout_s = float(
            options.get("settle_timeout_s", self.config.get("readback_timeout_s", 120.0))
        )
        poll_s = float(options.get("poll_interval_s", self.config.get("poll_interval_s", 0.25)))
        min_settle_s = float(options.get("min_settle_s", 1.0))

        try:
            client.connect()
            limits = client.wavelength_limits()
            if not limits[0] <= float(wavelength_nm) <= limits[1]:
                raise LaserControlError(
                    "CoBrite range is {:.3f}-{:.3f} nm; {:.3f} nm is outside it.".format(
                        limits[0], limits[1], float(wavelength_nm)
                    )
                )

            if power_dbm is not None:
                client.set_power(float(power_dbm))
            power_setpoint = client.power()

            state = client.output_state()
            if enable_output and state != 1:
                client.set_output(True)
                state = client.output_state()

            client.set_wavelength(wavelength_nm)
            client.operation_complete()
            settled = client.wait_until_settled(
                wavelength_nm,
                tolerance_nm=tolerance,
                timeout_s=settle_timeout_s,
                poll_interval_s=poll_s,
                min_settle_s=min_settle_s,
            )
            readback = float(settled["readback_nm"])
            if abs(readback - float(wavelength_nm)) > tolerance:
                raise LaserControlError(
                    "CoBrite read-back {:.6f} nm is off target {:.6f} nm (tolerance {:.6f} nm).".format(
                        readback, float(wavelength_nm), tolerance
                    )
                )

            if settle_s > 0:
                time.sleep(settle_s)

            actual_power = None
            power_note = ""
            if state == 1:
                try:
                    actual_power = client.actual_power()
                except CoBriteError as exc:
                    power_note = str(exc)
            else:
                power_note = "laser output is OFF; no light is reaching the calibration path"

            if check_power and state == 1 and actual_power is not None:
                if abs(actual_power - float(power_setpoint)) > power_tolerance_db:
                    raise LaserControlError(
                        "CoBrite output power {:.2f} dBm deviates from the {:.2f} dBm setpoint "
                        "by more than {:.1f} dB; check the laser, shutter or patch cabling.".format(
                            actual_power, float(power_setpoint), power_tolerance_db
                        )
                    )

            return {
                "requested_nm": float(wavelength_nm),
                "readback_nm": readback,
                "readback_source": "CoBrite SCPI WAV?",
                "output": client.address,
                "link": client.connection_label(),
                "limits_nm": [float(limits[0]), float(limits[1])],
                "settle_s": float(settled["settle_s"]),
                "busy_polls": int(settled["busy_polls"]),
                "power_setpoint_dbm": float(power_setpoint),
                "max_power_dbm": float(client.max_power_dbm),
                "actual_power_dbm": actual_power,
                "output_enabled": state == 1,
                "note": power_note,
            }
        except CoBriteError as exc:
            raise LaserControlError(str(exc))

    def _http_session(self) -> Any:
        if self._session is None:
            try:
                import requests
            except ImportError as exc:
                raise LaserControlError(
                    "HTTP laser mode requires the 'requests' package."
                ) from exc
            self._session = requests.Session()
        return self._session

    def _request(self, request_config: Dict[str, Any], context: Dict[str, Any]) -> Any:
        if not isinstance(request_config, dict):
            raise LaserControlError("HTTP set_request/read_request must be an object.")
        url = _render(request_config.get("url", ""), context)
        method = str(request_config.get("method", "GET")).upper()
        if not url or "CHANGE-ME" in str(url):
            raise LaserControlError("Configure the laser HTTP request URL before using HTTP mode.")
        if method not in ("GET", "POST", "PUT", "PATCH"):
            raise LaserControlError("Unsupported HTTP method {!r}.".format(method))

        params = _render(request_config.get("params"), context)
        json_body = _render(request_config.get("json"), context)
        data = _render(request_config.get("data"), context)
        headers = dict(self.config.get("headers", {}))
        headers.update(request_config.get("headers", {}))

        response = self._http_session().request(
            method=method,
            url=url,
            params=params,
            json=json_body,
            data=data,
            headers=headers,
            timeout=float(self.config.get("timeout_s", 5.0)),
        )
        response.raise_for_status()
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text.strip()

    def _read_wavelength(self, output: Any) -> float:
        request_config = self.config.get("read_request")
        if not request_config:
            raise LaserControlError(
                "HTTP mode requires read_request so the set wavelength can be verified."
            )
        context = {"output": output}
        payload = self._request(request_config, context)
        path = str(self.config.get("response_wavelength_path", ""))
        value = _nested_value(payload, path)
        try:
            return float(value)
        except (TypeError, ValueError):
            raise LaserControlError(
                "Could not parse wavelength from response value {!r}.".format(value)
            )

    def set_and_wait(self, wavelength_nm: float, output: Any = None) -> Dict[str, Any]:
        """Set the requested wavelength and return requested/read-back metadata."""
        requested = float(wavelength_nm)
        output = self.config.get("output") if output is None else output
        tolerance = float(self.config.get("tolerance_nm", 0.1))

        if self.mode == "manual":
            print("\nSet laser output {!r} to {:.6f} nm using its normal controls.".format(
                output, requested
            ))
            input("Wait for the laser to settle, then press Enter to enter its read-back.")
            readback_text = input("Actual wavelength read-back in nm: ").strip()
            try:
                readback = float(readback_text)
            except ValueError:
                raise LaserControlError("Enter the wavelength shown by the laser in nm.")
            if abs(readback - requested) > tolerance:
                raise LaserControlError(
                    "Laser read-back {:.6f} nm is {:.6f} nm from requested {:.6f} nm "
                    "(tolerance {:.6f} nm).".format(
                        readback, abs(readback - requested), requested, tolerance
                    )
                )
            return {
                "requested_nm": requested,
                "readback_nm": readback,
                "readback_source": "operator-entered laser display",
                "output": output,
            }

        if self.mode == "cobrite":
            return self._cobrite_set_and_wait(requested, output, tolerance)

        set_request = self.config.get("set_request")
        if not set_request:
            raise LaserControlError("HTTP mode requires a configured set_request.")
        context = {"wavelength_nm": requested, "output": output}
        self._request(set_request, context)

        settle_s = float(self.config.get("settle_s", 1.0))
        if settle_s > 0:
            time.sleep(settle_s)

        timeout_s = float(self.config.get("readback_timeout_s", 30.0))
        poll_s = max(0.05, float(self.config.get("poll_interval_s", 0.5)))
        deadline = time.monotonic() + timeout_s
        last_readback = None
        while time.monotonic() <= deadline:
            last_readback = self._read_wavelength(output)
            if abs(last_readback - requested) <= tolerance:
                return {
                    "requested_nm": requested,
                    "readback_nm": last_readback,
                    "readback_source": "HTTP status",
                    "output": output,
                }
            time.sleep(poll_s)

        raise LaserControlError(
            "Laser read-back did not reach {:.6f} nm within {:.1f} s; last value was {!r}.".format(
                requested, timeout_s, last_readback
            )
        )

    def close(self) -> None:
        if self._cobrite is not None:
            self._cobrite.close()
            self._cobrite = None
        if self._session is not None:
            self._session.close()
            self._session = None
