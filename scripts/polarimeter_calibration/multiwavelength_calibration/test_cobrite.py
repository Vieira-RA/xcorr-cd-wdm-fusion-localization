#!/usr/bin/env python3
"""Hardware-free tests for the CoBrite SCPI client.

A fake instrument is served on localhost: it implements the subset of commands
from ``cobrite/Manual_CoBrite.pdf`` that the client uses, including the tuning
state machine (``BUSY?`` = 1 while tuning, then 0) and error replies.
"""
import contextlib
import csv
import io
import json
import os
import socket
import sys
import tempfile
import textwrap
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from cobrite import CoBriteClient, CoBriteError, probe
from laser_client import LaserControlError, LaserController
import sweep_calibration

IDN = "IDP-COBRITE CBDX-NC-NN-NN-NN-FA, SN 19160001, F/W Ver 1.0.0(101), HW Ver 1.00"


class FakeCoBrite:
    """Threaded fake CoBrite speaking the documented SCPI dialect."""

    def __init__(
        self,
        limits=(1528.0, 1565.0),
        wavelength=1550.0,
        settle_polls=2,
        power_setpoint=14.0,
        actual_power=13.8,
        output_state=1,
        echo=False,
    ):
        self.limits = tuple(float(value) for value in limits)
        self.wavelength = float(wavelength)
        self.settle_polls = int(settle_polls)
        self.power_setpoint = float(power_setpoint)
        self.actual_power = float(actual_power)
        self.output_state = int(output_state)
        self.echo = bool(echo)
        self._remaining_polls = 0
        self.commands = []

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(8)
        self.port = self._server.getsockname()[1]
        self._stopped = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def close(self):
        self._stopped = True
        try:
            self._server.close()
        except OSError:
            pass

    def config(self, **overrides):
        settings = {
            "host": "127.0.0.1",
            "port": self.port,
            "address": "1,1,1",
            "timeout_s": 5.0,
            "min_settle_s": 0.0,
            "poll_interval_s": 0.01,
            "settle_s": 0.0,
            "settle_timeout_s": 10.0,
            "check_power": False,
        }
        settings.update(overrides)
        return settings

    def _serve(self):
        while not self._stopped:
            try:
                connection, _ = self._server.accept()
            except OSError:
                return
            with connection:
                self._handle(connection)

    def _handle(self, connection):
        buffer = b""
        while not self._stopped:
            try:
                data = connection.recv(4096)
            except OSError:
                return
            if not data:
                return
            buffer += data
            while b";" in buffer:
                raw, buffer = buffer.split(b";", 1)
                command = raw.decode("ascii", errors="replace").strip()
                if not command:
                    continue
                reply = self._dispatch(command)
                payload = command + ";" + reply + ";" if self.echo else reply + ";"
                connection.sendall(payload.encode("ascii"))

    def _dispatch(self, command):
        self.commands.append(command)
        head, _, tail = command.partition(" ")
        token = head.upper()
        argument = tail.strip()

        if token in ("ECHO", "PASS", "INTI", "DEFAULT"):
            return ""
        if token == "*IDN?":
            return IDN
        if token == "*OPC?":
            return "1"
        if token == ":SYST:ERR?":
            return "0, no error"
        if token == "WAV:LIM?":
            return "{},{}".format(*self.limits)
        if token == "WAV?":
            return "{:.4f}".format(self.wavelength)
        if token == "WAV":
            fields = argument.split(",")
            try:
                target = float(fields[-1])
            except (IndexError, ValueError):
                return "ERR 100, unknown command"
            if not self.limits[0] <= target <= self.limits[1]:
                return "ERR 101, Parameter out of range"
            self.wavelength = target          # the set value is visible at once
            self._remaining_polls = self.settle_polls
            return ""
        if token == "BUSY?":
            if self._remaining_polls > 0:
                self._remaining_polls -= 1
                return "1"
            return "0"
        if token == "POW?":
            return "{:.2f}".format(self.power_setpoint)
        if token == "APOW?":
            return "{:.2f}".format(self.actual_power)
        if token == "POW":
            try:
                self.power_setpoint = float(argument.split(",")[-1])
            except (IndexError, ValueError):
                return "ERR 100, unknown command"
            return ""
        if token == "STAT?":
            return str(self.output_state)
        if token == "STAT":
            try:
                self.output_state = int(float(argument.split(",")[-1]))
            except (IndexError, ValueError):
                return "ERR 100, unknown command"
            return ""
        if token == "FREQ?":
            return "{:.4f}".format(299792.458 / self.wavelength)
        return "ERR 100, unknown command"


class CoBriteClientTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeCoBrite()
        self.addCleanup(self.fake.close)

    def client(self, **overrides):
        client = CoBriteClient(self.fake.config(**overrides))
        self.addCleanup(client.close)
        return client.connect()

    def test_connect_reads_identity_and_limits(self):
        client = self.client()
        self.assertIn("IDP-COBRITE", client.idn())
        self.assertEqual(client.wavelength_limits(), (1528.0, 1565.0))
        self.assertAlmostEqual(client.wavelength(), 1550.0, places=4)
        self.assertEqual(client.output_state(), 1)
        self.assertFalse(client.busy())

    def test_limits_are_cached_after_the_first_query(self):
        client = self.client()
        client.wavelength_limits()
        queried = [c for c in self.fake.commands if c.startswith("WAV:LIM?")]
        client.wavelength_limits()
        self.assertEqual(len(queried), 1)

    def test_set_wavelength_waits_for_busy_to_clear(self):
        client = self.client()
        client.set_wavelength(1552.4)
        result = client.wait_until_settled(1552.4, tolerance_nm=0.01)
        self.assertAlmostEqual(result["readback_nm"], 1552.4, places=4)
        self.assertGreaterEqual(result["busy_polls"], 1)
        self.assertTrue(result["saw_busy"])

    def test_set_wavelength_rejects_out_of_range_before_sending(self):
        client = self.client()
        with self.assertRaises(CoBriteError):
            client.set_wavelength(1600.0)
        self.assertFalse([c for c in self.fake.commands if c.upper().startswith("WAV ")])
        self.assertAlmostEqual(client.wavelength(), 1550.0, places=4)

    def test_instrument_error_reply_is_raised(self):
        self.fake.limits = (1400.0, 1600.0)   # widen so the client pre-check passes
        client = self.client()
        self.fake.limits = (1528.0, 1565.0)   # the instrument itself now refuses
        # Cached limits say 1400-1600, so the set is attempted and rejected.
        client._limits = (1400.0, 1600.0)
        with self.assertRaises(CoBriteError):
            client.set_wavelength(1590.0)

    def test_settle_timeout_when_target_never_verifies(self):
        self.fake.settle_polls = 10 ** 9
        client = self.client()
        client.set_wavelength(1551.0)
        with self.assertRaises(CoBriteError):
            client.wait_until_settled(1555.0, timeout_s=0.6, poll_interval_s=0.05)

    def test_echo_mode_does_not_break_parsing(self):
        self.fake.echo = True
        client = self.client()
        self.assertAlmostEqual(client.wavelength(), 1550.0, places=4)
        self.assertEqual(client.wavelength_limits(), (1528.0, 1565.0))
        self.assertEqual(client.output_state(), 1)

    def test_wait_until_settled_requires_several_polls(self):
        """WAV? reports the target while still tuning, so BUSY? must gate it."""
        self.fake.settle_polls = 3
        client = self.client()
        client.set_wavelength(1548.0)
        result = client.wait_until_settled(1548.0, tolerance_nm=0.01, min_polls=2)
        self.assertGreaterEqual(result["busy_polls"], 3)

    def test_probe_reports_instrument_state(self):
        info = probe(self.fake.config())
        self.assertIn("IDP-COBRITE", info["idn"])
        self.assertEqual(info["address"], "1,1,1")
        self.assertEqual(info["limits_nm"], [1528.0, 1565.0])
        self.assertAlmostEqual(info["wavelength_nm"], 1550.0, places=4)
        self.assertAlmostEqual(info["actual_power_dbm"], 13.8, places=2)
        self.assertEqual(info["output_state"], 1)

    def test_connection_failure_is_reported_clearly(self):
        client = CoBriteClient(
            {"host": "127.0.0.1", "port": 1, "timeout_s": 0.5}
        )
        with self.assertRaises(CoBriteError):
            client.connect()


class LaserControllerCoBriteTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeCoBrite()
        self.addCleanup(self.fake.close)

    def controller(self, **cobrite):
        settings = self.fake.config(**cobrite)
        controller = LaserController({"mode": "cobrite", "cobrite": settings})
        self.addCleanup(controller.close)
        return controller

    def test_mode_aliases(self):
        self.assertEqual(LaserController({"mode": "scpi"}).mode, "cobrite")
        with self.assertRaises(LaserControlError):
            LaserController({"mode": "nonsense"})

    def test_set_and_wait_returns_readback_metadata(self):
        controller = self.controller()
        meta = controller.set_and_wait(1549.2)
        self.assertAlmostEqual(meta["requested_nm"], 1549.2, places=4)
        self.assertAlmostEqual(meta["readback_nm"], 1549.2, places=4)
        self.assertEqual(meta["readback_source"], "CoBrite SCPI WAV?")
        self.assertEqual(meta["output"], "1,1,1")
        self.assertEqual(meta["limits_nm"], [1528.0, 1565.0])
        self.assertEqual(meta["output_enabled"], 1)
        self.assertIn("settle_s", meta)

    def test_out_of_range_raises_laser_control_error(self):
        controller = self.controller()
        with self.assertRaises(LaserControlError):
            controller.set_and_wait(1500.0)

    def test_power_verification_flags_a_dead_output(self):
        self.fake.actual_power = 1.0     # 13 dB below a 14 dBm setpoint
        controller = self.controller(check_power=True, power_tolerance_db=3.0)
        with self.assertRaises(LaserControlError):
            controller.set_and_wait(1550.0)

    def test_power_verification_accepts_a_healthy_output(self):
        controller = self.controller(check_power=True, power_tolerance_db=3.0)
        meta = controller.set_and_wait(1550.0)
        self.assertAlmostEqual(meta["actual_power_dbm"], 13.8, places=2)

    def test_output_off_is_reported_in_the_metadata(self):
        self.fake.output_state = 0
        controller = self.controller()
        meta = controller.set_and_wait(1550.0)
        self.assertEqual(meta["output_enabled"], 0)
        self.assertIn("OFF", meta["note"])

    def test_enable_output_switches_the_laser_on(self):
        self.fake.output_state = 0
        controller = self.controller(enable_output=True)
        meta = controller.set_and_wait(1550.0)
        self.assertEqual(meta["output_enabled"], 1)
        self.assertEqual(self.fake.output_state, 1)

    def test_power_setpoint_is_sent_before_tuning(self):
        controller = self.controller(power_dbm=9.5)
        meta = controller.set_and_wait(1550.0)
        self.assertAlmostEqual(meta["power_setpoint_dbm"], 9.5, places=3)
        self.assertAlmostEqual(self.fake.power_setpoint, 9.5, places=3)
        order = [c.split(" ")[0].upper() for c in self.fake.commands]
        self.assertLess(order.index("POW"), order.index("WAV"))

    def test_describe_reads_the_instrument(self):
        info = self.controller().describe()
        self.assertIn("IDP-COBRITE", info["idn"])
        self.assertEqual(info["limits_nm"], [1528.0, 1565.0])

    def test_manual_mode_does_not_open_a_connection(self):
        controller = LaserController({"mode": "manual"})
        self.assertEqual(controller.describe(), {"mode": "manual"})
        controller.close()


class SweepWithLaserTests(unittest.TestCase):
    """End-to-end: real sweep code, fake CoBrite, stub acquisition script.

    The stub replaces ``fused_acquisition_controlled.py`` so the whole
    subprocess + cwd + CSV-discovery + fit + conditioning + report path runs
    without any instrument being present.
    """

    def setUp(self):
        self.fake = FakeCoBrite(actual_power=9.9)   # 10 dBm setpoint, healthy output
        self.addCleanup(self.fake.close)
        self.module_dir = str(Path(__file__).resolve().parent)

    def _write_config(self, directory, run_root, stub_path):
        config = {
            "wavelengths_nm": [1550.0],
            "n_sops": 12,
            "validation_fraction": 0.25,
            "random_seed": 4,
            "continue_on_error": False,
            "simulate": False,
            "max_condition": 3.0,
            "fail_on_poor_conditioning": False,
            "acquisition_script": str(stub_path),
            "output_root": str(run_root),
            "operator_checklist": [],
            "laser": {
                "mode": "cobrite",
                "tolerance_nm": 0.05,
                "settle_s": 0.0,
                "readback_timeout_s": 20.0,
                "poll_interval_s": 0.01,
                "cobrite": self.fake.config(
                    power_dbm=10.0, check_power=True, power_tolerance_db=3.0
                ),
            },
        }
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def _write_stub(self, directory):
        stub = Path(directory) / "fake_acquisition.py"
        stub.write_text(
            textwrap.dedent(
                """
                import sys
                sys.path.insert(0, {module_dir!r})
                from synthetic import simulate_fused_acquisition

                simulate_fused_acquisition(
                    output_dir=".",
                    wavelength_nm=1550.0,
                    n_sops=int(sys.argv[1]),
                    seed=21,
                )
                """
            ).format(module_dir=self.module_dir),
            encoding="utf-8",
        )
        return stub

    def test_sweep_sets_the_laser_and_writes_all_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs"
            stub = self._write_stub(directory)
            config_path = self._write_config(directory, run_root, stub)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                exit_code = sweep_calibration.main(["--config", str(config_path)])

            self.assertEqual(exit_code, 0, stdout.getvalue())
            self.assertTrue(
                any(command.upper().startswith("WAV ") for command in self.fake.commands),
                "the sweep never sent a wavelength command",
            )
            wavelength_command = [
                command for command in self.fake.commands if command.upper().startswith("WAV ")
            ][-1]
            self.assertIn("1550", wavelength_command)

            run_dirs = [path for path in run_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertTrue((run_dir / "summary.csv").is_file())
            self.assertTrue((run_dir / "all_calibration_matrices.npz").is_file())
            self.assertTrue((run_dir / "conditioning_report.csv").is_file())
            self.assertTrue((run_dir / "sweep_metadata.json").is_file())

            metadata = json.loads((run_dir / "sweep_metadata.json").read_text())
            self.assertFalse(metadata["simulated"])
            self.assertEqual(metadata["laser_mode"], "cobrite")
            self.assertIn("IDP-COBRITE", metadata["laser"]["idn"])

            per_lambda = json.loads(
                (run_dir / "lambda_1550p000nm" / "run_metadata.json").read_text()
            )
            self.assertEqual(per_lambda["laser"]["readback_source"], "CoBrite SCPI WAV?")
            self.assertAlmostEqual(per_lambda["laser"]["readback_nm"], 1550.0, places=4)

            with open(run_dir / "summary.csv", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(rows[0]["readback_source"], "CoBrite SCPI WAV?")
            self.assertEqual(rows[0]["simulated"], "False")

    def test_sweep_refuses_a_config_above_the_10_dbm_lab_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs"
            stub = self._write_stub(directory)
            config_path = self._write_config(directory, run_root, stub)
            config = json.loads(config_path.read_text())
            config["laser"]["cobrite"]["power_dbm"] = 12.0
            config_path.write_text(json.dumps(config), encoding="utf-8")

            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                exit_code = sweep_calibration.main(["--config", str(config_path)])

            self.assertEqual(exit_code, 2)
            self.assertIn("10 dBm lab limit", stderr.getvalue())
            self.assertFalse([c for c in self.fake.commands if c.upper().startswith("POW ")])

    def test_sweep_refuses_to_start_when_the_laser_is_unreachable(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "runs"
            stub = self._write_stub(directory)
            config_path = self._write_config(directory, run_root, stub)
            config = json.loads(config_path.read_text())
            config["laser"]["cobrite"]["port"] = 1   # nothing listens there
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exit_code = sweep_calibration.main(["--config", str(config_path)])

            self.assertEqual(exit_code, 2)
            self.assertFalse([c for c in self.fake.commands if c.upper().startswith("WAV ")])


if __name__ == "__main__":
    unittest.main()


class FakeSerialHandle:
    """Stand-in for a pyserial handle that answers in the CoBrite serial dialect."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.written = []
        self.timeout = 1.0
        self.closed = False
        self._pending = b""

    def write(self, data):
        self.written.append(data)
        command = data.decode("ascii").strip().rstrip(";")
        if command.upper().startswith("ECHO") or command.upper().startswith("PASS"):
            self._pending += b";\r\n"                      # bare acknowledgement
        else:
            self._pending += self.replies.pop(0).encode("ascii")

    def flush(self):
        pass

    def read(self, size=4096):
        if not self._pending:
            return b""
        payload, self._pending = self._pending, b""
        return payload

    def close(self):
        self.closed = True


class SerialTransportTests(unittest.TestCase):
    def _client(self, replies, **overrides):
        handle = FakeSerialHandle(replies)
        fake_serial = types.SimpleNamespace(
            Serial=lambda port, baudrate=115200, timeout=1.0, write_timeout=1.0: handle
        )
        config = {
            "transport": "serial",
            "com_port": "COM4",
            "timeout_s": 2.0,
            "min_settle_s": 0.0,
            "poll_interval_s": 0.01,
            "settle_s": 0.0,
        }
        config.update(overrides)
        with mock.patch.dict(sys.modules, {"serial": fake_serial}):
            client = CoBriteClient(config)
            client.connect()
        self.addCleanup(client.close)
        return client, handle

    def test_serial_link_is_reported(self):
        client, handle = self._client(
            ["IDP-COBRITE CBDX2-NC-NC-FA, SN 1, F/W 1.0;\r\n"], com_port="COM7"
        )
        self.assertEqual(client.idn().split(",")[0], "IDP-COBRITE CBDX2-NC-NC-FA")
        self.assertIn("serial COM7", client.connection_label())

    def test_ack_then_value_is_parsed(self):
        client, _ = self._client(
            [
                "IDP-COBRITE CBDX2;\r\n",
                "1524.498,1572.063;\r\n",
                "1550.0000;\r\n",
            ]
        )
        self.assertEqual(client.wavelength_limits(), (1524.498, 1572.063))
        self.assertAlmostEqual(client.wavelength(), 1550.0, places=4)

    def test_bare_acknowledgement_does_not_hang_a_set_command(self):
        # First reply answers *IDN? at connect, the second acknowledges the set.
        client, handle = self._client(["IDP-COBRITE CBDX2;\r\n", ";\r\n"])
        client._limits = (1524.0, 1572.0)
        client.set_wavelength(1550.0)          # ack only, must return promptly
        self.assertTrue(any(b"WAV" in command for command in handle.written))

    def test_missing_com_port_is_reported(self):
        with mock.patch.dict(sys.modules, {"serial": types.SimpleNamespace(Serial=lambda *a, **k: None)}):
            with self.assertRaises(Exception) as context:
                CoBriteClient({"transport": "serial"}).connect()
        self.assertIn("com_port", str(context.exception))

    def test_pyserial_absent_gives_an_actionable_error(self):
        with mock.patch.dict(sys.modules, {"serial": None}):
            with self.assertRaises(Exception) as context:
                CoBriteClient({"transport": "serial", "com_port": "COM4"}).connect()
        self.assertIn("pyserial", str(context.exception))
