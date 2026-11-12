#!/usr/bin/env python3
"""Hardware-free tests for the transport-neutral scope access layer."""
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scope_io


class FakeVxi11Instrument:
    def __init__(self):
        self.timeout = 0
        self.written = []
        self.closed = False

    def write(self, command):
        self.written.append(command)

    def ask(self, command):
        return "TEKTRONIX,DPO7254,C000001,CF:91.1CT"

    def read_raw(self, size=None):
        return b"\n"

    def close(self):
        self.closed = True


class FakeVisaResource:
    def __init__(self, timeout_ms=20000):
        self.timeout = timeout_ms
        self.write_termination = None
        self.read_termination = None
        self.written = []
        self.read_terminations = []
        self.closed = False

    def write(self, command):
        self.written.append(command)

    def query(self, command):
        return "TEKTRONIX,DPO7254,C000001,CF:91.1CT"

    def read_raw(self, size=None):
        self.read_terminations.append(self.read_termination)
        return b"#800001000"

    def close(self):
        self.closed = True


class FakeResourceManager:
    def __init__(self, resources, resource):
        self._resources = resources
        self._resource = resource
        self.opened = []

    def list_resources(self, pattern="?*"):
        return tuple(self._resources)

    def open_resource(self, name):
        self.opened.append(name)
        return self._resource


class Vxi11BackendTests(unittest.TestCase):
    def test_timeout_is_exposed_in_seconds_and_passthrough(self):
        instrument = FakeVxi11Instrument()
        scope = scope_io._Vxi11Scope(instrument, "198.192.1.1")
        scope.timeout = 12.5
        self.assertAlmostEqual(instrument.timeout, 12.5)
        self.assertAlmostEqual(scope.timeout, 12.5)

    def test_write_ask_and_close_are_forwarded(self):
        instrument = FakeVxi11Instrument()
        scope = scope_io._Vxi11Scope(instrument, "198.192.1.1")
        scope.write("ACQUIRE:STATE STOP")
        self.assertIn("TEKTRONIX", scope.ask("*IDN?"))
        self.assertEqual(instrument.written, ["ACQUIRE:STATE STOP"])
        scope.close()
        self.assertTrue(instrument.closed)

    def test_read_errors_are_wrapped(self):
        instrument = FakeVxi11Instrument()

        def boom(size=None):
            raise RuntimeError("read timed out")

        instrument.read_raw = boom
        scope = scope_io._Vxi11Scope(instrument, "198.192.1.1")
        with self.assertRaises(scope_io.ScopeTimeout):
            scope.read_raw(1)


class PyVisaBackendTests(unittest.TestCase):
    def test_timeout_is_converted_to_milliseconds(self):
        resource = FakeVisaResource()
        scope = scope_io._PyVisaScope(resource, "USB0::0x0699::0x0364::C000001::INSTR")
        scope.timeout = 30.0
        self.assertEqual(resource.timeout, 30000)
        self.assertAlmostEqual(scope.timeout, 30.0)
        scope.timeout = 0.5
        self.assertEqual(resource.timeout, 500)

    def test_text_queries_use_newline_termination(self):
        resource = FakeVisaResource()
        scope = scope_io._PyVisaScope(resource, "USB0::x::INSTR")
        self.assertEqual(resource.write_termination, "\n")
        self.assertEqual(resource.read_termination, "\n")
        self.assertIn("TEKTRONIX", scope.ask("*IDN?"))

    def test_binary_reads_temporarily_disable_the_terminator(self):
        resource = FakeVisaResource()
        scope = scope_io._PyVisaScope(resource, "USB0::x::INSTR")
        payload = scope.read_raw(2)
        self.assertEqual(payload, b"#800001000")
        # terminator disabled for the binary read, restored afterwards
        self.assertEqual(resource.read_terminations, [None])
        self.assertEqual(resource.read_termination, "\n")

    def test_close_is_forwarded(self):
        resource = FakeVisaResource()
        scope = scope_io._PyVisaScope(resource, "USB0::x::INSTR")
        scope.close()
        self.assertTrue(resource.closed)


class OpenScopeTests(unittest.TestCase):
    def test_resource_ranking_prefers_tektronix_usb(self):
        manager = FakeResourceManager(
            [
                "ASRL3::INSTR",
                "TCPIP0::192.168.0.9::INSTR",
                "USB0::0x1234::0x5678::X::INSTR",
                "USB0::0x0699::0x0364::C000001::INSTR",
            ],
            FakeVisaResource(),
        )
        ranked = scope_io._candidate_resources(manager, "198.192.1.1")
        self.assertEqual(ranked[0], "USB0::0x0699::0x0364::C000001::INSTR")

    def test_resource_ranking_tries_the_direct_vxi11_form_first(self):
        # VISA often lists nothing for a directly addressed LAN instrument, so the
        # canonical VXI-11 resource strings are synthesised from the IP and tried
        # before whatever VISA happened to discover.
        manager = FakeResourceManager(
            ["TCPIP0::198.192.1.1::INSTR", "TCPIP0::10.0.0.5::INSTR"], FakeVisaResource()
        )
        ranked = scope_io._candidate_resources(manager, "198.192.1.1")
        self.assertEqual(ranked[0], "TCPIP0::198.192.1.1::inst0::INSTR")
        self.assertIn("TCPIP0::198.192.1.1::INSTR", ranked)

    def test_abbreviated_tcpip_form_is_normalised(self):
        manager = FakeResourceManager(["TCPIP::198.192.1.1::inst0::INSTR"], FakeVisaResource())
        ranked = scope_io._candidate_resources(manager, "198.192.1.1")
        self.assertTrue(all(not r.upper().startswith("TCPIP::") for r in ranked))

    def test_pyvisa_transport_with_explicit_resource(self):
        resource = FakeVisaResource()
        manager = FakeResourceManager([], resource)
        fake_pyvisa = types.SimpleNamespace(ResourceManager=lambda: manager)
        with mock.patch.dict(sys.modules, {"pyvisa": fake_pyvisa}):
            scope = scope_io.open_scope(
                transport="pyvisa",
                resource="USB0::0x0699::0x0364::C000001::INSTR",
                timeout=5.0,
                verbose=False,
            )
        self.assertEqual(manager.opened, ["USB0::0x0699::0x0364::C000001::INSTR"])
        self.assertEqual(scope.address, "USB0::0x0699::0x0364::C000001::INSTR")
        self.assertEqual(resource.timeout, 5000)

    def test_unreachable_instrument_reports_what_it_tried(self):
        manager = FakeResourceManager([], FakeVisaResource())

        def refuse(name):
            raise RuntimeError("instrument unreachable")

        manager.open_resource = refuse
        fake_pyvisa = types.SimpleNamespace(ResourceManager=lambda *args: manager)
        with mock.patch.dict(sys.modules, {"pyvisa": fake_pyvisa}):
            with self.assertRaises(scope_io.ScopeError) as context:
                scope_io.open_scope(ip="192.0.2.1", transport="pyvisa",
                                    resource=None, verbose=False)
        message = str(context.exception)
        self.assertIn("No VISA backend could reach", message)
        self.assertIn("192.0.2.1", message)

    def test_vxi11_transport_uses_the_ip(self):
        instrument = FakeVxi11Instrument()

        class FakeInstrument:
            def __new__(cls, ip):
                cls.ip = ip
                return instrument

        fake_vxi11 = types.SimpleNamespace(Instrument=FakeInstrument)
        with mock.patch.dict(sys.modules, {"vxi11": fake_vxi11, "pyvisa": None}):
            scope = scope_io.open_scope(ip="169.254.102.200", transport="vxi11",
                                        timeout=7.0, verbose=False)
        self.assertEqual(FakeInstrument.ip, "169.254.102.200")
        self.assertEqual(instrument.timeout, 7.0)
        self.assertEqual(scope.address, "169.254.102.200")

    def test_environment_variables_are_honoured(self):
        instrument = FakeVxi11Instrument()

        class FakeInstrument:
            def __new__(cls, ip):
                cls.ip = ip
                return instrument

        fake_vxi11 = types.SimpleNamespace(Instrument=FakeInstrument)
        env = {
            "PM_SCOPE_IP": "169.254.102.201",
            "PM_SCOPE_TRANSPORT": "vxi11",
        }
        with mock.patch.dict(os.environ, env), \
                mock.patch.dict(sys.modules, {"vxi11": fake_vxi11}):
            scope_io.open_scope(verbose=False)
        self.assertEqual(FakeInstrument.ip, "169.254.102.201")

    def test_no_transport_available_reports_everything_it_tried(self):
        with mock.patch.dict(sys.modules, {"vxi11": None, "pyvisa": None}):
            with self.assertRaises(scope_io.ScopeError) as context:
                scope_io.open_scope(transport="auto", verbose=False)
        message = str(context.exception)
        self.assertIn("PM_SCOPE_RESOURCE", message)


if __name__ == "__main__":
    unittest.main()
