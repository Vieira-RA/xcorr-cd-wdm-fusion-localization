#!/usr/bin/env python3
"""Hardware-free tests for the pre-flight instrument tools.

Covers the parts of ``scope_probe.py`` and ``discover_laser.py`` that can be
verified without instruments: scale arithmetic, clipping/headroom decisions, and
TCP discovery against the fake CoBrite from ``test_cobrite``.
"""
import ipaddress
import unittest

import discover_laser
import scope_probe
from test_cobrite import FakeCoBrite

import arp_sweep


def _measurement(peak_v, mean_v=0.0, std_v=1e-5, samples=10000, rail=0):
    return {
        "samples": samples,
        "mean_v": mean_v,
        "std_v": std_v,
        "min_v": -peak_v,
        "max_v": peak_v,
        "peak_v": peak_v,
        "standard_error_v": std_v / samples ** 0.5 if samples else float("nan"),
        "rail_low": rail,
        "rail_high": 0,
    }


class ScaleSuggestionTests(unittest.TestCase):
    def test_nice_scale_picks_the_1_2_5_series(self):
        self.assertAlmostEqual(scope_probe._nice_scale(0.004), 0.005)
        self.assertAlmostEqual(scope_probe._nice_scale(0.005), 0.005)
        self.assertAlmostEqual(scope_probe._nice_scale(0.0051), 0.01)
        self.assertAlmostEqual(scope_probe._nice_scale(0.011), 0.02)
        self.assertAlmostEqual(scope_probe._nice_scale(0.031), 0.05)
        self.assertAlmostEqual(scope_probe._nice_scale(0.0), 0.001)
        self.assertLessEqual(scope_probe._nice_scale(1000.0), 10.0)

    def test_target_headroom_scale_reproduces_the_peak(self):
        peak = 40e-3
        scale = scope_probe._nice_scale(peak / (scope_probe.TARGET_HEADROOM * 4.0))
        usage = peak / (4.0 * scale)
        self.assertLessEqual(usage, scope_probe.TARGET_HEADROOM + 1e-9)


class HeadroomAnalysisTests(unittest.TestCase):
    def test_signal_below_the_headroom_target_passes(self):
        # 30 mV peak at 20 mV/div => 37.5% of the +/-80 mV full scale.
        measurements = {channel: _measurement(30e-3, mean_v=5e-3) for channel in (1, 2, 3, 4)}
        result = scope_probe.analyse(measurements, dict(scope_probe.DEFAULT_SCALES))
        self.assertFalse(result["any_clipped"])
        self.assertAlmostEqual(result["worst_usage_fraction"], 30e-3 / 80e-3, places=6)
        self.assertTrue(result["scale_ok"])

    def test_signal_near_full_scale_is_tight_but_not_clipped(self):
        # 65 mV peak => 81% of full scale: no rails, but beyond the 70% target.
        measurements = {channel: _measurement(65e-3) for channel in (1, 2, 3, 4)}
        result = scope_probe.analyse(measurements, dict(scope_probe.DEFAULT_SCALES))
        self.assertFalse(result["any_clipped"])
        self.assertFalse(result["scale_ok"])

    def test_low_level_signal_passes_and_suggests_smaller_scale(self):
        measurements = {channel: _measurement(5e-3) for channel in (1, 2, 3, 4)}
        result = scope_probe.analyse(measurements, dict(scope_probe.DEFAULT_SCALES))
        self.assertFalse(result["any_clipped"])
        self.assertTrue(result["scale_ok"])
        # 5 mV peak at 70% of full scale => 5/(0.7*4) = 1.79 mV/div -> 2 mV/div
        self.assertAlmostEqual(result["suggested_scales"][1], 0.002)

    def test_rail_counts_are_reported_as_clipping(self):
        measurements = {channel: _measurement(50e-3) for channel in (1, 2, 3, 4)}
        measurements[3] = _measurement(50e-3, rail=12)
        result = scope_probe.analyse(measurements, dict(scope_probe.DEFAULT_SCALES))
        self.assertTrue(result["any_clipped"])
        self.assertTrue([row["clipped"] for row in result["rows"] if row["channel"] == 3][0])
        self.assertFalse([row["clipped"] for row in result["rows"] if row["channel"] == 1][0])

    def test_usage_above_full_scale_is_clipping_even_without_rails(self):
        measurements = {channel: _measurement(120e-3) for channel in (1, 2, 3, 4)}
        result = scope_probe.analyse(measurements, dict(scope_probe.DEFAULT_SCALES))
        self.assertTrue(result["any_clipped"])

    def test_suggests_a_larger_scale_for_a_clipping_channel(self):
        measurements = {channel: _measurement(120e-3) for channel in (1, 2, 3, 4)}
        result = scope_probe.analyse(measurements, dict(scope_probe.DEFAULT_SCALES))
        self.assertGreater(result["suggested_scales"][1], 0.02)

    def test_snr_uses_the_standard_error_of_the_record_mean(self):
        measurements = {channel: _measurement(10e-3, mean_v=10e-3, std_v=1e-3) for channel in (1, 2, 3, 4)}
        result = scope_probe.analyse(measurements, dict(scope_probe.DEFAULT_SCALES))
        # std 1 mV, N = 10000 -> SEM 10 uV -> SNR 1000
        self.assertAlmostEqual(result["rows"][0]["snr_of_mean"], 1000.0, places=3)


class DiscoveryTests(unittest.TestCase):
    def test_candidate_subnets_narrow_wide_masks_to_a_slash_24(self):
        adapters = [
            {"name": "Ethernet 14", "ip": "169.254.22.216", "netmask": "255.255.0.0"},
            {"name": "Ethernet 15", "ip": "192.168.7.5", "netmask": "255.255.255.0"},
        ]
        networks = discover_laser.candidate_subnets(adapters, ["172.16.4.0/24", "nonsense"])
        self.assertIn(ipaddress.IPv4Network("169.254.22.0/24"), networks)
        self.assertIn(ipaddress.IPv4Network("192.168.7.0/24"), networks)
        self.assertIn(ipaddress.IPv4Network("172.16.4.0/24"), networks)
        self.assertEqual(len(networks), 3)

    def test_adapters_are_tagged_as_usb_when_the_name_says_so(self):
        adapters = [{"name": "Ethernet 14 Remote NDIS Compatible Device", "ip": "169.254.22.216"}]
        for adapter in adapters:
            name = adapter["name"].lower()
            adapter["kind"] = (
                "usb/rndis"
                if any(hint in name for hint in discover_laser.USB_HINTS)
                else "other"
            )
        self.assertEqual(adapters[0]["kind"], "usb/rndis")

    def test_probe_tcp_reads_the_identity_string(self):
        fake = FakeCoBrite()
        self.addCleanup(fake.close)
        identity = discover_laser.probe_tcp("127.0.0.1", fake.port, timeout=1.0)
        self.assertIsNotNone(identity)
        self.assertIn("IDP-COBRITE", identity)

    def test_probe_tcp_returns_none_when_nothing_listens(self):
        self.assertIsNone(discover_laser.probe_tcp("127.0.0.1", 1, timeout=0.3))


class ArpSweepTests(unittest.TestCase):
    WINDOWS_TABLE = """
Interface: 198.192.1.2 --- 0xe
  Internet Address      Physical Address      Type
  198.192.1.1           00-0f-bb-12-34-56     dynamic
  198.192.1.2           80-6d-97-05-ad-aa     dynamic
  198.192.255.255       ff-ff-ff-ff-ff-ff     static
  224.0.0.22            01-00-5e-00-00-16     static

Interface: 169.254.22.216 --- 0x54
  Internet Address      Physical Address      Type
  169.254.9.2           02-9c-24-07-b9-d4     dynamic
  169.254.255.255       ff-ff-ff-ff-ff-ff     static
"""

    def test_windows_table_is_parsed_and_placeholder_rows_dropped(self):
        entries = arp_sweep.parse_arp_table(self.WINDOWS_TABLE)
        addresses = {entry["ip"] for entry in entries}
        self.assertIn("198.192.1.1", addresses)
        self.assertIn("169.254.9.2", addresses)
        # broadcast / multicast placeholders are filtered out
        self.assertNotIn("198.192.255.255", addresses)
        self.assertNotIn("224.0.0.22", addresses)
        # the interface's own address is filtered out
        self.assertNotIn("198.192.1.2", addresses)

    def test_mac_is_normalised_to_dash_form(self):
        entries = arp_sweep.parse_arp_table(self.WINDOWS_TABLE)
        scope_like = [e for e in entries if e["ip"] == "198.192.1.1"][0]
        self.assertEqual(scope_like["mac"], "00-0f-bb-12-34-56")
        self.assertEqual(scope_like["kind"], "dynamic")

    def test_subnet_filter_keeps_only_the_wired_segment(self):
        import ipaddress

        entries = arp_sweep.parse_arp_table(
            self.WINDOWS_TABLE, ipaddress.IPv4Network("198.192.1.0/24")
        )
        self.assertEqual([entry["ip"] for entry in entries], ["198.192.1.1"])

    def test_entries_are_attributed_to_their_interface_block(self):
        entries = arp_sweep.parse_arp_table(self.WINDOWS_TABLE)
        by_ip = {entry["ip"]: entry["interface_ip"] for entry in entries}
        self.assertEqual(by_ip["198.192.1.1"], "198.192.1.2")
        self.assertEqual(by_ip["169.254.9.2"], "169.254.22.216")

    def test_posix_table_is_also_understood(self):
        table = (
            "? (198.192.1.1) at 00:0f:bb:12:34:56 [ether] on eth0\n"
            "? (198.192.1.9) at ff:ff:ff:ff:ff:ff [ether] on eth0\n"
        )
        entries = arp_sweep.parse_arp_table(table)
        self.assertEqual([entry["ip"] for entry in entries], ["198.192.1.1"])
        self.assertEqual(entries[0]["mac"], "00-0f-bb-12-34-56")

    def test_empty_table_yields_nothing(self):
        self.assertEqual(arp_sweep.parse_arp_table(""), [])

    def test_probe_ports_reports_only_listening_ports(self):
        fake = FakeCoBrite()
        self.addCleanup(fake.close)
        self.assertEqual(arp_sweep.probe_ports("127.0.0.1", (fake.port,), timeout=1.0), [fake.port])
        self.assertEqual(arp_sweep.probe_ports("127.0.0.1", (1,), timeout=0.3), [])


if __name__ == "__main__":
    unittest.main()


class CrossWavelengthTests(unittest.TestCase):
    def test_angular_error_is_zero_for_identical_vectors(self):
        import numpy as np
        import cross_wavelength_validation as cwv

        stokes = np.array([[1.0, 1.0], [0.6, -0.6], [0.8, 0.0], [0.0, 0.8]])
        np.testing.assert_allclose(cwv._angular_error_deg(stokes, stokes), 0.0, atol=1e-9)

    def test_angular_error_is_ninety_degrees_for_orthogonal_polarization(self):
        import numpy as np
        import cross_wavelength_validation as cwv

        a = np.array([[1.0], [1.0], [0.0], [0.0]])
        b = np.array([[1.0], [0.0], [1.0], [0.0]])
        np.testing.assert_allclose(cwv._angular_error_deg(a, b), 90.0, atol=1e-6)

    def test_cross_validation_recovers_a_perfect_calibration(self):
        import numpy as np
        import cross_wavelength_validation as cwv

        rng = np.random.default_rng(7)
        true_matrix = np.array([[2.0, 0.2, 0.1, 0.0],
                                [0.1, 1.5, 0.2, 0.1],
                                [0.0, 0.2, 1.6, 0.1],
                                [0.1, 0.0, 0.2, 1.4]])
        entries = []
        for index, wavelength in enumerate((1530.0, 1550.0)):
            detector = rng.normal(size=(4, 20))
            stokes = true_matrix @ detector
            entries.append({"wavelength_nm": wavelength, "C": true_matrix.copy(),
                            "D": detector, "S": stokes})
        result = cwv.cross_validate(entries)
        np.testing.assert_allclose(result["mean_angle_deg"], 0.0, atol=1e-6)
        self.assertEqual(result["wavelengths_nm"], [1530.0, 1550.0])
        self.assertLess(float(np.nanmax(result["normalized_stokes_rmse"])), 1e-9)
