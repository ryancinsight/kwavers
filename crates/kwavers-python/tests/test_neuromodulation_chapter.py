from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))

import ch26_neuromodulation as nm  # noqa: E402


def test_default_protocol_satisfies_acoustic_safety_values():
    protocol = nm.Protocol()
    mi = float(nm.mechanical_index(protocol.pressure_pa, protocol.frequency_hz))
    ispta_w_cm2 = float(nm.intensity_sppa_w_m2(protocol.pressure_pa) * protocol.duty_cycle / 1.0e4)

    assert 0.40 < mi < 0.45
    assert 0.10 < ispta_w_cm2 < 0.20
    assert mi < 1.9
    assert ispta_w_cm2 < 0.72


def test_gaussian_focus_peaks_at_target_and_decays_off_axis():
    protocol = nm.Protocol(pressure_pa=300.0e3)
    field = nm.gaussian_focus(protocol, shape=(21, 21, 21))
    pressure = field["pressure"]
    center = tuple(dim // 2 for dim in pressure.shape)

    assert float(pressure[center]) == protocol.pressure_pa
    assert float(pressure[0, 0, 0]) < 1.0e3
    assert np.count_nonzero(pressure > protocol.pressure_pa / 2.0) > 0


def test_channel_probability_is_bounded_and_pressure_monotone():
    pressures = np.array([0.0, 150.0e3, 300.0e3, 600.0e3])
    tension = nm.acoustic_energy_tension_mn_m(pressures)
    probabilities = nm.open_probability(tension, nm.CHANNELS[0])
    drive = nm.coupled_channel_drive(pressures)

    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.all(np.diff(probabilities) > 0.0)
    assert float(drive[-1]) > float(drive[1])


def test_neural_response_is_input_sensitive_and_value_semantic():
    low = nm.neural_response(nm.Protocol(pressure_pa=150.0e3), duration_s=1.0, dt_s=0.002)
    high = nm.neural_response(nm.Protocol(pressure_pa=500.0e3), duration_s=1.0, dt_s=0.002)

    assert float(high["calcium"].max()) > float(low["calcium"].max())
    assert float(high["voltage_mv"].max()) > float(low["voltage_mv"].max())
    assert float(high["response_probability"].max()) > float(low["response_probability"].max())


def test_pennes_temperature_inspects_computed_thermal_dose():
    protocol = nm.Protocol()
    default_ispta = float(nm.intensity_sppa_w_m2(protocol.pressure_pa) * protocol.duty_cycle)
    hot_ispta = float(nm.intensity_sppa_w_m2(800.0e3) * 0.20)

    default = nm.pennes_temperature(default_ispta, protocol.sonication_s, dt_s=0.05)
    hot = nm.pennes_temperature(hot_ispta, protocol.sonication_s, dt_s=0.05)

    assert float(default.temperature_c[-1] - nm.T0_C) < 0.5
    assert float(default.cem43_min[-1]) < 0.25
    assert float(hot.temperature_c[-1]) > float(default.temperature_c[-1])
    assert float(hot.cem43_min[-1]) > float(default.cem43_min[-1])


def test_cavitation_guardrail_increases_across_mi_boundary():
    mi = np.array([0.3, 0.7, 1.2])
    risk = nm.cavitation_risk(mi)

    assert float(risk[0]) < 0.01
    assert 0.45 < float(risk[1]) < 0.55
    assert float(risk[2]) > 0.99
