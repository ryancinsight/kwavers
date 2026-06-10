"""Value-semantic tests for the electrical (Hodgkin–Huxley + NICE) pathway.

Mirrors the Rust value-semantic tests in
`kwavers_physics::acoustics::therapy::neuromodulation`, exercised through the
PyO3 bindings. All physics runs in Rust; this file only marshals arguments and
asserts value semantics.
"""

from __future__ import annotations

import numpy as np
import pykwavers as kw


# ── Hodgkin–Huxley neuron ────────────────────────────────────────────────────

def test_hodgkin_huxley_suprathreshold_fires():
    _, voltage, spikes = kw.hodgkin_huxley_response(15.0, 0.01, 50.0)
    assert spikes.size >= 2  # repetitive firing above rheobase
    assert float(np.max(voltage)) > 20.0  # action-potential overshoot
    assert float(np.max(voltage)) < 60.0  # below E_Na


def test_hodgkin_huxley_subthreshold_does_not_fire():
    _, voltage, spikes = kw.hodgkin_huxley_response(1.0, 0.01, 50.0)
    assert spikes.size == 0
    assert float(np.max(voltage)) < -50.0  # stays near rest


# ── Bilayer-sonophore capacitance geometry (Plaksin Eq. 8) ───────────────────

def test_bilayer_capacitance_curve_is_monotone_decreasing():
    cm0, a, delta = 1.0, 32.0e-9, 1.26e-9
    z = np.array([0.0, 0.5e-9, 1.0e-9, 2.0e-9, 4.0e-9])
    cm = kw.bilayer_capacitance_curve(z, cm0, a, delta)
    assert abs(cm[0] - cm0) < 1e-9  # C_m(0) = C_m0
    assert np.all(np.diff(cm) < 0.0)  # capacitance falls as the gap widens


# ── NICE intramembrane-cavitation coupling ───────────────────────────────────

def test_nice_zero_deflection_is_quiescent():
    _, voltage, _, spikes = kw.nice_bilayer_sonophore_response(
        0.5, 0.0, 4.0e-5, 1.0, 12.0, 18.0
    )
    assert spikes.size == 0
    assert float(np.max(np.abs(voltage + 65.0))) < 0.5  # stays at rest (−65 mV)


def test_nice_post_stimulus_ap_with_pulse_duration_dependence():
    # 0.5 ms pulse: insufficient charge accumulation → no AP.
    _, _, _, short_spikes = kw.nice_bilayer_sonophore_response(
        0.5, 1.0, 4.0e-5, 1.0, 1.5, 11.0
    )
    assert short_spikes.size == 0
    # 5 ms pulse: accumulated charge depolarises after offset → post-stimulus AP.
    _, voltage, _, long_spikes = kw.nice_bilayer_sonophore_response(
        0.5, 1.0, 4.0e-5, 1.0, 6.0, 14.0
    )
    assert long_spikes.size >= 1
    assert np.all(long_spikes >= 6.0)  # spike occurs after sonication offset
    assert float(np.max(voltage)) > 20.0


def test_sonic_matches_carrier_resolved_single_burst():
    # The cycle-averaged SONIC reduction reproduces the carrier-resolved NICE
    # result for a single burst (same spike count, AP timing within 1 ms).
    _, _, _, nice_spikes = kw.nice_bilayer_sonophore_response(
        0.5, 1.0, 4.0e-5, 1.0, 6.0, 14.0
    )
    _, voltage, _, sonic_spikes = kw.nice_sonic_response(
        0.5, 1.0, 5.0e-3, 1.0, 6.0, 14.0
    )
    assert nice_spikes.size == sonic_spikes.size >= 1
    assert abs(float(nice_spikes[0]) - float(sonic_spikes[0])) < 1.0
    assert float(np.max(voltage)) > 20.0


def test_nice_charge_accumulates_with_deflection():
    def end_charge(deflection_nm: float) -> float:
        time, _, charge, _ = kw.nice_bilayer_sonophore_response(
            0.5, deflection_nm, 4.0e-5, 1.0, 9.0, 11.0
        )
        mask = time < 9.0
        return float(charge[mask][-1])

    q_rest = 1.0 * -65.0
    d1 = end_charge(1.0) - q_rest
    d4 = end_charge(4.0) - q_rest
    assert d1 > 1.0  # net charge accumulation
    assert d4 > d1  # grows with leaflet deflection


# ── Pulse-train dosimetry (Blackmore Table 1; Atkinson-Clement 2025) ─────────

def test_theta_burst_dosimetry_matches_paper():
    z = 1000.0 * 1540.0
    p_peak = float(np.sqrt(2.0 * z * 54.51e4))  # I_SPPA = 54.51 W/cm²
    dose = kw.pulse_train_dosimetry(
        500.0e3, 20.0e-3, 5.0, 80.0, 0.0, 1, p_peak, 1000.0, 1540.0
    )
    assert abs(dose["total_duty_cycle"] - 0.10) < 1e-9
    assert abs(dose["total_time_s"] - 80.0) < 1e-6
    assert abs(dose["isppa_w_cm2"] - 54.51) < 1e-2
    assert abs(dose["ispta_w_cm2"] - 5.451) < 1e-2
    # I_SPTA = 5451 mW/cm² exceeds the conservative FDA *diagnostic* limit
    # (720 mW/cm²); neuromodulation routinely operates above it (Blackmore 2019
    # notes the diagnostic limits are unnecessarily restrictive for neuromod).
    assert dose["within_fda_limits"] is False


def test_high_pressure_breaches_fda_limits():
    dose = kw.pulse_train_dosimetry(
        500.0e3, 20.0e-3, 5.0, 80.0, 0.0, 1, 5.0e6, 1000.0, 1540.0
    )
    assert dose["mechanical_index"] > 1.9
    assert dose["within_fda_limits"] is False


def test_cortical_rs_vs_fs_selectivity():
    # RS and FS cortical neurons respond differently to the same drive.
    _, v_rs, _, _ = kw.cortical_sonic_response("rs", 0.5, 1.5, 5.0e-3, 1.0, 30.0, 45.0)
    _, v_fs, _, _ = kw.cortical_sonic_response("fs", 0.5, 1.5, 5.0e-3, 1.0, 30.0, 45.0)
    assert np.all(np.isfinite(v_rs)) and np.all(np.isfinite(v_fs))
    win_rs = float(np.mean(v_rs[(v_rs.size // 10):]))
    win_fs = float(np.mean(v_fs[(v_fs.size // 10):]))
    assert abs(win_rs - win_fs) > 0.5


def test_quasistatic_deflection_is_pressure_driven():
    # Deflection is solved from acoustic pressure: rectified (0 in compression),
    # monotone and nm-scale in rarefaction.
    p = np.array([300.0e3, -100.0e3, -300.0e3, -500.0e3])
    z = kw.bls_deflection_curve(p)
    assert z[0] == 0.0  # compression ⇒ flat
    assert z[1] > 0.0 and z[2] > z[1] and z[3] > z[2]  # rarefaction grows
    assert 5.0e-9 < z[3] < 20.0e-9  # ≈ Plaksin Fig. 1 order (12 nm @ 500 kPa)


def test_quasistatic_nice_evokes_post_stimulus_ap():
    _, voltage, _, spikes = kw.nice_quasistatic_response(
        500.0e3, 0.5, 4.0e-5, 1.0, 8.0, 16.0
    )
    assert spikes.size >= 1
    assert np.all(spikes >= 1.0)  # post-onset
    assert float(np.max(voltage)) > 20.0


def test_dynamic_bls_nice_evokes_post_stimulus_ap():
    # Exact transient bilayer-sonophore dynamics (full Rayleigh-Plesset ODE).
    # 300 kPa keeps the leaflet in the clean expansion regime (no steric-wall
    # contact); 500 kPa at the squid rest would over-drive into the wall.
    _, voltage, _, spikes = kw.nice_dynamic_response(300.0e3, 0.5, 4.0e-5, 1.0, 8.0, 16.0)
    assert spikes.size >= 1
    assert np.all(spikes >= 1.0)
    assert float(np.max(voltage)) > 20.0


def test_itrusst_safety_assessment():
    ok = kw.itrusst_safety(0.5, 1.0, 0.0)
    assert ok["mechanical_ok"] and ok["thermal_ok"] and ok["overall_ok"]
    mech_breach = kw.itrusst_safety(2.5, 0.5, 0.0)
    assert (not mech_breach["mechanical_ok"]) and (not mech_breach["overall_ok"])
    therm_breach = kw.itrusst_safety(0.5, 3.0, 5.0)
    assert (not therm_breach["thermal_ok"]) and (not therm_breach["overall_ok"])


def test_neuromod_threshold_pressure():
    # A firing threshold exists for the squid neuron at 0.5 MHz; it lies in range,
    # and pressures above it fire while well-below do not (verified in Rust).
    thr = kw.neuromod_threshold_pressure_pa(
        "squid", 0.5, p_lo_pa=50.0e3, p_hi_pa=600.0e3, n_iter=6,
        offset_ms=6.0, t_end_ms=14.0,
    )
    assert thr is not None
    assert 50.0e3 < thr < 600.0e3
    # Below the bracket lower bound the search returns None (no threshold found).
    none_thr = kw.neuromod_threshold_pressure_pa(
        "squid", 0.5, p_lo_pa=50.0e3, p_hi_pa=60.0e3, n_iter=4,
        offset_ms=6.0, t_end_ms=14.0,
    )
    assert none_thr is None
    import pytest
    with pytest.raises(Exception):
        kw.neuromod_threshold_pressure_pa("bogus", 0.5)
