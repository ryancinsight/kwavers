from __future__ import annotations

import sys
from pathlib import Path


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_chapter07_minnaert_markers_use_rust_inverse_helper():
    source = (BOOK_DIR / "ch07_theranostics.py").read_text(encoding="utf-8")

    assert "kw.minnaert_radius_for_frequency_m" in source
    assert "np.sqrt(3 * KAPPA * P0 / RHO_L)" not in source


def test_chapter07_cem43_uses_vector_rust_binding():
    source = (BOOK_DIR / "ch07_theranostics.py").read_text(encoding="utf-8")

    assert "kw.closed_loop_cem43_fixture(" in source
    assert "def _cumulative_cem43" not in source
    assert "kw.compute_cem43(T_arr[: i + 1]" not in source
    assert "np.random.default_rng" not in source
    assert "rng.standard_normal" not in source


def test_chapter07_pcd_spectra_and_controller_use_rust_bindings():
    source = (BOOK_DIR / "ch07_theranostics.py").read_text(encoding="utf-8")

    assert "kw.keller_miksis_pcd_spectrum(" in source
    assert "kw.keller_miksis_pcd_controller_trace(" in source
    assert "np.fft.rfft" not in source
    assert "np.fft.rfftfreq" not in source
    assert "def _km_signals" not in source
    assert "for pulse in range(N_PULSES - 1)" not in source


def test_minnaert_inverse_binding_roundtrips_frequency():
    import pykwavers as kw

    gamma = 1.4
    p0_pa = 101_325.0
    rho = 998.0
    expected_radius_m = 3.0e-6
    frequency_hz = kw.minnaert_resonance_hz(expected_radius_m, gamma, p0_pa, rho)

    radius_m = kw.minnaert_radius_for_frequency_m(frequency_hz, gamma, p0_pa, rho)
    roundtrip_hz = kw.minnaert_resonance_hz(radius_m, gamma, p0_pa, rho)

    # Same analytical relation solved in opposite directions through the Rust
    # binding; 1e-12 relative tolerance is far below any visual marker precision.
    assert abs(radius_m - expected_radius_m) <= expected_radius_m * 1.0e-12
    assert abs(roundtrip_hz - frequency_hz) <= frequency_hz * 1.0e-12


def test_minnaert_inverse_binding_rejects_invalid_domain_with_zero():
    import pykwavers as kw

    assert kw.minnaert_radius_for_frequency_m(0.0, 1.4, 101_325.0, 998.0) == 0.0
    assert kw.minnaert_radius_for_frequency_m(1.0e6, -1.0, 101_325.0, 998.0) == 0.0
    assert kw.minnaert_radius_for_frequency_m(1.0e6, 1.4, float("nan"), 998.0) == 0.0


def test_cem43_cumulative_binding_matches_total_dose():
    import numpy as np
    import pykwavers as kw

    temperatures_c = np.array([37.0, 43.0, 44.0, 42.0, 55.0], dtype=float)
    cumulative = np.asarray(kw.cem43_cumulative(temperatures_c, 0.5))
    total = kw.compute_cem43(temperatures_c, 0.5)

    assert cumulative.shape == temperatures_c.shape
    assert np.all(np.diff(cumulative) > 0.0)
    assert abs(float(cumulative[-1]) - total) <= max(abs(total), 1.0) * 1.0e-12


def test_closed_loop_cem43_fixture_binding_is_seeded_and_uses_cem43():
    import numpy as np
    import pykwavers as kw

    fixture = kw.closed_loop_cem43_fixture(60, 0.5, 37.0, 60.0, seed=42)
    repeated = kw.closed_loop_cem43_fixture(60, 0.5, 37.0, 60.0, seed=42)
    shifted_seed = kw.closed_loop_cem43_fixture(60, 0.5, 37.0, 60.0, seed=43)

    np.testing.assert_allclose(np.asarray(fixture["time_s"])[:3], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(np.asarray(fixture["fixed_temperature_c"])[[0, 30]], [37.0, 60.0])
    np.testing.assert_allclose(np.asarray(fixture["underdrive_temperature_c"])[40], 56.0)

    for key in (
        "time_s",
        "fixed_temperature_c",
        "feedback_temperature_c",
        "underdrive_temperature_c",
        "fixed_cem43_min",
        "feedback_cem43_min",
        "underdrive_cem43_min",
    ):
        np.testing.assert_allclose(np.asarray(fixture[key]), np.asarray(repeated[key]))

    assert not np.array_equal(
        np.asarray(fixture["feedback_temperature_c"]),
        np.asarray(shifted_seed["feedback_temperature_c"]),
    )
    for temperature_key, dose_key in (
        ("fixed_temperature_c", "fixed_cem43_min"),
        ("feedback_temperature_c", "feedback_cem43_min"),
        ("underdrive_temperature_c", "underdrive_cem43_min"),
    ):
        expected = np.asarray(kw.cem43_cumulative(np.asarray(fixture[temperature_key]), 0.5))
        observed = np.asarray(fixture[dose_key])
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
        assert np.all(np.diff(observed) >= 0.0)


def test_keller_miksis_pcd_bindings_return_finite_trace():
    import numpy as np
    import pykwavers as kw

    spectrum = kw.keller_miksis_pcd_spectrum(
        3.0e-6,
        50.0e3,
        1.0e6,
        3,
        64,
        1,
        101_325.0,
        998.0,
        0.0725,
        1.002e-3,
        1.4,
        2338.0,
        1500.0,
    )
    freq = np.asarray(spectrum["frequency_hz"])
    psd_db = np.asarray(spectrum["normalized_psd_db"])

    assert freq.ndim == 1
    assert psd_db.shape == freq.shape
    assert np.all(np.diff(freq) > 0.0)
    assert np.isfinite(psd_db).all()
    assert float(spectrum["stable_signal"]) >= 0.0
    assert float(spectrum["inertial_signal"]) >= 0.0

    trace = kw.keller_miksis_pcd_controller_trace(
        3.0e-6,
        1.0e6,
        4,
        50.0e3,
        3,
        64,
        1,
        0.05,
        0.3,
        1.05,
        0.80,
        10.0e3,
        500.0e3,
        101_325.0,
        998.0,
        0.0725,
        1.002e-3,
        1.4,
        2338.0,
        1500.0,
    )

    pressure_kpa = np.asarray(trace["pressure_kpa"])
    assert np.array_equal(np.asarray(trace["pulse_index"]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert pressure_kpa.shape == (4,)
    assert np.all((10.0 <= pressure_kpa) & (pressure_kpa <= 500.0))
    assert np.isfinite(np.asarray(trace["stable_signal"])).all()
    assert np.isfinite(np.asarray(trace["inertial_signal"])).all()
