from pathlib import Path

import numpy as np
import pykwavers as kw


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "book"
    / "ch23_passive_acoustic_mapping.py"
)


def test_chapter_uses_rust_owned_cavitation_spectrum() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "kw.normalized_cavitation_emission_spectrum(" in source
    assert "def _cavitation_spectrum" not in source
    assert "def lorentz" not in source
    assert "harmonics = np.arange" not in source


def test_chapter_uses_rust_owned_passive_point_source_rf() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    start = source.index("print(\"[fig02]")
    end = source.index("# ── Figure 03", start)
    body = source[start:end]

    assert "kw.passive_cavitation_point_source_rf(" in body
    assert "kw.passive_acoustic_map_das(" in body
    assert "passive[i] =" not in body
    assert "np.linalg.norm(sensor_xyz[i] - src)" not in body
    assert "np.sin(2.0 * np.pi * F0 * (t_axis - tau))" not in body


def test_chapter_uses_rust_owned_eigenspace_spectrum() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    start = source.index("print(\"[fig04]")
    end = source.index("# ── Figure 05", start)
    body = source[start:end]

    assert "kw.eigenspace_covariance_eigenvalues(" in body
    assert "def _build_csd_matrix" not in source
    assert "RNG.standard_normal" not in body
    assert "np.exp(1j * 2.0 * np.pi * F0 * r / C0)" not in body
    assert "kw.hermitian_eigenvalues_complex(" not in body


def test_chapter_uses_rust_owned_vcz_coherence() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    start = source.index("print(\"[fig03]")
    end = source.index("# ── Figure 04", start)
    body = source[start:end]

    assert "kw.van_cittert_zernike_coherence(" in body
    assert "def _vcz_coherence" not in source
    assert "np.sinc(" not in source
    assert "Requires: numpy, matplotlib, scipy" not in source


def test_chapter_uses_rust_owned_cavitation_dose_fixture() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    start = source.index("print(\"[fig06]")
    body = source[start:]

    assert "kw.passive_cavitation_dose_fixture(" in body
    assert "np.random.default_rng" not in source
    assert "def _cavitation_dose" not in body
    assert "RNG.poisson" not in body
    assert "RNG.exponential" not in body


def test_normalized_cavitation_emission_spectrum_distinguishes_regimes() -> None:
    frequencies_hz = np.linspace(0.1e6, 5.5e6, 4000)
    stable = np.asarray(
        kw.normalized_cavitation_emission_spectrum(frequencies_hz, 1.0e6, "stable")
    )
    inertial = np.asarray(
        kw.normalized_cavitation_emission_spectrum(frequencies_hz, 1.0e6, "inertial")
    )

    assert stable.shape == frequencies_hz.shape
    assert inertial.shape == frequencies_hz.shape
    assert np.isclose(float(stable.max()), 1.0, rtol=0.0, atol=1e-12)
    assert np.isclose(float(inertial.max()), 1.0, rtol=0.0, atol=1e-12)

    interharmonic = int(np.argmin(np.abs(frequencies_hz - 4.75e6)))
    assert inertial[interharmonic] > 10.0 * stable[interharmonic]


def test_normalized_cavitation_emission_spectrum_rejects_unknown_regime() -> None:
    frequencies_hz = np.array([1.0e6])
    try:
        kw.normalized_cavitation_emission_spectrum(frequencies_hz, 1.0e6, "unknown")
    except ValueError as exc:
        assert "regime must be 'stable' or 'inertial'" in str(exc)
    else:
        raise AssertionError("unknown cavitation regime must be rejected")


def test_passive_point_source_rf_binding_matches_closed_form_sample() -> None:
    receiver_xyz = np.asarray([[0.0, 0.0, 0.01]], dtype=np.float64)
    source_xyz = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

    rf = np.asarray(
        kw.passive_cavitation_point_source_rf(
            receiver_xyz,
            source_xyz,
            3,
            100_000.0,
            1_000.0,
            1_000.0,
            3.0,
        )
    )

    centered_time_s = -0.01 / 1_000.0
    envelope_scale_s = 3.0 / (2.0 * 1_000.0)
    envelope = np.exp(-0.5 * (centered_time_s / envelope_scale_s) ** 2)
    expected = envelope * np.sin(2.0 * np.pi * 1_000.0 * centered_time_s) / 0.01
    assert rf.shape == (1, 3)
    np.testing.assert_allclose(rf[0, 0], expected, rtol=0.0, atol=1.0e-12)


def test_eigenspace_covariance_eigenvalues_match_theorem_split() -> None:
    eigenvalues = np.asarray(kw.eigenspace_covariance_eigenvalues(8, 3, 10.0, 1.0))

    assert eigenvalues.shape == (8,)
    np.testing.assert_allclose(eigenvalues[:3], np.full(3, 11.0), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(eigenvalues[3:], np.ones(5), rtol=0.0, atol=0.0)


def test_eigenspace_covariance_eigenvalues_reject_invalid_rank() -> None:
    try:
        kw.eigenspace_covariance_eigenvalues(8, 8, 10.0, 1.0)
    except ValueError as exc:
        assert "0 < n_sources < n_elements" in str(exc)
    else:
        raise AssertionError("invalid eigenspace source rank must be rejected")


def test_van_cittert_zernike_coherence_matches_sinc_law() -> None:
    source_extent_m = 1.0e-3
    depth_m = 40.0e-3
    wavelength_m = 1_500.0 / 1.0e6
    first_zero_m = wavelength_m * depth_m / source_extent_m
    delta_x_m = np.array([0.0, 0.5 * first_zero_m, first_zero_m], dtype=np.float64)

    coherence = np.asarray(
        kw.van_cittert_zernike_coherence(
            delta_x_m,
            source_extent_m,
            depth_m,
            wavelength_m,
        )
    )
    expected = np.sinc(source_extent_m * delta_x_m / (wavelength_m * depth_m))

    np.testing.assert_allclose(coherence, expected, rtol=0.0, atol=1.0e-15)
    assert coherence[0] == 1.0
    assert abs(coherence[-1]) <= 1.0e-15


def test_van_cittert_zernike_coherence_rejects_invalid_geometry() -> None:
    try:
        kw.van_cittert_zernike_coherence(np.array([0.0]), 1.0e-3, 0.0, 1.5e-3)
    except ValueError as exc:
        assert "depth_m must be positive and finite" in str(exc)
    else:
        raise AssertionError("invalid VCZ depth must be rejected")


def test_passive_cavitation_dose_fixture_is_normalized_and_seeded() -> None:
    time_s = np.array([0.0, 0.5, 1.0, 2.0, 4.0], dtype=np.float64)

    trace = kw.passive_cavitation_dose_fixture(time_s, 1.0, 0.1, 2.0, seed=7)
    repeated = kw.passive_cavitation_dose_fixture(time_s, 1.0, 0.1, 2.0, seed=7)
    shifted_seed = kw.passive_cavitation_dose_fixture(time_s, 1.0, 0.1, 2.0, seed=8)

    np.testing.assert_allclose(np.asarray(trace["time_s"]), time_s, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(trace["stable_dose"]),
        np.array([0.25, 0.25, 0.5, 0.75, 1.0]),
        rtol=0.0,
        atol=1.0e-15,
    )
    for key in ("stable_dose", "inertial_trial1_dose", "inertial_trial2_dose"):
        observed = np.asarray(trace[key])
        np.testing.assert_allclose(observed, np.asarray(repeated[key]), rtol=0.0, atol=0.0)
        assert observed[-1] == 1.0
        assert np.all(np.diff(observed) >= 0.0)
        assert np.all((0.0 <= observed) & (observed <= 1.0))

    assert not np.array_equal(
        np.asarray(trace["inertial_trial1_dose"]),
        np.asarray(shifted_seed["inertial_trial1_dose"]),
    )


def test_passive_cavitation_dose_fixture_rejects_invalid_time_axis() -> None:
    try:
        kw.passive_cavitation_dose_fixture(np.array([0.0, -1.0]), 1.0, 0.1)
    except ValueError as exc:
        assert "time axis" in str(exc)
    else:
        raise AssertionError("negative time axis must be rejected")
