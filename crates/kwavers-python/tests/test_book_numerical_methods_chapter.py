from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_chapter02_numerical_curves_use_rust_wave_helpers():
    source = (BOOK_DIR / "ch02_numerical_methods.py").read_text(encoding="utf-8")

    required = (
        "kw.fdtd_cfl_stability_region_2d",
        "kw.centered_fd_modified_wavenumber",
        "kw.kspace_temporal_correction",
    )
    forbidden = (
        "def fd_symbol",
        "cx * cx + cz * cz <= 1.0",
        "np.sin(x[mask]) / x[mask]",
        "(8.0 * np.sin(theta) - np.sin(2.0 * theta)) / 6.0",
        "(45.0 * np.sin(theta) - 9.0 * np.sin(2.0 * theta)",
    )
    for token in required:
        assert token in source
    for token in forbidden:
        assert token not in source


def test_centered_fd_modified_wavenumber_binding_matches_symbols():
    import pykwavers as kw

    theta = np.array([0.0, np.pi / 2.0, np.pi], dtype=np.float64)
    second = np.asarray(kw.centered_fd_modified_wavenumber(theta, 2))
    fourth = np.asarray(kw.centered_fd_modified_wavenumber(theta, 4))
    sixth = np.asarray(kw.centered_fd_modified_wavenumber(theta, 6))

    assert np.allclose(second, np.sin(theta), rtol=0.0, atol=1.0e-12)
    assert np.allclose(
        fourth,
        (8.0 * np.sin(theta) - np.sin(2.0 * theta)) / 6.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.allclose(
        sixth,
        (45.0 * np.sin(theta) - 9.0 * np.sin(2.0 * theta) + np.sin(3.0 * theta))
        / 30.0,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_kspace_temporal_correction_binding_matches_sinc():
    import pykwavers as kw

    theta = np.array([0.0, np.pi], dtype=np.float64)
    correction = np.asarray(kw.kspace_temporal_correction(theta, 0.5))
    expected = np.array([1.0, np.sin(np.pi / 4.0) / (np.pi / 4.0)])

    assert np.allclose(correction, expected, rtol=0.0, atol=1.0e-12)


def test_fdtd_cfl_stability_region_binding_marks_component_ball():
    import pykwavers as kw

    cfl_x = np.array([0.0, 0.8, 1.0], dtype=np.float64)
    cfl_z = np.array([0.0, 0.8], dtype=np.float64)
    stable = np.asarray(kw.fdtd_cfl_stability_region_2d(cfl_x, cfl_z)).reshape((3, 2))

    assert np.array_equal(stable, np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]))


def test_numerical_method_bindings_reject_invalid_input():
    import pykwavers as kw

    with pytest.raises(ValueError, match="order"):
        kw.centered_fd_modified_wavenumber(np.array([0.0], dtype=np.float64), 8)
    with pytest.raises(ValueError, match="finite"):
        kw.kspace_temporal_correction(np.array([0.0], dtype=np.float64), np.inf)
    with pytest.raises(ValueError, match="finite"):
        kw.fdtd_cfl_stability_region_2d(
            np.array([0.0], dtype=np.float64),
            np.array([np.nan], dtype=np.float64),
        )
