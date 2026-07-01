from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_chapter01_wave_figures_use_rust_wave_helpers():
    source = (BOOK_DIR / "ch01_wave_physics_fundamentals.py").read_text(
        encoding="utf-8"
    )

    required = (
        "kw.standing_wave_1d",
        "kw.gaussian_modulated_pulse_1d",
        "kw.dalembert_split_solution_1d",
    )
    forbidden = (
        "p0 * np.sin(k * x)",
        "np.exp(-((x - x0)",
        "np.interp(xb - ct",
        "np.interp(xb + ct",
        "0.5 * (g_right + g_left)",
    )
    for token in required:
        assert token in source
    for token in forbidden:
        assert token not in source


def test_gaussian_modulated_pulse_binding_matches_formula():
    import pykwavers as kw

    x = np.array([-1.0e-3, 0.0, 1.0e-3], dtype=np.float64)
    pulse = np.asarray(kw.gaussian_modulated_pulse_1d(x, 0.0, 2.0e-3, 4.0e-3, 1.0e5))
    expected = 1.0e5 * np.exp(-(x**2) / (2.0 * (2.0e-3) ** 2)) * np.cos(
        2.0 * np.pi * x / 4.0e-3
    )

    assert np.allclose(pulse, expected, rtol=1.0e-12, atol=1.0e-10)
    assert pulse[1] == 1.0e5


def test_standing_wave_binding_matches_chapter01_contract():
    import pykwavers as kw

    x = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float64)
    p0 = 2.0e5
    k = 2.0 * np.pi
    omega_t = np.pi / 3.0

    pressure = np.asarray(kw.standing_wave_1d(p0, k, x, omega_t))
    expected = p0 * np.sin(k * x) * np.cos(omega_t)

    assert np.allclose(pressure, expected, rtol=1.0e-12, atol=1.0e-10)
    assert pressure[0] == 0.0


def test_dalembert_split_solution_binding_moves_half_copies():
    import pykwavers as kw

    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    g = np.array([0.0, 0.0, 10.0, 0.0, 0.0], dtype=np.float64)
    pressure = np.asarray(kw.dalembert_split_solution_1d(x, g, 1.0))

    assert np.array_equal(pressure, np.array([0.0, 5.0, 0.0, 5.0, 0.0]))


def test_wave_helper_bindings_reject_invalid_input():
    import pykwavers as kw

    with pytest.raises(ValueError, match="sigma_m"):
        kw.gaussian_modulated_pulse_1d(np.array([0.0], dtype=np.float64), 0.0, 0.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="strictly increasing"):
        kw.dalembert_split_solution_1d(
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            1.0,
        )
