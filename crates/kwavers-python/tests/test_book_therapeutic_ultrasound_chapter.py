from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_chapter06_pressure_conversion_uses_rust_helper():
    source = (BOOK_DIR / "ch06_therapeutic_ultrasound.py").read_text(encoding="utf-8")

    assert "kw.acoustic_pressure_amplitude_from_intensity" in source
    assert "np.sqrt(2.0 * RHO0 * C0" not in source


def test_acoustic_pressure_from_intensity_binding_roundtrips_intensity():
    import pykwavers as kw

    rho = 1060.0
    sound_speed = 1540.0
    pressure = np.asarray([0.0, 1.0e6, 2.0e6], dtype=np.float64)
    intensity = np.asarray(kw.acoustic_intensity_from_amplitude(pressure, rho, sound_speed))
    roundtrip = np.asarray(
        kw.acoustic_pressure_amplitude_from_intensity(intensity, rho, sound_speed)
    )

    # Same f64 formulas in opposite order; 1e-9 Pa is below one femtofraction of
    # the 1 MPa scale used by the figure inputs.
    assert np.allclose(roundtrip, pressure, rtol=1.0e-12, atol=1.0e-9)


def test_acoustic_pressure_from_intensity_binding_rejects_invalid_inputs():
    import pykwavers as kw

    with pytest.raises(ValueError):
        kw.acoustic_pressure_amplitude_from_intensity(np.asarray([-1.0]), 1060.0, 1540.0)
    with pytest.raises(ValueError):
        kw.acoustic_pressure_amplitude_from_intensity(np.asarray([np.nan]), 1060.0, 1540.0)
    with pytest.raises(ValueError):
        kw.acoustic_pressure_amplitude_from_intensity(np.asarray([1.0]), 0.0, 1540.0)
