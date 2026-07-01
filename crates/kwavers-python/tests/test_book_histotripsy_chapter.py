from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_chapter21_histotripsy_pressure_inversion_uses_rust_helper():
    source = (BOOK_DIR / "ch21_histotripsy_comparison.py").read_text(encoding="utf-8")

    assert "kw.acoustic_pressure_amplitude_from_intensity" in source
    assert "np.sqrt(2.0 * RHO0 * C0 * I_S_M)" not in source


def test_histotripsy_shock_intensity_roundtrips_through_rust_helper():
    import pykwavers as kw

    rho = 1060.0
    sound_speed = 1540.0
    shock_intensity = np.asarray([25.0e7], dtype=np.float64)
    pressure = np.asarray(
        kw.acoustic_pressure_amplitude_from_intensity(shock_intensity, rho, sound_speed)
    )
    roundtrip = np.asarray(kw.acoustic_intensity_from_amplitude(pressure, rho, sound_speed))

    assert np.allclose(roundtrip, shock_intensity, rtol=1.0e-12, atol=1.0e-6)
