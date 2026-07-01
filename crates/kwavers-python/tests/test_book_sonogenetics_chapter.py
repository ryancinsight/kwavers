from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_chapter18_pressure_from_intensity_uses_rust_helper():
    source = (BOOK_DIR / "ch18_sonogenetics.py").read_text(encoding="utf-8")

    assert "kw.acoustic_pressure_amplitude_from_intensity" in source
    assert "np.sqrt(2.0 * rho * c * intensity_w_m2)" not in source


def test_sonogenetics_pressure_conversion_matches_intensity_roundtrip():
    import pykwavers as kw

    rho = 1000.0
    sound_speed = 1500.0
    intensity = np.asarray([1.0e2, 1.0e4, 1.0e6], dtype=np.float64)
    pressure = np.asarray(
        kw.acoustic_pressure_amplitude_from_intensity(intensity, rho, sound_speed)
    )
    roundtrip = np.asarray(kw.acoustic_intensity_from_amplitude(pressure, rho, sound_speed))

    # Same Rust f64 closed form in opposite directions; this tolerance is far
    # below any plotted intensity tick and guards against Python-side formula drift.
    assert np.allclose(roundtrip, intensity, rtol=1.0e-12, atol=1.0e-9)
