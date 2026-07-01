from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


BOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "book"
sys.path.insert(0, str(BOOK_DIR))


def test_chapter08_spreading_uses_rust_helper():
    source = (BOOK_DIR / "ch08_acoustic_propagation.py").read_text(encoding="utf-8")

    assert "kw.geometric_spreading_intensity_envelopes" in source
    assert "kw.spherical_wave_pressure" not in source
    assert "pykwavers optional" not in source


def test_geometric_spreading_binding_matches_normalized_laws():
    import pykwavers as kw

    radii = np.asarray([0.01, 0.02, 0.04], dtype=np.float64)
    spherical, cylindrical = (
        np.asarray(values, dtype=float)
        for values in kw.geometric_spreading_intensity_envelopes(radii)
    )

    assert np.array_equal(spherical, np.asarray([1.0, 0.25, 0.0625]))
    assert np.array_equal(cylindrical, np.asarray([1.0, 0.5, 0.25]))


def test_geometric_spreading_binding_rejects_invalid_radii():
    import pykwavers as kw

    with pytest.raises(ValueError):
        kw.geometric_spreading_intensity_envelopes(np.asarray([], dtype=np.float64))
    with pytest.raises(ValueError):
        kw.geometric_spreading_intensity_envelopes(np.asarray([0.0], dtype=np.float64))
    with pytest.raises(ValueError):
        kw.geometric_spreading_intensity_envelopes(np.asarray([1.0, np.nan], dtype=np.float64))
