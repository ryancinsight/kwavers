from pathlib import Path

import numpy as np
import pykwavers as kw


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "book"
    / "ch14_sensors_and_measurements.py"
)


def test_chapter14_pressure_velocity_panel_uses_rust_pair_helper() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    start = source.index("def fig03_pressure_velocity")
    end = source.index("# ── Figure 04", start)
    body = source[start:end]

    assert "kw.plane_wave_pressure_velocity_1d(" in body
    assert "P0 * np.sin(k * x" not in body
    assert "P0 / Z * np.sin(k * x" not in body


def test_plane_wave_pressure_velocity_binding_preserves_impedance_ratio() -> None:
    x = np.asarray([0.0, 0.25, 0.5], dtype=np.float64)
    pressure, velocity = kw.plane_wave_pressure_velocity_1d(
        1.0e5,
        np.pi,
        x,
        np.pi / 2.0,
        998.0,
        1500.0,
    )

    pressure = np.asarray(pressure)
    velocity = np.asarray(velocity)
    assert pressure.shape == x.shape
    assert velocity.shape == x.shape
    np.testing.assert_allclose(velocity, pressure / (998.0 * 1500.0), rtol=0.0, atol=1.0e-14)
    assert abs(float(pressure[0])) < 1.0e-10
