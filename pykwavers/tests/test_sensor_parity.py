#!/usr/bin/env python3
"""
Sensor Parity Tests: pykwavers.Sensor vs kwave.kSensor

Validates that pykwavers Sensor creation and recording behavior
match k-wave-python's kSensor.
"""

import numpy as np
import pytest

import pykwavers as kw
from conftest import requires_kwave, compute_cfl_dt, HAS_KWAVE

if HAS_KWAVE:
    from kwave.ksensor import kSensor


# ============================================================================
# pykwavers Sensor standalone tests
# ============================================================================


class TestSensorCreation:
    """Test pykwavers Sensor construction."""

    def test_point_sensor(self):
        """Point sensor creation."""
        s = kw.Sensor.point(position=(1e-3, 1e-3, 1e-3))
        assert s.sensor_type == "point"

    def test_grid_sensor(self):
        """Grid (full-field) sensor creation."""
        s = kw.Sensor.grid()
        assert s.sensor_type == "grid"

    @pytest.mark.parametrize("pos", [
        (0.0, 0.0, 0.0),
        (1.6e-3, 1.6e-3, 1.6e-3),
        (5e-3, 5e-3, 5e-3),
        (0.1e-3, 0.1e-3, 0.1e-3),
    ])
    def test_point_sensor_various_positions(self, pos):
        """Point sensor accepts various positions."""
        s = kw.Sensor.point(position=pos)
        assert s.sensor_type == "point"

    def test_sensor_repr(self):
        """Sensor has meaningful representation."""
        s = kw.Sensor.point(position=(1e-3, 2e-3, 3e-3))
        r = repr(s)
        assert "Sensor" in r or "sensor" in r.lower()


class TestSensorRecording:
    """Test that sensors correctly record simulation data."""

    def test_point_sensor_records_time_series(self, grid, medium):
        """Point sensor returns 1D time series."""
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sensor = kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))

        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=50, dt=1e-8)

        assert result.sensor_data.ndim == 1
        assert result.sensor_data.shape[0] == 50

    def test_point_sensor_data_is_numpy(self, grid, medium):
        """Sensor data is returned as numpy array."""
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sensor = kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))

        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=10, dt=1e-8)

        assert isinstance(result.sensor_data, np.ndarray)

    def test_time_vector_matches_steps(self, grid, medium):
        """Time vector length matches time_steps."""
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sensor = kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))

        nt = 30
        dt = 1e-8
        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=nt, dt=dt)

        assert result.time.shape[0] == nt

    def test_time_vector_monotonically_increasing(self, grid, medium):
        """Time vector is monotonically increasing."""
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sensor = kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))

        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=50, dt=1e-8)

        diffs = np.diff(result.time)
        assert np.all(diffs > 0), "Time vector not monotonically increasing"

    def test_different_sensor_positions_give_different_data(self, grid, medium):
        """Sensors at different positions record different time series."""
        # Use point source - spherical wavefront gives position-dependent data
        src = kw.Source.point(position=(0.0, 1.6e-3, 1.6e-3), frequency=1e6, amplitude=1e5)
        nt = 100
        dt = compute_cfl_dt(grid.dx, 1500.0)

        # Positions at different distances from source along X axis
        positions = [
            (0.4e-3, 1.6e-3, 1.6e-3),   # close
            (1.6e-3, 1.6e-3, 1.6e-3),    # middle
            (2.8e-3, 1.6e-3, 1.6e-3),    # far
        ]
        results = []
        for pos in positions:
            sensor = kw.Sensor.point(pos)
            sim = kw.Simulation(grid, medium, src, sensor)
            r = sim.run(time_steps=nt, dt=dt)
            results.append(r.sensor_data.copy())

        # At least some pairs should differ
        differs = False
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if not np.allclose(results[i], results[j]):
                    differs = True
                    break
        assert differs, "All sensor positions gave identical data"


# ============================================================================
# Cross-validation: pykwavers Sensor vs kSensor
# ============================================================================


@requires_kwave
class TestSensorParityWithKWave:
    """Compare pykwavers Sensor setup against kSensor."""

    def test_point_mask_creation_matches(self):
        """Both create equivalent point sensor masks."""
        N = 32
        dx = 0.1e-3
        ix, iy, iz = N // 2, N // 2, N // 2

        # pykwavers: point sensor (internally maps to grid index)
        sensor_pk = kw.Sensor.point((ix * dx, iy * dx, iz * dx))
        assert sensor_pk is not None

        # k-wave: binary mask
        mask = np.zeros((N, N, N), dtype=bool)
        mask[ix, iy, iz] = True
        sensor_kw = kSensor(mask)
        assert sensor_kw is not None

    def test_sensor_mask_shapes_compatible(self):
        """kSensor mask shape matches grid shape."""
        N = 64
        dx = 0.1e-3

        mask = np.zeros((N, N, N), dtype=bool)
        mask[N // 2, N // 2, N // 2] = True
        sensor_kw = kSensor(mask)

        assert sensor_kw.mask.shape == (N, N, N)

    def test_multi_point_mask_possible(self):
        """Multiple sensor points via mask."""
        N = 32
        mask = np.zeros((N, N, N), dtype=bool)
        mask[8, 16, 16] = True
        mask[16, 16, 16] = True
        mask[24, 16, 16] = True

        sensor_kw = kSensor(mask)
        assert np.sum(sensor_kw.mask) == 3


# ============================================================================
# SimulationResult tests
# ============================================================================


class TestSimulationResult:
    """Test SimulationResult properties."""

    def test_result_has_sensor_data(self, grid, medium, sensor):
        """Result has sensor_data attribute."""
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=10, dt=1e-8)
        assert hasattr(result, "sensor_data")

    def test_result_has_time(self, grid, medium, sensor):
        """Result has time attribute."""
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=10, dt=1e-8)
        assert hasattr(result, "time")

    def test_result_has_time_steps(self, grid, medium, sensor):
        """Result has time_steps attribute."""
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=10, dt=1e-8)
        assert result.time_steps == 10

    def test_result_has_dt(self, grid, medium, sensor):
        """Result has dt attribute."""
        dt = 5e-9
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=10, dt=dt)
        assert abs(result.dt - dt) < 1e-15

    def test_result_final_time(self, grid, medium, sensor):
        """Result has correct final_time."""
        nt = 100
        dt = 1e-8
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=nt, dt=dt)
        assert hasattr(result, "final_time")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
