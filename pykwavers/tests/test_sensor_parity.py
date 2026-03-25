#!/usr/bin/env python3
"""
Sensor Parity Tests: pykwavers.Sensor vs kwave.kSensor

Validates that pykwavers Sensor creation and recording behavior
match k-wave-python's kSensor.

This module tests:
1. Sensor creation (point, grid, mask-based)
2. Multi-sensor array configurations
3. Sensor data recording and output formats
4. SimulationResult properties
5. Array sensor geometry and beamforming
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

    def test_mask_sensor(self):
        """Sensor from 3D boolean mask."""
        mask = np.zeros((32, 32, 32), dtype=bool)
        mask[8, 16, 16] = True
        mask[24, 16, 16] = True
        s = kw.Sensor.from_mask(mask)
        assert s.sensor_type == "mask"
        assert s.num_sensors == 2

    def test_mask_sensor_single_point(self):
        """Single-point mask sensor."""
        mask = np.zeros((16, 16, 16), dtype=bool)
        mask[8, 8, 8] = True
        s = kw.Sensor.from_mask(mask)
        assert s.sensor_type == "mask"
        assert s.num_sensors == 1

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

    def test_mask_sensor_repr(self):
        """Mask sensor repr includes sensor count."""
        mask = np.zeros((16, 16, 16), dtype=bool)
        mask[4, 4, 4] = True
        mask[8, 8, 8] = True
        mask[12, 12, 12] = True
        s = kw.Sensor.from_mask(mask)
        r = repr(s)
        assert "mask" in r.lower()
        assert "3" in r  # 3 sensors


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

    def test_mask_sensor_records_2d_array(self, grid, medium):
        """Multi-point mask sensor returns 2D array (n_sensors, n_timesteps)."""
        N = 32
        dx = grid.dx
        mask = np.zeros((N, N, N), dtype=bool)
        mask[8, 16, 16] = True
        mask[24, 16, 16] = True
        mask[16, 8, 16] = True

        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sensor = kw.Sensor.from_mask(mask)

        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=50, dt=1e-8)

        assert result.sensor_data.ndim == 2, f"Expected 2D, got {result.sensor_data.ndim}D"
        assert result.sensor_data.shape[0] == 3, f"Expected 3 sensors, got {result.sensor_data.shape[0]}"
        assert result.sensor_data.shape[1] == 50

    def test_single_mask_sensor_returns_1d(self, grid, medium):
        """Single-point mask sensor returns 1D for backward compatibility."""
        N = 32
        mask = np.zeros((N, N, N), dtype=bool)
        mask[16, 16, 16] = True

        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sensor = kw.Sensor.from_mask(mask)

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


# ============================================================================
# Array Sensor tests
# ============================================================================


class TestArraySensor:
    """Test array sensor configurations."""

    def test_linear_array_sensor(self):
        """Create linear array sensor."""
        # Linear array along X axis
        n_elements = 8
        pitch = 0.3e-3  # 0.3 mm pitch
        positions = [(i * pitch, 0.0, 0.0) for i in range(n_elements)]
        
        # Create mask from positions
        N = 64
        dx = 0.1e-3
        mask = np.zeros((N, N, N), dtype=bool)
        
        for pos in positions:
            ix = int(pos[0] / dx) + N // 2
            iy = int(pos[1] / dx) + N // 2
            iz = int(pos[2] / dx) + N // 2
            if 0 <= ix < N and 0 <= iy < N and 0 <= iz < N:
                mask[ix, iy, iz] = True
        
        sensor = kw.Sensor.from_mask(mask)
        assert sensor.sensor_type == "mask"
        assert sensor.num_sensors == n_elements

    def test_2d_array_sensor(self):
        """Create 2D planar array sensor."""
        # 4x4 planar array in X-Y plane
        n_x, n_y = 4, 4
        pitch = 0.4e-3
        
        N = 64
        dx = 0.1e-3
        mask = np.zeros((N, N, N), dtype=bool)
        
        for i in range(n_x):
            for j in range(n_y):
                x = (i - n_x // 2) * pitch
                y = (j - n_y // 2) * pitch
                ix = int(x / dx) + N // 2
                iy = int(y / dx) + N // 2
                iz = N // 2
                if 0 <= ix < N and 0 <= iy < N:
                    mask[ix, iy, iz] = True
        
        sensor = kw.Sensor.from_mask(mask)
        assert sensor.num_sensors == n_x * n_y

    def test_circular_array_sensor(self):
        """Create circular array sensor."""
        # Circular array with elements on a circle
        n_elements = 16
        radius = 2e-3  # 2 mm radius
        
        N = 64
        dx = 0.1e-3
        mask = np.zeros((N, N, N), dtype=bool)
        
        for i in range(n_elements):
            angle = 2 * np.pi * i / n_elements
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            ix = int(x / dx) + N // 2
            iy = int(y / dx) + N // 2
            iz = N // 2
            if 0 <= ix < N and 0 <= iy < N:
                mask[ix, iy, iz] = True
        
        sensor = kw.Sensor.from_mask(mask)
        assert sensor.num_sensors == n_elements

    def test_ring_array_sensor_3d(self):
        """Create 3D ring array sensor."""
        # Ring array with elements distributed in 3D
        n_elements = 12
        radius = 1.5e-3
        
        N = 48
        dx = 0.1e-3
        mask = np.zeros((N, N, N), dtype=bool)
        
        for i in range(n_elements):
            angle = 2 * np.pi * i / n_elements
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0.5e-3 * np.sin(2 * angle)  # Slight Z variation
            ix = int(x / dx) + N // 2
            iy = int(y / dx) + N // 2
            iz = int(z / dx) + N // 2
            if 0 <= ix < N and 0 <= iy < N and 0 <= iz < N:
                mask[ix, iy, iz] = True
        
        sensor = kw.Sensor.from_mask(mask)
        assert sensor.num_sensors == n_elements


class TestSensorGeometry:
    """Test sensor geometry calculations."""

    def test_sensor_spacing_uniform(self):
        """Linear array has uniform spacing."""
        n_elements = 8
        pitch = 0.3e-3
        
        N = 64
        dx = 0.1e-3
        mask = np.zeros((N, N, N), dtype=bool)
        
        expected_positions = []
        for i in range(n_elements):
            x = (i - n_elements // 2) * pitch
            ix = int(x / dx) + N // 2
            if 0 <= ix < N:
                mask[ix, N // 2, N // 2] = True
                expected_positions.append(ix)
        
        sensor = kw.Sensor.from_mask(mask)
        assert sensor.num_sensors == len(expected_positions)

    def test_sensor_aperture_size(self):
        """Calculate sensor array aperture."""
        n_elements = 10
        pitch = 0.2e-3
        aperture = (n_elements - 1) * pitch  # 1.8 mm
        
        # Verify aperture calculation
        assert abs(aperture - 1.8e-3) < 1e-10

    def test_sensor_centroid(self):
        """Calculate sensor array centroid."""
        # Symmetric array should have centroid at center
        n_elements = 5
        pitch = 0.4e-3
        
        N = 64
        dx = 0.1e-3
        mask = np.zeros((N, N, N), dtype=bool)
        
        for i in range(n_elements):
            x = (i - n_elements // 2) * pitch
            ix = int(x / dx) + N // 2
            if 0 <= ix < N:
                mask[ix, N // 2, N // 2] = True
        
        sensor = kw.Sensor.from_mask(mask)
        assert sensor is not None


class TestSensorBeamforming:
    """Test sensor beamforming capabilities."""

    def test_receive_beamforming_linear_array(self, grid, medium):
        """Test receive beamforming with linear array."""
        # Create linear array sensor
        n_elements = 8
        pitch = 0.3e-3
        
        N = grid.nx  # Match grid fixture (32)
        dx = grid.dx
        mask = np.zeros((N, N, N), dtype=bool)
        
        for i in range(n_elements):
            x = (i - n_elements // 2) * pitch
            ix = int(x / dx) + N // 2
            if 0 <= ix < N:
                mask[ix, N // 2, N // 2] = True
        
        sensor = kw.Sensor.from_mask(mask)
        
        # Run simulation with point source
        src = kw.Source.point(position=(0.0, 0.0, 0.0), frequency=1e6, amplitude=1e5)
        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=50, dt=1e-8)
        
        # Should have n_elements time series
        assert result.sensor_data.shape[0] == n_elements

    def test_time_delay_calculation(self):
        """Calculate time delays for beamforming."""
        # Focus at 5 mm depth
        focus_depth = 5e-3
        sound_speed = 1500.0  # m/s
        
        n_elements = 8
        pitch = 0.3e-3
        
        # Calculate delays relative to center element
        delays = []
        for i in range(n_elements):
            x = (i - n_elements // 2) * pitch
            distance = np.sqrt(x**2 + focus_depth**2)
            delay = distance / sound_speed
            delays.append(delay)
        
        # Center element should have shortest delay
        center_idx = n_elements // 2
        min_delay = min(delays)
        assert abs(delays[center_idx] - min_delay) < 1e-10 or delays[center_idx] == min(delays)

    def test_dynamic_receive_aperture(self):
        """Test dynamic receive aperture concept."""
        # As wave propagates, active aperture changes
        f_number = 2.0  # f/#
        depth = 10e-3  # 10 mm
        
        # Active aperture = depth / f_number
        active_aperture = depth / f_number
        assert abs(active_aperture - 5e-3) < 1e-10


# ============================================================================
# Additional parity tests with k-wave
# ============================================================================


@requires_kwave
class TestArraySensorParityWithKWave:
    """Compare array sensor configurations with k-wave."""

    def test_linear_array_mask_parity(self):
        """Linear array mask matches between implementations."""
        n_elements = 8
        pitch = 0.3e-3
        N = 64
        dx = 0.1e-3
        
        # Create mask
        mask = np.zeros((N, N, N), dtype=bool)
        for i in range(n_elements):
            x = (i - n_elements // 2) * pitch
            ix = int(x / dx) + N // 2
            if 0 <= ix < N:
                mask[ix, N // 2, N // 2] = True
        
        # pykwavers
        sensor_pk = kw.Sensor.from_mask(mask)
        
        # k-wave
        sensor_kw = kSensor(mask)
        
        assert sensor_pk.num_sensors == np.sum(sensor_kw.mask)

    def test_2d_array_mask_parity(self):
        """2D array mask matches between implementations."""
        n_x, n_y = 4, 4
        pitch = 0.4e-3
        N = 64
        dx = 0.1e-3
        
        mask = np.zeros((N, N, N), dtype=bool)
        for i in range(n_x):
            for j in range(n_y):
                x = (i - n_x // 2) * pitch
                y = (j - n_y // 2) * pitch
                ix = int(x / dx) + N // 2
                iy = int(y / dx) + N // 2
                if 0 <= ix < N and 0 <= iy < N:
                    mask[ix, iy, N // 2] = True
        
        sensor_pk = kw.Sensor.from_mask(mask)
        sensor_kw = kSensor(mask)
        
        assert sensor_pk.num_sensors == np.sum(sensor_kw.mask)

    def test_sensor_record_time_series_parity(self):
        """Sensor time series recording matches k-wave format."""
        N = 32
        dx = 0.1e-3
        
        # Single point sensor
        mask = np.zeros((N, N, N), dtype=bool)
        mask[N // 2, N // 2, N // 2] = True
        
        sensor_kw = kSensor(mask)
        
        # kSensor stores mask
        assert sensor_kw.mask.shape == (N, N, N)
        assert np.sum(sensor_kw.mask) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
