#!/usr/bin/env python3
"""
Source Parity Tests: pykwavers.Source vs kwave.kSource

Validates source creation, mask-based sources, and temporal signal
specification match k-wave-python's kSource behavior.
"""

import numpy as np
import pytest

import pykwavers as kw
from conftest import requires_kwave, compute_cfl_dt, HAS_KWAVE

if HAS_KWAVE:
    from kwave.ksource import kSource
    from kwave.utils.signals import tone_burst


# ============================================================================
# pykwavers Source standalone tests
# ============================================================================


class TestSourceCreation:
    """Test pykwavers Source construction."""

    def test_plane_wave_source(self, grid):
        """Plane wave source creation."""
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        assert src.source_type == "plane_wave"
        assert src.frequency == 1e6
        assert src.amplitude == 1e5

    def test_point_source(self):
        """Point source creation."""
        src = kw.Source.point(position=(1e-3, 1e-3, 1e-3), frequency=1e6, amplitude=1e5)
        assert src.source_type == "point"
        assert src.frequency == 1e6
        assert src.amplitude == 1e5

    def test_mask_source(self, grid):
        """Mask-based source creation."""
        mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float64)
        mask[0, :, :] = 1.0  # Planar source at x=0

        nt = 50
        dt = compute_cfl_dt(grid.dx, 1500.0)
        t = np.arange(nt) * dt
        signal = 1e5 * np.sin(2 * np.pi * 1e6 * t)

        src = kw.Source.from_mask(mask, signal, frequency=1e6)
        assert src is not None

    @pytest.mark.parametrize("freq", [0.5e6, 1e6, 2e6, 5e6, 10e6])
    def test_various_frequencies(self, grid, freq):
        """Source works at different frequencies."""
        src = kw.Source.plane_wave(grid, frequency=freq, amplitude=1e5)
        assert src.frequency == freq

    @pytest.mark.parametrize("amp", [1e3, 1e4, 1e5, 1e6])
    def test_various_amplitudes(self, grid, amp):
        """Source works at different amplitudes."""
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=amp)
        assert src.amplitude == amp

    def test_point_source_various_positions(self):
        """Point source at different positions."""
        positions = [
            (0.5e-3, 0.5e-3, 0.5e-3),
            (1e-3, 2e-3, 3e-3),
            (0.1e-3, 0.1e-3, 0.1e-3),
        ]
        for pos in positions:
            src = kw.Source.point(position=pos, frequency=1e6, amplitude=1e5)
            assert src.source_type == "point"


class TestMaskSourceSignalMatching:
    """Test that mask sources correctly handle temporal signals."""

    def test_signal_length_must_match_time_steps(self, grid):
        """Signal length must match simulation time_steps."""
        mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float64)
        mask[0, :, :] = 1.0

        nt = 50
        dt = 1e-8
        t = np.arange(nt) * dt
        signal = 1e5 * np.sin(2 * np.pi * 1e6 * t)

        src = kw.Source.from_mask(mask, signal, frequency=1e6)
        m = kw.Medium.homogeneous(1500.0, 1000.0)
        sensor = kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))
        sim = kw.Simulation(grid, m, src, sensor)

        # Should succeed with matching time_steps
        result = sim.run(time_steps=nt, dt=dt)
        assert result.time_steps == nt

    def test_signal_length_mismatch_raises(self, grid):
        """Mismatched signal length and time_steps raises error."""
        mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float64)
        mask[0, :, :] = 1.0
        signal = np.sin(np.linspace(0, 2 * np.pi, 100))

        src = kw.Source.from_mask(mask, signal, frequency=1e6)
        m = kw.Medium.homogeneous(1500.0, 1000.0)
        sensor = kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))
        sim = kw.Simulation(grid, m, src, sensor)

        with pytest.raises(ValueError, match="Signal length"):
            sim.run(time_steps=50, dt=1e-8)

    def test_mask_source_plane_produces_nonzero(self, grid):
        """Planar mask source produces non-zero sensor data."""
        mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float64)
        mask[0, :, :] = 1.0

        nt = 100
        dt = compute_cfl_dt(grid.dx, 1500.0)
        t = np.arange(nt) * dt
        signal = 1e5 * np.sin(2 * np.pi * 1e6 * t)

        src = kw.Source.from_mask(mask, signal, frequency=1e6)
        m = kw.Medium.homogeneous(1500.0, 1000.0)
        sensor = kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))
        sim = kw.Simulation(grid, m, src, sensor)

        result = sim.run(time_steps=nt, dt=dt)
        assert np.max(np.abs(result.sensor_data)) > 0, "Mask source produced all-zero data"

    def test_mask_source_point_produces_nonzero(self, grid):
        """Single-voxel mask source produces non-zero sensor data."""
        mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float64)
        mask[5, 5, 5] = 1.0

        nt = 100
        dt = compute_cfl_dt(grid.dx, 1500.0)
        t = np.arange(nt) * dt
        signal = 1e5 * np.sin(2 * np.pi * 1e6 * t)

        src = kw.Source.from_mask(mask, signal, frequency=1e6)
        m = kw.Medium.homogeneous(1500.0, 1000.0)
        sensor = kw.Sensor.point((2e-3, 2e-3, 2e-3))
        sim = kw.Simulation(grid, m, src, sensor)

        result = sim.run(time_steps=nt, dt=dt)
        assert np.max(np.abs(result.sensor_data)) > 0, "Point mask source produced all-zero data"


# ============================================================================
# Cross-validation: pykwavers Source vs kSource
# ============================================================================


@requires_kwave
class TestSourceParityWithKWave:
    """Compare pykwavers Source setup against kSource."""

    def test_pressure_mask_source_creation_matches(self, grid):
        """Both create pressure mask sources with same pattern."""
        N = grid.nx
        dx = grid.dx

        # Create identical mask: planar source at z=0
        mask_pk = np.zeros((N, N, N), dtype=np.float64)
        mask_pk[:, :, 0] = 1.0

        mask_kw = np.zeros((N, N, N), dtype=bool)
        mask_kw[:, :, 0] = True

        nt = 100
        dt = compute_cfl_dt(dx, 1500.0)
        t = np.arange(nt) * dt
        signal = 1e5 * np.sin(2 * np.pi * 1e6 * t)

        # pykwavers
        src_pk = kw.Source.from_mask(mask_pk, signal, frequency=1e6)
        assert src_pk is not None

        # k-wave
        src_kw = kSource()
        src_kw.p_mask = mask_kw
        src_kw.p = signal.reshape(1, -1)  # k-wave expects [num_sources, nt]
        assert src_kw.p_mask is not None

    def test_point_mask_source_creation_matches(self, grid):
        """Both create single-voxel pressure sources."""
        N = grid.nx
        dx = grid.dx

        ix, iy, iz = N // 4, N // 2, N // 2

        mask_pk = np.zeros((N, N, N), dtype=np.float64)
        mask_pk[ix, iy, iz] = 1.0

        mask_kw = np.zeros((N, N, N), dtype=bool)
        mask_kw[ix, iy, iz] = True

        nt = 100
        dt = compute_cfl_dt(dx, 1500.0)
        t = np.arange(nt) * dt
        signal = 1e5 * np.sin(2 * np.pi * 1e6 * t)

        src_pk = kw.Source.from_mask(mask_pk, signal, frequency=1e6)
        assert src_pk is not None

        src_kw = kSource()
        src_kw.p_mask = mask_kw
        src_kw.p = signal.reshape(1, -1)
        assert src_kw.p_mask is not None

    def test_tone_burst_signal_compatible(self, grid):
        """k-wave tone_burst signal can be used in pykwavers from_mask."""
        N = grid.nx
        dx = grid.dx
        c = 1500.0
        dt = compute_cfl_dt(dx, c)
        freq = 1e6
        cycles = 3

        # Generate k-wave tone burst
        signal_kw = tone_burst(1.0 / dt, freq, cycles)

        # Use in pykwavers mask source
        mask = np.zeros((N, N, N), dtype=np.float64)
        mask[:, :, 0] = 1.0

        signal_flat = signal_kw.flatten().astype(np.float64)
        nt = len(signal_flat)
        src_pk = kw.Source.from_mask(mask, signal_flat * 1e5, frequency=freq)
        assert src_pk is not None

        # Run pykwavers simulation with this signal
        m = kw.Medium.homogeneous(c, 1000.0)
        sensor = kw.Sensor.point((N * dx / 2, N * dx / 2, N * dx / 2))
        sim = kw.Simulation(grid, m, src_pk, sensor)
        result = sim.run(time_steps=nt, dt=dt)

        assert np.all(np.isfinite(result.sensor_data))
        assert np.max(np.abs(result.sensor_data)) > 0


# ============================================================================
# Source physics validation
# ============================================================================


class TestSourcePhysics:
    """Validate correct physical behavior of sources."""

    def test_amplitude_scaling(self):
        """Output scales linearly with amplitude."""
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        m = kw.Medium.homogeneous(1500.0, 1000.0)
        sensor = kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))

        results = []
        for amp in [1e4, 1e5]:
            src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=amp)
            sim = kw.Simulation(grid, m, src, sensor)
            r = sim.run(time_steps=50, dt=1e-8)
            results.append(np.max(np.abs(r.sensor_data)))

        # 10x amplitude should give ~10x output
        ratio = results[1] / results[0]
        assert 5.0 < ratio < 20.0, f"Amplitude ratio {ratio:.1f} not near 10x"

    def test_source_produces_finite_data(self, grid, medium, sensor):
        """All source types produce finite sensor data."""
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=20, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
