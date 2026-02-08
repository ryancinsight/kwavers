#!/usr/bin/env python3
"""
Utility and Integration Tests

Tests for helper utilities, error metrics computation, and integration
between pykwavers components.
"""

import numpy as np
import pytest

import pykwavers as kw
from conftest import compute_cfl_dt, compute_error_metrics


# ============================================================================
# Error metrics utility tests
# ============================================================================


class TestErrorMetrics:
    """Test the error metrics computation utility."""

    def test_identical_signals(self):
        """Identical signals give zero error and perfect correlation."""
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        metrics = compute_error_metrics(signal, signal)
        assert metrics["l2_error"] < 1e-10
        assert metrics["correlation"] > 0.999

    def test_opposite_signals(self):
        """Opposite signals give high error and negative correlation."""
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        metrics = compute_error_metrics(signal, -signal)
        assert metrics["l2_error"] > 1.0
        assert metrics["correlation"] < -0.9

    def test_scaled_signals(self):
        """Scaled signals have high correlation but error proportional to scale."""
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        metrics = compute_error_metrics(signal, 2.0 * signal)
        assert metrics["correlation"] > 0.999
        assert metrics["l2_error"] > 0.5  # L2 = ||2s - s|| / ||s|| = 1.0

    def test_uncorrelated_signals(self):
        """Random signals have low correlation."""
        rng = np.random.RandomState(42)
        s1 = rng.randn(1000)
        s2 = rng.randn(1000)
        metrics = compute_error_metrics(s1, s2)
        assert abs(metrics["correlation"]) < 0.1

    def test_different_lengths(self):
        """Handles signals of different lengths (truncates to min)."""
        s1 = np.sin(np.linspace(0, 4 * np.pi, 100))
        s2 = np.sin(np.linspace(0, 4 * np.pi, 80))
        metrics = compute_error_metrics(s1, s2)
        assert "l2_error" in metrics
        assert "correlation" in metrics

    def test_zero_signal(self):
        """Handles zero reference signal gracefully."""
        s1 = np.zeros(100)
        s2 = np.zeros(100)
        metrics = compute_error_metrics(s1, s2)
        assert metrics["l2_error"] == 0.0


# ============================================================================
# CFL time step utility
# ============================================================================


class TestCFLDt:
    """Test CFL time step computation."""

    def test_water_standard(self):
        """CFL dt for water at 0.1mm spacing."""
        dt = compute_cfl_dt(0.1e-3, 1500.0)
        assert dt > 0
        assert dt == pytest.approx(0.3 * 0.1e-3 / 1500.0)

    def test_higher_speed_smaller_dt(self):
        """Higher sound speed gives smaller dt."""
        dt_water = compute_cfl_dt(0.1e-3, 1500.0)
        dt_bone = compute_cfl_dt(0.1e-3, 3000.0)
        assert dt_bone < dt_water

    def test_larger_spacing_larger_dt(self):
        """Larger spacing allows larger dt."""
        dt_fine = compute_cfl_dt(0.05e-3, 1500.0)
        dt_coarse = compute_cfl_dt(0.2e-3, 1500.0)
        assert dt_coarse > dt_fine

    def test_custom_cfl(self):
        """Custom CFL number."""
        dt_default = compute_cfl_dt(0.1e-3, 1500.0, cfl=0.3)
        dt_aggressive = compute_cfl_dt(0.1e-3, 1500.0, cfl=0.5)
        assert dt_aggressive > dt_default


# ============================================================================
# Simulation integration tests
# ============================================================================


class TestSimulationIntegration:
    """End-to-end integration tests for simulation pipeline."""

    def test_complete_pipeline(self):
        """Full pipeline: grid -> medium -> source -> sensor -> run -> result."""
        grid = kw.Grid(nx=16, ny=16, nz=16, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        medium = kw.Medium.homogeneous(1500.0, 1000.0)
        source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sensor = kw.Sensor.point((0.8e-3, 0.8e-3, 0.8e-3))

        sim = kw.Simulation(grid, medium, source, sensor)
        result = sim.run(time_steps=10, dt=1e-8)

        assert result.time_steps == 10
        assert result.sensor_data.shape == (10,)
        assert result.time.shape == (10,)
        assert np.all(np.isfinite(result.sensor_data))
        assert np.all(np.isfinite(result.time))

    def test_pipeline_with_mask_source(self):
        """Pipeline with mask-based source."""
        N = 16
        dx = 0.1e-3
        nt = 50
        dt = compute_cfl_dt(dx, 1500.0)

        grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
        medium = kw.Medium.homogeneous(1500.0, 1000.0)

        mask = np.zeros((N, N, N), dtype=np.float64)
        mask[0, :, :] = 1.0
        t = np.arange(nt) * dt
        signal = 1e5 * np.sin(2 * np.pi * 1e6 * t)
        source = kw.Source.from_mask(mask, signal, frequency=1e6)

        sensor = kw.Sensor.point((0.8e-3, 0.8e-3, 0.8e-3))
        sim = kw.Simulation(grid, medium, source, sensor)
        result = sim.run(time_steps=nt, dt=dt)

        assert result.time_steps == nt
        assert np.all(np.isfinite(result.sensor_data))

    def test_pipeline_with_pml_size(self):
        """Pipeline with custom PML size."""
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        medium = kw.Medium.homogeneous(1500.0, 1000.0)
        source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sensor = kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))

        for pml_size in [4, 6, 10]:
            sim = kw.Simulation(grid, medium, source, sensor, pml_size=pml_size)
            result = sim.run(time_steps=20, dt=1e-8)
            assert np.all(np.isfinite(result.sensor_data))

    def test_pipeline_all_solver_types(self):
        """All solver types run to completion."""
        grid = kw.Grid(nx=16, ny=16, nz=16, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        medium = kw.Medium.homogeneous(1500.0, 1000.0)
        source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sensor = kw.Sensor.point((0.8e-3, 0.8e-3, 0.8e-3))

        for solver in [kw.SolverType.FDTD, kw.SolverType.PSTD, kw.SolverType.Hybrid]:
            sim = kw.Simulation(grid, medium, source, sensor, solver=solver)
            result = sim.run(time_steps=10, dt=1e-8)
            assert result.time_steps == 10
            assert np.all(np.isfinite(result.sensor_data))

    def test_reproducibility(self):
        """Same configuration gives same result."""
        grid = kw.Grid(nx=16, ny=16, nz=16, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        medium = kw.Medium.homogeneous(1500.0, 1000.0)
        source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sensor = kw.Sensor.point((0.8e-3, 0.8e-3, 0.8e-3))

        sim1 = kw.Simulation(grid, medium, source, sensor)
        r1 = sim1.run(time_steps=50, dt=1e-8)

        sim2 = kw.Simulation(grid, medium, source, sensor)
        r2 = sim2.run(time_steps=50, dt=1e-8)

        np.testing.assert_array_equal(r1.sensor_data, r2.sensor_data)


# ============================================================================
# Feature gap documentation tests
# ============================================================================


class TestFeatureGaps:
    """
    Document features present in k-wave-python but not yet in pykwavers.
    These tests mark known gaps -- they should PASS (via xfail or skip)
    and serve as a roadmap for expanding pykwavers.
    """

    @pytest.mark.xfail(reason="pykwavers lacks initial pressure p0 support")
    def test_initial_pressure_p0(self):
        """pykwavers should support initial pressure distribution (p0)."""
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        medium = kw.Medium.homogeneous(1500.0, 1000.0)
        sensor = kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))

        # This would be the desired API:
        p0 = np.zeros((32, 32, 32))
        p0[16, 16, 16] = 1e5
        source = kw.Source.initial_pressure(p0)  # noqa: F841

    @pytest.mark.xfail(reason="pykwavers lacks heterogeneous medium support via Python API")
    def test_heterogeneous_medium(self):
        """pykwavers should support spatially varying sound speed/density."""
        c = np.ones((32, 32, 32)) * 1500.0
        c[16:, :, :] = 2000.0
        rho = np.ones((32, 32, 32)) * 1000.0

        medium = kw.Medium(sound_speed=c, density=rho)  # noqa: F841

    @pytest.mark.xfail(reason="pykwavers lacks velocity source support")
    def test_velocity_source(self):
        """pykwavers should support velocity sources (ux, uy, uz)."""
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        mask = np.zeros((32, 32, 32))
        mask[0, :, :] = 1.0
        signal = np.sin(np.linspace(0, 2 * np.pi, 50))

        source = kw.Source.velocity_source(mask, signal, direction="z")  # noqa: F841

    @pytest.mark.xfail(reason="pykwavers lacks absorption parameters in Python API")
    def test_absorption_parameters(self):
        """pykwavers should expose alpha_coeff and alpha_power."""
        medium = kw.Medium.homogeneous(
            sound_speed=1500.0, density=1000.0,
            alpha_coeff=0.75, alpha_power=1.5,
        )
        assert hasattr(medium, "alpha_coeff")
        assert medium.alpha_coeff == 0.75

    @pytest.mark.xfail(reason="pykwavers lacks multi-sensor recording")
    def test_multi_sensor_mask(self):
        """pykwavers should support binary mask sensors (multiple points)."""
        mask = np.zeros((32, 32, 32), dtype=bool)
        mask[8, 16, 16] = True
        mask[16, 16, 16] = True
        mask[24, 16, 16] = True

        sensor = kw.Sensor.from_mask(mask)  # noqa: F841

    @pytest.mark.xfail(reason="pykwavers lacks direction parameter for plane_wave")
    def test_plane_wave_direction(self):
        """pykwavers should support plane wave direction parameter."""
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        source = kw.Source.plane_wave(
            grid, frequency=1e6, amplitude=1e5,
            direction=(1.0, 0.0, 0.0),  # x-direction
        )
        assert source is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
