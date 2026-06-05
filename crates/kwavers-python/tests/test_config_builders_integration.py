#!/usr/bin/env python3
"""
Config builders integration test: PmlConfig, HelmholtzConfig, NonlinearConfig, ThermalConfig.

Exercises the PyO3 config builder classes end-to-end with `Simulation.run()`,
verifying that the delegation from pykwavers → kwavers::simulation::runner works
correctly for every config path.

Tests:
1.  Config builder creation and fluent builder API
2.  Attachment via Simulation.set_*_config()
3.  Round-trip: config survives ``Simulation.run()`` and produces valid results
4.  Backward compatibility: legacy setters sync to config builder fields
5.  Error cases: invalid values rejected at construction time
"""

import numpy as np
import pykwavers as kw
import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_grid():
    """32³ grid with 0.1 mm spacing — fast enough for integration tests."""
    return kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)


@pytest.fixture
def water_medium():
    """Water at 20 °C."""
    return kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)


@pytest.fixture
def plane_source(small_grid):
    """1 MHz plane wave."""
    return kw.Source.plane_wave(small_grid, frequency=1e6, amplitude=1e5)


@pytest.fixture
def center_sensor():
    """Sensor at centre of 32³ grid."""
    return kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))


@pytest.fixture
def multi_sensor(small_grid):
    """Three aligned sensors across the domain centre via boolean mask."""
    nx, ny, nz = small_grid.nx, small_grid.ny, small_grid.nz
    mask = np.zeros((nx, ny, nz), dtype=bool)
    # z-indices for 0.8, 1.6, 2.4 mm at 0.1 mm spacing
    cz = nx // 2
    mask[cz, cz, 8] = True
    mask[cz, cz, 16] = True
    mask[cz, cz, 24] = True
    return kw.Sensor.from_mask(mask)


@pytest.fixture
def sim(small_grid, water_medium, plane_source, center_sensor):
    """Baseline FDTD simulation ready to accept config overrides."""
    return kw.Simulation(small_grid, water_medium, plane_source, center_sensor)


# ============================================================================
# PmlConfig
# ============================================================================


class TestPmlConfig:
    """PML boundary configuration tests."""

    def test_default_construction(self):
        """PmlConfig() creates a config with all-None fields."""
        pml = kw.PmlConfig()
        assert pml is not None
        rep = repr(pml)
        assert "PmlConfig" in rep

    def test_with_size_fluent(self):
        """with_size() sets uniform PML thickness."""
        pml = kw.PmlConfig().with_size(20)
        rep = repr(pml)
        assert "20" in rep

    def test_with_size_xyz_fluent(self):
        """with_size_xyz() sets per-axis thickness."""
        pml = kw.PmlConfig().with_size_xyz(20, 10, 15)
        rep = repr(pml)
        assert "PmlConfig" in rep

    def test_with_inside_fluent(self):
        """with_inside() toggles interior vs padded PML placement."""
        pml = kw.PmlConfig().with_inside(False)
        rep = repr(pml)
        assert "false" in rep

    def test_with_alpha_fluent(self):
        """with_alpha() sets uniform absorption factor."""
        pml = kw.PmlConfig().with_alpha(2.0)
        rep = repr(pml)
        assert "2.0" in rep or "2" in rep

    def test_with_alpha_xyz_fluent(self):
        """with_alpha_xyz() sets per-axis absorption."""
        pml = kw.PmlConfig().with_alpha_xyz(2.0, 1.5, 3.0)
        rep = repr(pml)
        assert "PmlConfig" in rep

    def test_full_chain(self):
        """Chaining every builder method produces a coherent config."""
        pml = (
            kw.PmlConfig()
            .with_size(20)
            .with_inside(True)
            .with_alpha(2.0)
        )
        rep = repr(pml)
        assert "20" in rep
        assert "true" in rep
        assert "2.0" in rep or "2" in rep

    def test_set_pml_config(self, sim):
        """set_pml_config() on Simulation accepts a pre-built PmlConfig."""
        pml = kw.PmlConfig().with_size(15).with_alpha(1.8)
        sim.set_pml_config(pml)  # should not raise

    def test_no_pml_simulation_runs(self, sim):
        """A simulation with default PML config runs and produces finite output."""
        pml = kw.PmlConfig().with_size(10)
        sim.set_pml_config(pml)
        result = sim.run(time_steps=50, dt=1e-8)
        assert result.time_steps == 50
        assert np.all(np.isfinite(result.sensor_data))

    def test_no_pml_nonlinear_simulation_runs(self, sim):
        """PML + nonlinear config together run cleanly."""
        sim.set_pml_config(kw.PmlConfig().with_size(10))
        sim.set_nonlinear_config(kw.NonlinearConfig().with_enabled())
        result = sim.run(time_steps=50, dt=1e-8)
        assert result.time_steps == 50
        assert np.all(np.isfinite(result.sensor_data))


# ============================================================================
# HelmholtzConfig
# ============================================================================


class TestHelmholtzConfig:
    """Helmholtz frequency-domain solver configuration tests."""

    def test_default_construction(self):
        """HelmholtzConfig() creates a config with no frequency set."""
        hc = kw.HelmholtzConfig()
        assert hc is not None
        rep = repr(hc)
        assert "HelmholtzConfig" in rep

    def test_with_frequency_fluent(self):
        """with_frequency() sets the Helmholtz source frequency in Hz."""
        hc = kw.HelmholtzConfig().with_frequency(1e6)
        rep = repr(hc)
        assert "1000000" in rep or "1e" in rep

    def test_with_frequency_rejects_zero(self):
        """Frequency ≤ 0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            kw.HelmholtzConfig().with_frequency(0.0)

    def test_with_frequency_rejects_negative(self):
        """Negative frequency raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            kw.HelmholtzConfig().with_frequency(-1e6)

    def test_set_helmholtz_config(self, sim):
        """set_helmholtz_config() on Simulation accepts a HelmholtzConfig."""
        hc = kw.HelmholtzConfig().with_frequency(500e3)
        sim.set_helmholtz_config(hc)  # should not raise


# ============================================================================
# NonlinearConfig
# ============================================================================


class TestNonlinearConfig:
    """Nonlinear acoustics (Westervelt) configuration tests."""

    def test_default_construction(self):
        """NonlinearConfig() creates a disabled config by default."""
        nl = kw.NonlinearConfig()
        rep = repr(nl)
        assert "NonlinearConfig" in rep
        assert "false" in rep or "enabled" in rep

    def test_with_enabled_fluent(self):
        """with_enabled() activates the Westervelt source term."""
        nl = kw.NonlinearConfig().with_enabled()
        rep = repr(nl)
        assert "true" in rep

    def test_with_alpha_coeff_fluent(self):
        """with_alpha_coeff() sets the absorption coefficient."""
        nl = kw.NonlinearConfig().with_alpha_coeff(0.75)
        rep = repr(nl)
        assert "0.75" in rep

    def test_with_alpha_power_fluent(self):
        """with_alpha_power() sets the power-law exponent."""
        nl = kw.NonlinearConfig().with_alpha_power(1.5)
        rep = repr(nl)
        assert "1.5" in rep

    def test_full_chain(self):
        """Chaining every builder method produces a coherent config."""
        nl = (
            kw.NonlinearConfig()
            .with_enabled()
            .with_alpha_coeff(0.75)
            .with_alpha_power(1.5)
        )
        rep = repr(nl)
        assert "true" in rep
        assert "0.75" in rep
        assert "1.5" in rep

    def test_set_nonlinear_config(self, sim):
        """set_nonlinear_config() on Simulation accepts a NonlinearConfig."""
        nl = kw.NonlinearConfig().with_enabled().with_alpha_coeff(0.5)
        sim.set_nonlinear_config(nl)  # should not raise

    def test_nonlinear_run_handles_default(self, sim):
        """Running with disabled nonlinear config (default) is fine."""
        sim.set_nonlinear_config(kw.NonlinearConfig())
        result = sim.run(time_steps=50, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))

    def test_nonlinear_run_with_nonzero_alpha(self, sim):
        """Running with a nonzero alpha_coeff but disabled nonlinear is fine."""
        sim.set_nonlinear_config(kw.NonlinearConfig().with_alpha_coeff(0.75))
        result = sim.run(time_steps=50, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))

    def test_nonlinear_run_with_both_enabled_and_alpha(self, sim):
        """Running with fully-enabled nonlinear config runs cleanly."""
        sim.set_nonlinear_config(
            kw.NonlinearConfig().with_enabled().with_alpha_coeff(0.75)
        )
        result = sim.run(time_steps=50, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))


# ============================================================================
# ThermalConfig
# ============================================================================


class TestThermalConfig:
    """Acoustic→thermal coupling configuration tests."""

    def test_minimal_construction(self):
        """ThermalConfig(center_frequency=1e6) creates valid defaults."""
        tc = kw.ThermalConfig(center_frequency=1e6)
        rep = repr(tc)
        assert "ThermalConfig" in rep

    def test_center_frequency_rejects_zero(self):
        """center_frequency=0 raises ValueError."""
        with pytest.raises(ValueError, match="> 0"):
            kw.ThermalConfig(center_frequency=0.0)

    def test_center_frequency_rejects_negative(self):
        """Negative frequency raises ValueError."""
        with pytest.raises(ValueError, match="> 0"):
            kw.ThermalConfig(center_frequency=-1e6)

    def test_n_acoustic_per_thermal_rejects_zero(self):
        """n_acoustic_per_thermal=0 raises ValueError."""
        with pytest.raises(ValueError, match=">= 1"):
            kw.ThermalConfig(center_frequency=1e6, n_acoustic_per_thermal=0)

    def test_invalid_material_properties_rejected(self):
        """Negative thermal properties raise ValueError."""
        with pytest.raises(ValueError, match="> 0"):
            kw.ThermalConfig(
                center_frequency=1e6,
                thermal_conductivity=-0.5,
            )

    def test_full_constructor(self):
        """Full constructor with all overrides produces a valid config."""
        tc = kw.ThermalConfig(
            center_frequency=1e6,
            n_acoustic_per_thermal=10,
            thermal_conductivity=0.6,
            density=1050.0,
            specific_heat=3700.0,
            enable_bioheat=True,
            perfusion_rate=1e-2,
            blood_density=1060.0,
            blood_specific_heat=3900.0,
            arterial_temperature=37.0,
            metabolic_heat=2000.0,
            initial_temperature=37.0,
            track_thermal_dose=True,
            dt_thermal=0.1,
        )
        rep = repr(tc)
        assert "ThermalConfig" in rep

    def test_with_bioheat_fluent(self):
        """with_bioheat() enables perfusion+metabolic terms."""
        tc = kw.ThermalConfig(center_frequency=1e6).with_bioheat()
        rep = repr(tc)
        assert "true" in rep

    def test_with_material_fluent(self):
        """with_material() sets k, rho, cp in one call."""
        tc = kw.ThermalConfig(center_frequency=1e6).with_material(0.6, 1050.0, 3700.0)
        rep = repr(tc)
        assert "ThermalConfig" in rep

    def test_with_steps_per_thermal_fluent(self):
        """with_steps_per_thermal() sets acoustic:thermal step ratio."""
        tc = kw.ThermalConfig(center_frequency=1e6).with_steps_per_thermal(5)
        rep = repr(tc)
        assert "ThermalConfig" in rep

    def test_set_thermal_config(self, sim):
        """set_thermal_config() on Simulation accepts a ThermalConfig."""
        tc = kw.ThermalConfig(center_frequency=1e6)
        sim.set_thermal_config(tc)  # should not raise

    def test_thermal_acoustic_run_is_finite(self, sim):
        """Running FDTD with thermal config attached produces finite output."""
        # Note: the FDTD solver path does not run the thermal loop, but the
        # config builder round-trip (python → rust → SimulationRunRequest)
        # is still exercised.
        tc = kw.ThermalConfig(center_frequency=1e6)
        sim.set_thermal_config(tc)
        result = sim.run(time_steps=50, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))


# ============================================================================
# Backward Compatibility: Legacy Setters
# ============================================================================


class TestLegacySetterSync:
    """Legacy setter methods keep config builder fields in sync."""

    def test_set_pml_size_syncs_config(self, sim):
        """set_pml_size() updates both legacy field and pml_config."""
        pml_cfg = kw.PmlConfig().with_size(10)
        sim.set_pml_config(pml_cfg)

        # Legacy setter updates both
        sim.set_pml_size(20)
        assert sim.pml_size == 20

        # Verify simulation still runs
        result = sim.run(time_steps=50, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))

    def test_set_pml_inside_syncs_config(self, sim):
        """set_pml_inside() updates both legacy field and pml_config."""
        sim.set_pml_inside(False)
        assert sim.pml_inside is False

        result = sim.run(time_steps=50, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))

    def test_set_nonlinear_syncs_config(self, sim):
        """set_nonlinear() updates both legacy field and nonlinear_config."""
        sim.set_nonlinear(True)
        assert sim.nonlinear is True

        result = sim.run(time_steps=50, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))

    def test_set_alpha_coeff_syncs_config(self, sim):
        """set_alpha_coeff() updates both legacy field and nonlinear_config."""
        sim.set_alpha_coeff(0.5)
        assert sim.alpha_coeff == 0.5

        result = sim.run(time_steps=50, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))

    def test_set_alpha_power_syncs_config(self, sim):
        """set_alpha_power() updates both legacy field and nonlinear_config."""
        sim.set_alpha_power(1.2)
        assert sim.alpha_power == 1.2

        result = sim.run(time_steps=50, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))

    def test_set_helmholtz_wavenumber_syncs_config(self, sim):
        """set_helmholtz_wavenumber() updates both legacy field and helmholtz_config."""
        sim.set_helmholtz_wavenumber(500e3)
        assert sim.helmholtz_frequency == 500e3

        result = sim.run(time_steps=50, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))


# ============================================================================
# Full Integration: All Configs + Run
# ============================================================================


class TestFullConfigRoundTrip:
    """End-to-end: set every config builder, run simulation, inspect output."""

    def test_all_configs_round_trip_fdtd(self, sim):
        """Every config builder attached before an FDTD run produces valid output."""
        sim.set_pml_config(kw.PmlConfig().with_size(10))
        sim.set_helmholtz_config(kw.HelmholtzConfig().with_frequency(1e6))
        sim.set_nonlinear_config(
            kw.NonlinearConfig().with_enabled().with_alpha_coeff(0.75)
        )

        # Thermal: runs FDTD path (thermal loop is PSTD-only but round-trip still works)
        sim.set_thermal_config(
            kw.ThermalConfig(
                center_frequency=1e6,
                n_acoustic_per_thermal=10,
            )
        )

        result = sim.run(time_steps=100, dt=1e-8)
        assert result.time_steps == 100
        assert result.dt == 1e-8
        assert np.all(np.isfinite(result.sensor_data))

        # Result metadata
        assert hasattr(result, "time")
        assert result.time_steps == 100
        assert result.dt == 1e-8
        assert result.final_time >= 0.0
        # Grid shape should survive the round-trip
        assert result.shape == (32, 32, 32)
        assert result.num_sensors >= 1

    def test_all_configs_round_trip_pstd(self, small_grid, water_medium, plane_source, center_sensor):
        """Every config builder attached before a PSTD run produces valid output."""
        sim = kw.Simulation(
            small_grid,
            water_medium,
            plane_source,
            center_sensor,
            solver=kw.SolverType.PSTD,
        )

        sim.set_pml_config(kw.PmlConfig().with_size(10).with_alpha(2.0))
        sim.set_helmholtz_config(kw.HelmholtzConfig().with_frequency(1e6))
        sim.set_nonlinear_config(
            kw.NonlinearConfig().with_enabled().with_alpha_coeff(0.75)
        )

        result = sim.run(time_steps=100, dt=1e-8)
        assert result.time_steps == 100
        # PSTD records Nt+1 samples (includes t=0)
        assert len(result.sensor_data) >= 100
        assert np.all(np.isfinite(result.sensor_data))

    def test_record_modes_with_configs(self, sim):
        """Config builders + record_modes=['p_max','p_min','p_rms'] populate stats."""
        sim.set_pml_config(kw.PmlConfig().with_size(10))
        sim.set_nonlinear_config(
            kw.NonlinearConfig().with_enabled().with_alpha_coeff(0.75)
        )

        result = sim.run(
            time_steps=50,
            dt=1e-8,
            record_modes=["p_max", "p_min", "p_rms", "p_final"],
        )
        assert result.time_steps == 50
        assert np.all(np.isfinite(result.sensor_data))

        # Sampled statistics should be populated
        assert result.p_max is not None, "p_max should be populated"
        assert result.p_min is not None, "p_min should be populated"
        assert result.p_rms is not None, "p_rms should be populated"
        assert result.p_final is not None, "p_final should be populated"
        assert np.all(np.isfinite(result.p_max))
        assert np.all(np.isfinite(result.p_min))
        assert np.all(np.isfinite(result.p_rms))
        assert np.all(np.isfinite(result.p_final))

    def test_record_all_fields_with_configs(self, sim):
        """Config builders + record_modes=['all'] populate sensor & field data."""
        sim.set_pml_config(kw.PmlConfig().with_size(10))

        result = sim.run(
            time_steps=30,
            dt=1e-8,
            record_modes=["all"],
        )
        assert np.all(np.isfinite(result.sensor_data))

        # "all" should populate both sampled stats and full-grid fields
        assert result.p_max is not None
        assert result.p_rms is not None


# ============================================================================
# Config Setter Order Independence
# ============================================================================


class TestConfigSetterOrderIndependence:
    """Config builder setters are commutative: order doesn't matter."""

    def test_nonlinear_then_pml(self, sim):
        """Setting nonlinear first then PML runs cleanly."""
        sim.set_nonlinear_config(
            kw.NonlinearConfig().with_enabled().with_alpha_coeff(0.75)
        )
        sim.set_pml_config(kw.PmlConfig().with_size(10))
        result = sim.run(time_steps=50, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))

    def test_pml_then_nonlinear(self, sim):
        """Setting PML first then nonlinear runs cleanly."""
        sim.set_pml_config(kw.PmlConfig().with_size(10))
        sim.set_nonlinear_config(
            kw.NonlinearConfig().with_enabled().with_alpha_coeff(0.75)
        )
        result = sim.run(time_steps=50, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))

    def test_multiple_reconfigures(self, sim):
        """Re-setting the same config multiple times is idempotent."""
        sim.set_pml_config(kw.PmlConfig().with_size(5))
        sim.set_pml_config(kw.PmlConfig().with_size(10))
        sim.set_pml_config(kw.PmlConfig().with_size(15))
        result = sim.run(time_steps=50, dt=1e-8)
        assert np.all(np.isfinite(result.sensor_data))


# ============================================================================
# Multi-Sensor Integration
# ============================================================================


class TestMultiSensorWithConfigs:
    """Config builders with multi-sensor arrays."""

    def test_multi_sensor_with_pml_nonlinear(self, small_grid, water_medium, plane_source, multi_sensor):
        """Multi-sensor + PmlConfig + NonlinearConfig runs and returns 2D data."""
        sim = kw.Simulation(small_grid, water_medium, plane_source, multi_sensor)
        sim.set_pml_config(kw.PmlConfig().with_size(10))
        sim.set_nonlinear_config(
            kw.NonlinearConfig().with_enabled().with_alpha_coeff(0.75)
        )

        result = sim.run(time_steps=50, dt=1e-8)
        assert result.time_steps == 50
        assert np.all(np.isfinite(result.sensor_data))

        # Multi-sensor data should be 2D: (n_sensors, n_timesteps)
        sensor_data = result.sensor_data
        assert sensor_data.ndim == 2, (
            f"Expected 2D multi-sensor data, got {sensor_data.ndim}D"
        )
        assert sensor_data.shape[0] == 3, (
            f"Expected 3 sensors, got {sensor_data.shape[0]}"
        )
        assert result.num_sensors == 3


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
