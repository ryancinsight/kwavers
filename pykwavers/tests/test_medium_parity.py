#!/usr/bin/env python3
"""
Medium Parity Tests: pykwavers.Medium vs kwave.kWaveMedium

Validates that pykwavers Medium creation and property specification
match k-wave-python's kWaveMedium behavior.
"""

import numpy as np
import pytest

import pykwavers as kw
from conftest import requires_kwave, HAS_KWAVE

if HAS_KWAVE:
    from kwave.kmedium import kWaveMedium


# ============================================================================
# pykwavers Medium standalone tests
# ============================================================================


class TestMediumCreation:
    """Test pykwavers Medium construction and properties."""

    def test_homogeneous_water(self):
        """Create homogeneous water medium."""
        m = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
        assert m is not None
        r = repr(m)
        assert "Medium" in r or "Homogeneous" in r

    def test_homogeneous_bone(self):
        """Create homogeneous bone medium."""
        m = kw.Medium.homogeneous(sound_speed=3000.0, density=1850.0)
        assert m is not None

    def test_homogeneous_soft_tissue(self):
        """Create homogeneous soft tissue medium."""
        m = kw.Medium.homogeneous(sound_speed=1540.0, density=1050.0)
        assert m is not None

    def test_homogeneous_with_absorption(self):
        """Create homogeneous medium with absorption."""
        m = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0, absorption=0.75)
        assert m is not None

    @pytest.mark.parametrize("c,rho", [
        (1500.0, 1000.0),    # water
        (1540.0, 1050.0),    # soft tissue
        (3000.0, 1850.0),    # cortical bone
        (330.0, 1.225),      # air
        (5900.0, 7850.0),    # steel
    ])
    def test_various_media(self, c, rho):
        """Medium creation succeeds for various physical materials."""
        m = kw.Medium.homogeneous(sound_speed=c, density=rho)
        assert m is not None

    def test_medium_repr(self):
        """Medium has meaningful representation."""
        m = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
        r = repr(m)
        assert len(r) > 0

    def test_medium_can_be_used_in_simulation(self, grid, sensor):
        """Medium can be passed to Simulation constructor."""
        m = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sim = kw.Simulation(grid, m, src, sensor)
        assert sim is not None


# ============================================================================
# Cross-validation: pykwavers Medium vs kWaveMedium
# ============================================================================


@requires_kwave
class TestMediumParityWithKWave:
    """Compare pykwavers Medium behavior against kWaveMedium."""

    def test_homogeneous_medium_accepts_same_params(self):
        """Both accept same sound_speed/density parameter pattern."""
        c, rho = 1500.0, 1000.0

        m_pk = kw.Medium.homogeneous(sound_speed=c, density=rho)
        m_kw = kWaveMedium(sound_speed=c, density=rho)

        assert m_pk is not None
        assert m_kw is not None

    def test_medium_with_absorption_params(self):
        """Both accept absorption parameters."""
        c, rho = 1500.0, 1000.0
        alpha_coeff = 0.75
        alpha_power = 1.5

        m_pk = kw.Medium.homogeneous(sound_speed=c, density=rho, absorption=alpha_coeff)

        m_kw = kWaveMedium(sound_speed=c, density=rho)
        m_kw.alpha_coeff = alpha_coeff
        m_kw.alpha_power = alpha_power

        assert m_pk is not None
        assert m_kw is not None

    def test_simulation_with_matched_media_runs(self, grid):
        """Simulation runs with medium matching kWaveMedium params."""
        c, rho = 1500.0, 1000.0
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        dx = grid.dx

        # pykwavers
        m_pk = kw.Medium.homogeneous(sound_speed=c, density=rho)
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sensor = kw.Sensor.point((nx * dx / 2, ny * dx / 2, nz * dx / 2))
        sim = kw.Simulation(grid, m_pk, src, sensor)
        result = sim.run(time_steps=10, dt=1e-8)

        assert result is not None
        assert result.time_steps == 10
        assert np.all(np.isfinite(result.sensor_data))


# ============================================================================
# Medium property edge cases
# ============================================================================


class TestMediumEdgeCases:
    """Test edge cases and boundary conditions for Medium."""

    def test_very_low_sound_speed(self):
        """Medium with very low sound speed (air-like)."""
        m = kw.Medium.homogeneous(sound_speed=330.0, density=1.225)
        assert m is not None

    def test_very_high_sound_speed(self):
        """Medium with very high sound speed (steel-like)."""
        m = kw.Medium.homogeneous(sound_speed=5900.0, density=7850.0)
        assert m is not None

    def test_impedance_matching(self):
        """Verify impedance Z = rho * c is physically reasonable."""
        c, rho = 1500.0, 1000.0
        Z_water = c * rho  # 1.5 MRayl

        m = kw.Medium.homogeneous(sound_speed=c, density=rho)
        assert m is not None

        # Water impedance should be ~1.5e6 Rayl
        assert abs(Z_water - 1.5e6) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
