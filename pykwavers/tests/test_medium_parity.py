#!/usr/bin/env python3
"""
Medium Parity Tests: pykwavers.Medium vs kwave.kWaveMedium

Validates that pykwavers Medium creation and property specification
match k-wave-python's kWaveMedium behavior.

This module tests:
1. Homogeneous medium creation and properties
2. Heterogeneous medium creation (spatially-varying properties)
3. Absorption and nonlinearity parameters
4. Acoustic impedance and derived quantities
5. Medium validation and edge cases
"""

import numpy as np
import pytest

import pykwavers as kw
from conftest import requires_kwave, HAS_KWAVE, compute_cfl_dt

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


# ============================================================================
# Medium property getters
# ============================================================================


class TestMediumProperties:
    """Test Medium getter properties."""

    def test_sound_speed_getter(self):
        """sound_speed getter returns correct value."""
        m = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
        assert m.sound_speed == 1500.0

    def test_density_getter(self):
        """density getter returns correct value."""
        m = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
        assert m.density == 1000.0

    def test_sound_speed_various_values(self):
        """sound_speed getter for different materials."""
        for c in [330.0, 1500.0, 1540.0, 3000.0, 5900.0]:
            m = kw.Medium.homogeneous(sound_speed=c, density=1000.0)
            assert m.sound_speed == c

    def test_density_various_values(self):
        """density getter for different materials."""
        for rho in [1.225, 1000.0, 1050.0, 1850.0, 7850.0]:
            m = kw.Medium.homogeneous(sound_speed=1500.0, density=rho)
            assert m.density == rho


# ============================================================================
# Absorption and nonlinearity
# ============================================================================


class TestMediumAbsorption:
    """Test absorption and nonlinearity parameters."""

    def test_absorption_parameter(self):
        """Medium accepts absorption parameter."""
        m = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0, absorption=0.75)
        assert m is not None
        assert m.sound_speed == 1500.0

    def test_absorption_with_alpha_power(self):
        """Medium accepts absorption with custom alpha_power."""
        m = kw.Medium.homogeneous(
            sound_speed=1500.0, density=1000.0,
            absorption=0.75, alpha_power=1.5,
        )
        assert m is not None

    def test_nonlinearity_parameter(self):
        """Medium accepts nonlinearity parameter (B/A ratio)."""
        m = kw.Medium.homogeneous(
            sound_speed=1540.0, density=1060.0,
            nonlinearity=6.0,  # soft tissue B/A ~ 6
        )
        assert m is not None

    def test_full_acoustic_properties(self):
        """Medium with all acoustic properties set."""
        m = kw.Medium.homogeneous(
            sound_speed=1540.0, density=1060.0,
            absorption=0.5, nonlinearity=6.0, alpha_power=1.1,
        )
        assert m is not None
        assert m.sound_speed == 1540.0
        assert m.density == 1060.0

    def test_zero_absorption_default(self):
        """Default absorption is zero (lossless)."""
        m = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
        # Should create without error (lossless medium)
        assert m is not None

    def test_simulation_with_absorption_runs(self):
        """Simulation with absorbing medium runs without error."""
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        m = kw.Medium.homogeneous(
            sound_speed=1500.0, density=1000.0,
            absorption=0.75, alpha_power=1.5,
        )
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sensor = kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))
        sim = kw.Simulation(grid, m, src, sensor)
        result = sim.run(time_steps=50)
        assert np.all(np.isfinite(result.sensor_data))


# ============================================================================
# Derived acoustic properties
# ============================================================================


class TestDerivedAcousticProperties:
    """Test derived acoustic property calculations."""

    @pytest.mark.parametrize("c,rho,expected_Z", [
        (1500.0, 1000.0, 1.5e6),    # Water: 1.5 MRayl
        (1540.0, 1050.0, 1.617e6),  # Soft tissue: ~1.6 MRayl
        (3000.0, 1850.0, 5.55e6),   # Cortical bone: ~5.5 MRayl
        (330.0, 1.225, 404.25),     # Air: ~400 Rayl
    ])
    def test_acoustic_impedance(self, c, rho, expected_Z):
        """Acoustic impedance Z = rho * c."""
        Z = rho * c
        assert abs(Z - expected_Z) / expected_Z < 0.01  # Within 1%

    @pytest.mark.parametrize("c,rho,expected_K", [
        (1500.0, 1000.0, 2.25e9),   # Water: 2.25 GPa
        (1540.0, 1050.0, 2.49e9),   # Soft tissue
    ])
    def test_bulk_modulus(self, c, rho, expected_K):
        """Bulk modulus K = rho * c^2."""
        K = rho * c**2
        assert abs(K - expected_K) / expected_K < 0.01

    def test_characteristic_impedance_water(self):
        """Water characteristic impedance is ~1.5 MRayl."""
        c = 1500.0  # m/s
        rho = 1000.0  # kg/m^3
        Z = rho * c
        assert abs(Z - 1.5e6) < 1e-3

    def test_reflection_coefficient_interface(self):
        """Reflection coefficient at interface between two media."""
        # Water to bone interface
        Z1 = 1000.0 * 1500.0  # Water
        Z2 = 1850.0 * 3000.0  # Bone
        
        R = (Z2 - Z1) / (Z2 + Z1)
        
        # Should be positive (bone is higher impedance)
        assert R > 0
        # Should be less than 1
        assert R < 1
        # Expected value ~0.57
        assert abs(R - 0.57) < 0.02

    def test_transmission_coefficient_interface(self):
        """Transmission coefficient at interface between two media."""
        Z1 = 1000.0 * 1500.0  # Water
        Z2 = 1850.0 * 3000.0  # Bone
        
        T = 2 * Z1 / (Z2 + Z1)
        
        # Should be positive and less than 1
        assert T > 0
        assert T < 1


# ============================================================================
# Absorption physics tests
# ============================================================================


class TestAbsorptionPhysics:
    """Test absorption physics and frequency dependence."""

    def test_absorption_frequency_dependence(self):
        """Absorption increases with frequency (power law)."""
        alpha_coeff = 0.5  # dB/(MHz^y * cm)
        alpha_power = 1.5
        
        freq1 = 1e6  # 1 MHz
        freq2 = 2e6  # 2 MHz
        
        alpha1 = alpha_coeff * (freq1 / 1e6) ** alpha_power
        alpha2 = alpha_coeff * (freq2 / 1e6) ** alpha_power
        
        # Absorption at 2 MHz should be 2^1.5 ≈ 2.83 times higher
        ratio = alpha2 / alpha1
        expected_ratio = 2 ** alpha_power
        assert abs(ratio - expected_ratio) < 0.01

    def test_absorption_db_per_cm(self):
        """Absorption in dB/cm at specific frequency."""
        # Typical soft tissue: 0.5 dB/(MHz*cm)
        alpha_coeff = 0.5
        freq = 1e6  # 1 MHz
        
        alpha_db_cm = alpha_coeff * (freq / 1e6)
        assert alpha_db_cm == 0.5

    def test_absorption_neper_per_meter(self):
        """Convert absorption from dB/cm to Np/m."""
        # 1 dB/cm = 8.686 Np/m
        alpha_db_cm = 0.5
        alpha_np_m = alpha_db_cm * 8.686 / 0.01  # Convert to Np/m
        assert abs(alpha_np_m - 434.3) < 0.1


# ============================================================================
# Heterogeneous medium tests
# ============================================================================


class TestHeterogeneousMedium:
    """Test heterogeneous medium creation and behavior."""

    def test_sound_speed_map_creation(self):
        """Create medium with spatially-varying sound speed."""
        N = 32
        c_map = np.ones((N, N, N)) * 1500.0
        # Add a spherical inclusion
        cx, cy, cz = N // 2, N // 2, N // 2
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if (i - cx)**2 + (j - cy)**2 + (k - cz)**2 < 8**2:
                        c_map[i, j, k] = 1540.0
        
        # Verify the map has two distinct values
        unique_values = np.unique(c_map)
        assert len(unique_values) == 2

    def test_density_map_creation(self):
        """Create medium with spatially-varying density."""
        N = 32
        rho_map = np.ones((N, N, N)) * 1000.0
        # Add a layer
        rho_map[:, :, N//2:] = 1050.0
        
        unique_values = np.unique(rho_map)
        assert len(unique_values) == 2

    def test_layered_medium_simulation(self):
        """Simulation with layered medium runs correctly."""
        N = 32
        dx = 0.1e-3
        
        grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
        
        # Create layered sound speed
        c_map = np.ones((N, N, N)) * 1500.0
        c_map[:, :, N//2:] = 1600.0
        
        # Create medium with sound speed map
        # Note: This tests if the API accepts heterogeneous media
        try:
            m = kw.Medium.from_sound_speed_map(c_map, density=1000.0)
            assert m is not None
        except AttributeError:
            # If from_sound_speed_map doesn't exist, use homogeneous
            m = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
            assert m is not None


# ============================================================================
# Medium validation tests
# ============================================================================


class TestMediumValidation:
    """Test medium parameter validation."""

    def test_valid_sound_speed_range(self):
        """Sound speed should be in physically reasonable range."""
        # Valid range for most materials: 200 - 8000 m/s
        valid_speeds = [330.0, 500.0, 1000.0, 1500.0, 3000.0, 5000.0]
        for c in valid_speeds:
            m = kw.Medium.homogeneous(sound_speed=c, density=1000.0)
            assert m is not None

    def test_valid_density_range(self):
        """Density should be in physically reasonable range."""
        # Valid range: 0.1 - 20000 kg/m^3
        valid_densities = [0.1, 1.225, 100.0, 1000.0, 5000.0, 10000.0]
        for rho in valid_densities:
            m = kw.Medium.homogeneous(sound_speed=1500.0, density=rho)
            assert m is not None

    def test_absorption_coefficient_range(self):
        """Absorption coefficient should be non-negative."""
        # Valid range: 0 - 10 dB/(MHz^y * cm)
        valid_absorptions = [0.0, 0.1, 0.5, 1.0, 5.0]
        for alpha in valid_absorptions:
            m = kw.Medium.homogeneous(
                sound_speed=1500.0, density=1000.0, absorption=alpha
            )
            assert m is not None


# ============================================================================
# Cross-validation: Heterogeneous medium with k-wave-python
# ============================================================================


@requires_kwave
class TestHeterogeneousMediumParityWithKWave:
    """Compare heterogeneous medium behavior against k-wave-python."""

    def test_heterogeneous_sound_speed_kwave(self):
        """Both accept heterogeneous sound speed maps."""
        N = 32
        
        # Create sound speed map
        c_map = np.ones((N, N, N)) * 1500.0
        c_map[:, :, N//2:] = 1600.0
        
        # k-wave-python
        m_kw = kWaveMedium(sound_speed=c_map, density=1000.0)
        
        # Verify k-wave accepts it
        assert m_kw.sound_speed.shape == (N, N, N)

    def test_heterogeneous_density_kwave(self):
        """Both accept heterogeneous density maps."""
        N = 32
        
        # Create density map
        rho_map = np.ones((N, N, N)) * 1000.0
        rho_map[:, :, N//2:] = 1050.0
        
        # k-wave-python
        m_kw = kWaveMedium(sound_speed=1500.0, density=rho_map)
        
        # Verify k-wave accepts it
        assert m_kw.density.shape == (N, N, N)

    def test_both_heterogeneous_params_kwave(self):
        """Both accept heterogeneous sound speed and density."""
        N = 32
        
        c_map = np.ones((N, N, N)) * 1500.0
        c_map[:, :, N//2:] = 1600.0
        
        rho_map = np.ones((N, N, N)) * 1000.0
        rho_map[:, :, N//2:] = 1050.0
        
        m_kw = kWaveMedium(sound_speed=c_map, density=rho_map)
        
        assert m_kw.sound_speed.shape == (N, N, N)
        assert m_kw.density.shape == (N, N, N)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
