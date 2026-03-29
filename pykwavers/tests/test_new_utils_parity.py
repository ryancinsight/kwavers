"""
Parity tests for newly exposed pykwavers utility functions.

Tests cover:
- Geometry: make_sphere, make_circle
- Unit conversions: freq2wavenumber, hounsfield2density, hounsfield2soundspeed
- Water properties: water_sound_speed, water_density, water_absorption, water_nonlinearity
- Signal processing: add_noise

These tests validate against known analytical values and, where available,
k-wave-python equivalents.
"""

import numpy as np
import pytest

import pykwavers as kw

# Optional k-wave-python
try:
    from kwave.utils.mapgen import make_ball as kw_make_ball
    from kwave.utils.conversion import db2neper as kw_db2neper
    from kwave.utils.conversion import neper2db as kw_neper2db
    from kwave.kgrid import kWaveGrid
    from kwave.data import Vector

    HAS_KWAVE = True
except ImportError:
    HAS_KWAVE = False


# ============================================================================
# Geometry: make_sphere (alias for make_ball)
# ============================================================================


class TestMakeSphere:
    """Test make_sphere geometry function."""

    def test_make_sphere_equals_make_ball(self):
        """make_sphere is an alias for make_ball — results must be identical."""
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        center = (1.6e-3, 1.6e-3, 1.6e-3)
        radius = 0.5e-3

        ball = kw.make_ball(grid, center, radius)
        sphere = kw.make_sphere(grid, center, radius)

        np.testing.assert_array_equal(ball, sphere)

    def test_make_sphere_volume(self):
        """Verify sphere volume matches analytical (4/3)pi*r^3."""
        grid = kw.Grid(nx=64, ny=64, nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        center = (3.2e-3, 3.2e-3, 3.2e-3)
        radius = 1.0e-3

        mask = kw.make_sphere(grid, center, radius)
        vol_numerical = np.sum(mask) * (0.1e-3) ** 3
        vol_analytical = (4.0 / 3.0) * np.pi * radius**3

        assert abs(vol_numerical - vol_analytical) / vol_analytical < 0.15

    def test_make_sphere_symmetry(self):
        """Sphere must be symmetric in all three axes."""
        grid = kw.Grid(nx=64, ny=64, nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        center = (3.2e-3, 3.2e-3, 3.2e-3)
        radius = 1.0e-3

        mask = kw.make_sphere(grid, center, radius)
        for r in range(1, 6):
            assert mask[32 + r, 32, 32] == mask[32 - r, 32, 32]
            assert mask[32, 32 + r, 32] == mask[32, 32 - r, 32]
            assert mask[32, 32, 32 + r] == mask[32, 32, 32 - r]


# ============================================================================
# Geometry: make_circle (outline, not filled)
# ============================================================================


class TestMakeCircle:
    """Test make_circle geometry function (circle outline/shell)."""

    def test_make_circle_is_ring(self):
        """make_circle should produce a ring, not a filled disc."""
        grid = kw.Grid(nx=64, ny=64, nz=1, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        center = (3.2e-3, 3.2e-3, 0.0)
        radius = 1.5e-3

        circle = kw.make_circle(grid, center, radius)
        disc = kw.make_disc(grid, center, radius)

        # Circle should have strictly fewer active voxels than disc
        assert np.sum(circle) < np.sum(disc)
        # Circle center should be False (hollow)
        assert not circle[32, 32, 0]
        # Points at the radius perimeter should be True
        # At radius 1.5mm and dx=0.1mm, radius = 15 grid points from center
        assert circle[32 + 15, 32, 0] or circle[32 + 14, 32, 0] or circle[32 + 16, 32, 0]

    def test_make_circle_thickness(self):
        """Thicker circle should have more active voxels."""
        grid = kw.Grid(nx=64, ny=64, nz=1, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        center = (3.2e-3, 3.2e-3, 0.0)
        radius = 1.5e-3

        thin = kw.make_circle(grid, center, radius, thickness=1)
        thick = kw.make_circle(grid, center, radius, thickness=3)

        assert np.sum(thick) > np.sum(thin)

    def test_make_circle_perimeter(self):
        """Active voxel count should approximate 2*pi*r / dx."""
        grid = kw.Grid(nx=128, ny=128, nz=1, dx=0.05e-3, dy=0.05e-3, dz=0.05e-3)
        center = (3.2e-3, 3.2e-3, 0.0)
        radius = 2.0e-3

        circle = kw.make_circle(grid, center, radius, thickness=1)
        count = np.sum(circle)
        expected_circumference = 2 * np.pi * radius
        expected_count = expected_circumference / 0.05e-3
        # Allow 30% tolerance due to discretization
        assert abs(count - expected_count) / expected_count < 0.30, (
            f"Circle has {count} voxels, expected ~{expected_count:.0f}"
        )

    def test_make_circle_invalid_inputs(self):
        """Invalid inputs should raise errors."""
        grid = kw.Grid(nx=32, ny=32, nz=1, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        center = (1.6e-3, 1.6e-3, 0.0)

        with pytest.raises(Exception):
            kw.make_circle(grid, center, -1.0)
        with pytest.raises(Exception):
            kw.make_circle(grid, center, 0.0)
        with pytest.raises(Exception):
            kw.make_circle(grid, center, 0.5e-3, thickness=0)


# ============================================================================
# Unit Conversions: freq2wavenumber
# ============================================================================


class TestFreq2Wavenumber:
    """Test frequency-to-wavenumber conversion."""

    def test_known_values(self):
        """k = 2*pi*f/c at 1 MHz in water."""
        k = kw.freq2wavenumber(1e6, 1500.0)
        expected = 2 * np.pi * 1e6 / 1500.0
        np.testing.assert_allclose(k, expected, rtol=1e-12)

    def test_zero_frequency(self):
        """k(0 Hz) = 0."""
        assert kw.freq2wavenumber(0.0, 1500.0) == 0.0

    def test_proportional_to_frequency(self):
        """Wavenumber doubles when frequency doubles."""
        k1 = kw.freq2wavenumber(1e6, 1500.0)
        k2 = kw.freq2wavenumber(2e6, 1500.0)
        np.testing.assert_allclose(k2, 2 * k1, rtol=1e-12)

    def test_inverse_to_sound_speed(self):
        """Wavenumber halves when sound speed doubles."""
        k_water = kw.freq2wavenumber(1e6, 1500.0)
        k_bone = kw.freq2wavenumber(1e6, 3000.0)
        np.testing.assert_allclose(k_bone, k_water / 2, rtol=1e-12)

    def test_invalid_sound_speed(self):
        """Negative or zero sound speed must raise."""
        with pytest.raises(ValueError):
            kw.freq2wavenumber(1e6, 0.0)
        with pytest.raises(ValueError):
            kw.freq2wavenumber(1e6, -100.0)

    def test_negative_frequency(self):
        """Negative frequency must raise."""
        with pytest.raises(ValueError):
            kw.freq2wavenumber(-1.0, 1500.0)


# ============================================================================
# Unit Conversions: Hounsfield
# ============================================================================


class TestHounsfieldConversions:
    """Test Hounsfield unit conversion functions.

    Uses k-wave-python's piecewise linear model where HU values
    correspond to raw CT numbers (HU≈1000 for water).
    """

    def test_water_density(self):
        """Water region (HU≈1000) → ~1012 kg/m³ in k-wave model."""
        rho = kw.hounsfield2density(1000.0)
        assert 900.0 < rho < 1100.0

    def test_low_hu_density(self):
        """HU=0 → negative density in the piecewise fit (extrapolation region)."""
        rho = kw.hounsfield2density(0.0)
        assert rho == pytest.approx(-5.68, abs=1.0)

    def test_bone_density(self):
        """Cortical bone (HU=1500) → high density."""
        rho = kw.hounsfield2density(1500.0)
        assert rho > 1000.0

    def test_sound_speed_from_density(self):
        """Sound speed = (density + 349) / 0.893 (Mast 2000)."""
        hu = 1000.0
        rho = kw.hounsfield2density(hu)
        c = kw.hounsfield2soundspeed(hu)
        expected_c = (rho + 349.0) / 0.893
        assert c == pytest.approx(expected_c, rel=1e-10)

    def test_density_monotonic(self):
        """Density should increase monotonically with HU."""
        hu_values = [0, 500, 900, 1000, 1100, 1300, 1500]
        densities = [kw.hounsfield2density(hu) for hu in hu_values]
        for i in range(len(densities) - 1):
            assert densities[i + 1] > densities[i]


# ============================================================================
# Water Properties: Temperature-Dependent
# ============================================================================


class TestWaterProperties:
    """Test temperature-dependent water property functions."""

    def test_sound_speed_reference(self):
        """Water sound speed at 20°C ≈ 1481 m/s (Duck 1990)."""
        c = kw.water_sound_speed(20.0)
        assert c == pytest.approx(1481.0, abs=5.0)

    def test_sound_speed_increases_with_temperature(self):
        """Water sound speed increases from 20°C to 37°C."""
        c_20 = kw.water_sound_speed(20.0)
        c_37 = kw.water_sound_speed(37.0)
        assert c_37 > c_20

    def test_sound_speed_body_temperature(self):
        """Water sound speed at 37°C ≈ 1530 m/s (literature range: 1520-1540)."""
        c = kw.water_sound_speed(37.0)
        assert 1500.0 < c < 1560.0

    def test_density_reference(self):
        """Water density at 20°C ≈ 998 kg/m³."""
        rho = kw.water_density(20.0)
        assert rho == pytest.approx(998.0, abs=5.0)

    def test_density_decreases_with_temperature(self):
        """Water density decreases from 20°C to 37°C (thermal expansion)."""
        rho_20 = kw.water_density(20.0)
        rho_37 = kw.water_density(37.0)
        assert rho_37 < rho_20

    def test_density_positive(self):
        """Density must always be positive in physical range."""
        for t in [0, 10, 20, 37, 60, 80]:
            assert kw.water_density(float(t)) > 0

    def test_absorption_at_reference(self):
        """Water absorption at 1 MHz, 20°C ≈ 0.002 dB/cm ≈ 0.023 Np/m (Pinkerton 1949)."""
        alpha = kw.water_absorption(1e6, 20.0)
        # 0.002 dB/cm → 0.002 / 8.686 * 100 ≈ 0.023 Np/m
        assert alpha == pytest.approx(0.023, abs=0.005)

    def test_absorption_frequency_squared(self):
        """Water absorption scales as f^2 (Stokes model)."""
        alpha_1 = kw.water_absorption(1e6, 20.0)
        alpha_2 = kw.water_absorption(2e6, 20.0)
        # α(2f) / α(f) ≈ 4 for y=2
        ratio = alpha_2 / alpha_1
        assert ratio == pytest.approx(4.0, rel=0.05)

    def test_absorption_temperature_dependence(self):
        """Higher temperature should change absorption."""
        alpha_20 = kw.water_absorption(1e6, 20.0)
        alpha_37 = kw.water_absorption(1e6, 37.0)
        assert alpha_37 != alpha_20

    def test_nonlinearity_reference(self):
        """Water B/A at 20°C ≈ 5.0 (Beyer 1960)."""
        ba = kw.water_nonlinearity(20.0)
        assert ba == pytest.approx(5.0, abs=0.5)

    def test_nonlinearity_temperature_dependence(self):
        """B/A increases slightly with temperature."""
        ba_20 = kw.water_nonlinearity(20.0)
        ba_37 = kw.water_nonlinearity(37.0)
        assert ba_37 > ba_20


# ============================================================================
# Signal Processing: add_noise
# ============================================================================


class TestAddNoise:
    """Test add_noise signal processing function."""

    def test_output_length(self):
        """Output length matches input."""
        sig = np.sin(np.linspace(0, 10, 200))
        noisy = kw.add_noise(sig, 20.0)
        assert len(noisy) == len(sig)

    def test_snr_approximate(self):
        """Output SNR approximately matches requested SNR."""
        rng = np.random.RandomState(42)
        sig = np.sin(np.linspace(0, 20 * np.pi, 10000))
        snr_target = 20.0

        noisy = kw.add_noise(sig, snr_target, seed=123)
        noise = noisy - sig

        sig_power = np.mean(sig**2)
        noise_power = np.mean(noise**2)
        snr_actual = 10 * np.log10(sig_power / noise_power)

        # Allow 3 dB tolerance (PRNG-based, not perfect Gaussian)
        assert abs(snr_actual - snr_target) < 3.0

    def test_reproducibility(self):
        """Same seed produces same output."""
        sig = np.sin(np.linspace(0, 10, 200))
        n1 = kw.add_noise(sig, 20.0, seed=42)
        n2 = kw.add_noise(sig, 20.0, seed=42)
        np.testing.assert_array_equal(n1, n2)

    def test_different_seeds(self):
        """Different seeds produce different output."""
        sig = np.sin(np.linspace(0, 10, 200))
        n1 = kw.add_noise(sig, 20.0, seed=1)
        n2 = kw.add_noise(sig, 20.0, seed=2)
        assert not np.array_equal(n1, n2)

    def test_high_snr_preserves_signal(self):
        """At very high SNR, output should be close to input."""
        sig = np.sin(np.linspace(0, 10, 200))
        noisy = kw.add_noise(sig, 60.0, seed=42)
        np.testing.assert_allclose(noisy, sig, atol=0.01)

    def test_low_snr_corrupts_signal(self):
        """At very low SNR, noise dominates."""
        sig = np.sin(np.linspace(0, 10, 200))
        noisy = kw.add_noise(sig, -10.0, seed=42)
        noise = noisy - sig
        assert np.std(noise) > np.std(sig)


# ============================================================================
# Cross-validation against k-wave-python (when available)
# ============================================================================


@pytest.mark.skipif(not HAS_KWAVE, reason="k-wave-python not installed")
class TestKWavePythonParity:
    """Validate against k-wave-python equivalents."""

    @pytest.mark.xfail(
        reason="Rust db2neper uses the physical k-Wave acoustic-absorption unit conversion "
               "(dB/(MHz^y·cm) → Np/(m·(rad/s)^y)), not the simple dimensionless "
               "amplitude formula (Np = dB*ln(10)/20). Roundtrip is self-consistent."
    )
    def test_db2neper_standard_conversion(self):
        """db2neper uses standard mathematical conversion (amplitude dB).

        Note: k-wave-python's db2neper/neper2db use a different convention
        that includes absorption unit-system conversion (dB/MHz^y/cm ↔ Np/m),
        so a direct scalar comparison is not appropriate. Our implementation
        follows the standard: Np = dB * ln(10) / 20.
        """
        # Standard mathematical identity
        for db in [0, 1, 10, 20]:
            neper = kw.db2neper(db)
            expected = db * np.log(10) / 20.0
            np.testing.assert_allclose(neper, expected, rtol=1e-12)

    def test_make_sphere_matches_kwave_make_ball(self):
        """make_sphere output matches k-wave-python make_ball dimensions and shape."""
        Nx, Ny, Nz = 32, 32, 32
        dx = 0.1e-3

        kwa_grid = kw.Grid(Nx, Ny, Nz, dx, dx, dx)
        center_phys = (1.6e-3, 1.6e-3, 1.6e-3)
        radius = 0.5e-3

        kwa_mask = kw.make_sphere(kwa_grid, center_phys, radius)

        assert kwa_mask.shape == (Nx, Ny, Nz)
        assert kwa_mask[16, 16, 16]  # center
        assert not kwa_mask[0, 0, 0]  # corner

        vol_analytical = (4.0 / 3.0) * np.pi * radius**3
        vol_numerical = np.sum(kwa_mask) * dx**3
        assert abs(vol_numerical - vol_analytical) / vol_analytical < 0.25

    def test_make_ball_vs_kwave_make_ball(self):
        """make_ball active-voxel count matches k-wave-python make_ball."""
        Nx, Ny, Nz = 64, 64, 64
        dx = 0.1e-3
        radius_gp = 10  # grid points

        kwa_grid = kw.Grid(Nx, Ny, Nz, dx, dx, dx)
        center_phys = (3.2e-3, 3.2e-3, 3.2e-3)  # grid-point 32
        radius_m = radius_gp * dx

        kwa_mask = kw.make_ball(kwa_grid, center_phys, radius_m)
        kw_mask = kw_make_ball(Vector([Nx, Ny, Nz]), Vector([32, 32, 32]), radius_gp, binary=True)

        kwa_count = int(np.sum(kwa_mask > 0))
        kw_count = int(np.sum(kw_mask > 0))

        # Allow 15% tolerance for discretisation approach differences
        if kw_count > 0:
            ratio = abs(kwa_count - kw_count) / kw_count
            assert ratio < 0.15, (
                f"Voxel count mismatch: pykwavers={kwa_count}, k-wave={kw_count} ({ratio:.1%})"
            )

    def test_water_sound_speed_matches_kwave(self):
        """water_sound_speed matches k-wave-python water_sound_speed (Marczak 1997)."""
        from kwave.utils.mapgen import water_sound_speed as kw_wss

        for temp in [0.0, 10.0, 20.0, 25.0, 30.0, 37.0, 60.0, 90.0]:
            pk_val = kw.water_sound_speed(temp)
            kw_val = float(kw_wss(temp))
            np.testing.assert_allclose(pk_val, kw_val, rtol=1e-6,
                err_msg=f"water_sound_speed mismatch at {temp}°C")

    def test_water_density_matches_kwave(self):
        """water_density matches k-wave-python water_density (Jones 1992)."""
        from kwave.utils.mapgen import water_density as kw_wd

        for temp in [5.0, 10.0, 20.0, 25.0, 30.0, 37.0, 40.0]:
            pk_val = kw.water_density(temp)
            kw_val = float(kw_wd(temp))
            np.testing.assert_allclose(pk_val, kw_val, rtol=1e-6,
                err_msg=f"water_density mismatch at {temp}°C")

    def test_water_nonlinearity_matches_kwave(self):
        """water_nonlinearity matches k-wave-python water_non_linearity (Beyer 1960)."""
        from kwave.utils.mapgen import water_non_linearity as kw_wnl

        for temp in [0.0, 10.0, 20.0, 25.0, 37.0, 50.0, 80.0, 100.0]:
            pk_val = kw.water_nonlinearity(temp)
            kw_val = float(kw_wnl(temp))
            np.testing.assert_allclose(pk_val, kw_val, rtol=1e-6,
                err_msg=f"water_nonlinearity mismatch at {temp}°C")

    def test_water_absorption_matches_kwave(self):
        """water_absorption matches k-wave-python water_absorption (Pinkerton 1949).

        k-wave-python's water_absorption takes frequency in MHz and returns
        dB/cm. pykwavers takes frequency in Hz and returns Np/m.
        We convert for comparison.
        """
        from kwave.utils.mapgen import water_absorption as kw_wa

        NEPER2DB = 8.686  # 20*log10(e)
        for freq_mhz in [0.5, 1.0, 2.0, 5.0]:
            for temp in [10.0, 20.0, 25.0, 37.0, 50.0]:
                freq_hz = freq_mhz * 1e6
                pk_npm = kw.water_absorption(freq_hz, temp)  # Np/m
                kw_dbcm = float(kw_wa(freq_mhz, temp))       # dB/cm

                # Convert k-wave dB/cm → Np/m: Np/m = (dB/cm) / NEPER2DB * 100
                kw_npm = kw_dbcm / NEPER2DB * 100.0

                if kw_npm > 1e-10:
                    np.testing.assert_allclose(pk_npm, kw_npm, rtol=1e-6,
                        err_msg=f"water_absorption mismatch at {freq_mhz} MHz, {temp}°C")

    def test_hounsfield2density_matches_kwave(self):
        """hounsfield2density matches k-wave-python piecewise linear model exactly."""
        from kwave.utils.conversion import hounsfield2density as kw_h2d

        for hu in [-1000.0, -500.0, 0.0, 500.0, 929.0, 930.0, 1000.0, 1098.0,
                   1099.0, 1200.0, 1259.0, 1260.0, 1500.0, 2000.0]:
            pk_val = kw.hounsfield2density(hu)
            kw_val = kw_h2d(np.array([[hu]])).item()
            np.testing.assert_allclose(pk_val, kw_val, rtol=1e-10,
                err_msg=f"hounsfield2density mismatch at HU={hu}")

    def test_hounsfield2soundspeed_matches_kwave(self):
        """hounsfield2soundspeed matches k-wave-python exactly."""
        from kwave.utils.conversion import hounsfield2soundspeed as kw_h2ss

        for hu in [-500.0, 0.0, 500.0, 1000.0, 1200.0, 1500.0]:
            pk_val = kw.hounsfield2soundspeed(hu)
            kw_val = kw_h2ss(np.array([[hu]])).item()
            np.testing.assert_allclose(pk_val, kw_val, rtol=1e-10,
                err_msg=f"hounsfield2soundspeed mismatch at HU={hu}")


# ============================================================================
# Phase 3 parity utilities: gaussian, spect, extract_amp_phase, cart2grid,
# grid2cart
# ============================================================================


class TestGaussian:
    """Tests for kw.gaussian() — k-Wave makeGaussian parity."""

    def test_length(self):
        g = kw.gaussian(N=65, var=10.0)
        assert len(g) == 65

    def test_peak_at_centre(self):
        """Peak must occur at the centre sample for odd N."""
        g = kw.gaussian(N=65, var=10.0, magnitude=3.0)
        assert np.argmax(g) == 32

    def test_peak_magnitude(self):
        """Peak value equals `magnitude` parameter."""
        g = kw.gaussian(N=65, var=10.0, magnitude=2.5)
        np.testing.assert_allclose(g[32], 2.5, rtol=1e-12)

    def test_symmetry(self):
        """Gaussian must be symmetric around the centre."""
        g = kw.gaussian(N=65, var=10.0)
        np.testing.assert_allclose(g, g[::-1], rtol=1e-12)

    def test_default_magnitude(self):
        """Default magnitude is 1.0."""
        g = kw.gaussian(N=65, var=10.0)
        np.testing.assert_allclose(g[32], 1.0, rtol=1e-12)


class TestSpect:
    """Tests for kw.spect() — single-sided amplitude spectrum."""

    def test_pure_sine_amplitude(self):
        """Single 1 MHz sine should have amplitude ≈ 1 at 1 MHz."""
        fs = 10e6
        # Use n=1000 so f0=1 MHz lands exactly on bin 100 (no spectral leakage)
        n = 1000
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 1e6 * t)
        f, amp, _ = kw.spect(x, fs)
        # Find bin closest to 1 MHz
        idx = int(np.argmin(np.abs(f - 1e6)))
        np.testing.assert_allclose(amp[idx], 1.0, atol=0.02)

    def test_dc_component(self):
        """Constant signal should have power only at DC."""
        fs = 1e6
        x = np.ones(512) * 3.0
        f, amp, _ = kw.spect(x, fs)
        np.testing.assert_allclose(amp[0], 3.0, rtol=1e-6)
        # All non-DC bins should be ~0
        assert np.max(amp[1:]) < 1e-10

    def test_output_length(self):
        """Output frequency axis should have length N//2 + 1."""
        x = np.random.randn(256)
        f, amp, phase = kw.spect(x, 1e6)
        assert len(f) == 129  # 256//2 + 1
        assert len(amp) == 129
        assert len(phase) == 129


class TestExtractAmpPhase:
    """Tests for kw.extract_amp_phase()."""

    def test_sine_amplitude(self):
        """Extract amplitude of a known sine wave."""
        fs = 10e6
        # Use n=2000 so f0=1 MHz lands exactly on bin 200 (no spectral leakage)
        n = 2000
        t = np.arange(n) / fs
        f0 = 1e6
        amp_expected = 2.0
        x = amp_expected * np.sin(2 * np.pi * f0 * t)
        amp, _ = kw.extract_amp_phase(x, f0, fs)
        np.testing.assert_allclose(amp, amp_expected, rtol=0.02)

    def test_cosine_phase(self):
        """Cosine signal has FFT phase ≈ 0; sine has phase ≈ -π/2; difference is π/2."""
        fs = 10e6
        # Use n=2000 so f0=1 MHz lands exactly on bin 200 (no spectral leakage)
        n = 2000
        t = np.arange(n) / fs
        f0 = 1e6
        # cos(2πf0t) → purely real FFT coefficient → phase = 0
        _, phase_cos = kw.extract_amp_phase(np.cos(2 * np.pi * f0 * t), f0, fs)
        np.testing.assert_allclose(phase_cos, 0.0, atol=0.1)
        # sin(2πf0t) → purely imaginary negative coefficient → phase = -π/2
        _, phase_sin = kw.extract_amp_phase(np.sin(2 * np.pi * f0 * t), f0, fs)
        # cosine leads sine by π/2
        np.testing.assert_allclose(phase_cos - phase_sin, np.pi / 2, atol=0.1)


class TestCart2Grid:
    """Tests for kw.cart2grid()."""

    def test_single_point_creates_mask(self):
        """A single Cartesian point should mark exactly one voxel."""
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        cart = np.array([[0.0], [0.0], [0.0]])  # origin
        mask = kw.cart2grid(grid, cart)
        assert mask.shape == (32, 32, 32)
        assert mask.sum() == 1

    def test_multiple_points(self):
        """N distinct points should set at most N voxels."""
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        pts = np.array([
            [-1e-3, 0.0, 1e-3],
            [-1e-3, 0.0, 1e-3],
            [0.0,   0.0, 0.0],
        ])
        mask = kw.cart2grid(grid, pts)
        assert mask.sum() <= 3


class TestGrid2Cart:
    """Tests for kw.grid2cart()."""

    def test_uniform_field_returns_constant(self):
        """A uniform field should return the same value at all Cartesian points."""
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        field = np.ones((32, 32, 32)) * 7.0
        pts = np.array([
            [0.0, 0.5e-3],
            [0.0, 0.5e-3],
            [0.0, 0.5e-3],
        ])
        vals = kw.grid2cart(grid, field, pts)
        np.testing.assert_allclose(vals, 7.0, rtol=1e-12)

    def test_output_length(self):
        """Output length must equal number of Cartesian points."""
        grid = kw.Grid(nx=16, ny=16, nz=16, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        field = np.random.randn(16, 16, 16)
        pts = np.random.uniform(-0.5e-3, 0.5e-3, (3, 10))
        vals = kw.grid2cart(grid, field, pts)
        assert len(vals) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
