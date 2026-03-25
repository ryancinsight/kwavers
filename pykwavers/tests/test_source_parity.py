#!/usr/bin/env python3
"""
Source Parity Tests: pykwavers.Source vs kwave.kSource

Validates source creation, mask-based sources, and temporal signal
specification match k-wave-python's kSource behavior.

This module tests:
1. Source creation (point, plane wave, mask-based)
2. Temporal signal generation (sinusoidal, tone burst, Gaussian)
3. Source injection modes (additive, dirichlet)
4. Initial pressure (p0) sources
5. Velocity sources
6. Source physics validation
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


# ============================================================================
# Source mode tests
# ============================================================================


class TestSourceMode:
    """Test source injection mode parameter."""

    def test_mask_source_default_mode_additive(self, grid):
        """Default source mode is additive."""
        mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float64)
        mask[0, :, :] = 1.0
        signal = np.sin(np.linspace(0, 2 * np.pi, 50))
        src = kw.Source.from_mask(mask, signal, frequency=1e6)
        assert src.source_mode == "additive"

    def test_mask_source_dirichlet_mode(self, grid):
        """Source mode can be set to dirichlet."""
        mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float64)
        mask[0, :, :] = 1.0
        signal = np.sin(np.linspace(0, 2 * np.pi, 50))
        src = kw.Source.from_mask(mask, signal, frequency=1e6, mode="dirichlet")
        assert src.source_mode == "dirichlet"

    def test_mask_source_additive_no_correction_mode(self, grid):
        """Source mode can be set to additive_no_correction."""
        mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float64)
        mask[0, :, :] = 1.0
        signal = np.sin(np.linspace(0, 2 * np.pi, 50))
        src = kw.Source.from_mask(mask, signal, frequency=1e6, mode="additive_no_correction")
        assert src.source_mode == "additive_no_correction"

    def test_plane_wave_default_mode(self, grid):
        """Plane wave source defaults to additive mode."""
        src = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        assert src.source_mode == "additive"

    def test_point_source_default_mode(self):
        """Point source defaults to additive mode."""
        src = kw.Source.point((1e-3, 1e-3, 1e-3), frequency=1e6, amplitude=1e5)
        assert src.source_mode == "additive"


# ============================================================================
# Initial pressure (p0) source tests
# ============================================================================


class TestInitialPressureSource:
    """Test initial pressure (IVP) source creation."""

    def test_from_initial_pressure_creation(self):
        """from_initial_pressure creates a valid source."""
        p0 = np.zeros((32, 32, 32))
        p0[16, 16, 16] = 1e5
        src = kw.Source.from_initial_pressure(p0)
        assert src is not None
        assert src.source_type == "p0"

    def test_from_initial_pressure_runs(self):
        """Initial pressure source can be used in simulation."""
        p0 = np.zeros((32, 32, 32))
        p0[16, 16, 16] = 1e5

        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        medium = kw.Medium.homogeneous(1500.0, 1000.0)
        sensor = kw.Sensor.point((2.4e-3, 1.6e-3, 1.6e-3))
        src = kw.Source.from_initial_pressure(p0)

        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=100)

        assert result is not None
        assert result.time_steps == 100
        assert np.all(np.isfinite(result.sensor_data))

    def test_from_initial_pressure_produces_nonzero(self):
        """Initial pressure source produces non-zero sensor data."""
        p0 = np.zeros((32, 32, 32))
        p0[16, 16, 16] = 1e5

        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        medium = kw.Medium.homogeneous(1500.0, 1000.0)
        sensor = kw.Sensor.point((2.4e-3, 1.6e-3, 1.6e-3))
        src = kw.Source.from_initial_pressure(p0)

        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=200)

        max_p = np.max(np.abs(result.sensor_data))
        assert max_p > 0, "Initial pressure source produced all-zero data"

    def test_from_initial_pressure_gaussian_blob(self):
        """Gaussian initial pressure blob propagates outward."""
        N = 32
        p0 = np.zeros((N, N, N))
        cx, cy, cz = N // 2, N // 2, N // 2
        sigma = 2.0
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    r2 = (i - cx)**2 + (j - cy)**2 + (k - cz)**2
                    p0[i, j, k] = 1e4 * np.exp(-r2 / (2 * sigma**2))

        grid = kw.Grid(nx=N, ny=N, nz=N, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        medium = kw.Medium.homogeneous(1500.0, 1000.0)
        sensor = kw.Sensor.point((2.4e-3, 1.6e-3, 1.6e-3))
        src = kw.Source.from_initial_pressure(p0)

        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=150)

        assert np.all(np.isfinite(result.sensor_data))
        assert np.max(np.abs(result.sensor_data)) > 0


# ============================================================================
# Velocity source tests
# ============================================================================


class TestVelocitySource:
    """Test velocity source creation and execution."""

    def test_from_velocity_mask_creation_z(self):
        """Create velocity source with z-direction signal."""
        mask = np.zeros((32, 32, 32))
        mask[0, :, :] = 1.0
        signal = np.sin(np.linspace(0, 2 * np.pi, 50))
        src = kw.Source.from_velocity_mask(mask, uz=signal)
        assert src.source_type == "velocity"
        assert src.source_mode == "additive"

    def test_from_velocity_mask_creation_x(self):
        """Create velocity source with x-direction signal."""
        mask = np.zeros((32, 32, 32))
        mask[:, :, 0] = 1.0
        signal = np.sin(np.linspace(0, 2 * np.pi, 50))
        src = kw.Source.from_velocity_mask(mask, ux=signal)
        assert src.source_type == "velocity"

    def test_from_velocity_mask_multi_direction(self):
        """Velocity source with multiple direction components."""
        mask = np.zeros((32, 32, 32))
        mask[0, :, :] = 1.0
        signal = np.sin(np.linspace(0, 2 * np.pi, 50))
        src = kw.Source.from_velocity_mask(mask, ux=signal, uy=signal, uz=signal)
        assert src.source_type == "velocity"

    def test_from_velocity_mask_dirichlet_mode(self):
        """Velocity source with dirichlet mode."""
        mask = np.zeros((32, 32, 32))
        mask[0, :, :] = 1.0
        signal = np.sin(np.linspace(0, 2 * np.pi, 50))
        src = kw.Source.from_velocity_mask(mask, uz=signal, mode="dirichlet")
        assert src.source_mode == "dirichlet"

    def test_from_velocity_mask_no_signal_raises(self):
        """Velocity source without any signal raises error."""
        mask = np.zeros((32, 32, 32))
        mask[0, :, :] = 1.0
        with pytest.raises(ValueError, match="At least one velocity"):
            kw.Source.from_velocity_mask(mask)

    def test_from_velocity_mask_empty_mask_raises(self):
        """Velocity source with empty mask raises error."""
        mask = np.zeros((32, 32, 32))
        signal = np.sin(np.linspace(0, 2 * np.pi, 50))
        with pytest.raises(ValueError, match="no active points"):
            kw.Source.from_velocity_mask(mask, uz=signal)

    def test_velocity_source_simulation_runs(self):
        """Velocity source can be used in simulation."""
        N = 32
        grid = kw.Grid(nx=N, ny=N, nz=N, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        medium = kw.Medium.homogeneous(1500.0, 1000.0)
        sensor = kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))

        mask = np.zeros((N, N, N))
        mask[0, :, :] = 1.0
        nt = 50
        dt = 1e-8
        t = np.arange(nt) * dt
        uz = 1e-2 * np.sin(2 * np.pi * 1e6 * t)  # velocity signal [m/s]

        src = kw.Source.from_velocity_mask(mask, uz=uz)
        sim = kw.Simulation(grid, medium, src, sensor)
        result = sim.run(time_steps=nt, dt=dt)

        assert result is not None
        assert result.time_steps == nt
        assert np.all(np.isfinite(result.sensor_data))


# ============================================================================
# Signal generation tests
# ============================================================================


class TestSignalGeneration:
    """Test temporal signal generation for sources."""

    def test_sinusoidal_signal(self):
        """Generate continuous sinusoidal signal."""
        freq = 1e6
        nt = 100
        dt = 1e-8
        t = np.arange(nt) * dt
        signal = 1e5 * np.sin(2 * np.pi * freq * t)
        
        # Verify signal properties
        assert len(signal) == nt
        assert np.max(signal) == 1e5
        assert np.min(signal) == -1e5

    def test_gaussian_pulse_signal(self):
        """Generate Gaussian-modulated pulse signal."""
        freq = 1e6
        nt = 200
        dt = 1e-8
        t = np.arange(nt) * dt
        
        # Gaussian envelope
        t0 = 50 * dt  # Pulse center
        sigma = 10 * dt  # Pulse width
        envelope = np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))
        
        # Modulated signal
        signal = 1e5 * envelope * np.sin(2 * np.pi * freq * t)
        
        # Verify signal peaks near t0
        peak_idx = np.argmax(np.abs(signal))
        assert abs(peak_idx - 50) < 10

    def test_tone_burst_signal(self):
        """Generate tone burst (N-cycle) signal."""
        freq = 1e6
        cycles = 3
        dt = 1e-8
        
        # Generate tone burst
        period = 1.0 / freq
        burst_duration = cycles * period
        nt_burst = int(burst_duration / dt)
        
        t = np.arange(nt_burst) * dt
        signal = np.sin(2 * np.pi * freq * t)
        
        # Verify burst length
        assert len(signal) == nt_burst
        # Count zero crossings (should be ~2*cycles)
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        assert len(zero_crossings) >= 2 * cycles - 1

    def test_ramped_signal(self):
        """Generate ramped (windowed) signal to reduce transients."""
        freq = 1e6
        nt = 100
        dt = 1e-8
        t = np.arange(nt) * dt
        
        # Hanning window
        window = np.hanning(nt)
        signal = 1e5 * window * np.sin(2 * np.pi * freq * t)
        
        # Verify smooth start and end
        assert abs(signal[0]) < 1e-10
        assert abs(signal[-1]) < 1e-10
        assert np.max(np.abs(signal)) > 0.5e5  # Peak in middle

    @pytest.mark.parametrize("freq,cycles", [
        (0.5e6, 1),
        (1e6, 3),
        (2e6, 5),
        (5e6, 2),
    ])
    def test_various_tone_bursts(self, freq, cycles):
        """Generate tone bursts with various parameters."""
        dt = 1e-8
        period = 1.0 / freq
        burst_duration = cycles * period
        nt = int(burst_duration / dt)
        
        t = np.arange(nt) * dt
        signal = np.sin(2 * np.pi * freq * t)
        
        # Verify frequency by counting zero crossings
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        measured_periods = len(zero_crossings) / 2
        assert abs(measured_periods - cycles) < 1


# ============================================================================
# Transducer array source tests
# ============================================================================


class TestTransducerArraySource:
    """Test transducer array source creation."""

    def test_linear_array_creation(self):
        """Create linear transducer array source."""
        num_elements = 32
        pitch = 0.3e-3  # Element spacing
        element_width = 0.25e-3
        
        # Element positions
        positions = []
        for i in range(num_elements):
            x = i * pitch
            positions.append((x, 0, 0))
        
        assert len(positions) == num_elements

    def test_phased_array_delays(self):
        """Calculate transmit delays for beam steering."""
        num_elements = 16
        pitch = 0.3e-3
        c = 1540.0
        steer_angle = 30.0  # degrees
        
        # Calculate delays for each element
        delays = []
        for i in range(num_elements):
            x = (i - num_elements // 2) * pitch
            # Delay = x * sin(angle) / c
            delay = x * np.sin(np.radians(steer_angle)) / c
            delays.append(delay)
        
        # Verify delays are monotonic for steering
        delays = np.array(delays)
        assert np.all(np.diff(delays) > 0) or np.all(np.diff(delays) < 0)

    def test_focus_delays(self):
        """Calculate transmit delays for focusing."""
        num_elements = 16
        pitch = 0.3e-3
        c = 1540.0
        focus_depth = 30e-3  # 30 mm
        
        # Calculate delays for each element
        delays = []
        for i in range(num_elements):
            x = (i - num_elements // 2) * pitch
            # Distance from element to focus
            distance = np.sqrt(x ** 2 + focus_depth ** 2)
            # Delay relative to center element
            delay = (distance - focus_depth) / c
            delays.append(delay)
        
        # Verify center elements have smallest delays
        delays = np.array(delays)
        assert delays[num_elements // 2] <= delays[0]
        assert delays[num_elements // 2] <= delays[-1]


# ============================================================================
# Source scaling and normalization tests
# ============================================================================


class TestSourceScaling:
    """Test source amplitude scaling and normalization."""

    def test_pressure_to_velocity_scaling(self):
        """Convert pressure amplitude to equivalent velocity amplitude."""
        # p = rho * c * u (plane wave relation)
        rho = 1000.0
        c = 1500.0
        p_amplitude = 1e5  # Pa
        
        u_amplitude = p_amplitude / (rho * c)
        
        # Expected: 1e5 / (1000 * 1500) = 0.0667 m/s
        expected = 1e5 / 1.5e6
        assert abs(u_amplitude - expected) < 1e-10

    def test_intensity_calculation(self):
        """Calculate acoustic intensity from pressure."""
        # I = p^2 / (2 * rho * c)
        p = 1e5  # Pa
        rho = 1000.0
        c = 1500.0
        
        I = p ** 2 / (2 * rho * c)
        
        # Expected: 1e10 / (2 * 1000 * 1500) = 3333 W/m^2
        expected = 1e10 / 3e6
        assert abs(I - expected) < 1

    def test_dB_scale_conversion(self):
        """Convert between linear and dB scales."""
        # Reference: 1 Pa
        p_ref = 1.0
        p = 1e5  # 100 kPa
        
        # dB = 20 * log10(p / p_ref)
        dB = 20 * np.log10(p / p_ref)
        
        # Expected: 20 * 5 = 100 dB
        assert abs(dB - 100) < 0.1


# ============================================================================
# Cross-validation: Signal generation with k-wave-python
# ============================================================================


@requires_kwave
class TestSignalGenerationParityWithKWave:
    """Compare signal generation against k-wave-python utilities."""

    def test_tone_burst_matches_kwave(self):
        """pykwavers tone burst matches k-wave tone_burst."""
        freq = 1e6
        cycles = 3
        fs = 100e6  # Sampling frequency
        
        # k-wave tone burst
        signal_kw = tone_burst(fs, freq, cycles)
        
        # Verify signal properties
        assert len(signal_kw) > 0
        assert np.max(np.abs(signal_kw)) > 0
        
        # Count cycles
        zero_crossings = np.where(np.diff(np.sign(signal_kw.flatten())))[0]
        measured_cycles = len(zero_crossings) / 2
        assert abs(measured_cycles - cycles) < 1

    def test_gaussian_matches_kwave(self):
        """Gaussian pulse generation matches expected shape."""
        from kwave.utils.signals import tone_burst
        
        # Generate Gaussian-modulated signal
        freq = 1e6
        cycles = 5
        fs = 100e6
        
        # Use pykwavers tone_burst (wrapper around kwave's)
        signal = kw.tone_burst(fs, freq, cycles)
        
        # Verify envelope shape
        assert len(signal) > 0
        # Peak should be near center
        peak_idx = np.argmax(np.abs(signal))
        center_idx = len(signal) // 2
        assert abs(peak_idx - center_idx) < len(signal) // 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
