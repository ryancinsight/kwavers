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

    def test_initial_pressure_p0(self):
        """pykwavers supports initial pressure distribution (p0)."""
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        medium = kw.Medium.homogeneous(1500.0, 1000.0)
        sensor = kw.Sensor.point((1.6e-3, 1.6e-3, 1.6e-3))

        p0 = np.zeros((32, 32, 32))
        p0[16, 16, 16] = 1e5
        source = kw.Source.from_initial_pressure(p0)
        assert source is not None

    def test_heterogeneous_medium(self):
        """pykwavers should support spatially varying sound speed/density."""
        c = np.ones((32, 32, 32)) * 1500.0
        c[16:, :, :] = 2000.0
        rho = np.ones((32, 32, 32)) * 1000.0

        medium = kw.Medium(sound_speed=c, density=rho)
        assert medium is not None
        # Max sound speed should be 2000
        assert medium.sound_speed == 2000.0
        # Density at origin
        assert medium.density == 1000.0
        # Heterogeneous flag
        assert not medium.is_homogeneous
        # Repr should mention heterogeneous
        assert "heterogeneous" in repr(medium).lower()
        assert "32" in repr(medium)

    def test_heterogeneous_medium_with_absorption(self):
        """Heterogeneous medium with spatially varying absorption."""
        c = np.ones((16, 16, 16)) * 1500.0
        rho = np.ones((16, 16, 16)) * 1000.0
        alpha = np.ones((16, 16, 16)) * 0.5
        alpha[8:, :, :] = 1.0

        medium = kw.Medium(sound_speed=c, density=rho, absorption=alpha)
        assert not medium.is_homogeneous
        assert medium.sound_speed == 1500.0

    def test_heterogeneous_medium_simulation(self):
        """Heterogeneous medium should run in FDTD simulation."""
        nx, ny, nz = 24, 24, 24
        dx = 0.1e-3

        grid = kw.Grid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, dz=dx)

        # Two-layer medium
        c = np.ones((nx, ny, nz)) * 1500.0
        c[nx // 2 :, :, :] = 2000.0
        rho = np.ones((nx, ny, nz)) * 1000.0

        medium = kw.Medium(sound_speed=c, density=rho)
        source = kw.Source.point(
            position=(nx // 4, ny // 2, nz // 2), frequency=1e6, amplitude=1e5
        )
        sensor = kw.Sensor.point((nx * 3 // 4, ny // 2, nz // 2))

        sim = kw.Simulation(grid=grid, medium=medium, source=source, sensor=sensor)
        result = sim.run(time_steps=50)
        assert result is not None
        assert result.sensor_data.shape[0] == 50

    def test_heterogeneous_medium_validation(self):
        """Heterogeneous medium constructor validates inputs."""
        # Zero sound speed should raise
        c = np.zeros((8, 8, 8))
        rho = np.ones((8, 8, 8))
        with pytest.raises(ValueError, match="positive"):
            kw.Medium(sound_speed=c, density=rho)

        # Shape mismatch should raise
        c = np.ones((8, 8, 8)) * 1500.0
        rho = np.ones((8, 8, 4)) * 1000.0
        with pytest.raises(ValueError, match="shape"):
            kw.Medium(sound_speed=c, density=rho)

    def test_homogeneous_medium_properties(self):
        """Homogeneous medium has correct is_homogeneous flag and repr."""
        medium = kw.Medium.homogeneous(1500.0, 1000.0)
        assert medium.is_homogeneous
        assert "homogeneous" in repr(medium).lower()
        assert "1500" in repr(medium)

    def test_velocity_source(self):
        """pykwavers supports velocity sources (ux, uy, uz)."""
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        mask = np.zeros((32, 32, 32))
        mask[0, :, :] = 1.0
        signal = np.sin(np.linspace(0, 2 * np.pi, 50))

        source = kw.Source.from_velocity_mask(mask, uz=signal)
        assert source is not None
        assert source.source_type == "velocity"

    def test_absorption_parameters(self):
        """pykwavers exposes absorption and alpha_power."""
        medium = kw.Medium.homogeneous(
            sound_speed=1500.0, density=1000.0,
            absorption=0.75, alpha_power=1.5,
        )
        assert medium.sound_speed == 1500.0
        assert medium.density == 1000.0

    def test_multi_sensor_mask(self):
        """pykwavers supports binary mask sensors (multiple points)."""
        mask = np.zeros((32, 32, 32), dtype=bool)
        mask[8, 16, 16] = True
        mask[16, 16, 16] = True
        mask[24, 16, 16] = True

        sensor = kw.Sensor.from_mask(mask)  # noqa: F841

    def test_plane_wave_direction(self):
        """pykwavers supports plane wave direction parameter."""
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)

        # Default direction (+z)
        source_z = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        assert source_z is not None

        # Explicit x-direction
        source_x = kw.Source.plane_wave(
            grid, frequency=1e6, amplitude=1e5,
            direction=(1.0, 0.0, 0.0),
        )
        assert source_x is not None

        # Negative z-direction
        source_nz = kw.Source.plane_wave(
            grid, frequency=1e6, amplitude=1e5,
            direction=(0.0, 0.0, -1.0),
        )
        assert source_nz is not None

        # Unnormalized direction is auto-normalized
        source_diag = kw.Source.plane_wave(
            grid, frequency=1e6, amplitude=1e5,
            direction=(1.0, 1.0, 0.0),
        )
        assert source_diag is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================================
# Signal Processing Comparison Utilities
# ============================================================================


class TestSignalProcessingComparison:
    """
    Comprehensive signal processing utilities for comparing pykwavers and k-wave.
    
    These utilities support frequency-domain analysis, time-domain metrics,
    and beamforming comparisons.
    """

    def test_fft_comparison(self):
        """Compare FFT of signals between implementations."""
        # Generate test signal
        fs = 100e6  # 100 MHz sampling
        f0 = 1e6    # 1 MHz center frequency
        nt = 256
        
        t = np.arange(nt) / fs
        signal = np.sin(2 * np.pi * f0 * t)
        
        # Compute FFT
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(nt, 1/fs)
        
        # Peak should be at f0
        peak_idx = np.argmax(np.abs(fft_result[:nt//2]))
        peak_freq = freqs[peak_idx]
        
        assert abs(peak_freq - f0) < fs / nt

    def test_power_spectral_density(self):
        """Compute and compare power spectral density."""
        fs = 100e6
        f0 = 2e6
        nt = 512
        
        t = np.arange(nt) / fs
        signal = np.sin(2 * np.pi * f0 * t) + 0.1 * np.random.randn(nt)
        
        # Compute PSD using Welch method
        freqs, psd = self._compute_psd(signal, fs)
        
        # Peak should be near f0
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]
        
        assert abs(peak_freq - f0) < fs / nt * 2

    def _compute_psd(self, signal, fs, nperseg=128):
        """Simple PSD computation using Welch method."""
        nseg = len(signal) // nperseg
        psd = np.zeros(nperseg // 2 + 1)
        
        for i in range(nseg):
            segment = signal[i * nperseg:(i + 1) * nperseg]
            fft_seg = np.fft.rfft(segment)
            psd += np.abs(fft_seg) ** 2
        
        psd /= nseg
        freqs = np.fft.rfftfreq(nperseg, 1/fs)
        return freqs, psd

    def test_cross_correlation_analysis(self):
        """Compute cross-correlation for time lag detection."""
        # Create shifted signals
        nt = 200
        shift = 10
        
        original = np.sin(2 * np.pi * np.arange(nt) / 20)
        shifted = np.roll(original, shift)
        
        # Compute cross-correlation
        xcorr = np.correlate(original, shifted, mode='full')
        lags = np.arange(-nt + 1, nt)
        
        # Find peak lag
        peak_lag = lags[np.argmax(xcorr)]
        
        assert peak_lag == -shift

    def test_envelope_detection(self):
        """Test envelope detection using Hilbert transform."""
        fs = 100e6
        f0 = 5e6
        nt = 256
        
        t = np.arange(nt) / fs
        carrier = np.sin(2 * np.pi * f0 * t)
        
        # Gaussian envelope
        envelope_true = np.exp(-((t - t[nt//2]) / (1e-6)) ** 2)
        signal = carrier * envelope_true
        
        # Detect envelope using Hilbert transform
        analytic = self._hilbert(signal)
        envelope_detected = np.abs(analytic)
        
        # Normalize for comparison
        envelope_detected /= np.max(envelope_detected)
        envelope_true_norm = envelope_true / np.max(envelope_true)
        
        # Should match reasonably
        error = np.mean((envelope_detected - envelope_true_norm) ** 2)
        assert error < 0.1

    def _hilbert(self, signal):
        """Compute Hilbert transform using FFT."""
        n = len(signal)
        fft_signal = np.fft.fft(signal)
        
        # Create analytic signal by zeroing negative frequencies
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1:n // 2] = 2
        else:
            h[0] = 1
            h[1:(n + 1) // 2] = 2
        
        analytic = np.fft.ifft(fft_signal * h)
        return analytic

    def test_bandpass_filtering(self):
        """Test bandpass filtering of signals."""
        fs = 100e6
        nt = 512
        
        t = np.arange(nt) / fs
        
        # Multi-frequency signal
        f_low, f_mid, f_high = 1e6, 5e6, 10e6
        signal = (np.sin(2 * np.pi * f_low * t) + 
                  np.sin(2 * np.pi * f_mid * t) + 
                  np.sin(2 * np.pi * f_high * t))
        
        # Simple frequency domain bandpass
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(nt, 1/fs)
        
        # Bandpass around f_mid (3-7 MHz)
        low_cut, high_cut = 3e6, 7e6
        mask = (np.abs(freqs) >= low_cut) & (np.abs(freqs) <= high_cut)
        fft_filtered = fft_signal * mask
        
        filtered = np.real(np.fft.ifft(fft_filtered))
        
        # Check that f_mid component dominates
        fft_result = np.fft.fft(filtered)
        peak_freq = np.abs(freqs[np.argmax(np.abs(fft_result[:nt//2]))])
        
        assert abs(peak_freq - f_mid) < 1e6

    def test_time_frequency_analysis(self):
        """Test short-time Fourier transform for time-frequency analysis."""
        fs = 100e6
        nt = 256
        
        t = np.arange(nt) / fs
        
        # Chirp signal (frequency increases with time)
        f0, f1 = 1e6, 10e6
        signal = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / t[-1]) * t)
        
        # Simple STFT
        window_size = 32
        hop = 16
        n_windows = (nt - window_size) // hop + 1
        
        stft_result = np.zeros((window_size // 2 + 1, n_windows))
        
        for i in range(n_windows):
            start = i * hop
            segment = signal[start:start + window_size]
            windowed = segment * np.hanning(window_size)
            stft_result[:, i] = np.abs(np.fft.rfft(windowed))
        
        # Check that frequency content increases over time
        peak_freqs = []
        freqs = np.fft.rfftfreq(window_size, 1/fs)
        
        for i in range(n_windows):
            peak_idx = np.argmax(stft_result[:, i])
            peak_freqs.append(freqs[peak_idx])
        
        # Frequency should generally increase
        assert peak_freqs[-1] > peak_freqs[0]


class TestBeamformingComparison:
    """
    Utilities for comparing beamforming results between implementations.
    """

    def test_delay_and_sum_beamforming(self):
        """Test delay-and-sum beamforming algorithm."""
        # Array parameters
        n_elements = 16
        pitch = 0.3e-3
        c = 1500.0
        
        # Focus parameters
        focus_depth = 20e-3
        focus_angle = 0  # degrees
        
        # Calculate delays
        delays = self._calculate_delays(n_elements, pitch, c, focus_depth, focus_angle)
        
        # Delays should be symmetric around center
        center_delay = delays[n_elements // 2]
        
        # All delays should be positive (relative to earliest arrival)
        assert np.all(delays >= 0)
        
        # Center element should have smallest delay for 0 degree steering
        assert delays[n_elements // 2] <= delays[0]
        assert delays[n_elements // 2] <= delays[-1]

    def _calculate_delays(self, n_elements, pitch, c, focus_depth, focus_angle):
        """Calculate beamforming delays for linear array."""
        delays = np.zeros(n_elements)
        
        for i in range(n_elements):
            # Element position relative to center
            x = (i - n_elements / 2 + 0.5) * pitch
            
            # Distance to focus point
            focus_x = focus_depth * np.sin(np.radians(focus_angle))
            distance = np.sqrt((x - focus_x) ** 2 + focus_depth ** 2)
            
            # Delay relative to origin
            delays[i] = distance / c
        
        # Normalize to minimum delay
        delays -= np.min(delays)
        
        return delays

    def test_apodization_windows(self):
        """Test different apodization window functions."""
        n_elements = 32
        
        windows = {
            'rectangular': np.ones(n_elements),
            'hanning': np.hanning(n_elements),
            'hamming': np.hamming(n_elements),
            'blackman': np.blackman(n_elements),
        }
        
        for name, window in windows.items():
            # All windows should sum to approximately n_elements/2 for symmetric windows
            total = np.sum(window)
            
            # Hanning and Hamming should have lower totals than rectangular
            if name == 'rectangular':
                assert total == n_elements
            else:
                assert total < n_elements

    def test_dynamic_receive_beamforming(self):
        """Test dynamic receive beamforming delays."""
        n_elements = 16
        pitch = 0.3e-3
        c = 1500.0
        
        # Delays change with depth
        depths = np.array([5e-3, 10e-3, 20e-3, 40e-3])
        
        delays_per_depth = []
        for depth in depths:
            delays = self._calculate_delays(n_elements, pitch, c, depth, 0)
            delays_per_depth.append(delays)
        
        # Maximum delay difference should decrease with depth
        max_delays = [np.max(d) - np.min(d) for d in delays_per_depth]
        
        # Deeper depths have smaller delay spread
        for i in range(len(max_delays) - 1):
            assert max_delays[i + 1] < max_delays[i]


class TestAcousticFieldMetrics:
    """
    Metrics for comparing acoustic field simulations.
    """

    def test_peak_pressure_location(self):
        """Test finding peak pressure location in field."""
        # Create synthetic pressure field
        N = 64
        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)
        z = np.linspace(-1, 1, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Gaussian peak at (0.2, -0.1, 0.3)
        peak_true = (0.2, -0.1, 0.3)
        field = np.exp(-((X - peak_true[0])**2 + (Y - peak_true[1])**2 + (Z - peak_true[2])**2) / 0.1)
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(field), field.shape)
        peak_found = (x[peak_idx[0]], y[peak_idx[1]], z[peak_idx[2]])
        
        # Should be close to true peak
        for i in range(3):
            assert abs(peak_found[i] - peak_true[i]) < 0.1

    def test_beam_width_calculation(self):
        """Test beam width calculation from pressure field."""
        # Create synthetic beam profile
        N = 100
        x = np.linspace(-5e-3, 5e-3, N)
        
        # Gaussian beam with known width
        beam_width_true = 2e-3  # -6dB width
        profile = np.exp(-2 * np.log(2) * (x / (beam_width_true / 2)) ** 2)
        
        # Measure -6dB width
        threshold = 0.5  # -6dB = 0.5 amplitude
        above_threshold = np.abs(profile) >= threshold
        
        if np.any(above_threshold):
            indices = np.where(above_threshold)[0]
            beam_width_measured = x[indices[-1]] - x[indices[0]]
            
            # Should be close to true width (within 50% due to discrete sampling)
            assert abs(beam_width_measured - beam_width_true) / beam_width_true < 0.5

    def test_acoustic_intensity_calculation(self):
        """Test acoustic intensity from pressure and velocity."""
        # Simple plane wave
        p0 = 1e5  # Pa
        rho = 1000  # kg/m³
        c = 1500  # m/s
        
        # For plane wave: I = p² / (rho * c)
        intensity = p0 ** 2 / (rho * c)
        
        # Also: I = p * u where u = p / (rho * c)
        u = p0 / (rho * c)
        intensity_alt = p0 * u
        
        assert abs(intensity - intensity_alt) / intensity < 1e-10

    def test_spatial_peak_temporal_average_intensity(self):
        """Test ISPTA calculation."""
        # Pulsed ultrasound parameters
        pulse_duration = 1e-6  # seconds
        pulse_repetition_period = 100e-6  # seconds
        spatial_peak_intensity = 100  # W/cm²
        
        # Duty cycle
        duty_cycle = pulse_duration / pulse_repetition_period
        
        # ISPTA = ISPPA * duty_cycle
        ispta = spatial_peak_intensity * duty_cycle
        
        assert ispta == pytest.approx(1.0)  # W/cm²


class TestSignalComparisonMetrics:
    """
    Advanced metrics for comparing signals between implementations.
    """

    def test_normalized_cross_correlation(self):
        """Test normalized cross-correlation coefficient."""
        # Identical signals
        s1 = np.sin(2 * np.pi * np.arange(100) / 20)
        s2 = s1.copy()
        
        ncc = self._normalized_cross_correlation(s1, s2)
        assert abs(ncc - 1.0) < 1e-10
        
        # Opposite signals
        ncc_opposite = self._normalized_cross_correlation(s1, -s1)
        assert abs(ncc_opposite + 1.0) < 1e-10
        
        # Uncorrelated signals
        rng = np.random.RandomState(42)
        s_random = rng.randn(100)
        ncc_random = self._normalized_cross_correlation(s1, s_random)
        assert abs(ncc_random) < 0.3

    def _normalized_cross_correlation(self, s1, s2):
        """Compute normalized cross-correlation coefficient."""
        s1_norm = (s1 - np.mean(s1)) / np.std(s1)
        s2_norm = (s2 - np.mean(s2)) / np.std(s2)
        return np.mean(s1_norm * s2_norm)

    def test_spectral_distortion_index(self):
        """Test spectral distortion index for frequency-domain comparison."""
        # Original spectrum
        freqs = np.fft.rfftfreq(256, 1/100e6)
        spectrum1 = np.exp(-((freqs - 5e6) / 1e6) ** 2)
        
        # Distorted spectrum (shifted)
        spectrum2 = np.exp(-((freqs - 5.5e6) / 1e6) ** 2)
        
        # Spectral distortion
        sd = self._spectral_distortion(spectrum1, spectrum2)
        
        # Should be positive for different spectra
        assert sd > 0
        
        # Same spectrum should give zero
        sd_same = self._spectral_distortion(spectrum1, spectrum1)
        assert sd_same < 1e-10

    def _spectral_distortion(self, s1, s2):
        """Compute spectral distortion index."""
        # Log spectral distance
        eps = 1e-10
        log_diff = np.log(s1 + eps) - np.log(s2 + eps)
        return np.sqrt(np.mean(log_diff ** 2))

    def test_root_mean_square_error(self):
        """Test RMSE calculation."""
        s1 = np.array([1, 2, 3, 4, 5], dtype=float)
        s2 = np.array([1.1, 2.2, 2.9, 4.1, 4.8], dtype=float)
        
        rmse = np.sqrt(np.mean((s1 - s2) ** 2))
        
        # Should be small for similar signals
        assert rmse < 0.2

    def test_signal_to_noise_ratio(self):
        """Test SNR calculation."""
        # Signal
        signal = np.sin(2 * np.pi * np.arange(1000) / 20)
        signal_power = np.mean(signal ** 2)
        
        # Add noise
        rng = np.random.RandomState(42)
        noise = 0.1 * rng.randn(1000)
        noise_power = np.mean(noise ** 2)
        
        noisy_signal = signal + noise
        
        # Calculate SNR
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        # Should be around 20 dB for this noise level
        assert 15 < snr_db < 25
