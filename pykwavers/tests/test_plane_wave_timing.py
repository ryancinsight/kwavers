#!/usr/bin/env python3
"""
Phase 4: Plane Wave Timing Validation Test

This test validates the fix for plane wave arrival timing by comparing
measured vs expected arrival times for boundary-only injection mode.

Mathematical Specification:
- Plane wave propagation in homogeneous medium
- Expected arrival time: t_arrival = distance / sound_speed
- Acceptance criteria: |measured - expected| / expected < 5%

Test Cases:
1. +Z direction propagation (default)
2. -Z direction propagation
3. +X direction propagation
4. Multi-frequency validation
5. Amplitude independence

Author: Ryan Clanton (@ryancinsight)
Date: 2024-02-04
Sprint: 217 Session 9 - Phase 4 Development
"""

import numpy as np
import pykwavers as kw
import pytest


class TestPlaneWaveTiming:
    """Test suite for plane wave timing validation."""

    @pytest.fixture
    def basic_setup(self):
        """Standard test configuration."""
        return {
            "nx": 64,
            "ny": 64,
            "nz": 64,
            "dx": 0.1e-3,  # 0.1 mm
            "dy": 0.1e-3,
            "dz": 0.1e-3,
            "sound_speed": 1500.0,  # m/s
            "density": 1000.0,  # kg/m³
            "absorption": 0.0,
        }

    def calculate_expected_arrival(self, source_plane_idx, sensor_idx, grid_spacing, sound_speed):
        """
        Calculate expected arrival time.

        Parameters
        ----------
        source_plane_idx : int
            Index of source plane (boundary)
        sensor_idx : int
            Index of sensor location
        grid_spacing : float
            Grid spacing [m]
        sound_speed : float
            Sound speed [m/s]

        Returns
        -------
        float
            Expected arrival time [s]
        """
        distance = abs(sensor_idx - source_plane_idx) * grid_spacing
        return distance / sound_speed

    def find_arrival_time(self, pressure, time, threshold=0.1):
        """
        Find arrival time from pressure time series.

        Uses threshold crossing of normalized pressure amplitude.

        Parameters
        ----------
        pressure : np.ndarray
            Pressure time series [Pa]
        time : np.ndarray
            Time vector [s]
        threshold : float
            Detection threshold (fraction of max amplitude)

        Returns
        -------
        float
            Measured arrival time [s]
        """
        # Normalize pressure
        p_max = np.max(np.abs(pressure))
        if p_max < 1e-10:
            return np.nan

        p_norm = np.abs(pressure) / p_max

        # Find first crossing of threshold
        crossings = np.where(p_norm > threshold)[0]
        if len(crossings) == 0:
            return np.nan

        return time[crossings[0]]

    def test_z_direction_positive(self, basic_setup):
        """Test +Z direction plane wave timing."""
        # Setup
        grid = kw.Grid(
            nx=basic_setup["nx"],
            ny=basic_setup["ny"],
            nz=basic_setup["nz"],
            dx=basic_setup["dx"],
            dy=basic_setup["dy"],
            dz=basic_setup["dz"],
        )

        medium = kw.Medium.homogeneous(
            sound_speed=basic_setup["sound_speed"],
            density=basic_setup["density"],
            absorption=basic_setup["absorption"],
        )

        # Source at z=0, propagating +z
        frequency = 1e6  # 1 MHz
        amplitude = 1e5  # 100 kPa
        source = kw.Source.plane_wave(grid, frequency=frequency, amplitude=amplitude)

        # Sensor at center z
        sensor_z_idx = basic_setup["nz"] // 2
        sensor_pos = (
            basic_setup["nx"] // 2 * basic_setup["dx"],
            basic_setup["ny"] // 2 * basic_setup["dy"],
            sensor_z_idx * basic_setup["dz"],
        )
        sensor = kw.Sensor.point(position=sensor_pos)

        # Calculate expected arrival
        expected_arrival = self.calculate_expected_arrival(
            source_plane_idx=0,
            sensor_idx=sensor_z_idx,
            grid_spacing=basic_setup["dz"],
            sound_speed=basic_setup["sound_speed"],
        )

        # Run simulation
        sim = kw.Simulation(grid, medium, source, sensor)

        # Duration: 3x expected arrival time
        duration = 3.0 * expected_arrival
        dt = 0.3 * basic_setup["dz"] / basic_setup["sound_speed"]  # CFL = 0.3
        time_steps = int(duration / dt)

        result = sim.run(time_steps=time_steps, dt=dt)

        # Find measured arrival
        measured_arrival = self.find_arrival_time(result.sensor_data, result.time, threshold=0.1)

        # Validate
        assert not np.isnan(measured_arrival), "Failed to detect wave arrival"

        relative_error = abs(measured_arrival - expected_arrival) / expected_arrival

        print(f"\n+Z Direction Test:")
        print(f"  Expected arrival: {expected_arrival * 1e6:.3f} us")
        print(f"  Measured arrival: {measured_arrival * 1e6:.3f} us")
        print(f"  Relative error:   {relative_error * 100:.2f}%")

        assert relative_error < 0.05, (
            f"Timing error {relative_error * 100:.2f}% exceeds 5% threshold. "
            f"Expected: {expected_arrival * 1e6:.3f} us, "
            f"Measured: {measured_arrival * 1e6:.3f} us"
        )

    def test_z_direction_negative(self, basic_setup):
        """Test -Z direction plane wave timing."""
        # Setup
        grid = kw.Grid(
            nx=basic_setup["nx"],
            ny=basic_setup["ny"],
            nz=basic_setup["nz"],
            dx=basic_setup["dx"],
            dy=basic_setup["dy"],
            dz=basic_setup["dz"],
        )

        medium = kw.Medium.homogeneous(
            sound_speed=basic_setup["sound_speed"],
            density=basic_setup["density"],
            absorption=basic_setup["absorption"],
        )

        # Source at z=max, propagating -z
        frequency = 1e6
        amplitude = 1e5
        source = kw.Source.plane_wave(
            grid, frequency=frequency, amplitude=amplitude, direction=(0.0, 0.0, -1.0)
        )

        # Sensor at center z
        sensor_z_idx = basic_setup["nz"] // 2
        sensor_pos = (
            basic_setup["nx"] // 2 * basic_setup["dx"],
            basic_setup["ny"] // 2 * basic_setup["dy"],
            sensor_z_idx * basic_setup["dz"],
        )
        sensor = kw.Sensor.point(position=sensor_pos)

        # Calculate expected arrival
        expected_arrival = self.calculate_expected_arrival(
            source_plane_idx=basic_setup["nz"] - 1,
            sensor_idx=sensor_z_idx,
            grid_spacing=basic_setup["dz"],
            sound_speed=basic_setup["sound_speed"],
        )

        # Run simulation
        sim = kw.Simulation(grid, medium, source, sensor)

        duration = 3.0 * expected_arrival
        dt = 0.3 * basic_setup["dz"] / basic_setup["sound_speed"]
        time_steps = int(duration / dt)

        result = sim.run(time_steps=time_steps, dt=dt)

        # Find measured arrival
        measured_arrival = self.find_arrival_time(result.sensor_data, result.time)

        # Validate
        assert not np.isnan(measured_arrival), "Failed to detect wave arrival"

        relative_error = abs(measured_arrival - expected_arrival) / expected_arrival

        print(f"\n-Z Direction Test:")
        print(f"  Expected arrival: {expected_arrival * 1e6:.3f} us")
        print(f"  Measured arrival: {measured_arrival * 1e6:.3f} us")
        print(f"  Relative error:   {relative_error * 100:.2f}%")

        assert relative_error < 0.05, (
            f"Timing error {relative_error * 100:.2f}% exceeds 5% threshold"
        )

    def test_x_direction_positive(self, basic_setup):
        """Test +X direction plane wave timing."""
        grid = kw.Grid(
            nx=basic_setup["nx"],
            ny=basic_setup["ny"],
            nz=basic_setup["nz"],
            dx=basic_setup["dx"],
            dy=basic_setup["dy"],
            dz=basic_setup["dz"],
        )

        medium = kw.Medium.homogeneous(
            sound_speed=basic_setup["sound_speed"],
            density=basic_setup["density"],
            absorption=basic_setup["absorption"],
        )

        # Source at x=0, propagating +x
        frequency = 1e6
        amplitude = 1e5
        source = kw.Source.plane_wave(
            grid, frequency=frequency, amplitude=amplitude, direction=(1.0, 0.0, 0.0)
        )

        # Sensor at center x
        sensor_x_idx = basic_setup["nx"] // 2
        sensor_pos = (
            sensor_x_idx * basic_setup["dx"],
            basic_setup["ny"] // 2 * basic_setup["dy"],
            basic_setup["nz"] // 2 * basic_setup["dz"],
        )
        sensor = kw.Sensor.point(position=sensor_pos)

        # Calculate expected arrival
        expected_arrival = self.calculate_expected_arrival(
            source_plane_idx=0,
            sensor_idx=sensor_x_idx,
            grid_spacing=basic_setup["dx"],
            sound_speed=basic_setup["sound_speed"],
        )

        # Run simulation
        sim = kw.Simulation(grid, medium, source, sensor)

        duration = 3.0 * expected_arrival
        dt = 0.3 * basic_setup["dx"] / basic_setup["sound_speed"]
        time_steps = int(duration / dt)

        result = sim.run(time_steps=time_steps, dt=dt)

        # Find measured arrival
        measured_arrival = self.find_arrival_time(result.sensor_data, result.time)

        # Validate
        assert not np.isnan(measured_arrival), "Failed to detect wave arrival"

        relative_error = abs(measured_arrival - expected_arrival) / expected_arrival

        print(f"\n+X Direction Test:")
        print(f"  Expected arrival: {expected_arrival * 1e6:.3f} us")
        print(f"  Measured arrival: {measured_arrival * 1e6:.3f} us")
        print(f"  Relative error:   {relative_error * 100:.2f}%")

        assert relative_error < 0.05, (
            f"Timing error {relative_error * 100:.2f}% exceeds 5% threshold"
        )

    @pytest.mark.parametrize("frequency", [0.5e6, 1e6, 2e6])
    def test_frequency_independence(self, basic_setup, frequency):
        """Test that arrival time is independent of frequency."""
        grid = kw.Grid(
            nx=basic_setup["nx"],
            ny=basic_setup["ny"],
            nz=basic_setup["nz"],
            dx=basic_setup["dx"],
            dy=basic_setup["dy"],
            dz=basic_setup["dz"],
        )

        medium = kw.Medium.homogeneous(
            sound_speed=basic_setup["sound_speed"],
            density=basic_setup["density"],
            absorption=basic_setup["absorption"],
        )

        amplitude = 1e5
        source = kw.Source.plane_wave(grid, frequency=frequency, amplitude=amplitude)

        sensor_z_idx = basic_setup["nz"] // 2
        sensor_pos = (
            basic_setup["nx"] // 2 * basic_setup["dx"],
            basic_setup["ny"] // 2 * basic_setup["dy"],
            sensor_z_idx * basic_setup["dz"],
        )
        sensor = kw.Sensor.point(position=sensor_pos)

        expected_arrival = self.calculate_expected_arrival(
            source_plane_idx=0,
            sensor_idx=sensor_z_idx,
            grid_spacing=basic_setup["dz"],
            sound_speed=basic_setup["sound_speed"],
        )

        sim = kw.Simulation(grid, medium, source, sensor)

        duration = 3.0 * expected_arrival
        dt = 0.3 * basic_setup["dz"] / basic_setup["sound_speed"]
        time_steps = int(duration / dt)

        result = sim.run(time_steps=time_steps, dt=dt)

        measured_arrival = self.find_arrival_time(result.sensor_data, result.time)

        assert not np.isnan(measured_arrival), f"Failed to detect arrival at {frequency * 1e-6} MHz"

        relative_error = abs(measured_arrival - expected_arrival) / expected_arrival

        print(f"\nFrequency {frequency * 1e-6:.1f} MHz:")
        print(f"  Expected arrival: {expected_arrival * 1e6:.3f} us")
        print(f"  Measured arrival: {measured_arrival * 1e6:.3f} us")
        print(f"  Relative error:   {relative_error * 100:.2f}%")

        assert relative_error < 0.05, (
            f"Timing error at {frequency * 1e-6} MHz: {relative_error * 100:.2f}%"
        )

    @pytest.mark.parametrize("amplitude", [1e4, 1e5, 1e6])
    def test_amplitude_independence(self, basic_setup, amplitude):
        """Test that arrival time is independent of amplitude."""
        grid = kw.Grid(
            nx=basic_setup["nx"],
            ny=basic_setup["ny"],
            nz=basic_setup["nz"],
            dx=basic_setup["dx"],
            dy=basic_setup["dy"],
            dz=basic_setup["dz"],
        )

        medium = kw.Medium.homogeneous(
            sound_speed=basic_setup["sound_speed"],
            density=basic_setup["density"],
            absorption=basic_setup["absorption"],
        )

        frequency = 1e6
        source = kw.Source.plane_wave(grid, frequency=frequency, amplitude=amplitude)

        sensor_z_idx = basic_setup["nz"] // 2
        sensor_pos = (
            basic_setup["nx"] // 2 * basic_setup["dx"],
            basic_setup["ny"] // 2 * basic_setup["dy"],
            sensor_z_idx * basic_setup["dz"],
        )
        sensor = kw.Sensor.point(position=sensor_pos)

        expected_arrival = self.calculate_expected_arrival(
            source_plane_idx=0,
            sensor_idx=sensor_z_idx,
            grid_spacing=basic_setup["dz"],
            sound_speed=basic_setup["sound_speed"],
        )

        sim = kw.Simulation(grid, medium, source, sensor)

        duration = 3.0 * expected_arrival
        dt = 0.3 * basic_setup["dz"] / basic_setup["sound_speed"]
        time_steps = int(duration / dt)

        result = sim.run(time_steps=time_steps, dt=dt)

        measured_arrival = self.find_arrival_time(result.sensor_data, result.time)

        assert not np.isnan(measured_arrival), f"Failed to detect arrival at {amplitude} Pa"

        relative_error = abs(measured_arrival - expected_arrival) / expected_arrival

        print(f"\nAmplitude {amplitude * 1e-3:.0f} kPa:")
        print(f"  Expected arrival: {expected_arrival * 1e6:.3f} us")
        print(f"  Measured arrival: {measured_arrival * 1e6:.3f} us")
        print(f"  Relative error:   {relative_error * 100:.2f}%")

        assert relative_error < 0.05, f"Timing error at {amplitude} Pa: {relative_error * 100:.2f}%"

    def test_different_distances(self, basic_setup):
        """Test timing accuracy at different propagation distances."""
        # Test at 25%, 50%, 75% of domain
        test_positions = [0.25, 0.5, 0.75]

        for frac in test_positions:
            sensor_z_idx = int(basic_setup["nz"] * frac)

            grid = kw.Grid(
                nx=basic_setup["nx"],
                ny=basic_setup["ny"],
                nz=basic_setup["nz"],
                dx=basic_setup["dx"],
                dy=basic_setup["dy"],
                dz=basic_setup["dz"],
            )

            medium = kw.Medium.homogeneous(
                sound_speed=basic_setup["sound_speed"],
                density=basic_setup["density"],
                absorption=basic_setup["absorption"],
            )

            frequency = 1e6
            amplitude = 1e5
            source = kw.Source.plane_wave(grid, frequency=frequency, amplitude=amplitude)

            sensor_pos = (
                basic_setup["nx"] // 2 * basic_setup["dx"],
                basic_setup["ny"] // 2 * basic_setup["dy"],
                sensor_z_idx * basic_setup["dz"],
            )
            sensor = kw.Sensor.point(position=sensor_pos)

            expected_arrival = self.calculate_expected_arrival(
                source_plane_idx=0,
                sensor_idx=sensor_z_idx,
                grid_spacing=basic_setup["dz"],
                sound_speed=basic_setup["sound_speed"],
            )

            sim = kw.Simulation(grid, medium, source, sensor)

            duration = 3.0 * expected_arrival
            dt = 0.3 * basic_setup["dz"] / basic_setup["sound_speed"]
            time_steps = int(duration / dt)

            result = sim.run(time_steps=time_steps, dt=dt)

            measured_arrival = self.find_arrival_time(result.sensor_data, result.time)

            assert not np.isnan(measured_arrival), f"Failed to detect arrival at {frac * 100}%"

            relative_error = abs(measured_arrival - expected_arrival) / expected_arrival

            print(f"\nDistance {frac * 100:.0f}% of domain:")
            print(f"  Expected arrival: {expected_arrival * 1e6:.3f} us")
            print(f"  Measured arrival: {measured_arrival * 1e6:.3f} us")
            print(f"  Relative error:   {relative_error * 100:.2f}%")

            assert relative_error < 0.05, (
                f"Timing error at {frac * 100}%: {relative_error * 100:.2f}%"
            )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
