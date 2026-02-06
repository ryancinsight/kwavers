#!/usr/bin/env python3
"""
Minimal smoke test for pykwavers Python bindings.

This test validates that the basic PyO3 bindings work correctly
and can execute a simple simulation with sensor data recording.

Author: Ryan Clanton (@ryancinsight)
Date: 2026-02-04
Sprint: 217 Session 9 - Phase 2 PyO3 Integration
"""

import sys

import numpy as np

try:
    import pykwavers as kw
except ImportError as e:
    print(f"Failed to import pykwavers: {e}")
    print("\nTo build and install pykwavers:")
    print("  cd pykwavers")
    print("  maturin build --release")
    print("  pip install ../target/wheels/pykwavers-*.whl")
    sys.exit(1)


def test_grid_creation():
    """Test grid creation and basic properties."""
    print("Testing Grid creation...")
    grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)

    assert grid.nx == 32
    assert grid.ny == 32
    assert grid.nz == 32
    assert grid.dx == 0.1e-3
    assert grid.total_points() == 32 * 32 * 32

    print(f"  [OK] Grid: {grid}")


def test_medium_creation():
    """Test medium creation."""
    print("Testing Medium creation...")
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)

    print(f"  [OK] Medium: {medium}")


def test_source_creation(grid):
    """Test source creation."""
    print("Testing Source creation...")
    source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)

    assert source.frequency == 1e6
    assert source.amplitude == 1e5
    assert source.source_type == "plane_wave"

    print(f"  [OK] Source: {source}")


def test_sensor_creation():
    """Test sensor creation."""
    print("Testing Sensor creation...")
    sensor = kw.Sensor.point(
        position=(0.0016, 0.0016, 0.0016)
    )  # Center of 32^3 grid with 0.1mm spacing

    assert sensor.sensor_type == "point"

    print(f"  [OK] Sensor: {sensor}")


def test_simulation_run(grid, medium, source, sensor):
    """Test simulation execution with sensor data recording."""
    print("Testing Simulation.run()...")

    sim = kw.Simulation(grid, medium, source, sensor)
    print(f"  Created: {sim}")

    # Run small simulation (10 time steps for speed)
    time_steps = 10
    dt = 1e-8  # 10 ns

    print(f"  Running {time_steps} time steps with dt={dt * 1e9:.2f} ns...")
    result = sim.run(time_steps=time_steps, dt=dt)

    print(f"  [OK] Result: {result}")

    # Validate result structure
    assert hasattr(result, "sensor_data"), "Result should have sensor_data"
    assert hasattr(result, "time"), "Result should have time"
    assert result.time_steps == time_steps
    assert result.dt == dt

    # Validate sensor_data is a NumPy array
    sensor_data = result.sensor_data
    time_vec = result.time

    print(f"  Sensor data type: {type(sensor_data)}")
    print(f"  Sensor data shape: {sensor_data.shape}")
    print(f"  Time vector shape: {time_vec.shape}")

    assert isinstance(sensor_data, np.ndarray), "sensor_data should be NumPy array"
    assert isinstance(time_vec, np.ndarray), "time should be NumPy array"
    assert sensor_data.shape == (time_steps,), (
        f"Expected shape ({time_steps},), got {sensor_data.shape}"
    )
    assert time_vec.shape == (time_steps,), f"Expected shape ({time_steps},), got {time_vec.shape}"

    # Check data values
    print(f"  Sensor data range: [{sensor_data.min():.2e}, {sensor_data.max():.2e}]")
    print(f"  Time range: [{time_vec.min():.2e}s, {time_vec.max():.2e}s]")

    # Basic sanity checks
    assert np.all(np.isfinite(sensor_data)), "Sensor data should be finite"
    assert np.all(np.isfinite(time_vec)), "Time vector should be finite"
    assert np.all(time_vec >= 0), "Time should be non-negative"

    print("  [OK] All validation checks passed!")



def main():
    """Run all tests."""
    print("=" * 80)
    print("pykwavers Smoke Test")
    print("=" * 80)
    print()

    try:
        # Test component creation
        test_grid_creation()
        test_medium_creation()
        test_source_creation(grid=kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3))
        test_sensor_creation()

        print()

        # Test simulation execution
        grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
        medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
        source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
        sensor = kw.Sensor.point(position=(0.0016, 0.0016, 0.0016))
        result = test_simulation_run(grid, medium, source, sensor)

        print()
        print("=" * 80)
        print("[SUCCESS] All tests passed!")
        print("=" * 80)
        print()
        print("Summary:")
        print(f"  - Grid: {grid.nx}×{grid.ny}×{grid.nz} points")
        print(f"  - Sensor data: {result.sensor_data.shape} array")
        print(f"  - Time steps: {result.time_steps}")
        print(f"  - Total time: {result.final_time * 1e6:.2f} μs")
        print()
        print("Phase 2 PyO3 integration successful!")

        return 0

    except AssertionError as e:
        print()
        print("=" * 80)
        print(f"[FAIL] Test failed: {e}")
        print("=" * 80)
        return 1

    except Exception as e:
        print()
        print("=" * 80)
        print(f"[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
