#!/usr/bin/env python3
"""
Example: 2D Transducer Array Usage in pykwavers

This example demonstrates the native 2D transducer array implementation
and compares it with k-wave-python concepts.

Key Features Demonstrated:
- Creating transducer arrays
- Electronic steering and focusing
- Apodization windows
- Active element masking
- Integration with simulations
"""

import numpy as np
import pykwavers as kw


def example_basic_array_creation():
    """Example 1: Basic array creation and configuration."""
    print("=" * 60)
    print("Example 1: Basic Transducer Array Creation")
    print("=" * 60)

    # Create a standard 32-element linear array
    array = kw.TransducerArray2D(
        number_elements=32,
        element_width=0.3e-3,  # 0.3 mm width
        element_length=10e-3,  # 10 mm elevation
        element_spacing=0.5e-3,  # 0.5 mm spacing (λ/2 at 1.54 MHz)
        sound_speed=1540.0,  # Soft tissue
        frequency=1e6,  # 1 MHz
    )

    print(f"Created array: {array}")
    print(f"Number of elements: {array.number_elements}")
    print(f"Element spacing: {array.element_spacing * 1e3:.2f} mm")
    print(f"Total aperture: {array.aperture_width * 1e3:.1f} mm")


def example_beam_steering():
    """Example 2: Electronic beam steering."""
    print("\n" + "=" * 60)
    print("Example 2: Electronic Beam Steering")
    print("=" * 60)

    array = kw.TransducerArray2D(
        number_elements=64,
        element_width=0.2e-3,
        element_length=8e-3,
        element_spacing=0.3e-3,
        sound_speed=1540.0,
        frequency=2.5e6,
    )

    # Test steering at different angles
    steering_angles = [0, 10, 20, 30]

    for angle in steering_angles:
        array.set_steering_angle(angle)
        print(f"Steering angle: {angle:3.1f}°")

    print("\nArray configured for dynamic steering!")


def example_beam_focusing():
    """Example 3: Electronic beam focusing."""
    print("\n" + "=" * 60)
    print("Example 3: Electronic Beam Focusing")
    print("=" * 60)

    array = kw.TransducerArray2D(
        number_elements=64,
        element_width=0.2e-3,
        element_length=8e-3,
        element_spacing=0.3e-3,
        sound_speed=1540.0,
        frequency=2.5e6,
    )

    # Set focus at different depths
    focus_distances = [10e-3, 20e-3, 30e-3, 40e-3]  # mm to m

    print("Focusing configuration:")
    for depth in focus_distances:
        array.set_focus_distance(depth)
        print(f"  Focus depth: {depth * 1e3:4.1f} mm")


def example_apodization():
    """Example 4: Apodization (windowing)."""
    print("\n" + "=" * 60)
    print("Example 4: Apodization Windows")
    print("=" * 60)

    array = kw.TransducerArray2D(
        number_elements=32,
        element_width=0.3e-3,
        element_length=10e-3,
        element_spacing=0.5e-3,
        sound_speed=1540.0,
        frequency=1e6,
    )

    # Available apodization windows
    windows = ["Rectangular", "Hanning", "Hamming", "Blackman"]

    print("Available apodization windows:")
    for window in windows:
        array.set_transmit_apodization(window)
        print(f"  - {window}")


def example_active_elements():
    """Example 5: Active element masking."""
    print("\n" + "=" * 60)
    print("Example 5: Active Element Masking")
    print("=" * 60)

    array = kw.TransducerArray2D(
        number_elements=32,
        element_width=0.3e-3,
        element_length=10e-3,
        element_spacing=0.5e-3,
        sound_speed=1540.0,
        frequency=1e6,
    )

    # Different masking patterns
    patterns = {
        "Full aperture": [True] * 32,
        "Half aperture (center)": [False] * 8 + [True] * 16 + [False] * 8,
        "Half aperture (left)": [True] * 16 + [False] * 16,
        "Sparse (every 2nd)": [i % 2 == 0 for i in range(32)],
    }

    print("Active element patterns:")
    for name, mask in patterns.items():
        array.set_active_elements(mask)
        active_count = sum(mask)
        print(f"  {name}: {active_count} elements active")


def example_integration():
    """Example 6: Integration with simulation."""
    print("\n" + "=" * 60)
    print("Example 6: Integration with Simulation")
    print("=" * 60)

    # Simulation parameters
    N = 64
    dx = 0.5e-3
    c = 1540.0
    rho = 1000.0
    freq = 1e6

    # Create grid and medium
    grid = kw.Grid(nx=N, ny=N, nz=2, dx=dx, dy=dx, dz=dx)
    medium = kw.Medium.homogeneous(sound_speed=c, density=rho)

    # Create source with transducer array
    source_mask = np.zeros((N, N, 2))
    source_mask[:, :, 0] = 1.0  # Source plane at z=0

    # Time signal
    dt = 0.3 * dx / c / np.sqrt(3)
    nt = 200
    t = np.arange(nt) * dt
    signal = 1e5 * np.sin(2 * np.pi * freq * t)

    source = kw.Source.from_mask(source_mask, signal, frequency=freq)

    # Sensor point
    sensor = kw.Sensor.point((N * dx / 2, N * dx / 2, 1e-3))

    # Create and run simulation
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.FDTD)
    result = sim.run(time_steps=nt, dt=dt)

    print(f"Simulation completed!")
    print(f"  Sensor data shape: {result.sensor_data.shape}")
    print(f"  Max pressure: {np.max(np.abs(result.sensor_data)):.2e} Pa")


def example_comparison_with_kwave():
    """Example 7: Conceptual comparison with k-wave-python."""
    print("\n" + "=" * 60)
    print("Example 7: Comparison with k-wave-python")
    print("=" * 60)

    print("k-wave-python approach:")
    print("  1. Create kWaveGrid")
    print("  2. Define kWaveTransducerSimple")
    print("  3. Define NotATransducer with beam parameters")
    print("  4. Run simulation with transducer as source")

    print("\npykwavers approach:")
    print("  1. Create TransducerArray2D")
    print("  2. Configure focus, steering, apodization")
    print("  3. Use with Source.from_mask or similar")
    print("  4. Run simulation")

    print("\nKey differences:")
    print("  - Single class instead of multiple objects")
    print("  - Direct property configuration")
    print("  - Native Rust implementation")
    print("  - Simpler API with same functionality")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("pykwavers 2D Transducer Array Examples")
    print("=" * 60)

    example_basic_array_creation()
    example_beam_steering()
    example_beam_focusing()
    example_apodization()
    example_active_elements()
    example_integration()
    example_comparison_with_kwave()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
