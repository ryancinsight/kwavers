#!/usr/bin/env python3
"""
Validation Test: Plane Wave Propagation in Homogeneous Medium

This test validates pykwavers against k-wave-python for a simple plane wave
propagation scenario in a homogeneous medium.
"""

import numpy as np
import pytest

try:
    import pykwavers as kw
    HAS_PYKWAVERS = True
except ImportError:
    HAS_PYKWAVERS = False

try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_options import SimulationOptions
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.data import Vector
    HAS_KWAVE = True
except ImportError:
    HAS_KWAVE = False


pytestmark = [
    pytest.mark.skipif(not HAS_PYKWAVERS, reason="pykwavers not installed"),
    pytest.mark.skipif(not HAS_KWAVE, reason="k-wave-python not installed"),
]


class TestPlaneWaveValidation:
    """Validation tests comparing pykwavers with k-wave-python."""

    def test_homogeneous_medium_plane_wave_2d(self):
        """Test plane wave propagation in 2D homogeneous medium."""
        nx, ny = 128, 128
        dx = 0.1e-3
        frequency = 1e6
        amplitude = 1e5
        sound_speed = 1500.0
        density = 1000.0

        cfl = 0.3
        dt = cfl * dx / sound_speed
        time_steps = 500

        # pykwavers simulation
        grid_kw = kw.Grid(nx, ny, 1, dx, dx, dx)
        medium_kw = kw.Medium.homogeneous(sound_speed, density)
        source_kw = kw.Source.plane_wave(grid_kw, frequency=frequency, amplitude=amplitude)
        sensor_kw = kw.Sensor.point((nx * dx / 2, ny * dx / 2, 0))

        sim_kw = kw.Simulation(grid_kw, medium_kw, source_kw, sensor_kw, solver=kw.SolverType.PSTD)
        result_kw = sim_kw.run(time_steps=time_steps, dt=dt)

        # k-wave-python simulation
        kgrid = kWaveGrid(Vector([nx, ny, 1]), Vector([dx, dx, dx]))
        kgrid.setTime(time_steps, dt)

        medium = kWaveMedium(sound_speed=sound_speed, density=density)

        source = kSource()
        source.p_mask = np.zeros((nx, ny, 1), dtype=bool)
        source.p_mask[0, :, 0] = True
        t_array = np.arange(0, time_steps * dt, dt)
        source.p = amplitude * np.sin(2 * np.pi * frequency * t_array)

        sensor = kSensor()
        sensor.mask = np.zeros((nx, ny, 1), dtype=bool)
        sensor.mask[nx // 2, ny // 2, 0] = True

        simulation_options = SimulationOptions(pml_auto=True, pml_inside=False)
        execution_options = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0)

        sensor_data = kspaceFirstOrder3D(
            medium=medium,
            kgrid=kgrid,
            source=source,
            sensor=sensor,
            simulation_options=simulation_options,
            execution_options=execution_options,
        )

        result_kwave = sensor_data['p'].flatten()
        result_pykwavers = np.array(result_kw.sensor_data)

        # Validation
        assert result_pykwavers.shape == result_kwave.shape

        # Check correlation
        if np.std(result_pykwavers) > 0 and np.std(result_kwave) > 0:
            correlation = np.corrcoef(result_pykwavers, result_kwave)[0, 1]
            assert correlation > 0.8, f"Signal correlation too low: {correlation:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
