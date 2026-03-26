"""
Parity test for heterogeneous medium simulations between kwavers and k-wave-python.

This test validates that kwavers produces results within 0.90 correlation to k-wave-python
for heterogeneous media with spatially varying properties.

Reference: IMPLEMENTATION_ROADMAP.md Phase 1 - Priority 1: HeterogeneousMedium
"""

import numpy as np
import pytest

# Import pykwavers
import pykwavers as kw

# Import k-wave-python (if available)
try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.utils.signals import tone_burst
    from kwave.utils.filters import smooth
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options import SimulationOptions, SimulationExecutionOptions
    KWave_AVAILABLE = True
except ImportError:
    KWave_AVAILABLE = False
    pytest.skip("k-wave-python not installed", allow_module_level=True)

from scipy.signal import correlate


def calculate_correlation(array1, array2):
    """Calculate normalized correlation between two arrays."""
    if array1.shape != array2.shape:
        raise ValueError(f"Shape mismatch: {array1.shape} vs {array2.shape}")
    
    # Flatten arrays
    a1 = array1.flatten()
    a2 = array2.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(a1) | np.isnan(a2))
    a1 = a1[mask]
    a2 = a2[mask]
    
    if len(a1) == 0:
        return 0.0
    
    # Calculate Pearson correlation
    correlation = np.corrcoef(a1, a2)[0, 1]
    
    return correlation if not np.isnan(correlation) else 0.0


class TestHeterogeneousMediumParity:
    """Test heterogeneous medium parity between kwavers and k-wave-python."""
    
    def test_simple_layered_medium(self):
        """
        Test a simple layered medium: water (z<5mm) and tissue (z>=5mm).
        
        Expected correlation: > 0.90
        """
        # Grid parameters
        nx, ny, nz = 64, 64, 64
        dx = dy = dz = 0.1e-3  # 0.1 mm grid spacing
        
        # Create grids
        kw_grid = kw.Grid(nx, ny, nz, dx, dy, dz)
        kgrid = kWaveGrid([nx, ny, nz], [dx, dy, dz])
        
        # Create layered medium
        # Layer 1 (z < 5mm): Water
        # Layer 2 (z >= 5mm): Soft tissue
        z_positions = np.arange(nz) * dz
        
        # kwavers medium (using arrays)
        c_kwavers = np.ones((nx, ny, nz)) * 1500.0
        rho_kwavers = np.ones((nx, ny, nz)) * 1000.0
        alpha_kwavers = np.zeros((nx, ny, nz))
        
        # Apply tissue properties to upper half
        tissue_mask = z_positions >= 5e-3
        c_kwavers[:, :, tissue_mask] = 1540.0
        rho_kwavers[:, :, tissue_mask] = 1060.0
        alpha_kwavers[:, :, tissue_mask] = 0.5
        
        kwavers_medium = kw.Medium(c_kwavers, rho_kwavers, alpha_kwavers)
        
        # k-wave medium
        kmedium = kWaveMedium(
            sound_speed=c_kwavers,
            density=rho_kwavers,
            alpha_coeff=alpha_kwavers,
            alpha_power=1.0
        )
        
        # Create point source in water layer
        source_pos = [0.002, 0.002, 0.002]  # 2mm depth in water
        frequency = 1e6  # 1 MHz
        
        # kwavers source
        kwavers_source = kw.Source.point(source_pos, frequency, amplitude=1e5)
        
        # k-wave source
        ksource = kSource()
        ksource.p_mask = np.zeros((nx, ny, nz), dtype=bool)
        ksource.p_mask[
            int(source_pos[0]/dx), 
            int(source_pos[1]/dy), 
            int(source_pos[2]/dz)
        ] = True
        ksource.p = tone_burst(1/kgrid.dt, frequency, 5).reshape(1, -1)
        
        # Create sensors
        sensor_pos = [0.006, 0.006, 0.008]  # In tissue layer
        
        # kwavers sensor
        kwavers_sensor = kw.Sensor.point(sensor_pos)
        
        # k-wave sensor
        ksensor = kSensor()
        ksensor.mask = np.zeros((nx, ny, nz), dtype=bool)
        ksensor.mask[
            int(sensor_pos[0]/dx),
            int(sensor_pos[1]/dy),
            int(sensor_pos[2]/dz)
        ] = True
        
        # Run simulations
        # kwavers
        kwavers_sim = kw.Simulation(kw_grid, kwavers_medium, kwavers_source, kwavers_sensor)
        kwavers_result = kwavers_sim.run(time_steps=500, dt=1e-8)
        
        # k-wave
        sim_options = SimulationOptions(
            pml_size=10,
            pml_alpha=2.0,
            save_to_disk=False
        )
        execution_options = SimulationExecutionOptions(
            is_gpu_simulation=False
        )
        
        ksensor_data = kspaceFirstOrder3D(
            kgrid=kgrid,
            medium=kmedium,
            source=ksource,
            sensor=ksensor,
            simulation_options=sim_options,
            execution_options=execution_options
        )
        
        # Compare results
        kwavers_signal = kwavers_result.sensor_data.flatten()
        kwave_signal = ksensor_data['p'].flatten()
        
        correlation = calculate_correlation(kwavers_signal, kwave_signal)
        
        print(f"Layered medium correlation: {correlation:.4f}")
        assert correlation > 0.90, f"Correlation {correlation:.4f} below threshold 0.90"
    
    def test_gradient_medium(self):
        """
        Test a medium with linear gradient in sound speed.
        
        Expected correlation: > 0.90
        """
        # Grid parameters
        nx, ny, nz = 48, 48, 48
        dx = dy = dz = 0.15e-3
        
        # Create grids
        kw_grid = kw.Grid(nx, ny, nz, dx, dy, dz)
        kgrid = kWaveGrid([nx, ny, nz], [dx, dy, dz])
        
        # Create gradient medium: c varies linearly with depth
        z_positions = np.arange(nz) * dz
        c_base = 1500.0
        c_gradient = 500.0  # Increase by 500 m/s over depth
        
        c_array = np.zeros((nx, ny, nz))
        for k in range(nz):
            c_array[:, :, k] = c_base + (z_positions[k] / (nz * dz)) * c_gradient
        
        rho_array = np.ones((nx, ny, nz)) * 1000.0
        
        # kwavers medium
        kwavers_medium = kw.Medium(c_array, rho_array)
        
        # k-wave medium
        kmedium = kWaveMedium(
            sound_speed=c_array,
            density=rho_array,
            alpha_coeff=0.0,
            alpha_power=1.0
        )
        
        # Plane wave source
        frequency = 500e3  # 500 kHz
        
        kwavers_source = kw.Source.plane_wave(kw_grid, frequency, amplitude=1e5)
        
        ksource = kSource()
        # Create plane wave at z=0
        ksource.p_mask = np.zeros((nx, ny, nz), dtype=bool)
        ksource.p_mask[:, :, 0] = True
        ksource.p = tone_burst(1/kgrid.dt, frequency, 3).reshape(1, -1)
        
        # Line sensor through center
        kwavers_sensor = kw.Sensor.grid()  # Record full field
        
        ksensor = kSensor()
        ksensor.mask = np.ones((nx, ny, nz), dtype=bool)
        
        # Run simulations
        kwavers_sim = kw.Simulation(kw_grid, kwavers_medium, kwavers_source, kwavers_sensor)
        kwavers_result = kwavers_sim.run(time_steps=400, dt=2e-8)
        
        sim_options = SimulationOptions(
            pml_size=10,
            pml_alpha=2.0,
            save_to_disk=False
        )
        execution_options = SimulationExecutionOptions(
            is_gpu_simulation=False
        )
        
        ksensor_data = kspaceFirstOrder3D(
            kgrid=kgrid,
            medium=kmedium,
            source=ksource,
            sensor=ksensor,
            simulation_options=sim_options,
            execution_options=execution_options
        )
        
        # Compare center line
        center_x, center_y = nx // 2, ny // 2
        kwavers_line = kwavers_result.pressure[center_x, center_y, :, :]
        kwave_line = ksensor_data['p'].reshape(nx, ny, nz, -1)[center_x, center_y, :, :]
        
        correlation = calculate_correlation(kwavers_line, kwave_line)
        
        print(f"Gradient medium correlation: {correlation:.4f}")
        assert correlation > 0.90, f"Correlation {correlation:.4f} below threshold 0.90"
    
    def test_inclusion_medium(self):
        """
        Test a medium with a spherical inclusion (tumor-like structure).
        
        Expected correlation: > 0.90
        """
        # Grid parameters
        nx, ny, nz = 56, 56, 56
        dx = dy = dz = 0.12e-3
        
        # Create grids
        kw_grid = kw.Grid(nx, ny, nz, dx, dy, dz)
        kgrid = kWaveGrid([nx, ny, nz], [dx, dy, dz])
        
        # Background: soft tissue
        c_array = np.ones((nx, ny, nz)) * 1540.0
        rho_array = np.ones((nx, ny, nz)) * 1060.0
        alpha_array = np.ones((nx, ny, nz)) * 0.5
        
        # Create spherical inclusion (tumor)
        center = np.array([nx, ny, nz]) // 2
        radius = 10  # grid points
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                    if dist <= radius:
                        c_array[i, j, k] = 1600.0  # Tumor has higher sound speed
                        rho_array[i, j, k] = 1100.0
                        alpha_array[i, j, k] = 1.0
        
        # Smooth the interfaces
        c_array = smooth(c_array, 2)
        rho_array = smooth(rho_array, 2)
        
        # kwavers medium
        kwavers_medium = kw.Medium(c_array, rho_array, alpha_array)
        
        # k-wave medium
        kmedium = kWaveMedium(
            sound_speed=c_array,
            density=rho_array,
            alpha_coeff=alpha_array,
            alpha_power=1.0
        )
        
        # Point source outside inclusion
        source_pos = [0.002, 0.002, 0.002]
        frequency = 2e6  # 2 MHz
        
        kwavers_source = kw.Source.point(source_pos, frequency, amplitude=1e5)
        
        ksource = kSource()
        ksource.p_mask = np.zeros((nx, ny, nz), dtype=bool)
        ksource.p_mask[
            int(source_pos[0]/dx),
            int(source_pos[1]/dy),
            int(source_pos[2]/dz)
        ] = True
        ksource.p = tone_burst(1/kgrid.dt, frequency, 4).reshape(1, -1)
        
        # Sensors at multiple positions
        sensor_positions = [
            [0.003, 0.003, 0.003],
            [0.004, 0.004, 0.004],
            [center[0]*dx, center[1]*dy, center[2]*dz]
        ]
        
        # Create mask sensor
        sensor_mask = np.zeros((nx, ny, nz), dtype=bool)
        for pos in sensor_positions:
            i, j, k = int(pos[0]/dx), int(pos[1]/dy), int(pos[2]/dz)
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                sensor_mask[i, j, k] = True
        
        kwavers_sensor = kw.Sensor.from_mask(sensor_mask)
        
        ksensor = kSensor()
        ksensor.mask = sensor_mask
        
        # Run simulations
        kwavers_sim = kw.Simulation(kw_grid, kwavers_medium, kwavers_source, kwavers_sensor)
        kwavers_result = kwavers_sim.run(time_steps=600, dt=1e-8)
        
        sim_options = SimulationOptions(
            pml_size=10,
            pml_alpha=2.0,
            save_to_disk=False
        )
        execution_options = SimulationExecutionOptions(
            is_gpu_simulation=False
        )
        
        ksensor_data = kspaceFirstOrder3D(
            kgrid=kgrid,
            medium=kmedium,
            source=ksource,
            sensor=ksensor,
            simulation_options=sim_options,
            execution_options=execution_options
        )
        
        # Compare sensor data
        kwavers_signals = kwavers_result.sensor_data
        kwave_signals = ksensor_data['p']
        
        correlation = calculate_correlation(kwavers_signals, kwave_signals)
        
        print(f"Inclusion medium correlation: {correlation:.4f}")
        assert correlation > 0.90, f"Correlation {correlation:.4f} below threshold 0.90"


@pytest.mark.skipif(not KWave_AVAILABLE, reason="k-wave-python not installed")
def test_heterogeneous_medium_basic():
    """Basic test that heterogeneous medium can be created and used."""
    nx, ny, nz = 32, 32, 32
    dx = dy = dz = 0.2e-3
    
    grid = kw.Grid(nx, ny, nz, dx, dy, dz)
    
    # Create simple heterogeneous medium
    c = np.ones((nx, ny, nz)) * 1500.0
    c[16:, :, :] = 1600.0  # Lower half has higher sound speed
    
    rho = np.ones((nx, ny, nz)) * 1000.0
    rho[16:, :, :] = 1050.0
    
    medium = kw.Medium(c, rho)
    
    # Verify medium properties
    assert not medium.is_homogeneous
    assert medium.sound_speed == 1600.0  # max sound speed
    
    # Create source and sensor
    source = kw.Source.point((0.001, 0.001, 0.001), 1e6, 1e5)
    sensor = kw.Sensor.point((0.003, 0.003, 0.003))
    
    # Create and run simulation
    sim = kw.Simulation(grid, medium, source, sensor)
    result = sim.run(time_steps=100, dt=1e-8)
    
    # Verify we got results
    assert result.sensor_data is not None
    assert len(result.sensor_data) > 0


if __name__ == "__main__":
    # Run tests
    if KWave_AVAILABLE:
        print("Running heterogeneous medium parity tests...")
        test_class = TestHeterogeneousMediumParity()
        
        try:
            test_class.test_simple_layered_medium()
            print("✓ Layered medium test passed")
        except Exception as e:
            print(f"✗ Layered medium test failed: {e}")
        
        try:
            test_class.test_gradient_medium()
            print("✓ Gradient medium test passed")
        except Exception as e:
            print(f"✗ Gradient medium test failed: {e}")
        
        try:
            test_class.test_inclusion_medium()
            print("✓ Inclusion medium test passed")
        except Exception as e:
            print(f"✗ Inclusion medium test failed: {e}")
    else:
        print("k-wave-python not installed, running basic test only...")
        test_heterogeneous_medium_basic()
        print("✓ Basic heterogeneous medium test passed")
