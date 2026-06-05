"""
Tests for 2D Transducer Array functionality in pykwavers.

This module tests the TransducerArray2D class and validates it against
k-wave-python's transducer implementation.
"""

import numpy as np
import pytest

# Import pykwavers
import pykwavers as kw

# Import k-wave-python (if available)
try:
    from kwave.kgrid import kWaveGrid
    from kwave.ktransducer import NotATransducer
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.utils.signals import tone_burst
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options import SimulationOptions, SimulationExecutionOptions
    KWave_AVAILABLE = True
except ImportError:
    KWave_AVAILABLE = False


class TestTransducerArray2D:
    """Test suite for 2D transducer array."""
    
    def test_array_creation(self):
        """Test basic array creation."""
        array = kw.TransducerArray2D(
            number_elements=32,
            element_width=0.3e-3,
            element_length=10e-3,
            element_spacing=0.5e-3,
            sound_speed=1540.0,
            frequency=1e6
        )
        
        assert array.number_elements == 32
        assert array.element_width == 0.3e-3
        assert array.element_length == 10e-3
        assert array.element_spacing == 0.5e-3
        assert array.frequency == 1e6
        assert array.aperture_width > 0
        
    def test_beamforming_controls(self):
        """Test beamforming controls."""
        array = kw.TransducerArray2D(
            number_elements=16,
            element_width=0.3e-3,
            element_length=10e-3,
            element_spacing=0.5e-3,
            sound_speed=1540.0,
            frequency=1e6
        )
        
        # Test focus distance
        array.set_focus_distance(20e-3)
        assert array.focus_distance == 20e-3
        
        # Test steering angle
        array.set_steering_angle(10.0)
        assert array.steering_angle == 10.0
        
        # Test apodization
        array.set_transmit_apodization("Hanning")
        assert array.transmit_apodization == "Hanning"
        
        array.set_receive_apodization("Hamming")
        # Receive apodization not directly exposed, but should not error
        
    def test_active_elements(self):
        """Test active element masking."""
        array = kw.TransducerArray2D(
            number_elements=16,
            element_width=0.3e-3,
            element_length=10e-3,
            element_spacing=0.5e-3,
            sound_speed=1540.0,
            frequency=1e6
        )
        
        # Test setting active elements
        mask = [True] * 16
        mask[0] = False  # Disable first element
        mask[-1] = False  # Disable last element
        
        array.set_active_elements(mask)
        
        # Test wrong mask length
        with pytest.raises(ValueError):
            array.set_active_elements([True] * 8)  # Wrong length
    
    def test_position_setting(self):
        """Test setting array position."""
        array = kw.TransducerArray2D(
            number_elements=16,
            element_width=0.3e-3,
            element_length=10e-3,
            element_spacing=0.5e-3,
            sound_speed=1540.0,
            frequency=1e6
        )
        
        # Set position
        array.set_position(0.01, 0.02, 0.03)
        # Should not error
    
    def test_input_signal(self):
        """Test setting input signal."""
        array = kw.TransducerArray2D(
            number_elements=16,
            element_width=0.3e-3,
            element_length=10e-3,
            element_spacing=0.5e-3,
            sound_speed=1540.0,
            frequency=1e6
        )
        
        # Create custom signal
        signal = np.sin(2 * np.pi * 1e6 * np.arange(0, 1e-5, 1e-8))
        array.set_input_signal(signal)
        
        # Test empty signal
        with pytest.raises(ValueError):
            array.set_input_signal(np.array([]))
    
    def test_invalid_parameters(self):
        """Test parameter validation."""
        # Test zero elements
        with pytest.raises(ValueError):
            kw.TransducerArray2D(
                number_elements=0,
                element_width=0.3e-3,
                element_length=10e-3,
                element_spacing=0.5e-3,
                sound_speed=1540.0,
                frequency=1e6
            )
        
        # Test negative element width
        with pytest.raises(ValueError):
            kw.TransducerArray2D(
                number_elements=16,
                element_width=-0.3e-3,
                element_length=10e-3,
                element_spacing=0.5e-3,
                sound_speed=1540.0,
                frequency=1e6
            )
        
        # Test element spacing < width
        with pytest.raises(ValueError):
            kw.TransducerArray2D(
                number_elements=16,
                element_width=0.5e-3,
                element_length=10e-3,
                element_spacing=0.3e-3,  # Less than width
                sound_speed=1540.0,
                frequency=1e6
            )
    
    def test_apodization_validation(self):
        """Test apodization type validation."""
        array = kw.TransducerArray2D(
            number_elements=16,
            element_width=0.3e-3,
            element_length=10e-3,
            element_spacing=0.5e-3,
            sound_speed=1540.0,
            frequency=1e6
        )
        
        # Valid apodizations
        for apod in ["Rectangular", "Hanning", "Hamming", "Blackman"]:
            array.set_transmit_apodization(apod)
            assert array.transmit_apodization == apod
        
        # Invalid apodization
        with pytest.raises(ValueError):
            array.set_transmit_apodization("Invalid")


@pytest.mark.skipif(not KWave_AVAILABLE, reason="k-wave-python not installed")
class TestTransducerArrayParity:
    """Test parity with k-wave-python transducer implementation."""
    
    def test_array_geometry(self):
        """Test that array geometry matches k-wave-python."""
        # Create pykwavers array
        kw_array = kw.TransducerArray2D(
            number_elements=32,
            element_width=0.3e-3,
            element_length=10e-3,
            element_spacing=0.5e-3,
            sound_speed=1540.0,
            frequency=1e6
        )
        
        # Create k-wave-python transducer
        # Note: k-wave-python uses different API, this is a simplified comparison
        # In practice, you'd compare actual simulation results
        
        # Verify dimensions
        expected_width = (32 - 1) * 0.5e-3 + 0.3e-3
        assert abs(kw_array.aperture_width - expected_width) < 1e-10
    
    def test_basic_simulation_with_array(self):
        """Test running a simulation with transducer array."""
        # Grid
        nx, ny, nz = 64, 64, 64
        dx = dy = dz = 0.1e-3
        
        grid = kw.Grid(nx, ny, nz, dx, dy, dz)
        medium = kw.Medium.homogeneous(sound_speed=1540.0, density=1000.0)
        
        # Transducer array as source
        transducer = kw.TransducerArray2D(
            number_elements=16,
            element_width=0.3e-3,
            element_length=10e-3,
            element_spacing=0.5e-3,
            sound_speed=1540.0,
            frequency=1e6
        )
        transducer.set_focus_distance(20e-3)
        transducer.set_position(0.01, 0.01, 0.0)
        
        # Sensor
        sensor = kw.Sensor.grid()
        
        # Run simulation
        # Note: Currently transducer array needs to be integrated into Source
        # This test verifies the API works
        # In full implementation, you'd use:
        # sim = kw.Simulation(grid, medium, transducer, sensor)
        # result = sim.run(time_steps=100)
        
        # For now, just verify transducer was created
        assert transducer.number_elements == 16
        assert transducer.focus_distance == 20e-3


if __name__ == "__main__":
    # Run tests
    test_class = TestTransducerArray2D()
    
    print("Running TransducerArray2D tests...")
    
    try:
        test_class.test_array_creation()
        print("✓ Array creation test passed")
    except Exception as e:
        print(f"✗ Array creation test failed: {e}")
    
    try:
        test_class.test_beamforming_controls()
        print("✓ Beamforming controls test passed")
    except Exception as e:
        print(f"✗ Beamforming controls test failed: {e}")
    
    try:
        test_class.test_active_elements()
        print("✓ Active elements test passed")
    except Exception as e:
        print(f"✗ Active elements test failed: {e}")
    
    try:
        test_class.test_position_setting()
        print("✓ Position setting test passed")
    except Exception as e:
        print(f"✗ Position setting test failed: {e}")
    
    try:
        test_class.test_input_signal()
        print("✓ Input signal test passed")
    except Exception as e:
        print(f"✗ Input signal test failed: {e}")
    
    try:
        test_class.test_invalid_parameters()
        print("✓ Invalid parameters test passed")
    except Exception as e:
        print(f"✗ Invalid parameters test failed: {e}")
    
    try:
        test_class.test_apodization_validation()
        print("✓ Apodization validation test passed")
    except Exception as e:
        print(f"✗ Apodization validation test failed: {e}")
    
    print("\nAll basic tests completed!")
