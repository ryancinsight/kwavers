import pytest
import numpy as np
import pykwavers as kw

def test_pid_controller():
    """Test step response of PID controller binding"""
    pid = kw.PIDController(
        kp=2.0, ki=1.0, kd=0.5, setpoint=1.0, 
        sample_time=0.01, output_min=0.0, output_max=10.0, 
        integral_limit=10.0
    )
    
    measurement = 0.0
    dt = 0.01
    
    # Simple explicit first-order system integration
    for i in range(500):
        # Update returns: (control_signal, p_term, i_term, d_term)
        ctrl, p, i_term, d = pid.update(measurement)
        rate = ctrl - measurement
        measurement += dt * rate
        
        if i > 100 and abs(measurement - 1.0) < 0.01:
            break
            
    assert abs(measurement - 1.0) < 0.1, f"Failed to converge: {measurement}"

def test_bubble_field():
    """Test bubble field native bindings"""
    field = kw.BubbleField(10, 10, 10)
    assert field.num_bubbles() == 0
    
    field.add_center_bubble()
    assert field.num_bubbles() == 1

def test_resample_to_target_grid():
    """Test resampling identical grid mapping with Trilinear interpolator"""
    source = np.ones((4, 4, 4), dtype=np.float64)
    
    # 4x4 Identity transform matrix in 1D list (column-major standard)
    identity = [
        1.0, 0.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 0.0, 
        0.0, 0.0, 1.0, 0.0, 
        0.0, 0.0, 0.0, 1.0
    ]
    
    resampled = kw.resample_to_target_grid(source, identity, (4, 4, 4))
    
    assert resampled.shape == (4, 4, 4)
    np.testing.assert_allclose(resampled, 1.0, atol=1e-10)


def test_kspace_line_recon_zero_input():
    """Test that the line-reconstruction binding preserves the zero field."""
    sensor = np.zeros((6, 4), dtype=np.float64)
    recon = kw.kspace_line_recon(sensor, dy=0.1e-3, dt=1.0e-8, c=1500.0)

    assert recon.shape == (6, 4)
    np.testing.assert_allclose(recon, 0.0, atol=0.0)


def test_time_reversal_reconstruction_zero_input():
    """Test that the time-reversal binding preserves the zero field."""
    grid = kw.Grid(nx=4, ny=3, nz=1, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    sensor_data = np.zeros((3, 8), dtype=np.float64)
    sensor_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.1e-3, 0.0],
            [0.0, 0.2e-3, 0.0],
        ],
        dtype=np.float64,
    )

    recon = kw.time_reversal_reconstruction(
        sensor_data,
        sensor_positions,
        grid,
        sound_speed=1500.0,
        sampling_frequency=1.0e8,
        pml_size=2,
    )

    assert recon.shape == (4, 3, 1)
    np.testing.assert_allclose(recon, 0.0, atol=0.0)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
