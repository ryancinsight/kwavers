import numpy as np
import pykwavers as kw

def check_mask():
    Nx, Ny, Nz = 64, 64, 64
    dx, dy, dz = 0.5e-3, 0.5e-3, 0.5e-3
    c0 = 1540.0
    rho0 = 1000.0
    
    grid = kw.Grid(Nx, Ny, Nz, dx, dy, dz)
    medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)
    
    # Transducer props
    array = kw.TransducerArray2D(
        16,          # number_elements
        2 * dx,      # element_width
        12 * dz,     # element_length
        2 * dx,      # element_spacing
        c0,          # sound_speed
        1e6          # frequency
    )
    array.set_position(Nx//2 * dx, (Ny//2 - 16) * dy, (Nz//2 - 6) * dz)
    
    # Use array as sensor to check mask size via Simulation
    # Use a tuple for position
    source = kw.Source.point((0.01, 0.01, 0.01), frequency=1e6, amplitude=0)
    sim = kw.Simulation(grid, medium, source=source, sensor=array, pml_size=10)
    
    res = sim.run(time_steps=1)
    print(f"Recorded sensor entries: {res.sensor_data.shape[0]}")
    print(f"Expected (16 * 24): {16 * 24}")

if __name__ == "__main__":
    check_mask()
