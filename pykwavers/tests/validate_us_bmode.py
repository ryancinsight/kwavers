import numpy as np
import pykwavers as kw
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ktransducer import kWaveTransducerSimple, NotATransducer
from kwave.options.simulation_options import SimulationOptions
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.data import Vector
from kwave.utils.signals import tone_burst
import logging

def test_us_bmode_linear_transducer_parity():
    # Simulation settings
    Nx, Ny, Nz = 64, 64, 64
    dx, dy, dz = 0.5e-3, 0.5e-3, 0.5e-3
    c0 = 1540.0
    rho0 = 1000.0
    
    # 1. k-Wave setup
    kw_grid = kWaveGrid([Nx, Ny, Nz], [dx, dy, dz])
    t_end = 20e-6
    kw_grid.makeTime(c0, t_end=t_end, cfl=0.3)
    Nt = kw_grid.Nt
    dt = kw_grid.dt
    
    kw_medium = kWaveMedium(sound_speed=c0, density=rho0)
    
    # Input signal - Normalized to 1.0 peak
    freq = 1e6
    cycles = 4
    input_signal_raw = tone_burst(1/dt, freq, cycles).flatten()
    
    # k-Wave mass source injection
    # source.p = (source_strength / (c0 * rho0)) * input_signal
    source_strength = 1e6
    kw_input = (source_strength / (c0 * rho0)) * input_signal_raw
    
    # Transducer props - Centered in all dimensions
    transducer_props = {
        'number_elements': 16,
        'element_width': 2,
        'element_length': 12,
        'element_spacing': 0,
        'radius': float('inf'),
        'position': [Nx//2, Ny//2 - 16, Nz//2 - 6]
    }
    kw_transducer = kWaveTransducerSimple(kw_grid, **transducer_props)
    
    not_transducer_props = {
        'sound_speed': c0,
        'focus_distance': 20e-3,
        'elevation_focus_distance': 19e-3,
        'steering_angle': 0,
        'transmit_apodization': 'Hanning',
        'receive_apodization': 'Rectangular',
        'active_elements': np.ones((16, 1)),
        'input_signal': kw_input
    }
    kw_not_transducer = NotATransducer(kw_transducer, kw_grid, **not_transducer_props)
    
    # Run k-Wave
    from kwave.options.simulation_options import SimulationOptions
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    
    sim_opts = SimulationOptions(
        data_cast="double",
        save_to_disk=True,
        smooth_p0=False,
        pml_inside=True,
        pml_size=10
    )
    exec_opts = SimulationExecutionOptions(num_threads=24)

    print("Running k-Wave-python simulation...")
    kw_not_transducer.record = ["p"]
    kw_sensor_data = kspaceFirstOrder3D(
        kgrid=kw_grid, 
        source=kw_not_transducer, 
        sensor=kw_not_transducer, 
        medium=kw_medium, 
        simulation_options=sim_opts,
        execution_options=exec_opts
    )
    
    if isinstance(kw_sensor_data, dict):
        kw_sensor_data = kw_sensor_data['p']
    
    kw_sensor_data = kw_sensor_data.T
    
    # 2. pykwavers setup
    kwa_grid = kw.Grid(nx=Nx, ny=Ny, nz=Nz, dx=dx, dy=dy, dz=dz)
    kwa_medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)
    
    kwa_array = kw.TransducerArray2D(
        number_elements=16,
        element_width=2*dx,
        element_length=12*dz,
        element_spacing=2*dx,
        sound_speed=c0,
        frequency=freq
    )
    
    # Match k-Wave transducer position EXACTLY:
    # kw_transducer position = [Nx//2, Ny//2-16, Nz//2-6] in grid indices
    # Elements spread along X (azimuthal), depth propagation in +Y
    pos_x = (Nx // 2) * dx          # azimuthal center
    pos_y = (Ny // 2 - 16) * dy     # front face of transducer (depth start)
    pos_z = (Nz // 2 - 6) * dz      # elevation center offset
    kwa_array.set_position(pos_x, pos_y, pos_z)
    
    kwa_array.set_focus_distance(20e-3)
    kwa_array.set_transmit_apodization("Hanning")
    kwa_array.set_receive_apodization("Rectangular")
    
    # PSTD Source Scaling:
    # The PSTD stepper internally applies mass_source_scale = 2*dt / (N*c0*dx_min).
    # Do NOT pre-scale by 2*dt*c0/dx — that causes double-division.
    # Positive sign: PSTD mass source convention matches k-Wave's additive density source.
    kwa_input_final = input_signal_raw.astype(np.float64) * source_strength


    
    kwa_array.set_input_signal(kwa_input_final)
    
    print("Running pykwavers simulation...")
    sim = kw.Simulation(kwa_grid, kwa_medium, source=kwa_array, sensor=kwa_array, solver=kw.SolverType.PSTD, pml_size=10)
    res = sim.run(time_steps=Nt, dt=dt)
    kwa_sensor_raw = res.sensor_data
    print(f"k-Wave sensor data shape:   {kw_sensor_data.shape}")
    print(f"pykwavers sensor data shape: {kwa_sensor_raw.shape}")

    n_elements = 16
    kw_nodes_per_elem  = kw_sensor_data.shape[0] // n_elements
    kwa_nodes_per_elem = kwa_sensor_raw.shape[0] // n_elements
    print(f"k-Wave nodes/elem: {kw_nodes_per_elem}  kwa nodes/elem: {kwa_nodes_per_elem}")

    # Save raw data for diagnostics
    np.save("kw_res_raw.npy",  kw_sensor_data)
    np.save("kwa_res_raw.npy", kwa_sensor_raw)
    
    # 3. Post-process — average over each element's nodes
    kw_sensor_processed  = np.zeros((n_elements, Nt))
    kwa_sensor_processed = np.zeros((n_elements, Nt))
    
    for i in range(n_elements):
        kw_s  = i * kw_nodes_per_elem
        kw_e  = kw_s + kw_nodes_per_elem
        kwa_s = i * kwa_nodes_per_elem
        kwa_e = kwa_s + kwa_nodes_per_elem
        kw_sensor_processed[i, :]  = np.mean(kw_sensor_data[kw_s:kw_e, :], axis=0)
        kwa_sensor_processed[i, :] = np.mean(kwa_sensor_raw[kwa_s:kwa_e, :], axis=0)
    
    # 4. Compare
    print("\nParity Check (Averaged Elements):")
    max_kw = np.max(np.abs(kw_sensor_processed))
    max_kwa = np.max(np.abs(kwa_sensor_processed))
    print(f"Max Amplitude (k-Wave):    {max_kw:.6e}")
    print(f"Max Amplitude (pykwavers): {max_kwa:.6e}")
    
    if np.isnan(max_kwa):
        print("FAILURE: pykwavers returned NaNs")
    else:
        diff = np.abs(kw_sensor_processed - kwa_sensor_processed)
        max_diff = np.max(diff)
        rel_diff = max_diff / max_kw if max_kw > 0 else 0
        print(f"Max Abs Difference:        {max_diff:.6e}")
        print(f"Max Rel Difference:        {rel_diff:.6e}")
        
        # Check correlation
        correlation = np.corrcoef(kw_sensor_processed.flatten(), kwa_sensor_processed.flatten())[0, 1]
        print(f"Correlation:               {correlation:.6f}")
        
    np.save("kw_res_final.npy", kw_sensor_processed)
    np.save("kwa_res_final.npy", kwa_sensor_processed)

if __name__ == "__main__":
    test_us_bmode_linear_transducer_parity()
