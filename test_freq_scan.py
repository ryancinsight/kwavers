"""Frequency scan to understand if dispersion is ppw-dependent."""
import sys
sys.path.insert(0, 'd:/kwavers/pykwavers/python')
import numpy as np

try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
    from kwave.utils.signals import tone_burst
except ImportError as e:
    print(f"k-wave not available: {e}")
    sys.exit(1)

import pykwavers as kw

Nx = Ny = Nz = 64
dx = 1e-3
c0 = 1500.0
rho0 = 1000.0
pml_size = 10
n_cycles = 5

# Test at two frequencies
# f0=0.05 MHz: ppw = c/(f*dx) = 1500/(0.05e6*1e-3) = 30 ppw
# f0=0.5 MHz: ppw = 1500/(0.5e6*1e-3) = 3 ppw

# Use a fixed dt that works for both
c_dt_dx = 0.3  # CFL
dt = c_dt_dx * dx / c0  # = 0.3 * 1e-3 / 1500 = 2e-7 s
t_end = 30e-6  # 150 steps

Nt = int(t_end / dt)
print(f"dt={dt:.4e} s, Nt={Nt}")

# Pre-run k-Wave ONCE with f0=0.5 MHz (saves time)
# We'll reuse for both frequencies since k-Wave takes ages

def run_comparison(f0, label):
    ppw = c0 / (f0 * dx)
    print(f"\n--- {label} (f0={f0/1e6:.2f} MHz, ppw={ppw:.1f}) ---")
    
    kw_grid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
    # Override dt and Nt to match kwavers
    kw_grid.t_array = np.arange(0, Nt+1) * dt  # from 0 to (Nt-1)*dt  # Nt steps from dt to Nt*dt
    
    input_signal = tone_burst(1.0 / dt, f0, n_cycles).flatten()
    if len(input_signal) < Nt:
        input_signal = np.pad(input_signal, (0, Nt - len(input_signal)))
    else:
        input_signal = input_signal[:Nt]

    kw_medium = kWaveMedium(sound_speed=c0, density=rho0, alpha_coeff=0.0, alpha_power=1.5)
    src_mask = np.zeros((Nx, Ny, Nz))
    src_mask[Nx//2, Ny//2, Nz//2] = 1
    kw_source_obj = kSource()
    kw_source_obj.p_mask = src_mask
    kw_source_obj.p = input_signal.reshape(1, -1)
    kw_source_obj.p_mode = "additive"

    sen_mask = np.zeros((Nx, Ny, Nz))
    sen_mask[Nx//2 + 4, Ny//2, Nz//2] = 1
    kw_sensor_obj = kSensor(sen_mask)
    kw_sensor_obj.record = ["p"]

    kw_res = kspaceFirstOrder3D(
        medium=kw_medium, kgrid=kw_grid, source=kw_source_obj, sensor=kw_sensor_obj,
        simulation_options=SimulationOptions(pml_inside=True, pml_size=pml_size,
                                              data_cast="single", save_to_disk=True),
        execution_options=SimulationExecutionOptions(is_gpu_simulation=False),
    )
    kw_p = kw_res["p"]
    if kw_p.ndim == 1: kw_p = kw_p.reshape(1, -1)
    if kw_p.shape[0] > kw_p.shape[1]: kw_p = kw_p.T

    kwa_grid = kw.Grid(nx=Nx, ny=Ny, nz=Nz, dx=dx, dy=dx, dz=dx)
    kwa_medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)
    py_src_mask = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    py_src_mask[Nx//2, Ny//2, Nz//2] = 1.0
    kwa_source_obj = kw.Source.from_mask(py_src_mask, input_signal.copy(), f0, mode="additive")
    py_sen_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    py_sen_mask[Nx//2+4, Ny//2, Nz//2] = True
    kwa_sensor_obj = kw.Sensor.from_mask(py_sen_mask)
    kwa_sim = kw.Simulation(kwa_grid, kwa_medium, kwa_source_obj, kwa_sensor_obj, solver=kw.SolverType.PSTD)
    kwa_sim.set_pml_size(pml_size)
    kwa_sim.set_pml_inside(True)
    kwa_res = kwa_sim.run(time_steps=Nt, dt=dt)
    kwa_p = kwa_res.sensor_data
    if kwa_p.ndim == 1: kwa_p = kwa_p.reshape(1, -1)
    if kwa_p.shape[0] > kwa_p.shape[1]: kwa_p = kwa_p.T

    n_kw = kw_p.shape[1]
    n_kwa = kwa_p.shape[1]
    if n_kwa == n_kw + 1: kw_a, kwa_a = kw_p, kwa_p[:, 1:]
    elif n_kw == n_kwa + 1: kw_a, kwa_a = kw_p[:, 1:], kwa_p
    elif n_kw == n_kwa: kw_a, kwa_a = kw_p[:, 1:], kwa_p[:, :-1]
    else: n = min(n_kw, n_kwa); kw_a, kwa_a = kw_p[:, :n], kwa_p[:, :n]

    ref = kw_a[0].ravel()
    tst = kwa_a[0].ravel()
    corr = float(np.corrcoef(ref, tst)[0,1]) if np.std(ref)>1e-30 and np.std(tst)>1e-30 else 0
    rms_r = float(np.sqrt(np.mean(tst**2)) / np.sqrt(np.mean(ref**2)))
    amp_r = float(np.max(np.abs(tst)) / np.max(np.abs(ref)))
    print(f"  Sensor +4: corr={corr:.4f}, rms_ratio={rms_r:.4f}, amp_ratio={amp_r:.4f}")

run_comparison(0.05e6, "Low freq (30 ppw)")
run_comparison(0.5e6, "High freq (3 ppw)")
