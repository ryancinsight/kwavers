"""Quick diagnostic: tone burst comparison with additive_no_correction mode."""
import sys
sys.path.insert(0, 'd:/kwavers/pykwavers/python')
import numpy as np
import pykwavers as kw

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

Nx = Ny = Nz = 64
dx = 1e-3
c0 = 1500.0
rho0 = 1000.0
pml_size = 10
f0 = 0.5e6
n_cycles = 3
t_end = 40e-6
sensor_offsets = [4, 8]

kw_grid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
kw_grid.makeTime(c0, t_end=t_end)
kw_dt = float(kw_grid.dt)
Nt = int(kw_grid.Nt)

input_signal = tone_burst(1.0 / kw_dt, f0, n_cycles).flatten()
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
for off in sensor_offsets:
    sen_mask[Nx//2 + off, Ny//2, Nz//2] = 1
kw_sensor_obj = kSensor(sen_mask)
kw_sensor_obj.record = ["p"]

kw_res = kspaceFirstOrder3D(
    medium=kw_medium, kgrid=kw_grid, source=kw_source_obj, sensor=kw_sensor_obj,
    simulation_options=SimulationOptions(pml_inside=True, pml_size=pml_size,
                                          data_cast="single", save_to_disk=True),
    execution_options=SimulationExecutionOptions(is_gpu_simulation=False),
)
kw_p = kw_res["p"]
if kw_p.ndim == 1:
    kw_p = kw_p.reshape(1, -1)
if kw_p.shape[0] > kw_p.shape[1]:
    kw_p = kw_p.T

kwa_grid = kw.Grid(nx=Nx, ny=Ny, nz=Nz, dx=dx, dy=dx, dz=dx)
kwa_medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)

# Test with additive_no_correction to isolate source_kappa effect
py_src_mask = np.zeros((Nx, Ny, Nz), dtype=np.float64)
py_src_mask[Nx//2, Ny//2, Nz//2] = 1.0

for mode_label, mode in [("additive", "additive"), ("no_correction", "additive_no_correction")]:
    kwa_source_obj = kw.Source.from_mask(py_src_mask, input_signal.copy(), f0, mode=mode)
    py_sen_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    for off in sensor_offsets:
        py_sen_mask[Nx//2+off, Ny//2, Nz//2] = True
    kwa_sensor_obj = kw.Sensor.from_mask(py_sen_mask)
    kwa_sim = kw.Simulation(kwa_grid, kwa_medium, kwa_source_obj, kwa_sensor_obj, solver=kw.SolverType.PSTD)
    kwa_sim.set_pml_size(pml_size)
    kwa_sim.set_pml_inside(True)
    kwa_res = kwa_sim.run(time_steps=Nt, dt=kw_dt)
    kwa_p = kwa_res.sensor_data
    if kwa_p.ndim == 1:
        kwa_p = kwa_p.reshape(1, -1)
    if kwa_p.shape[0] > kwa_p.shape[1]:
        kwa_p = kwa_p.T

    n_kw = kw_p.shape[1]
    n_kwa = kwa_p.shape[1]
    if n_kwa == n_kw + 1:
        kwa_aligned = kwa_p[:, 1:]
        kw_aligned = kw_p
    elif n_kw == n_kwa + 1:
        kw_aligned = kw_p[:, 1:]
        kwa_aligned = kwa_p
    elif n_kw == n_kwa:
        kw_aligned = kw_p[:, 1:]
        kwa_aligned = kwa_p[:, :-1]
    else:
        n = min(n_kw, n_kwa)
        kw_aligned = kw_p[:, :n]
        kwa_aligned = kwa_p[:, :n]

    print(f"\nMode: {mode_label}")
    for i, off in enumerate(sensor_offsets):
        ref = kw_aligned[i].ravel()
        tst = kwa_aligned[i].ravel()
        corr = float(np.corrcoef(ref, tst)[0,1])
        rms_ratio = float(np.sqrt(np.mean(tst**2)) / np.sqrt(np.mean(ref**2)))
        print(f"  Sensor {i+1} (+{off} cells): corr={corr:.4f}, rms_ratio={rms_ratio:.4f}")

