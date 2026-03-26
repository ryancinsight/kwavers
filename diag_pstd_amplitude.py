#!/usr/bin/env python3
"""PSTD amplitude diagnostic vs k-Wave for plane wave case."""
import numpy as np, sys

try:
    import pykwavers as kw
except ImportError:
    print("pykwavers not available"); sys.exit(1)

try:
    from kwave.kgrid import kWaveGrid; from kwave.kmedium import kWaveMedium
    from kwave.ksource import kSource; from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_options import SimulationOptions
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.data import Vector; HAS_KWAVE = True
except ImportError:
    HAS_KWAVE = False; print("k-wave not available")

N = 32; dx = 0.2e-3; c = 1500.0; rho = 1000.0; freq = 1e6; amp = 1e5; pml_size = 6
cfl = 0.3; dt = cfl * dx / c; nt = int(10e-6 / dt)  # longer to see steady state

print(f"N={N}, dx={dx:.2e}, dt={dt:.4e}, nt={nt}")
print(f"Wave arrival: step {16*dx/c/dt:.1f}")

p_mask = np.zeros((N, N, N)); p_mask[:, :, 0] = 1.0
num_sources = int(np.sum(p_mask))
ix, iy, iz = N//2, N//2, N//2
sensor_mask = np.zeros((N, N, N)); sensor_mask[ix, iy, iz] = 1.0
sensor_pos = (ix*dx, iy*dx, iz*dx)
t_arr = np.arange(nt) * dt
signal = amp * np.sin(2*np.pi*freq*t_arr)

if HAS_KWAVE:
    kgrid = kWaveGrid(Vector([N, N, N]), Vector([dx, dx, dx]))
    kgrid.setTime(nt, dt)
    km = kWaveMedium(sound_speed=c, density=rho)
    ks = kSource(); ks.p_mask = p_mask.astype(bool)
    ks.p = np.tile(signal.flatten(), (num_sources, 1))
    ksen = kSensor(sensor_mask.astype(bool)); ksen.record = ["p"]
    so = SimulationOptions(pml_inside=True, pml_size=pml_size, data_cast="single", save_to_disk=True)
    eo = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0, show_sim_log=False)
    rkw = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=ks, sensor=ksen,
                              simulation_options=so, execution_options=eo)
    p_kw = np.array(rkw["p"]).flatten()
    print(f"k-Wave shape: {p_kw.shape}, max={np.max(np.abs(p_kw)):.4e}")

grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
medium = kw.Medium.homogeneous(sound_speed=c, density=rho)
source = kw.Source.from_mask(p_mask.astype(np.float64), signal.flatten(), frequency=freq)
sensor = kw.Sensor.point(position=sensor_pos)
sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD, pml_size=pml_size)
result = sim.run(time_steps=nt, dt=dt)
p_kwa_full = np.array(result.sensor_data).flatten()
p_kwa = p_kwa_full[1:] if len(p_kwa_full) > nt else p_kwa_full
print(f"pykwavers shape: {p_kwa.shape}, max={np.max(np.abs(p_kwa)):.4e}")

if HAS_KWAVE:
    n = min(len(p_kw), len(p_kwa))
    p_kw_t = p_kw[:n]; p_kwa_t = p_kwa[:n]
    max_r = np.max(np.abs(p_kwa_t)) / max(np.max(np.abs(p_kw_t)), 1e-15)
    corr = np.corrcoef(p_kw_t, p_kwa_t)[0, 1]
    print(f"\nAlignment: max_ratio={max_r:.4f}, corr={corr:.4f}")

    # Print every 5th step from 50 to 200
    print(f"\n{'step':>5}  {'kwave':>12}  {'kwavers':>12}  {'ratio':>8}")
    for i in range(50, min(n, 200), 5):
        ratio = p_kwa_t[i]/p_kw_t[i] if abs(p_kw_t[i]) > 100 else float('nan')
        print(f"{i:>5}  {p_kw_t[i]:>12.4e}  {p_kwa_t[i]:>12.4e}  {ratio:>8.4f}")

    # Compute RMS in steady-state window (steps 80-200)
    w_start, w_end = 80, min(n, 200)
    rms_kw = np.sqrt(np.mean(p_kw_t[w_start:w_end]**2))
    rms_kwa = np.sqrt(np.mean(p_kwa_t[w_start:w_end]**2))
    print(f"\nSteady-state RMS (steps {w_start}-{w_end}):")
    print(f"  k-Wave RMS:  {rms_kw:.4e}")
    print(f"  kwavers RMS: {rms_kwa:.4e}")
    print(f"  ratio: {rms_kwa/rms_kw:.4f}")
