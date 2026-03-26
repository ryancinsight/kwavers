#!/usr/bin/env python3
"""Test kwavers PSTD without PML to isolate source amplitude issue."""
import numpy as np, sys
sys.path.insert(0, 'd:/kwavers/pykwavers/python')

try:
    from pykwavers._pykwavers import Grid, Medium, Source, Sensor, Simulation, SolverType
except Exception as e:
    print(f"Error importing pykwavers: {e}"); sys.exit(1)

N = 32; dx = 0.2e-3; c = 1500.0; rho = 1000.0; freq = 1e6; amp = 1e5
cfl = 0.3; dt = cfl * dx / c; nt = 250

t_arr = np.arange(nt) * dt
signal = amp * np.sin(2*np.pi*freq*t_arr)

# Source: z=0 plane
p_mask = np.zeros((N, N, N)); p_mask[:, :, 0] = 1.0

# Sensor: center
ix, iy, iz = N//2, N//2, N//2
sensor_pos = (ix*dx, iy*dx, iz*dx)

# What the expected steady-state amplitude should be for a plane wave
# with mass source scaling: Δp = 2*c*dt/dx * p_src per step
# At steady state, the sensor sees a sinusoidal signal
# The amplitude should be amp (approximately), adjusted by PML and propagation

print(f"Source amplitude: {amp:.0f} Pa")
print(f"Expected scaling: 2*c*dt/dx = {2*c*dt/dx:.4f}")
print(f"Effective source per step: {2*c*dt/dx * amp:.1f} Pa")
print()

for pml in [0, 6]:
    grid = Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
    medium = Medium.homogeneous(sound_speed=c, density=rho)
    source = Source.from_mask(p_mask.astype(np.float64), signal.flatten(), frequency=freq)
    sensor = Sensor.point(position=sensor_pos)
    sim = Simulation(grid, medium, source, sensor, solver=SolverType.PSTD, pml_size=pml)
    result = sim.run(time_steps=nt, dt=dt)
    p_kwa = np.array(result.sensor_data).flatten()
    p_kwa = p_kwa[1:] if len(p_kwa) > nt else p_kwa

    # Compute RMS in steady-state window
    arr_step = int(16*dx/c/dt)
    w_start = arr_step + 10; w_end = min(nt, w_start + 100)
    rms = np.sqrt(np.mean(p_kwa[w_start:w_end]**2))
    max_abs = np.max(np.abs(p_kwa[w_start:w_end]))

    print(f"pml_size={pml}: max={max_abs:.4e}, rms={rms:.4e} (steps {w_start}-{w_end})")
    if pml == 0:
        p_nopml = p_kwa.copy()
        rms_nopml = rms
    else:
        rms_pml = rms
        print(f"  PML effect: rms ratio = {rms/rms_nopml:.4f}")

# Also run k-Wave for comparison
try:
    from kwave.kgrid import kWaveGrid; from kwave.kmedium import kWaveMedium
    from kwave.ksource import kSource; from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_options import SimulationOptions
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.data import Vector

    sensor_mask = np.zeros((N, N, N)); sensor_mask[ix, iy, iz] = 1.0
    num_sources = int(np.sum(p_mask))

    for pml in [0, 6]:
        kgrid = kWaveGrid(Vector([N, N, N]), Vector([dx, dx, dx]))
        kgrid.setTime(nt, dt)
        km = kWaveMedium(sound_speed=c, density=rho)
        ks = kSource(); ks.p_mask = p_mask.astype(bool)
        ks.p = np.tile(signal.flatten(), (num_sources, 1))
        ksen = kSensor(sensor_mask.astype(bool)); ksen.record = ["p"]
        so = SimulationOptions(pml_inside=True, pml_size=pml, data_cast="single", save_to_disk=True)
        eo = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0, show_sim_log=False)
        rkw = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=ks, sensor=ksen,
                                  simulation_options=so, execution_options=eo)
        p_kw = np.array(rkw["p"]).flatten()

        arr_step = int(16*dx/c/dt)
        w_start = arr_step + 10; w_end = min(nt, w_start + 100)
        rms = np.sqrt(np.mean(p_kw[w_start:w_end]**2))
        max_abs = np.max(np.abs(p_kw[w_start:w_end]))
        print(f"k-Wave pml={pml}: max={max_abs:.4e}, rms={rms:.4e}")

except ImportError:
    print("k-wave not available")
