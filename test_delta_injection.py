#!/usr/bin/env python3
"""Test 1-step delta injection: compare magnitude directly."""
import numpy as np
import pykwavers as kw
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.data import Vector

N = 32; dx = 1e-3; c = 1500.0; rho = 1000.0; pml_size = 6
cfl = 0.3; dt = cfl * dx / c
A = 1e5  # source amplitude Pa

# Delta signal: inject once at t=0
nt = 5  # Very short simulation

# Create multi-step signal but only 1 nonzero step
signal = np.zeros(nt); signal[0] = A

# ---- Point source (center of domain) ----
# Use center of domain (far from PML)
sx, sy, sz = 16, 16, 16
sensor_pts = [(16,16,14),(16,16,15),(16,16,17),(16,16,18)]

for label, source_mask_fn, source_pts in [
    ("point", lambda: np.array([[[1 if (i==sx and j==sy and k==sz) else 0 
                                   for k in range(N)] for j in range(N)] for i in range(N)], dtype=float), 1),
]:
    sm = source_mask_fn()
    
    # k-Wave  
    kgrid = kWaveGrid(Vector([N,N,N]), Vector([dx,dx,dx]))
    kgrid.setTime(nt, dt)
    km = kWaveMedium(sound_speed=c, density=rho)
    ks = kSource()
    ks.p_mask = sm.astype(bool)
    ks.p = signal.reshape(1, -1)
    ks.p_mode = "additive-no-correction"
    
    # Sensor at source location and nearby
    all_sensors = np.zeros((N,N,N), dtype=bool)
    all_sensors[sx,sy,sz] = True  # source location
    for (xi,yi,zi) in sensor_pts:
        all_sensors[xi,yi,zi] = True
    ksen = kSensor(all_sensors); ksen.record = ["p"]
    so = SimulationOptions(pml_inside=True, pml_size=pml_size, data_cast="single", save_to_disk=True)
    eo = SimulationExecutionOptions(is_gpu_simulation=False, verbose_level=0, show_sim_log=False)
    r = kspaceFirstOrder3D(kgrid=kgrid, medium=km, source=ks, sensor=ksen,
                           simulation_options=so, execution_options=eo)
    p_kw = np.array(r["p"])  # shape (n_sensors, nt)
    
    print(f"\nk-Wave ({label}): source at ({sx},{sy},{sz})")
    print(f"  After step 0 injection: p at source = {p_kw[0,0]:.4e} (expected {A*2*cfl:.4e} = A*2CFL)")
    print(f"  Step 0-4 at source: {p_kw[0,:5]}")
    print(f"  Step 0-4 at (16,16,14): {p_kw[1,:5] if p_kw.shape[0] > 1 else 'N/A'}")

    # kwavers
    grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
    medium = kw.Medium.homogeneous(sound_speed=c, density=rho)
    source = kw.Source.from_mask(sm, signal, frequency=freq if 'freq' in dir() else 0.5e6, 
                                  mode="additive_no_correction")
    sensors_kwa = [kw.Sensor.point(position=(sx*dx, sy*dx, sz*dx))]
    for (xi,yi,zi) in sensor_pts:
        sensors_kwa.append(kw.Sensor.point(position=(xi*dx, yi*dx, zi*dx)))
    
    # Run for source sensor only first
    sensor = kw.Sensor.point(position=(sx*dx, sy*dx, sz*dx))
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD, pml_size=pml_size)
    result = sim.run(time_steps=nt, dt=dt)
    p_kwa = np.array(result.sensor_data).flatten().astype(float)
    if len(p_kwa) > nt: p_kwa = p_kwa[1:]
    
    print(f"\nkwavers ({label}): source at ({sx},{sy},{sz})")
    print(f"  Step 0-4 at source: {p_kwa[:5]}")
    print(f"  Ratio kwa/kw at step 0: {p_kwa[0]/(p_kw[0,0]+1e-30):.4f}")
