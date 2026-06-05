"""
Source amplitude diagnostic: compare kwavers vs k-Wave at very early time steps.

Runs both sims with Nt=25 steps (wave hasn't reached sensors yet or just arriving).
Prints raw sensor data to identify whether source scaling is correct.
"""
import numpy as np
import sys

try:
    import pykwavers as kw
except ImportError:
    print("ERROR: pykwavers not found"); sys.exit(1)

try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
    from kwave.utils.signals import tone_burst
except ImportError:
    print("ERROR: k-wave-python not found"); sys.exit(1)

Nx = 64; dx = 1e-3; c0 = 1500.0; rho0 = 1000.0; pml_size = 10
f0 = 0.5e6; n_cycles = 3
# Use k-Wave's dt for consistency
kw_grid_tmp = kWaveGrid([Nx, Nx, Nx], [dx, dx, dx])
kw_grid_tmp.makeTime(c0, t_end=40e-6)
kw_dt = float(kw_grid_tmp.dt)
Nt_full = int(kw_grid_tmp.Nt)  # 202

Nt_short = 25  # Just enough to see source effect at sensor 1 (4 cells, arrives ~13 steps)

input_signal = tone_burst(1.0 / kw_dt, f0, n_cycles).flatten()
if len(input_signal) < Nt_full:
    input_signal = np.pad(input_signal, (0, Nt_full - len(input_signal)))
else:
    input_signal = input_signal[:Nt_full]
signal_short = input_signal[:Nt_short]

print(f"dt={kw_dt:.4e} s, Nt_full={Nt_full}, Nt_short={Nt_short}")
print(f"Signal peak={input_signal.max():.4e}, first 5 values: {input_signal[:5]}")

# Sensor offsets from source
sensor_offsets = [4, 8, 12]

# --- k-Wave ---
print("\n--- k-Wave ---")
kgrid = kWaveGrid([Nx, Nx, Nx], [dx, dx, dx])
kgrid.setTime(Nt_short, kw_dt)

src_mask = np.zeros((Nx, Nx, Nx))
src_mask[Nx//2, Nx//2, Nx//2] = 1
kw_source_obj = kSource()
kw_source_obj.p_mask = src_mask
kw_source_obj.p = signal_short.reshape(1, -1)
kw_source_obj.p_mode = "additive"

sen_mask = np.zeros((Nx, Nx, Nx))
for off in sensor_offsets:
    sen_mask[Nx//2 + off, Nx//2, Nx//2] = 1
# Also record the source point itself
sen_mask[Nx//2, Nx//2, Nx//2] = 1  # source location
kw_sensor_obj = kSensor(sen_mask)
kw_sensor_obj.record = ["p"]

kw_res = kspaceFirstOrder3D(
    medium=kWaveMedium(sound_speed=c0, density=rho0),
    kgrid=kgrid,
    source=kw_source_obj,
    sensor=kw_sensor_obj,
    simulation_options=SimulationOptions(
        pml_inside=True, pml_size=pml_size, data_cast="double", save_to_disk=True,
    ),
    execution_options=SimulationExecutionOptions(is_gpu_simulation=False, delete_data=True, verbose_level=0),
)
kw_p = np.array(kw_res["p"])
# k-Wave returns (Nt, n_sensors), flip to (n_sensors, Nt)
if kw_p.shape[0] > kw_p.shape[1]:
    kw_p = kw_p.T
print(f"k-Wave sensor data shape: {kw_p.shape} (n_sensors, Nt)")
print(f"Sensors order (k-Wave): source_point, offset4, offset8, offset12")

# --- pykwavers ---
print("\n--- pykwavers ---")
kwa_grid = kw.Grid(nx=Nx, ny=Nx, nz=Nx, dx=dx, dy=dx, dz=dx)
kwa_medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)

py_src_mask = np.zeros((Nx, Nx, Nx), dtype=np.float64)
py_src_mask[Nx//2, Nx//2, Nx//2] = 1.0

kwa_source_obj = kw.Source.from_mask(py_src_mask, signal_short.copy(), f0, mode="additive")

py_sen_mask = np.zeros((Nx, Nx, Nx), dtype=bool)
py_sen_mask[Nx//2, Nx//2, Nx//2] = True  # source point
for off in sensor_offsets:
    py_sen_mask[Nx//2 + off, Nx//2, Nx//2] = True
kwa_sensor_obj = kw.Sensor.from_mask(py_sen_mask)

kwa_sim = kw.Simulation(kwa_grid, kwa_medium, kwa_source_obj, kwa_sensor_obj, solver=kw.SolverType.PSTD)
kwa_sim.set_pml_size(pml_size)
kwa_sim.set_pml_inside(True)

kwa_res = kwa_sim.run(time_steps=Nt_short, dt=kw_dt)
kwa_p_raw = np.array(kwa_res.sensor_data)
print(f"kwavers sensor data shape: {kwa_p_raw.shape}")

# Drop initial t=0 state if present
n_kw = kw_p.shape[1]
n_kwa = kwa_p_raw.shape[1]
print(f"k-Wave Nt={n_kw}, kwavers Nt={n_kwa}")

if n_kwa == n_kw + 1:
    kwa_p = kwa_p_raw[:, 1:]  # drop t=0
elif n_kwa == n_kw:
    kwa_p = kwa_p_raw
else:
    print(f"WARNING: unexpected length mismatch ({n_kw} vs {n_kwa})")
    kwa_p = kwa_p_raw[:, :n_kw]

print(f"\nAfter alignment: kw={kw_p.shape}, kwa={kwa_p.shape}")

# Print first 20 values at source point (sensor index 0)
print("\n--- SOURCE POINT PRESSURE (first 20 steps) ---")
print(f"{'Step':>6} {'k-Wave':>14} {'kwavers':>14} {'ratio':>10}")
for t in range(min(20, kw_p.shape[1])):
    kw_val = kw_p[0, t]
    kwa_val = kwa_p[0, t]
    ratio = kwa_val / kw_val if abs(kw_val) > 1e-30 else float('nan')
    print(f"{t:6d} {kw_val:14.6e} {kwa_val:14.6e} {ratio:10.4f}")

# Print steps 10-20 at sensor 1 (offset=4 cells)
print("\n--- SENSOR 1 (offset=4 cells) PRESSURE (steps 10-25) ---")
print(f"{'Step':>6} {'k-Wave':>14} {'kwavers':>14} {'ratio':>10}")
for t in range(10, min(25, kw_p.shape[1])):
    kw_val = kw_p[1, t]
    kwa_val = kwa_p[1, t]
    ratio = kwa_val / kw_val if abs(kw_val) > 1e-30 else float('nan')
    print(f"{t:6d} {kw_val:14.6e} {kwa_val:14.6e} {ratio:10.4f}")

# RMS comparison
for i, label in enumerate(["source", "off4", "off8", "off12"]):
    kw_rms = np.sqrt(np.mean(kw_p[i]**2))
    kwa_rms = np.sqrt(np.mean(kwa_p[i]**2))
    if kw_rms > 1e-30:
        print(f"\n{label}: rms_ratio={kwa_rms/kw_rms:.4f}  kw_rms={kw_rms:.4e}  kwa_rms={kwa_rms:.4e}")
