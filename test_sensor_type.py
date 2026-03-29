#!/usr/bin/env python3
"""Check if sensor type (point vs mask) affects kwavers plane wave amplitude."""
import numpy as np
import pykwavers as kw

N = 32; dx = 1e-3; c = 1500.0; rho = 1000.0
pml_size = 6; cfl = 0.3; dt = cfl * dx / c
freq = 0.5e6; amp = 1e5; nt = 100
t_arr = np.arange(nt) * dt
signal = amp * np.sin(2*np.pi*freq*t_arr)

# Plane wave source (full z=0 face)
sm_plane = np.zeros((N,N,N)); sm_plane[:,:,0] = 1.0
sx, sy, sz = N//2, N//2, N//2
sensor_pos = (sx*dx, sy*dx, sz*dx)

# Mask sensor at center
sensor_mask = np.zeros((N,N,N), dtype=bool); sensor_mask[sx,sy,sz] = True

def run_kwa(use_point_sensor):
    grid = kw.Grid(nx=N, ny=N, nz=N, dx=dx, dy=dx, dz=dx)
    medium = kw.Medium.homogeneous(sound_speed=c, density=rho)
    source = kw.Source.from_mask(sm_plane.astype(np.float64), signal,
                                  frequency=freq, mode='additive_no_correction')
    if use_point_sensor:
        sensor = kw.Sensor.point(position=sensor_pos)
    else:
        sensor = kw.Sensor.from_mask(sensor_mask)
    sim = kw.Simulation(grid, medium, source, sensor,
                        solver=kw.SolverType.PSTD, pml_size=pml_size)
    result = sim.run(time_steps=nt, dt=dt)
    d = np.array(result.sensor_data).flatten().astype(float)
    if len(d) > nt: d = d[1:]
    return d

p_point = run_kwa(use_point_sensor=True)
p_mask = run_kwa(use_point_sensor=False)

rms_point = np.sqrt(np.mean(p_point[nt//2:]**2))
rms_mask = np.sqrt(np.mean(p_mask[nt//2:]**2))

print(f"Sensor.point RMS:    {rms_point:.4e}")
print(f"Sensor.from_mask RMS: {rms_mask:.4e}")
print(f"Ratio (point/mask): {rms_point/rms_mask:.4f}")
