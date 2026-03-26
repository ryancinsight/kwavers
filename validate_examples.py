#!/usr/bin/env python3
"""
Automated Parity Validation against k-wave-python

This script verifies that the pykwavers rust bindings produce mathematically 
identical results to the k-wave-python reference implementations for 
standard simulation examples, adhering to an L-infinity norm error < 1e-4.
"""

import numpy as np
import sys
import os

# Ensure we can import the local external/k-wave-python
sys.path.insert(0, os.path.abspath('external/k-wave-python'))

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.data import Vector
import pykwavers as kwa

def test_ivp_photoacoustic_waveforms():
    print("Running Mathematical Parity Test: IVP Photoacoustic Waveforms")
    
    # Grid initialization
    nx, ny, nz = 32, 32, 32
    dx = 0.1e-3
    kgrid = kWaveGrid(Vector([nx, ny, nz]), Vector([dx, dx, dx]))
    kgrid.makeTime(1500, cfl=0.3, t_end=20 * dx / 1500)
    
    kwa_grid = kwa.Grid(nx, ny, nz, dx, dx, dx)

    # Medium initialization
    medium = kWaveMedium(sound_speed=1500.0, density=1000.0)
    kwa_medium = kwa.Medium.homogeneous(1500.0, 1000.0)

    # Initial Pressure Source (Photoacoustic)
    # create a small sphere
    p0 = np.zeros((nx, ny, nz))
    cx, cy, cz = nx//2, ny//2, nz//2
    r_sphere = 5
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if (i-cx)**2 + (j-cy)**2 + (k-cz)**2 <= r_sphere**2:
                    p0[i, j, k] = 1.0
                    
    source = kSource()
    source.p0 = p0
    kwa_source = kwa.Source.from_initial_pressure(p0)

    # Sensor
    sensor_mask = np.zeros((nx, ny, nz), dtype=bool)
    sensor_mask[cx, cy, cz] = True # center
    sensor_mask[cx+7, cy, cz] = True # slightly outside
    
    sensor = kSensor(mask=sensor_mask)
    kwa_sensor = kwa.Sensor.from_mask(sensor_mask)

    sim_options = SimulationOptions(
        pml_inside=True, 
        pml_size=10, 
        data_cast='double', 
        save_to_disk=False,
        smooth_p0=False,
    )
    exec_options = SimulationExecutionOptions(is_run_simulation=True)

    print("Computing reference solution (k-wave-python)...")
    kw_result = kspaceFirstOrder3D(
        kgrid=kgrid, medium=medium, source=source, sensor=sensor,
        simulation_options=sim_options,
        execution_options=exec_options
    )
    
    kw_data = kw_result['p'] if isinstance(kw_result, dict) else kw_result.p
    if kw_data.ndim == 1:
        kw_data = kw_data.reshape(-1, 1)
    else:
        kw_data = kw_data.T

    print("Computing candidate solution (pykwavers PSTD)...")
    kwa_sim = kwa.Simulation(kwa_grid, kwa_medium, kwa_source, kwa_sensor, solver=kwa.SolverType.PSTD, pml_size=10)
    kwa_result = kwa_sim.run(time_steps=kgrid.Nt, dt=kgrid.dt)
    kwa_data = kwa_result.sensor_data

    # Align temporal index due to recording phase difference
    kw_aligned = kw_data[:, 1:]
    kwa_aligned = kwa_data[:, :-1]

    l_infinity_norm = np.max(np.abs(kw_aligned - kwa_aligned))
    
    print(f"L-Infinity Error Norm: {l_infinity_norm:.4e}")
    if l_infinity_norm < 1e-4:
        print("✅ PASS: Solutions are mathematically identical within tolerance.")
        return True
    else:
        print("❌ FAIL: Solutions diverge beyond acceptable numerical thresholds.")
        return False

if __name__ == '__main__':
    success = test_ivp_photoacoustic_waveforms()
    sys.exit(0 if success else 1)
