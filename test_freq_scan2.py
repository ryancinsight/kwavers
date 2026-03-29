"""Frequency scan: compare low-ppw vs high-ppw."""
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
c0 = 1500.0; rho0 = 1000.0; pml_size = 10; n_cycles = 5
c_dt_dx = 0.3; dt = c_dt_dx * dx / c0

def run_cmp(f0):
    ppw = c0 / (f0 * dx)
    t_end = max(30e-6, n_cycles / f0 * 3)  # ensure pulse fully passes sensor
    kw_grid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
    kw_grid.makeTime(c0, cfl=c_dt_dx, t_end=t_end)
    kw_dt = float(kw_grid.dt)
    Nt = int(kw_grid.Nt)
    
    sig = tone_burst(1.0/kw_dt, f0, n_cycles).flatten()
    if len(sig) < Nt: sig = np.pad(sig, (0, Nt-len(sig)))
    else: sig = sig[:Nt]
    
    kw_med = kWaveMedium(sound_speed=c0, density=rho0, alpha_coeff=0.0, alpha_power=1.5)
    sm = np.zeros((Nx,Ny,Nz)); sm[Nx//2,Ny//2,Nz//2] = 1
    ks = kSource(); ks.p_mask = sm; ks.p = sig.reshape(1,-1); ks.p_mode = "additive"
    senm = np.zeros((Nx,Ny,Nz)); senm[Nx//2+4,Ny//2,Nz//2] = 1
    ksen = kSensor(senm); ksen.record = ["p"]
    
    res = kspaceFirstOrder3D(medium=kw_med, kgrid=kw_grid, source=ks, sensor=ksen,
        simulation_options=SimulationOptions(pml_inside=True, pml_size=pml_size,
                                              data_cast="single", save_to_disk=True),
        execution_options=SimulationExecutionOptions(is_gpu_simulation=False))
    kw_p = res["p"]
    if kw_p.ndim == 1: kw_p = kw_p.reshape(1,-1)
    if kw_p.shape[0] > kw_p.shape[1]: kw_p = kw_p.T
    
    kwa_grid2 = kw.Grid(nx=Nx, ny=Ny, nz=Nz, dx=dx, dy=dx, dz=dx)
    kwa_med2 = kw.Medium.homogeneous(sound_speed=c0, density=rho0)
    psm = np.zeros((Nx,Ny,Nz), dtype=np.float64); psm[Nx//2,Ny//2,Nz//2] = 1.0
    kwa_src = kw.Source.from_mask(psm, sig.copy(), f0, mode="additive")
    psenm = np.zeros((Nx,Ny,Nz), dtype=bool); psenm[Nx//2+4,Ny//2,Nz//2] = True
    kwa_sen = kw.Sensor.from_mask(psenm)
    sim2 = kw.Simulation(kwa_grid2, kwa_med2, kwa_src, kwa_sen, solver=kw.SolverType.PSTD)
    sim2.set_pml_size(pml_size); sim2.set_pml_inside(True)
    r2 = sim2.run(time_steps=Nt, dt=kw_dt)
    kwa_p = r2.sensor_data
    if kwa_p.ndim == 1: kwa_p = kwa_p.reshape(1,-1)
    if kwa_p.shape[0] > kwa_p.shape[1]: kwa_p = kwa_p.T
    
    nk = kw_p.shape[1]; nka = kwa_p.shape[1]
    if nka == nk+1: ka, kwa_a = kw_p, kwa_p[:,1:]
    elif nk == nka+1: ka, kwa_a = kw_p[:,1:], kwa_p
    elif nk == nka: ka, kwa_a = kw_p[:,1:], kwa_p[:,:-1]
    else: n = min(nk,nka); ka, kwa_a = kw_p[:,:n], kwa_p[:,:n]
    
    ref = ka[0].ravel(); tst = kwa_a[0].ravel()
    corr = float(np.corrcoef(ref,tst)[0,1]) if np.std(ref)>1e-30 else 0
    rms_r = float(np.sqrt(np.mean(tst**2))/np.sqrt(np.mean(ref**2))) if np.sqrt(np.mean(ref**2))>1e-30 else float('inf')
    amp_r = float(np.max(np.abs(tst))/np.max(np.abs(ref))) if np.max(np.abs(ref))>1e-30 else float('inf')
    print(f"f0={f0/1e6:.2f}MHz ppw={ppw:.1f}: corr={corr:.4f} rms={rms_r:.4f} amp={amp_r:.4f}")

run_cmp(0.1e6)   # 10 ppw
run_cmp(0.5e6)   # 3 ppw
