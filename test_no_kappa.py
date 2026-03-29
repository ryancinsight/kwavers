"""Test with kappa=1 in both codes to isolate k-space correction effect."""
import sys
sys.path.insert(0, 'd:/kwavers/pykwavers/python')
import numpy as np
import pykwavers as kw

from kwave.kgrid import kWaveGrid; from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor; from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.signals import tone_burst

Nx=Ny=Nz=64; dx=1e-3; c0=1500.0; rho0=1000.0; pml_size=10; f0=0.5e6; n_cycles=3; t_end=40e-6

kw_grid=kWaveGrid([Nx,Ny,Nz],[dx,dx,dx])
kw_grid.makeTime(c0,t_end=t_end)
kw_dt=float(kw_grid.dt); Nt=int(kw_grid.Nt)
sig=tone_burst(1.0/kw_dt,f0,n_cycles).flatten()
if len(sig)<Nt: sig=np.pad(sig,(0,Nt-len(sig)))
else: sig=sig[:Nt]

sm=np.zeros((Nx,Ny,Nz)); sm[Nx//2,Ny//2,Nz//2]=1
ks=kSource(); ks.p_mask=sm; ks.p=sig.reshape(1,-1); ks.p_mode="additive"
senm=np.zeros((Nx,Ny,Nz)); senm[Nx//2+4,Ny//2,Nz//2]=1
ksen=kSensor(senm); ksen.record=["p"]

# k-Wave WITHOUT k-space correction (kappa=1, source_kappa=1)
res_nokappa=kspaceFirstOrder3D(medium=kWaveMedium(sound_speed=c0,density=rho0,alpha_coeff=0.0,alpha_power=1.5),
    kgrid=kw_grid,source=ks,sensor=ksen,
    simulation_options=SimulationOptions(pml_inside=True,pml_size=pml_size,data_cast="single",
                                          save_to_disk=True,use_kspace=False),
    execution_options=SimulationExecutionOptions(is_gpu_simulation=False))
kw_p_nokappa=res_nokappa["p"]
if kw_p_nokappa.ndim==1: kw_p_nokappa=kw_p_nokappa.reshape(1,-1)
if kw_p_nokappa.shape[0]>kw_p_nokappa.shape[1]: kw_p_nokappa=kw_p_nokappa.T

kwa_grid2=kw.Grid(nx=Nx,ny=Ny,nz=Nz,dx=dx,dy=dx,dz=dx)
psm=np.zeros((Nx,Ny,Nz),dtype=np.float64); psm[Nx//2,Ny//2,Nz//2]=1.0
# additive_no_correction = no source_kappa applied
kwa_src=kw.Source.from_mask(psm,sig.copy(),f0,mode="additive_no_correction")
psenm=np.zeros((Nx,Ny,Nz),dtype=bool); psenm[Nx//2+4,Ny//2,Nz//2]=True
sim2=kw.Simulation(kwa_grid2,kw.Medium.homogeneous(sound_speed=c0,density=rho0),kwa_src,kw.Sensor.from_mask(psenm),solver=kw.SolverType.PSTD)
sim2.set_pml_size(pml_size); sim2.set_pml_inside(True)

# Need to disable spectral correction in kwavers too
# Use environment variable or check if API supports it
# For now, test with additive (with source_kappa but kappa from propagation still active)
r2=sim2.run(time_steps=Nt,dt=kw_dt)
kwa_p=r2.sensor_data
if kwa_p.ndim==1: kwa_p=kwa_p.reshape(1,-1)
if kwa_p.shape[0]>kwa_p.shape[1]: kwa_p=kwa_p.T

nk=kw_p_nokappa.shape[1]; nka=kwa_p.shape[1]
if nka==nk+1: ka,kwa_a=kw_p_nokappa,kwa_p[:,1:]
elif nk==nka+1: ka,kwa_a=kw_p_nokappa[:,1:],kwa_p
elif nk==nka: ka,kwa_a=kw_p_nokappa[:,1:],kwa_p[:,:-1]
else: n=min(nk,nka); ka,kwa_a=kw_p_nokappa[:,:n],kwa_p[:,:n]

ref=ka[0].ravel(); tst=kwa_a[0].ravel()
corr=float(np.corrcoef(ref,tst)[0,1]) if np.std(ref)>1e-30 else 0
rms_r=float(np.sqrt(np.mean(tst**2))/np.sqrt(np.mean(ref**2)))
amp_r=float(np.max(np.abs(tst))/np.max(np.abs(ref)))
print(f"No kappa in kWave, no source_kappa in kwavers: corr={corr:.4f} rms={rms_r:.4f} amp={amp_r:.4f}")
print(f"  kw_peak={np.max(np.abs(ref)):.4e}, kwa_peak={np.max(np.abs(tst)):.4e}")
