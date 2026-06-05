"""
3D Tone Burst Point Source parity test.

Verifies kwavers PSTD matches k-Wave C++ for a tone burst point source
with matched 64x64x64 grids (pml_inside=True, pml_size=10).

Thresholds confirmed by canonical_comparison.py with Nyquist fix (2026-03-27):
  correlation > 0.999, amp_ratio in [0.99, 1.01], max_diff < 1e-6 Pa
"""
import numpy as np
import pytest

try:
    import pykwavers as kw
    PYKWAVERS_AVAILABLE = True
except ImportError:
    PYKWAVERS_AVAILABLE = False

try:
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
    from kwave.utils.signals import tone_burst
    KWAVE_AVAILABLE = True
except ImportError:
    KWAVE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (PYKWAVERS_AVAILABLE and KWAVE_AVAILABLE),
    reason="pykwavers and k-wave-python both required"
)

# Sensor offsets along x from source
SENSOR_OFFSETS = [4, 8, 12]


def run_point_source_comparison():
    """Run 3D tone burst comparison and return aligned sensor data."""
    Nx = Ny = Nz = 64
    dx = 1e-3
    c0 = 1500.0
    rho0 = 1000.0
    pml_size = 10
    f0 = 0.5e6
    n_cycles = 3
    t_end = 40e-6

    kw_grid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
    kw_grid.makeTime(c0, t_end=t_end)
    kw_dt = float(kw_grid.dt)
    Nt = int(kw_grid.Nt)

    input_signal = tone_burst(1.0 / kw_dt, f0, n_cycles).flatten()
    if len(input_signal) < Nt:
        input_signal = np.pad(input_signal, (0, Nt - len(input_signal)))
    else:
        input_signal = input_signal[:Nt]

    src_mask = np.zeros((Nx, Ny, Nz)); src_mask[Nx//2, Ny//2, Nz//2] = 1
    sen_mask = np.zeros((Nx, Ny, Nz))
    for off in SENSOR_OFFSETS:
        sen_mask[Nx//2 + off, Ny//2, Nz//2] = 1

    # k-Wave
    kw_source_obj = kSource()
    kw_source_obj.p_mask = src_mask
    kw_source_obj.p = input_signal.reshape(1, -1)
    kw_source_obj.p_mode = "additive"
    kw_sensor_obj = kSensor(sen_mask); kw_sensor_obj.record = ["p"]
    kw_res = kspaceFirstOrder3D(
        medium=kWaveMedium(sound_speed=c0, density=rho0),
        kgrid=kw_grid,
        source=kw_source_obj,
        sensor=kw_sensor_obj,
        simulation_options=SimulationOptions(
            pml_inside=True, pml_size=pml_size,
            data_cast="double", save_to_disk=True,
        ),
        execution_options=SimulationExecutionOptions(
            is_gpu_simulation=False, delete_data=True, verbose_level=0,
        ),
    )
    kw_p = np.array(kw_res["p"])
    if kw_p.ndim == 1: kw_p = kw_p.reshape(1,-1)
    if kw_p.shape[0] == Nt and kw_p.shape[1] != Nt: kw_p = kw_p.T

    # kwavers
    kwa_grid = kw.Grid(nx=Nx, ny=Ny, nz=Nz, dx=dx, dy=dx, dz=dx)
    kwa_medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)
    py_src = np.zeros((Nx,Ny,Nz),dtype=np.float64); py_src[Nx//2,Ny//2,Nz//2]=1.
    kwa_source_obj = kw.Source.from_mask(py_src, input_signal.copy(), f0, mode="additive")
    py_sen = np.zeros((Nx,Ny,Nz),dtype=bool)
    for off in SENSOR_OFFSETS: py_sen[Nx//2+off,Ny//2,Nz//2]=True
    kwa_sensor_obj = kw.Sensor.from_mask(py_sen)
    kwa_sim = kw.Simulation(kwa_grid, kwa_medium, kwa_source_obj, kwa_sensor_obj,
                            solver=kw.SolverType.PSTD)
    kwa_sim.set_pml_size(pml_size); kwa_sim.set_pml_inside(True)
    kwa_res = kwa_sim.run(time_steps=Nt, dt=kw_dt)
    kwa_p = np.array(kwa_res.sensor_data)
    if kwa_p.ndim == 1: kwa_p = kwa_p.reshape(1,-1)

    # Align: kwavers records after each step (Nt+1 samples including t=0).
    # k-Wave records Nt samples (steps 1..Nt). kwa_data[:, 1:] aligns with kw_data.
    n_kw = kw_p.shape[1]
    n_kwa = kwa_p.shape[1]
    if n_kwa == n_kw + 1:
        # kwavers has initial t=0 plus Nt steps; k-Wave has Nt steps
        return kw_p, kwa_p[:, 1:]
    elif n_kw == n_kwa:
        return kw_p[:, 1:], kwa_p[:, :-1]
    else:
        n = min(n_kw, n_kwa)
        return kw_p[:, :n], kwa_p[:, 1:n+1]


@pytest.mark.slow
def test_point_source_correlation():
    """Tone burst: Pearson correlation > 0.999 for all sensors."""
    kw_data, kwa_data = run_point_source_comparison()
    n = min(kw_data.shape[0], kwa_data.shape[0])
    for i in range(n):
        corr = float(np.corrcoef(kw_data[i], kwa_data[i])[0, 1])
        offset = SENSOR_OFFSETS[i] if i < len(SENSOR_OFFSETS) else i
        assert corr > 0.999, (
            f"Sensor offset={offset}: correlation {corr:.6f} < 0.999"
        )


@pytest.mark.slow
def test_point_source_amplitude():
    """Tone burst: peak amplitude ratio within 1% for all sensors."""
    kw_data, kwa_data = run_point_source_comparison()
    n = min(kw_data.shape[0], kwa_data.shape[0])
    for i in range(n):
        peak_kw  = float(np.abs(kw_data[i]).max())
        peak_kwa = float(np.abs(kwa_data[i]).max())
        ratio = peak_kwa / (peak_kw + 1e-30)
        offset = SENSOR_OFFSETS[i] if i < len(SENSOR_OFFSETS) else i
        assert 0.99 <= ratio <= 1.01, (
            f"Sensor offset={offset}: amplitude ratio {ratio:.6f} outside [0.99, 1.01] "
            f"(k-Wave={peak_kw:.4e}, kwavers={peak_kwa:.4e})"
        )
