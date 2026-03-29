"""
3D Plane Wave parity test.

Verifies kwavers PSTD matches k-Wave C++ for a tone burst plane wave
(full y-z face source) with matched 32x32x32 grids (pml_inside=True,
pml_size=6).

Thresholds confirmed by pw_diag.py (2026-03-27):
  correlation > 0.999, amp_ratio in [0.99, 1.01], max_diff < 1e-3 Pa
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


def run_plane_wave_comparison():
    """Run 3D plane wave comparison and return aligned sensor data."""
    Nx = Ny = Nz = 32
    dx = 2e-3
    c0 = 1500.0
    rho0 = 1000.0
    pml_size = 6
    f0 = 0.5e6
    n_cycles = 3
    t_end = 20e-6

    kw_grid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
    kw_grid.makeTime(c0, t_end=t_end)
    kw_dt = float(kw_grid.dt)
    Nt = int(kw_grid.Nt)

    input_signal = tone_burst(1.0 / kw_dt, f0, n_cycles).flatten()
    if len(input_signal) < Nt:
        input_signal = np.pad(input_signal, (0, Nt - len(input_signal)))
    else:
        input_signal = input_signal[:Nt]

    # Plane wave: full y-z face at x=pml_size
    src_mask = np.zeros((Nx, Ny, Nz))
    src_mask[pml_size, :, :] = 1.0
    sen_mask = np.zeros((Nx, Ny, Nz))
    sen_mask[Nx // 2, Ny // 2, Nz // 2] = 1

    # k-Wave
    kw_source_obj = kSource()
    kw_source_obj.p_mask = src_mask
    kw_source_obj.p = input_signal.reshape(1, -1)
    kw_source_obj.p_mode = "additive"
    kw_sensor_obj = kSensor(sen_mask)
    kw_sensor_obj.record = ["p"]
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
    kw_p = np.array(kw_res["p"]).flatten().reshape(1, -1)

    # kwavers
    kwa_grid = kw.Grid(nx=Nx, ny=Ny, nz=Nz, dx=dx, dy=dx, dz=dx)
    kwa_medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)
    kwa_source_obj = kw.Source.from_mask(
        src_mask.astype(np.float64), input_signal.copy(), f0, mode="additive"
    )
    kwa_sensor_obj = kw.Sensor.from_mask(sen_mask.astype(bool))
    kwa_sim = kw.Simulation(kwa_grid, kwa_medium, kwa_source_obj, kwa_sensor_obj,
                            solver=kw.SolverType.PSTD)
    kwa_sim.set_pml_size(pml_size)
    kwa_sim.set_pml_inside(True)
    kwa_res = kwa_sim.run(time_steps=Nt, dt=kw_dt)
    kwa_p = np.array(kwa_res.sensor_data).flatten().reshape(1, -1)

    # Align: kwavers records Nt+1 samples (includes t=0), k-Wave records Nt
    n_kw = kw_p.shape[1]
    n_kwa = kwa_p.shape[1]
    if n_kwa == n_kw + 1:
        return kw_p, kwa_p[:, 1:]
    elif n_kw == n_kwa:
        return kw_p[:, 1:], kwa_p[:, :-1]
    else:
        n = min(n_kw, n_kwa)
        return kw_p[:, :n], kwa_p[:, 1:n+1]


@pytest.mark.slow
def test_plane_wave_correlation():
    """Plane wave: Pearson correlation > 0.999 at central sensor."""
    kw_data, kwa_data = run_plane_wave_comparison()
    corr = float(np.corrcoef(kw_data[0], kwa_data[0])[0, 1])
    assert corr > 0.999, f"Plane wave correlation {corr:.6f} < 0.999"


@pytest.mark.slow
def test_plane_wave_amplitude():
    """Plane wave: peak amplitude ratio within 1% for central sensor."""
    kw_data, kwa_data = run_plane_wave_comparison()
    peak_kw = float(np.abs(kw_data[0]).max())
    peak_kwa = float(np.abs(kwa_data[0]).max())
    ratio = peak_kwa / (peak_kw + 1e-30)
    assert 0.99 <= ratio <= 1.01, (
        f"Plane wave amplitude ratio {ratio:.6f} outside [0.99, 1.01] "
        f"(k-Wave={peak_kw:.4e}, kwavers={peak_kwa:.4e})"
    )
