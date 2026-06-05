"""
Canonical Absorbing Medium parity test.

Verifies kwavers PSTD matches k-Wave C++ for a homogeneous absorbing medium
(alpha_coeff=0.5 dB/(MHz·cm), alpha_power=1.5) with matched 32x32x32 grids
(pml_inside=True, pml_size=6).

Thresholds:
  correlation > 0.99, amp_ratio in [0.90, 1.10]
  (wider amplitude tolerance: absorption coefficient accuracy depends on
   discrete approximation of the power-law frequency dependence)
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

ALPHA_COEFF = 0.5   # dB/(MHz·cm)
ALPHA_POWER = 1.5   # power-law exponent


def run_absorption_comparison():
    """Run absorbing medium comparison and return aligned sensor data."""
    Nx = Ny = Nz = 32
    dx = 1e-3
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

    # Point source near PML edge; sensor 8 cells away
    src_ix = pml_size + 2
    sen_ix = src_ix + 8
    src_mask = np.zeros((Nx, Ny, Nz))
    src_mask[src_ix, Ny // 2, Nz // 2] = 1.0
    sen_mask = np.zeros((Nx, Ny, Nz))
    sen_mask[sen_ix, Ny // 2, Nz // 2] = 1

    # k-Wave
    kw_source_obj = kSource()
    kw_source_obj.p_mask = src_mask
    kw_source_obj.p = input_signal.reshape(1, -1)
    kw_source_obj.p_mode = "additive"
    kw_sensor_obj = kSensor(sen_mask)
    kw_sensor_obj.record = ["p"]
    kw_medium_obj = kWaveMedium(
        sound_speed=c0, density=rho0,
        alpha_coeff=ALPHA_COEFF, alpha_power=ALPHA_POWER,
    )
    kw_res = kspaceFirstOrder3D(
        medium=kw_medium_obj,
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
    kwa_medium = kw.Medium.homogeneous(
        sound_speed=c0, density=rho0,
        absorption=ALPHA_COEFF, alpha_power=ALPHA_POWER,
    )
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
def test_absorption_correlation():
    """Absorbing medium: Pearson correlation > 0.99 at offset sensor."""
    kw_data, kwa_data = run_absorption_comparison()
    corr = float(np.corrcoef(kw_data[0], kwa_data[0])[0, 1])
    assert corr > 0.99, f"Absorption correlation {corr:.6f} < 0.99"


@pytest.mark.slow
def test_absorption_amplitude():
    """Absorbing medium: peak amplitude ratio within 10%."""
    kw_data, kwa_data = run_absorption_comparison()
    peak_kw = float(np.abs(kw_data[0]).max())
    peak_kwa = float(np.abs(kwa_data[0]).max())
    ratio = peak_kwa / (peak_kw + 1e-30)
    assert 0.90 <= ratio <= 1.10, (
        f"Absorption amplitude ratio {ratio:.6f} outside [0.90, 1.10] "
        f"(k-Wave={peak_kw:.4e}, kwavers={peak_kwa:.4e})"
    )
