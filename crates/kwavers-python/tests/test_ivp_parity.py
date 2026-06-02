"""
IVP (initial value problem / photoacoustic) parity test.

Verifies kwavers PSTD produces near-identical results to k-Wave C++ for
an initial pressure (Gaussian ball) simulation.

Thresholds confirmed by canonical_comparison.py with Nyquist fix (2026-03-27):
  correlation > 0.999, max_diff < 1e-4 Pa, amp_ratio in [0.99, 1.01]
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
    KWAVE_AVAILABLE = True
except ImportError:
    KWAVE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (PYKWAVERS_AVAILABLE and KWAVE_AVAILABLE),
    reason="pykwavers and k-wave-python both required"
)


def run_ivp_comparison():
    """Run IVP comparison and return aligned sensor data from both simulators."""
    Nx = 64
    dx = 1e-3 / Nx  # 15.625 µm
    c0 = 1500.0
    rho0 = 1000.0
    pml_size = 10
    dt = 2e-9
    Nt = 150
    source_radius = 2
    sensor_offset = 10

    r2_1d = (np.arange(Nx) - Nx // 2) ** 2
    xx, yy, zz = np.meshgrid(r2_1d, r2_1d, r2_1d, indexing="ij")
    p0 = np.exp(-(xx + yy + zz) / (2 * source_radius ** 2))

    # k-Wave
    kw_grid = kWaveGrid([Nx, Nx, Nx], [dx, dx, dx])
    kw_grid.setTime(Nt, dt)
    kw_sensor = kSensor()
    kw_sensor.mask = np.zeros((Nx, Nx, Nx), dtype=bool)
    kw_sensor.mask[Nx // 2 + sensor_offset, Nx // 2, Nx // 2] = True
    kw_sensor.record = ["p"]
    kw_source_obj = kSource()
    kw_source_obj.p0 = p0.copy()
    kw_res = kspaceFirstOrder3D(
        medium=kWaveMedium(sound_speed=c0),
        kgrid=kw_grid,
        source=kw_source_obj,
        sensor=kw_sensor,
        simulation_options=SimulationOptions(
            data_cast="double", save_to_disk=True,
            smooth_p0=False, pml_inside=True, pml_size=pml_size,
        ),
        execution_options=SimulationExecutionOptions(
            is_gpu_simulation=False, delete_data=True, verbose_level=0,
        ),
    )
    kw_data = np.array(kw_res["p"])
    if kw_data.ndim == 1:
        kw_data = kw_data.reshape(1, -1)
    if kw_data.shape[0] > kw_data.shape[1]:
        kw_data = kw_data.T

    # kwavers
    kwa_grid = kw.Grid(nx=Nx, ny=Nx, nz=Nx, dx=dx, dy=dx, dz=dx)
    kwa_medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)
    sensor_x = (Nx // 2 + sensor_offset) * dx
    sensor_y = (Nx // 2) * dx
    sensor_z = (Nx // 2) * dx
    kwa_sensor_obj = kw.Sensor.point(position=(sensor_x, sensor_y, sensor_z))
    kwa_source_obj = kw.Source.from_initial_pressure(p0.copy())
    kwa_sim = kw.Simulation(
        grid=kwa_grid, medium=kwa_medium,
        source=kwa_source_obj, sensor=kwa_sensor_obj,
        solver=kw.SolverType.PSTD, pml_size=pml_size,
    )
    kwa_res = kwa_sim.run(time_steps=Nt, dt=dt)
    kwa_data = np.array(kwa_res.sensor_data)
    if kwa_data.ndim == 1:
        kwa_data = kwa_data.reshape(1, -1)

    # Align: kwa has Nt+1 samples (includes t=0 initial)
    n_kw = kw_data.shape[1]
    n_kwa = kwa_data.shape[1]
    if n_kwa == n_kw + 1:
        kw_aligned = kw_data[:, 1:]
        kwa_aligned = kwa_data[:, 1:-1]
    else:
        kw_aligned = kw_data[:, 1:]
        kwa_aligned = kwa_data[:, :-1]
    n = min(kw_aligned.shape[1], kwa_aligned.shape[1])
    return kw_aligned[:, :n], kwa_aligned[:, :n]


@pytest.mark.slow
def test_ivp_parity_correlation():
    """IVP parity: Pearson correlation must exceed 0.999."""
    kw_data, kwa_data = run_ivp_comparison()
    corr = float(np.corrcoef(kw_data.ravel(), kwa_data.ravel())[0, 1])
    assert corr > 0.999, f"IVP correlation {corr:.6f} < 0.999"


@pytest.mark.slow
def test_ivp_parity_max_diff():
    """IVP parity: maximum absolute pressure difference must be < 1e-4 Pa."""
    kw_data, kwa_data = run_ivp_comparison()
    max_diff = float(np.abs(kw_data - kwa_data).max())
    peak = float(np.abs(kw_data).max())
    assert max_diff < 1e-4, (
        f"IVP max_diff {max_diff:.3e} Pa >= 1e-4 Pa "
        f"(peak={peak:.4e} Pa, relative error={max_diff/peak:.4e})"
    )


@pytest.mark.slow
def test_ivp_parity_amplitude():
    """IVP parity: peak amplitude ratio must be within 1% of 1.0."""
    kw_data, kwa_data = run_ivp_comparison()
    peak_kw = float(np.abs(kw_data).max())
    peak_kwa = float(np.abs(kwa_data).max())
    ratio = peak_kwa / peak_kw
    assert 0.99 <= ratio <= 1.01, (
        f"IVP amplitude ratio {ratio:.6f} outside [0.99, 1.01] "
        f"(k-Wave peak={peak_kw:.4e}, kwavers peak={peak_kwa:.4e})"
    )
