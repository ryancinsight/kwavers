import numpy as np
import pykwavers as kw
import time
from example_parity_utils import DEFAULT_OUTPUT_DIR, save_side_by_side_parity_figure, save_text_report
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "compare_initial_pressure_traces.png"
REPORT_PATH = DEFAULT_OUTPUT_DIR / "compare_initial_pressure_metrics.txt"


def ensure_2d_sensors(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.shape[0] > arr.shape[1]:
        return arr.T
    return arr


def align_ivp_traces(kw_p: np.ndarray, py_p: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Align k-wave-python and pykwavers IVP recorder semantics.

    k-wave-python records the initial pressure state before propagation.
    pykwavers records after PSTD IVP initialization/step sequencing. The
    canonical matched comparison is therefore the overlapping propagated
    interval, not equal raw column indices.
    """
    kw = ensure_2d_sensors(kw_p)
    py = ensure_2d_sensors(py_p)
    n_kw = kw.shape[1]
    n_py = py.shape[1]
    if n_py == n_kw + 1:
        return kw[:, 1:], py[:, 1:-1], "kwave[1:] vs pykwavers[1:-1]"
    if n_kw == n_py + 1:
        return kw[:, 1:], py, "kwave[1:] vs pykwavers[:]"
    if n_kw == n_py:
        return kw[:, 1:], py[:, :-1], "kwave[1:] vs pykwavers[:-1]"
    n = min(n_kw, n_py)
    return kw[:, :n], py[:, :n], f"truncated common prefix n={n}"

def main():
    Nx = 64
    Ny = 32
    Nz = 32
    dx = 1e-3
    PML_SIZE = 10
    
    sound_speed = 1500.0
    density = 1000.0
    
    # 1. Define k-wave grid
    kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
    
    medium = kWaveMedium(sound_speed=sound_speed, density=density, alpha_coeff=0.0, alpha_power=1.5)
    
    kgrid.makeTime(medium.sound_speed)
    kw_dt = kgrid.dt
    
    # Define an initial pressure distribution
    p0 = np.zeros((Nx, Ny, Nz), dtype=float)
    # A small ball in the center
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if abs(i - Nx//2) <= 2:
                    p0[i, j, k] = 1.0
                    
    source = kSource()
    source.p0 = p0
    
    sensor_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    # Sensors remain inside the physical domain and outside the PML return
    # interval used for parity. With pml_inside=True, the rightmost usable x
    # cell is Nx - PML_SIZE - 1; the largest offset below leaves nine grid
    # cells before that boundary.
    sensor_offsets = (4, 8, 12)
    for offset in sensor_offsets:
        sensor_mask[Nx//2 + offset, Ny//2, Nz//2] = 1
    
    sensor = kSensor(sensor_mask)
    sensor.record = ["p"]
    
    simulation_options = SimulationOptions(
        pml_x_alpha=1.5, pml_y_alpha=1.5, pml_z_alpha=1.5,
        pml_x_size=PML_SIZE, pml_y_size=PML_SIZE, pml_z_size=PML_SIZE,
        pml_inside=True,
        save_to_disk=True,
        smooth_p0=False,
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
    
    print("=" * 80)
    print("Running k-wave-python reference simulation")
    print("=" * 80)
    
    start_time = time.perf_counter()
    sensor_data = kspaceFirstOrder3D(
        kgrid=kgrid,
        medium=medium,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=execution_options,
    )
    kw_time = time.perf_counter() - start_time
    kw_p = ensure_2d_sensors(sensor_data["p"])
    
    print("=" * 80)
    print("Running pykwavers native simulation")
    print("=" * 80)
    
    py_grid = kw.Grid(Nx, Ny, Nz, dx, dx, dx)
    py_medium = kw.Medium.homogeneous(sound_speed, density)
    
    py_source = kw.Source.from_initial_pressure(p0)
    py_sensor = kw.Sensor.from_mask(sensor_mask)
    
    sim = kw.Simulation(py_grid, py_medium, py_source, py_sensor, solver=kw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)
    
    start_time = time.perf_counter()
    py_result = sim.run(time_steps=kgrid.Nt, dt=kw_dt)
    py_time = time.perf_counter() - start_time
    py_p = ensure_2d_sensors(py_result.sensor_data)

    kw_aligned, py_aligned, alignment = align_ivp_traces(kw_p, py_p)
    print(f"Alignment: {alignment}")

    max_sensor_x = Nx // 2 + max(sensor_offsets)
    boundary_margin_cells = min(
        max_sensor_x - PML_SIZE,
        (Nx - PML_SIZE - 1) - max_sensor_x,
    )
    first_boundary_return_s = 2.0 * boundary_margin_cells * dx / sound_speed
    parity_window_steps = max(1, min(kw_aligned.shape[1], py_aligned.shape[1], int(0.8 * first_boundary_return_s / kw_dt)))
    print(
        f"Parity window: first {parity_window_steps} samples "
        f"(80% of earliest boundary-return time {first_boundary_return_s*1e6:.2f} us)"
    )
        
    num_sensors = min(kw_aligned.shape[0], py_aligned.shape[0])
    n_common_ts = parity_window_steps
    metrics_lines = []
    passed = True
    
    for i in range(num_sensors):
        kw_trace = kw_aligned[i, :n_common_ts]
        py_trace = py_aligned[i, :n_common_ts]
        
        diff = kw_trace - py_trace
        l2_err = np.linalg.norm(diff) / np.linalg.norm(kw_trace)
        
        corr = 1.0
        if np.std(kw_trace) > 1e-10 and np.std(py_trace) > 1e-10:
            corr = np.corrcoef(kw_trace, py_trace)[0, 1]
            
        kw_peak_idx = np.argmax(np.abs(kw_trace))
        py_peak_idx = np.argmax(np.abs(py_trace))
        
        kw_peak_t = kw_peak_idx * kw_dt * 1e6
        py_peak_t = py_peak_idx * kw_dt * 1e6
        
        kw_peak_val = kw_trace[kw_peak_idx]
        py_peak_val = py_trace[py_peak_idx]
        
        ratio = kw_peak_val / py_peak_val if abs(py_peak_val) > 1e-10 else 0.0
        
        print(f"Sensor {i+1}:")
        print(f"  - L2 Error: {l2_err:.4f}")
        print(f"  - Correlation: {corr:.4f}")
        print(f"  - Peak time: kw={kw_peak_t:.2f} us, py={py_peak_t:.2f} us, delta={abs(kw_peak_t - py_peak_t):.2f} us")
        print(f"  - Peak amp: kw={kw_peak_val:.2e}, py={py_peak_val:.2e}, ratio={ratio:.3f}")
        metrics_lines.extend(
            [
                f"sensor_{i+1}_l2_error: {l2_err:.6e}",
                f"sensor_{i+1}_pearson_r: {corr:.6f}",
                f"sensor_{i+1}_peak_time_us_kwave: {kw_peak_t:.6f}",
                f"sensor_{i+1}_peak_time_us_pykwavers: {py_peak_t:.6f}",
                f"sensor_{i+1}_peak_ratio_kwave_over_pykwavers: {ratio:.6f}",
            ]
        )
        if corr < 0.99 or l2_err > 0.02:
            passed = False

    figure_path = save_side_by_side_parity_figure(
        kw_aligned[:, :n_common_ts],
        py_aligned[:, :n_common_ts],
        FIGURE_PATH,
        title="initial pressure sensor-trace parity",
        reference_label="k-wave-python pressure",
        candidate_label="pykwavers pressure",
        cmap="seismic",
    )
    save_text_report(
        REPORT_PATH,
        "compare_initial_pressure parity metrics",
        [
            f"kwave_runtime_s: {kw_time:.6f}",
            f"pykwavers_runtime_s: {py_time:.6f}",
            f"alignment: {alignment}",
            f"sensor_offsets_cells: {sensor_offsets}",
            f"pml_size_cells: {PML_SIZE}",
            f"boundary_margin_cells: {boundary_margin_cells}",
            f"parity_window_steps: {parity_window_steps}",
            f"parity_status: {'PASS' if passed else 'FAIL'}",
            f"figure: {figure_path.name}",
            *metrics_lines,
        ],
    )
    print(f"Figure written to: {figure_path}")
    print(f"Metrics written to: {REPORT_PATH}")
    print(f"Parity status: {'PASS' if passed else 'FAIL'}")
        
    print("=" * 80)
    print("RAW TRACE DUMP FOR SENSOR 2 (FIRST 80 STEPS):")
    for step in range(80):
        if step >= n_common_ts:
            break
        t_us = step * kw_dt * 1e6
        kw_val = kw_aligned[1, step]
        py_val = py_aligned[1, step]
        print(f" t={t_us:05.2f} us | kw: {kw_val:+.4e} | py: {py_val:+.4e}")

    if not passed:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
