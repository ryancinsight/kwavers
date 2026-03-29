#!/usr/bin/env python3
"""
Canonical kwavers vs k-wave-python Comparison Harness
======================================================

Ground rules enforced by this script:
  1. IDENTICAL total grid size in both simulators (no Nx-2*PML correction)
  2. pml_inside=True, same pml_size (PML is counted within total Nx)
  3. Timing alignment: kw_data[:, 1:] vs kwa_data[:, :-1]
     - k-Wave records pressure at t=0 (before step 1)
     - kwavers records pressure after each step
     - Slicing aligns step indices
  4. Metrics: correlation, RMS ratio, max_abs_diff, peak_pressure

Usage:
    python canonical_comparison.py              # run all cases
    python canonical_comparison.py --case ivp   # IVP photoacoustic only
    python canonical_comparison.py --case point_source  # 3D tone burst only
"""

import argparse
import sys

import numpy as np

try:
    import pykwavers as kw
except ImportError:
    print("ERROR: pykwavers not found. Run `maturin develop` in the pykwavers virtualenv.")
    sys.exit(1)

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
    print("WARNING: k-wave-python not found. Install with `pip install k-wave-python`.")


def compute_metrics(ref: np.ndarray, test: np.ndarray) -> dict:
    """Compute parity metrics between reference (k-Wave) and test (kwavers) traces."""
    ref = np.asarray(ref, dtype=float).ravel()
    test = np.asarray(test, dtype=float).ravel()

    max_diff = float(np.max(np.abs(ref - test)))
    ref_rms = float(np.sqrt(np.mean(ref ** 2)))
    test_rms = float(np.sqrt(np.mean(test ** 2)))
    rms_ratio = test_rms / ref_rms if ref_rms > 1e-30 else float("inf")

    if np.std(ref) > 1e-30 and np.std(test) > 1e-30:
        corr = float(np.corrcoef(ref, test)[0, 1])
    else:
        corr = 0.0

    ref_peak = float(np.max(np.abs(ref)))
    test_peak = float(np.max(np.abs(test)))
    amp_ratio = test_peak / ref_peak if ref_peak > 1e-30 else float("inf")

    return {
        "corr": corr,
        "rms_ratio": rms_ratio,
        "max_diff": max_diff,
        "ref_peak": ref_peak,
        "test_peak": test_peak,
        "amp_ratio": amp_ratio,
    }


def ensure_2d_sensors(data: np.ndarray) -> np.ndarray:
    """Normalize raw sensor output to shape (n_sensors, time_steps).

    Handles three common layouts returned by k-Wave and kwavers:
    - 1D (time,)           → (1, time)
    - 2D (time, n_sensors) → (n_sensors, time)  [when time > n_sensors]
    - 2D (n_sensors, time) → unchanged
    """
    if data.ndim == 1:
        return data.reshape(1, -1)
    # Heuristic: if first dim looks like time steps (much larger than sensor count)
    # k-Wave returns (Nt_recorded, n_sensors); flip to (n_sensors, Nt_recorded)
    if data.shape[0] > data.shape[1]:
        return data.T
    return data


def align(kw_data: np.ndarray, kwa_data: np.ndarray):
    """Canonical timing alignment between k-Wave and kwavers sensor data.

    k-Wave records at t=0 before the first step; kwavers records after each step.
    When both arrays have the same length N:
        kw[1:]  (drops t=0)  aligns with  kwa[:-1] (drops last step)
    When kwa has one extra sample (IVP records initial state at t=0):
        kwa[1:] (drops t=0 extra) aligns with kw (already starts at t=dt)
    When kw has one extra sample:
        kw[1:] aligns with kwa
    """
    n_kw = kw_data.shape[1]
    n_kwa = kwa_data.shape[1]

    print(f"  [align] kw samples={n_kw}, kwa samples={n_kwa}")

    if n_kw == n_kwa:
        # Same length: k-Wave t=0 extra, kwavers step Nt extra
        return kw_data[:, 1:], kwa_data[:, :-1]
    elif n_kwa == n_kw + 1:
        # kwavers has initial state at t=0 in addition to Nt steps
        # kwa[1:] starts at t=dt, same as kw[0]
        return kw_data, kwa_data[:, 1:]
    elif n_kw == n_kwa + 1:
        # k-Wave has initial state at t=0
        return kw_data[:, 1:], kwa_data
    else:
        # Lengths differ by more than 1; truncate to minimum
        n = min(n_kw, n_kwa)
        print(f"  [align] WARNING: length mismatch ({n_kw} vs {n_kwa}), truncating to {n}")
        return kw_data[:, :n], kwa_data[:, :n]


def print_metrics(metrics: dict, label: str) -> None:
    print(f"    {label}:")
    print(f"      correlation : {metrics['corr']:.6f}")
    print(f"      rms_ratio   : {metrics['rms_ratio']:.6f}  (kwavers/k-Wave)")
    print(f"      amp_ratio   : {metrics['amp_ratio']:.6f}  (kwavers/k-Wave peak)")
    print(f"      max_diff    : {metrics['max_diff']:.3e} Pa")
    print(f"      k-Wave peak : {metrics['ref_peak']:.4e} Pa")
    print(f"      kwavers peak: {metrics['test_peak']:.4e} Pa")


# ============================================================================
# Case 1: IVP Photoacoustic — proven-perfect parity case
# ============================================================================

def run_ivp_case(gpu: bool = False) -> tuple:
    """
    IVP (initial value problem) photoacoustic comparison.

    Identical setup to validate_ivp_photoacoustic.py Run 6, which achieved
    max diff = 1.3e-5 Pa at peak ±6.06e-2 Pa.

    Grid: 64³ isotropic, dx=15.625 µm, dt=2 ns, 150 steps, pml_size=10
    Source: Gaussian ball p0, radius=2 grid points
    Sensor: point at Nx//2+10
    """
    print("\n" + "=" * 60)
    print("CASE 1: IVP Photoacoustic (Gaussian ball initial pressure)")
    print("=" * 60)

    if not KWAVE_AVAILABLE:
        print("  SKIP — k-wave-python not available")
        return None, None

    Nx = 64
    dx = 1e-3 / Nx   # 15.625 µm
    c0 = 1500.0
    rho0 = 1000.0
    pml_size = 10
    dt = 2e-9
    Nt = 150
    source_radius = 2   # grid points (Gaussian sigma)
    sensor_offset = 10  # grid points from center along x

    # Gaussian ball initial pressure
    r2_1d = (np.arange(Nx) - Nx // 2) ** 2
    xx, yy, zz = np.meshgrid(r2_1d, r2_1d, r2_1d, indexing="ij")
    p0 = np.exp(-(xx + yy + zz) / (2 * source_radius ** 2))

    # --- k-Wave ---
    print("  Running k-wave-python...")
    kw_grid = kWaveGrid([Nx, Nx, Nx], [dx, dx, dx])
    kw_grid.setTime(Nt, dt)

    kw_medium = kWaveMedium(sound_speed=c0)

    kw_sensor = kSensor()
    kw_sensor.mask = np.zeros((Nx, Nx, Nx), dtype=bool)
    kw_sensor.mask[Nx // 2 + sensor_offset, Nx // 2, Nx // 2] = True
    kw_sensor.record = ["p"]

    kw_source_obj = kSource()
    kw_source_obj.p0 = p0.copy()

    kw_res = kspaceFirstOrder3D(
        medium=kw_medium,
        kgrid=kw_grid,
        source=kw_source_obj,
        sensor=kw_sensor,
        simulation_options=SimulationOptions(
            data_cast="double",
            save_to_disk=True,
            smooth_p0=False,
            pml_inside=True,
            pml_size=pml_size,
        ),
        execution_options=SimulationExecutionOptions(
            is_gpu_simulation=gpu,
            delete_data=True,
            verbose_level=0,
        ),
    )
    kw_data = ensure_2d_sensors(kw_res["p"])

    # --- pykwavers ---
    print("  Running pykwavers...")
    kwa_grid = kw.Grid(nx=Nx, ny=Nx, nz=Nx, dx=dx, dy=dx, dz=dx)
    kwa_medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)

    sensor_x = (Nx // 2 + sensor_offset) * dx
    sensor_y = (Nx // 2) * dx
    sensor_z = (Nx // 2) * dx
    kwa_sensor_obj = kw.Sensor.point(position=(sensor_x, sensor_y, sensor_z))
    kwa_source_obj = kw.Source.from_initial_pressure(p0.copy())

    kwa_sim = kw.Simulation(
        grid=kwa_grid,
        medium=kwa_medium,
        source=kwa_source_obj,
        sensor=kwa_sensor_obj,
        solver=kw.SolverType.PSTD,
        pml_size=pml_size,
    )
    kwa_res = kwa_sim.run(time_steps=Nt, dt=dt)
    kwa_data = ensure_2d_sensors(kwa_res.sensor_data)

    # IVP-specific timing alignment.
    #
    # kwavers with IVP velocity initialization records Nt+1 samples:
    #   kwa[:, 0] = initial state (t=0)
    #   kwa[:, n] = pressure after step n  (n=1..Nt)
    #
    # k-Wave records Nt samples:
    #   kw[:, n] = pressure after step n+1  (n=0..Nt-1)
    #
    # Diagnostic (offset sweep) confirmed: kwa step n = kw step n+1.
    # Correct alignment: compare kw[:, 1:] with kwa[:, 1:-1]
    #   kw[:, 1:]  = [kw step 2 .. kw step Nt]  (length Nt-1)
    #   kwa[:, 1:-1] = [kwa step 1 .. kwa step Nt-1] = [kw step 2 .. kw step Nt] ✓
    n_kw = kw_data.shape[1]
    n_kwa = kwa_data.shape[1]
    print(f"  [align-ivp] kw={n_kw}, kwa={n_kwa}")
    if n_kwa == n_kw + 1:
        kw_aligned = kw_data[:, 1:]
        kwa_aligned = kwa_data[:, 1:-1]
    elif n_kwa == n_kw:
        kw_aligned = kw_data[:, 1:]
        kwa_aligned = kwa_data[:, :-1]
    else:
        kw_aligned, kwa_aligned = align(kw_data, kwa_data)

    n = min(kw_aligned.shape[0], kwa_aligned.shape[0])
    all_metrics = []
    for i in range(n):
        m = compute_metrics(kw_aligned[i], kwa_aligned[i])
        all_metrics.append(m)
        print_metrics(m, f"Sensor {i+1} (offset={sensor_offset} cells)")

    # Pass/fail
    passed = all(m["corr"] > 0.999 and m["max_diff"] < 1e-4 for m in all_metrics)
    print(f"\n  Result: {'PASS' if passed else 'FAIL'} "
          f"(target: corr>0.999, max_diff<1e-4 Pa)")
    return kw_aligned, kwa_aligned


# ============================================================================
# Case 2: 3D Tone Burst Point Source — matched grids
# ============================================================================

def run_point_source_case(gpu: bool = True) -> tuple:
    """
    3D tone burst point source comparison with matched grids.

    Previously failed due to grid size mismatch (128×96×96 vs 96×64×64).
    This case uses identical 64³ grids in both simulators.

    Grid: 64³ isotropic, dx=1mm, pml_size=10
    Source: 0.5 MHz tone burst, 3 cycles, point at grid center
    Sensors: at +4, +8, +12 cells from source along x
    """
    print("\n" + "=" * 60)
    print("CASE 2: 3D Tone Burst Point Source (matched 64³ grids)")
    print("=" * 60)

    if not KWAVE_AVAILABLE:
        print("  SKIP — k-wave-python not available")
        return None, None

    Nx = Ny = Nz = 64
    dx = 1e-3
    c0 = 1500.0
    rho0 = 1000.0
    pml_size = 10
    f0 = 0.5e6
    n_cycles = 3
    t_end = 40e-6
    sensor_offsets = [4, 8, 12]  # cells from source along x

    # --- k-Wave ---
    print("  Running k-wave-python...")
    # CANONICAL: kWaveGrid([Nx, Ny, Nz], ...) — same total size as kwavers
    kw_grid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
    kw_grid.makeTime(c0, t_end=t_end)
    kw_dt = float(kw_grid.dt)
    Nt = int(kw_grid.Nt)
    print(f"  dt={kw_dt:.4e} s, Nt={Nt}")

    input_signal = tone_burst(1.0 / kw_dt, f0, n_cycles).flatten()
    if len(input_signal) < Nt:
        input_signal = np.pad(input_signal, (0, Nt - len(input_signal)))
    else:
        input_signal = input_signal[:Nt]

    kw_medium = kWaveMedium(
        sound_speed=c0, density=rho0, alpha_coeff=0.0, alpha_power=1.5
    )

    src_mask = np.zeros((Nx, Ny, Nz))
    src_mask[Nx // 2, Ny // 2, Nz // 2] = 1
    kw_source_obj = kSource()
    kw_source_obj.p_mask = src_mask
    kw_source_obj.p = input_signal.reshape(1, -1)
    kw_source_obj.p_mode = "additive"

    sen_mask = np.zeros((Nx, Ny, Nz))
    for off in sensor_offsets:
        sen_mask[Nx // 2 + off, Ny // 2, Nz // 2] = 1
    kw_sensor_obj = kSensor(sen_mask)
    kw_sensor_obj.record = ["p"]

    kw_res = kspaceFirstOrder3D(
        medium=kw_medium,
        kgrid=kw_grid,
        source=kw_source_obj,
        sensor=kw_sensor_obj,
        simulation_options=SimulationOptions(
            pml_inside=True,
            pml_size=pml_size,
            data_cast="single",
            save_to_disk=True,
        ),
        execution_options=SimulationExecutionOptions(is_gpu_simulation=gpu),
    )
    kw_p = ensure_2d_sensors(kw_res["p"])

    # --- pykwavers ---
    print("  Running pykwavers...")
    kwa_grid = kw.Grid(nx=Nx, ny=Ny, nz=Nz, dx=dx, dy=dx, dz=dx)
    kwa_medium = kw.Medium.homogeneous(sound_speed=c0, density=rho0)

    py_src_mask = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    py_src_mask[Nx // 2, Ny // 2, Nz // 2] = 1.0
    kwa_source_obj = kw.Source.from_mask(
        py_src_mask, input_signal.copy(), f0, mode="additive"
    )

    py_sen_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    for off in sensor_offsets:
        py_sen_mask[Nx // 2 + off, Ny // 2, Nz // 2] = True
    kwa_sensor_obj = kw.Sensor.from_mask(py_sen_mask)

    kwa_sim = kw.Simulation(
        kwa_grid, kwa_medium, kwa_source_obj, kwa_sensor_obj,
        solver=kw.SolverType.PSTD,
    )
    kwa_sim.set_pml_size(pml_size)
    kwa_sim.set_pml_inside(True)

    kwa_res = kwa_sim.run(time_steps=Nt, dt=kw_dt)
    kwa_p = ensure_2d_sensors(kwa_res.sensor_data)

    # Align and compute metrics
    kw_aligned, kwa_aligned = align(kw_p, kwa_p)
    n = min(kw_aligned.shape[0], kwa_aligned.shape[0])
    all_metrics = []
    for i in range(n):
        m = compute_metrics(kw_aligned[i], kwa_aligned[i])
        all_metrics.append(m)
        print_metrics(m, f"Sensor {i+1} (offset={sensor_offsets[i]} cells)")

    # Pass/fail
    passed = all(m["corr"] > 0.90 and 0.80 < m["rms_ratio"] < 1.20 for m in all_metrics)
    print(f"\n  Result: {'PASS' if passed else 'FAIL'} "
          f"(target: corr>0.90, rms_ratio 0.80–1.20)")
    return kw_aligned, kwa_aligned


# ============================================================================
# Case 3: Heterogeneous Medium (two-layer water | tissue interface)
# ============================================================================

def run_heterogeneous_case(gpu: bool = False) -> tuple:
    """
    Two-layer heterogeneous medium comparison.

    Grid: 32³ isotropic, dx=2mm, pml_size=6, pml_inside=True
    Medium: x < Nx//2 → water (c=1500, ρ=1000), x ≥ Nx//2 → tissue (c=1550, ρ=1050)
    Source: 0.5 MHz tone burst, 3 cycles, at (pml_size+2, Ny//2, Nz//2)
    Sensor: at (Nx-pml_size-3, Ny//2, Nz//2) (tissue layer, near far PML)
    """
    print("\n" + "=" * 60)
    print("CASE 3: Heterogeneous Medium (two-layer water|tissue)")
    print("=" * 60)

    if not KWAVE_AVAILABLE:
        print("  SKIP — k-wave-python not available")
        return None, None

    Nx = Ny = Nz = 32
    dx = 2e-3
    pml_size = 6
    f0 = 0.5e6
    n_cycles = 3
    t_end = 20e-6

    c_water, rho_water = 1500.0, 1000.0
    c_tissue, rho_tissue = 1550.0, 1050.0

    c_arr = np.full((Nx, Ny, Nz), c_water)
    rho_arr = np.full((Nx, Ny, Nz), rho_water)
    c_arr[Nx // 2:, :, :] = c_tissue
    rho_arr[Nx // 2:, :, :] = rho_tissue

    c_max = c_tissue

    # --- k-Wave ---
    print("  Running k-wave-python...")
    kw_grid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
    kw_grid.makeTime(c_max, t_end=t_end)
    kw_dt = float(kw_grid.dt)
    Nt = int(kw_grid.Nt)
    print(f"  dt={kw_dt:.4e} s, Nt={Nt}")

    input_signal = tone_burst(1.0 / kw_dt, f0, n_cycles).flatten()
    if len(input_signal) < Nt:
        input_signal = np.pad(input_signal, (0, Nt - len(input_signal)))
    else:
        input_signal = input_signal[:Nt]

    src_ix = pml_size + 2
    sen_ix = Nx - pml_size - 3

    src_mask = np.zeros((Nx, Ny, Nz))
    src_mask[src_ix, Ny // 2, Nz // 2] = 1.0
    sen_mask = np.zeros((Nx, Ny, Nz))
    sen_mask[sen_ix, Ny // 2, Nz // 2] = 1

    kw_source_obj = kSource()
    kw_source_obj.p_mask = src_mask
    kw_source_obj.p = input_signal.reshape(1, -1)
    kw_source_obj.p_mode = "additive"
    kw_sensor_obj = kSensor(sen_mask)
    kw_sensor_obj.record = ["p"]

    kw_res = kspaceFirstOrder3D(
        medium=kWaveMedium(sound_speed=c_arr, density=rho_arr),
        kgrid=kw_grid,
        source=kw_source_obj,
        sensor=kw_sensor_obj,
        simulation_options=SimulationOptions(
            pml_inside=True, pml_size=pml_size,
            data_cast="double", save_to_disk=True,
        ),
        execution_options=SimulationExecutionOptions(
            is_gpu_simulation=gpu, delete_data=True, verbose_level=0,
        ),
    )
    kw_p = ensure_2d_sensors(np.array(kw_res["p"]))

    # --- pykwavers ---
    print("  Running pykwavers...")
    kwa_grid = kw.Grid(nx=Nx, ny=Ny, nz=Nz, dx=dx, dy=dx, dz=dx)
    kwa_medium = kw.Medium(
        sound_speed=c_arr.astype(np.float64),
        density=rho_arr.astype(np.float64),
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
    kwa_p = ensure_2d_sensors(kwa_res.sensor_data)

    kw_aligned, kwa_aligned = align(kw_p, kwa_p)
    n = min(kw_aligned.shape[0], kwa_aligned.shape[0])
    all_metrics = []
    for i in range(n):
        m = compute_metrics(kw_aligned[i], kwa_aligned[i])
        all_metrics.append(m)
        print_metrics(m, f"Sensor {i+1} (src_ix={src_ix}, sen_ix={sen_ix})")

    passed = all(m["corr"] > 0.99 and 0.95 < m["amp_ratio"] < 1.05 for m in all_metrics)
    print(f"\n  Result: {'PASS' if passed else 'FAIL'} "
          f"(target: corr>0.99, amp_ratio 0.95–1.05)")
    return kw_aligned, kwa_aligned


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical kwavers vs k-wave-python comparison"
    )
    parser.add_argument(
        "--case",
        choices=["ivp", "point_source", "heterogeneous", "all"],
        default="all",
        help="Which case to run (default: all)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU for k-Wave simulation (default: CPU)",
    )
    args = parser.parse_args()

    print("kwavers vs k-wave-python — Canonical Comparison Harness")
    print("=" * 60)
    print(f"GPU mode: {args.gpu}")

    results = {}

    if args.case in ("ivp", "all"):
        kw_data, kwa_data = run_ivp_case(gpu=args.gpu)
        results["ivp"] = (kw_data, kwa_data)

    if args.case in ("point_source", "all"):
        kw_data, kwa_data = run_point_source_case(gpu=args.gpu)
        results["point_source"] = (kw_data, kwa_data)

    if args.case in ("heterogeneous", "all"):
        kw_data, kwa_data = run_heterogeneous_case(gpu=args.gpu)
        results["heterogeneous"] = (kw_data, kwa_data)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
