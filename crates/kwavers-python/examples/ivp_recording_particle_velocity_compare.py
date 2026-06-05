"""
Recording Particle Velocity — pykwavers vs k-wave-python comparison.

Ported from: k-Wave/examples/example_ivp_recording_particle_velocity.m
Reference:   external/k-wave-python/examples/ivp_recording_particle_velocity.py

Physics
-------
A smoothed disc source at the grid centre emits an initial pressure pulse. Four binary
sensor points are placed at equal radii in the cardinal directions (+x, -x,
+y, -y). The x-axis sensors should show large ux and near-zero uy; the y-axis
sensors should show the opposite.

Grid adaptation (3-D embedding of 2-D problem)
-----------------------------------------------
k-wave-python runs a 2-D (Nx × Ny) simulation. pykwavers is 3-D only, so we
embed via a singleton z-dimension: (Nx, Ny, 1). The disc source becomes a
column in z. Sensor positions carry the same (x, y) coordinates with z = 0.

Acceptance criteria
-------------------
- Pressure (p): Pearson r ≥ 0.99 for all 4 sensors.
- Velocity (dominant direction only): Pearson r ≥ 0.95.
  - ux at ±x sensors (where ux dominates)
  - uy at ±y sensors (where uy dominates)
  - Near-zero-direction pairs (ux at ±y, uy at ±x) are NOT gated because
    both solvers produce near-zero signals there; their correlation is
    noise-dominated and carries no physics information.

Source-preprocessing invariant
------------------------------
k-wave-python smooths initial pressure by default for multidimensional IVP
runs. pykwavers receives explicit arrays and does not apply this implicit
preprocessing. To isolate solver and recorder parity, both engines receive the
same k-Wave Blackman-smoothed p0, and the k-wave call sets smooth_p0=False to
prevent double smoothing.

Algorithm reference
-------------------
Treeby & Cox (2010). k-Wave: MATLAB toolbox for simulation and reconstruction
of photoacoustic wave fields. J. Biomed. Opt. 15(2), 021314.
"""

import sys
import numpy as np
from pathlib import Path

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    save_side_by_side_parity_figure,
    save_text_report,
)

# Windows console UTF-8 compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── k-wave-python import ──────────────────────────────────────────────────────
kwave_root = Path(__file__).parents[3] / "external" / "k-wave-python"
if str(kwave_root) not in sys.path:
    sys.path.insert(0, str(kwave_root))

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.filters import smooth as kwave_smooth
from kwave.utils.mapgen import make_disc

# ── pykwavers import ──────────────────────────────────────────────────────────
pykwavers_root = Path(__file__).parents[1] / "python"
if str(pykwavers_root) not in sys.path:
    sys.path.insert(0, str(pykwavers_root))

import pykwavers as kw

# =============================================================================
# Simulation parameters (shared between both solvers)
#
# These match the k-wave-python reference example parameters (128×128 grid,
# SENSOR_RADIUS=40) so the 4 sensors sit inside the PML-free interior
# [20, 107] for the default PML thickness of 20 cells.
# =============================================================================
NX = 128         # grid points — matches reference example
NY = 128
DX = 0.1e-3      # [m] — grid spacing (matches reference)
DY = 0.1e-3
C0 = 1500.0      # [m/s]
RHO0 = 1000.0    # [kg/m³]
ALPHA_COEFF = 0.75   # [dB/(MHz^y cm)] — same as reference
ALPHA_POWER = 1.5
T_END = 6e-6     # [s]  — matches reference
SENSOR_RADIUS = 40  # [grid points] from centre (sensors at cx±40 ∈ [24,104])
PRESSURE_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "ivp_recording_particle_velocity_pressure_compare.png"
UX_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "ivp_recording_particle_velocity_ux_compare.png"
UY_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "ivp_recording_particle_velocity_uy_compare.png"
REPORT_PATH = DEFAULT_OUTPUT_DIR / "ivp_recording_particle_velocity_metrics.txt"


def make_sensor_mask_2d(nx: int, ny: int) -> np.ndarray:
    """Binary 2-D sensor mask: 4 cardinal points at SENSOR_RADIUS from centre.

    Uses cx = nx//2 = ny//2, matching make_disc_p0_2d so all 4 sensors are
    exactly SENSOR_RADIUS cells from the disc centre.
    For NX=128, PML=20: interior [20,107]; sensors at cx±40 ∈ [24,104] ⊂ [20,107].
    """
    mask = np.zeros((nx, ny), dtype=bool)
    cx = nx // 2  # 64 for NX=128 — matches disc centre
    cy = ny // 2
    mask[cx + SENSOR_RADIUS, cy] = True  # +x
    mask[cx - SENSOR_RADIUS, cy] = True  # -x
    mask[cx, cy + SENSOR_RADIUS] = True  # +y
    mask[cx, cy - SENSOR_RADIUS] = True  # -y
    return mask


def make_disc_p0_2d_raw(nx: int, ny: int) -> np.ndarray:
    """Initial pressure: disc at grid centre, radius 5 cells, magnitude 5.

    Matches the k-wave-python reference example (disc_magnitude=5, disc_radius=5).
    """
    cx = nx // 2
    cy = ny // 2
    return 5.0 * make_disc(Vector([nx, ny]), Vector([cx, cy]), 5).astype(float)


def make_disc_p0_2d(nx: int, ny: int) -> np.ndarray:
    """Return the k-Wave-smoothed initial pressure shared by both engines."""
    return np.asarray(kwave_smooth(make_disc_p0_2d_raw(nx, ny), restore_max=True), dtype=np.float64)


# =============================================================================
# k-wave-python reference
# =============================================================================

def run_kwave_python() -> dict:
    """Run k-wave-python 2-D simulation; return dict with 'p', 'ux', 'uy'."""
    kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
    kgrid.makeTime(C0, t_end=T_END)

    medium = kWaveMedium(sound_speed=C0, density=RHO0, alpha_coeff=ALPHA_COEFF, alpha_power=ALPHA_POWER)

    source = kSource()
    source.p0 = make_disc_p0_2d(NX, NY)

    sensor_mask_2d = make_sensor_mask_2d(NX, NY)
    sensor = kSensor(mask=sensor_mask_2d)
    sensor.record = ["p", "ux", "uy"]

    return kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend="python",
        device="cpu",
        quiet=True,
        pml_inside=True,
        smooth_p0=False,
    )


# =============================================================================
# pykwavers simulation
# =============================================================================

def run_pykwavers() -> kw.SimulationResult:
    """Run pykwavers 3-D (Nx × Ny × 1) simulation with velocity recording."""
    # Grid: embed 2-D problem in 3-D by using NZ = 1.
    grid = kw.Grid(nx=NX, ny=NY, nz=1, dx=DX, dy=DY, dz=DX)

    medium = kw.Medium.homogeneous(sound_speed=C0, density=RHO0)

    # Initial pressure: disc in (x, y), constant in z (singleton axis).
    p0_2d = make_disc_p0_2d(NX, NY)
    p0_3d = p0_2d[:, :, np.newaxis]  # (NX, NY, 1)
    source = kw.Source.from_initial_pressure(p0_3d)

    # Sensor mask: 4 binary points in (x, y), z = 0.
    mask_2d = make_sensor_mask_2d(NX, NY)
    mask_3d = mask_2d[:, :, np.newaxis]  # (NX, NY, 1)
    sensor = kw.Sensor.from_mask(mask_3d)
    sensor.set_record(["p", "ux", "uy"])

    # PSTD solver required: FDTD stepper does not call record_velocity_step.
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)

    # Compute time steps from CFL (matching k-wave-python auto-time).
    # k-wave-python uses dt = cfl * dx / c0 (no sqrt(ndim) factor).
    # The k-space pseudospectral method is stable for c0*dt/dx ≤ 1/sqrt(3)≈0.577;
    # cfl=0.3 satisfies this in all dimensions.
    cfl = 0.3
    dt = cfl * DX / C0
    nt = int(np.round(T_END / dt)) + 1

    return sim.run(time_steps=nt, dt=dt)


# =============================================================================
# Pearson correlation helper
# =============================================================================

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Element-wise Pearson correlation between two arrays of equal shape."""
    a_flat = np.asarray(a, dtype=float).ravel()
    b_flat = np.asarray(b, dtype=float).ravel()
    if a_flat.std() < 1e-30 and b_flat.std() < 1e-30:
        return 1.0  # both constant-zero — perfectly correlated
    if a_flat.std() < 1e-30 or b_flat.std() < 1e-30:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


# =============================================================================
# Main comparison
# =============================================================================

def main() -> None:
    print("=== ivp_recording_particle_velocity_compare ===")
    print(f"Grid: {NX}x{NY}x1  dx={DX*1e3:.2f} mm  t_end={T_END*1e6:.1f} us")
    print()

    print("Running k-wave-python (2-D) ...")
    kw_result = run_kwave_python()
    p_kw  = np.asarray(kw_result["p"])   # (4, Nt)
    ux_kw = np.asarray(kw_result["ux"])  # (4, Nt)
    uy_kw = np.asarray(kw_result["uy"])  # (4, Nt)
    print(f"  p  shape={p_kw.shape}  range=[{p_kw.min():.3f}, {p_kw.max():.3f}]")
    print(f"  ux shape={ux_kw.shape}  range=[{ux_kw.min():.3e}, {ux_kw.max():.3e}]")
    print(f"  uy shape={uy_kw.shape}  range=[{uy_kw.min():.3e}, {uy_kw.max():.3e}]")
    print()

    print("Running pykwavers (3-D embed) ...")
    kwr_result = run_pykwavers()
    # sensor_data shape: (4, Nt_kwr) — may differ from Nt_kw
    p_kwr  = np.asarray(kwr_result.sensor_data)  # (4, Nt_kwr)
    ux_kwr = np.asarray(kwr_result.ux) if kwr_result.ux is not None else None
    uy_kwr = np.asarray(kwr_result.uy) if kwr_result.uy is not None else None
    print(f"  p  shape={p_kwr.shape}  range=[{p_kwr.min():.3f}, {p_kwr.max():.3f}]")
    if ux_kwr is not None:
        print(f"  ux shape={ux_kwr.shape}  range=[{ux_kwr.min():.3e}, {ux_kwr.max():.3e}]")
    if uy_kwr is not None:
        print(f"  uy shape={uy_kwr.shape}  range=[{uy_kwr.min():.3e}, {uy_kwr.max():.3e}]")
    print()

    # Trim to the shorter time axis for comparison.
    nt_min = min(p_kw.shape[1], p_kwr.shape[1])
    p_kw_t  = p_kw[:, :nt_min]
    p_kwr_t = p_kwr[:, :nt_min]

    P_THRESHOLD = 0.99    # pressure: shared smoothed source and aligned sensors
    V_THRESHOLD = 0.95    # velocity dominant direction after source-preprocessing parity
    passed = True
    correlation_rows: list[str] = []
    # Diagnostic: print raw velocity RMS per sensor index (before reordering)
    if ux_kwr is not None and uy_kwr is not None:
        print("\n[diag] pykwavers raw sensor order (Fortran) vs k-wave (C-order):")
        for i in range(4):
            rms_p  = float(np.sqrt(np.mean(p_kwr[:, :nt_min][i]**2)))
            rms_ux = float(np.sqrt(np.mean(ux_kwr[:, :nt_min][i]**2)))
            rms_uy = float(np.sqrt(np.mean(uy_kwr[:, :nt_min][i]**2)))
            dom = "ux" if rms_ux > rms_uy else "uy"
            print(f"  pykwr[{i}]: rms_p={rms_p:.3e}  rms_ux={rms_ux:.3e}  rms_uy={rms_uy:.3e}  dom={dom}")
        print()
        for i in range(4):
            rms_p  = float(np.sqrt(np.mean(p_kw[:, :nt_min][i]**2)))
            rms_ux = float(np.sqrt(np.mean(ux_kw[:, :nt_min][i]**2)))
            rms_uy = float(np.sqrt(np.mean(uy_kw[:, :nt_min][i]**2)))
            dom = "ux" if rms_ux > rms_uy else "uy"
            print(f"  kwave[{i}]: rms_p={rms_p:.3e}  rms_ux={rms_ux:.3e}  rms_uy={rms_uy:.3e}  dom={dom}")
        print()

    # k-wave-python extracts sensor positions in numpy C-order (row-major):
    #   flat_C(i,j) = i*NY + j  (NX=128, NY=128, cx=64, cy=64, r=40)
    #   (24,64)=3136, (64,24)=8216, (64,104)=8296, (104,64)=13376
    #   Sorted: (24,64)=3136 → −x, (64,24)=8216 → −y, (64,104)=8296 → +y, (104,64)=13376 → +x
    #   → kwave order: [0=−x, 1=−y, 2=+y, 3=+x]
    #
    # pykwavers extracts sensor positions in Fortran order from the 3-D mask:
    #   flat_F(i,j,k) = i + j*NX  (NX=128)
    #   (24,64): 24+64*128=8216, (64,24): 64+24*128=3136, (64,104): 64+104*128=13376, (104,64): 104+64*128=8296
    #   Sorted: 3136→(64,24)=−y, 8216→(24,64)=−x, 8296→(104,64)=+x, 13376→(64,104)=+y
    #   → pykwavers order: [0=−y, 1=−x, 2=+x, 3=+y]
    #
    # Reindex pykwavers to match k-wave C-order: kwave[i] ↔ pykwavers[REORDER[i]]
    REORDER = [1, 0, 3, 2]  # kwave[-x]→pykwr[1], [-y]→pykwr[0], [+y]→pykwr[3], [+x]→pykwr[2]

    print("-- Pearson correlations (sensor-wise) ----------------------------------")
    sensor_labels = ["-x", "-y", "+y", "+x"]      # k-wave C-order
    sensor_expect_ux = [True, False, False, True]  # True ↔ ux should dominate

    # Reorder pykwavers rows to match k-wave C-order before all comparisons.
    p_kwr_t = p_kwr_t[REORDER]

    # Pressure
    print(f"\nPressure (p)  [threshold={P_THRESHOLD}]:")
    for i, label in enumerate(sensor_labels):
        r = pearson_r(p_kw_t[i], p_kwr_t[i])
        correlation_rows.append(f"p[{label}]_pearson_r: {r:.6f}")
        ok = "✓" if r >= P_THRESHOLD else "✗"
        print(f"  {ok} sensor {label}: r={r:.4f}")
        if r < P_THRESHOLD:
            passed = False

    # ux and uy — threshold applied only to the dominant-direction sensors.
    # At a y-axis sensor (±y), ux is near-zero in both solvers; its Pearson
    # correlation is noise-dominated and meaningless. The acceptance criterion
    # is: ux at ±x sensors ≥ THRESHOLD, uy at ±y sensors ≥ THRESHOLD.
    if ux_kwr is not None:
        ux_kwr_t = ux_kwr[:, :nt_min][REORDER]
        ux_kw_t  = ux_kw[:, :nt_min]
        print(f"\nParticle velocity ux  [threshold={V_THRESHOLD} dominant, no gate near-zero]:")
        for i, (label, expect_ux) in enumerate(zip(sensor_labels, sensor_expect_ux)):
            r = pearson_r(ux_kw_t[i], ux_kwr_t[i])
            correlation_rows.append(f"ux[{label}]_pearson_r: {r:.6f}")
            is_dominant = expect_ux  # ux dominant at ±x sensors
            ok = "✓" if (r >= V_THRESHOLD or not is_dominant) else "✗"
            dom_str = "(dominant)" if is_dominant else "(near-zero)"
            print(f"  {ok} sensor {label} {dom_str}: r={r:.4f}")
            if r < V_THRESHOLD and is_dominant:
                passed = False
    else:
        print("\nWARNING: ux not recorded by pykwavers (check record modes)")
        passed = False

    if uy_kwr is not None:
        uy_kwr_t = uy_kwr[:, :nt_min][REORDER]
        uy_kw_t  = uy_kw[:, :nt_min]
        print(f"\nParticle velocity uy  [threshold={V_THRESHOLD} dominant, no gate near-zero]:")
        for i, (label, expect_ux) in enumerate(zip(sensor_labels, sensor_expect_ux)):
            r = pearson_r(uy_kw_t[i], uy_kwr_t[i])
            correlation_rows.append(f"uy[{label}]_pearson_r: {r:.6f}")
            is_dominant = not expect_ux  # uy dominant at ±y sensors
            ok = "✓" if (r >= V_THRESHOLD or not is_dominant) else "✗"
            dom_str = "(dominant)" if is_dominant else "(near-zero)"
            print(f"  {ok} sensor {label} {dom_str}: r={r:.4f}")
            if r < V_THRESHOLD and is_dominant:
                passed = False
    else:
        print("\nWARNING: uy not recorded by pykwavers (check record modes)")
        passed = False

    # Physical sanity check: x-axis sensors have |ux| >> |uy|, y-axis opposite.
    print("\n-- Physical sanity: directional velocity dominance (pykwavers, reordered) --")
    if ux_kwr is not None and uy_kwr is not None:
        # Use already-reordered arrays (matched to k-wave C-order).
        ux_t = ux_kwr_t
        uy_t = uy_kwr_t
        for i, (label, expect_ux) in enumerate(
            zip(sensor_labels, sensor_expect_ux)
        ):
            rms_ux = float(np.sqrt(np.mean(ux_t[i] ** 2)))
            rms_uy = float(np.sqrt(np.mean(uy_t[i] ** 2)))
            dominant_is_ux = rms_ux > rms_uy
            ok = "✓" if dominant_is_ux == expect_ux else "✗"
            dominant_str = "ux" if dominant_is_ux else "uy"
            expected_str = "ux" if expect_ux else "uy"
            print(
                f"  {ok} sensor {label}: dominant={dominant_str} (expected {expected_str})"
                f"  rms_ux={rms_ux:.3e}  rms_uy={rms_uy:.3e}"
            )
            if dominant_is_ux != expect_ux:
                passed = False

    figure_paths = [
        save_side_by_side_parity_figure(
            p_kw_t,
            p_kwr_t,
            PRESSURE_FIGURE_PATH,
            title="ivp_recording_particle_velocity pressure parity",
            reference_label="k-wave-python p",
            candidate_label="pykwavers p",
            cmap="seismic",
        )
    ]
    if ux_kwr is not None:
        figure_paths.append(
            save_side_by_side_parity_figure(
                ux_kw_t,
                ux_kwr_t,
                UX_FIGURE_PATH,
                title="ivp_recording_particle_velocity ux parity",
                reference_label="k-wave-python ux",
                candidate_label="pykwavers ux",
                cmap="seismic",
            )
        )
    if uy_kwr is not None:
        figure_paths.append(
            save_side_by_side_parity_figure(
                uy_kw_t,
                uy_kwr_t,
                UY_FIGURE_PATH,
                title="ivp_recording_particle_velocity uy parity",
                reference_label="k-wave-python uy",
                candidate_label="pykwavers uy",
                cmap="seismic",
            )
        )
    for path in figure_paths:
        print(f"  image: {path}")

    status = "PASS" if passed else "FAIL"
    save_text_report(
        REPORT_PATH,
        "ivp_recording_particle_velocity parity report",
        [
            f"parity_status: {status}",
            f"grid: {NX} x {NY} x 1 dx={DX:.6e} m",
            "source_preprocessing: k-Wave Blackman smooth(restore_max=True), shared by both engines",
            "kwave_smooth_p0: False",
            f"pressure_threshold: {P_THRESHOLD}",
            f"dominant_velocity_threshold: {V_THRESHOLD}",
            *correlation_rows,
            *(f"figure: {path.name}" for path in figure_paths),
        ],
    )
    print(f"  report: {REPORT_PATH}")

    print()
    if passed:
        print(f"RESULT: PASS — p r>={P_THRESHOLD}, dominant-velocity r>={V_THRESHOLD}, physics checks passed.")
        sys.exit(0)
    else:
        print("RESULT: FAIL — one or more checks below threshold.")
        sys.exit(1)


if __name__ == "__main__":
    main()
