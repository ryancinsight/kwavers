#!/usr/bin/env python3
"""
tvsp_transducer_field_patterns_compare.py
==========================================
Side-by-side parity comparison for the upstream k-Wave example
``example_tvsp_transducer_field_patterns.m``: a curved-arc pressure
transducer driven by a continuous sine wave in a homogeneous absorbing
medium, recording the time-averaged beam pattern (``p_max``, ``p_rms``)
and the final wavefield (``p_final``).

⚠️  KNOWN PARITY GAP — pykwavers continuous-pressure-mask-source path

This script currently FAILS with a **4×10⁸-fold amplitude blow-up** on
the pykwavers side (``p_final`` peak ≈ 4.4e8 Pa vs k-wave-python's
1.04 Pa for an identical setup). The blow-up is independent of:

- ``mode = "additive" | "additive_no_correction"``    (tested, both fail)
- ``enable_nonlinear``                                 (default already False)
- ``pml_inside = True | False``                        (tested both)

This points to a real bug in pykwavers' Source.from_mask continuous-
source dispatch path — possibly the kspace source-correction term, the
PML-source-overlap handling, or the source-injection scaling per voxel.

The script is committed in this state as the parity-harness scaffold
for once that pykwavers bug is fixed; the FAIL status is the truthful
report. The infrastructure (shared inputs, dual-engine runs, 9-panel
side-by-side figure) is in place and produces meaningful diagnostic
output today (the figure shows the blow-up clearly).

When the bug lands a fix, this script should reach Pearson r ≥ 0.95
on all three recorded fields (p_final / p_max / p_rms) per the
configured PARITY_THRESHOLDS.

This is a classic ultrasound imaging demo — the focused beam pattern
is one of the most visually distinctive figures in acoustics tutorials.

Physical setup (matches the MATLAB script verbatim)
---------------------------------------------------
Grid    : 216×216 with ``dx = dy = 50e-3 / 216 ≈ 0.231 mm``
Medium  : homogeneous water with absorption
          ``c0 = 1500 m/s, ρ0 = 1000 kg/m³,
           α = 0.75 dB/(MHz^y cm), y = 1.5``
Source  : curved-arc transducer at ``arc_pos = (20, 20)`` with
          ``radius = 60 pts``, ``diameter = 81 pts``, focused on
          ``focus_pos = (Nx/2, Nx/2)``. Driven by a continuous sine
          ``p(t) = 0.5·sin(2π·0.25 MHz·t)``, low-pass-filtered via
          ``filterTimeSeries``.
Sensor  : full-grid sensor with ``record_modes = ["p_final", "p_max", "p_rms"]``.
Time    : ``kgrid.makeTime(c0)`` → CFL-stable ``Nt`` and ``dt``.
PML     : 20 grid points, **outside** the domain (default in MATLAB
          k-Wave; ``pml_inside=False``).

Comparison strategy
-------------------
Both engines receive identical inputs (arc mask via k-wave-python's
``make_arc`` utility, same low-pass-filtered driving signal, same
full-grid sensor mask). The three recorded fields (``p_final``,
``p_max``, ``p_rms``) are compared with image-level Pearson r,
RMS ratio, PSNR. Side-by-side figures show:

  Top row    — p_max:   k-wave-python | pykwavers | difference
  Middle row — p_rms:   k-wave-python | pykwavers | difference
  Bottom row — p_final: k-wave-python | pykwavers | difference

The arc transducer mask is overlaid on each panel for context.

Outputs
-------
* ``output/tvsp_transducer_field_patterns_compare.png`` — 9-panel figure
* ``output/tvsp_transducer_field_patterns_metrics.txt`` — per-mode
  Pearson r, RMS ratio, PSNR, runtimes.

Usage
-----
    python examples/tvsp_transducer_field_patterns_compare.py
    python examples/tvsp_transducer_field_patterns_compare.py --no-cache
    python examples/tvsp_transducer_field_patterns_compare.py --allow-failure
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw

# ---------------------------------------------------------------------------
# Physical constants (matches MATLAB example_tvsp_transducer_field_patterns.m)
# ---------------------------------------------------------------------------
NX = NY = 216
DX = DY = 50e-3 / NX  # m

C0 = 1500.0
RHO0 = 1000.0
ALPHA_COEFF = 0.75
ALPHA_POWER = 1.5

# Arc transducer geometry (1-indexed grid points, MATLAB convention)
ARC_POS = (20, 20)
ARC_RADIUS = 60
ARC_DIAMETER = 81
FOCUS_POS = (NX // 2, NX // 2)

SOURCE_FREQ = 0.25e6  # 0.25 MHz
SOURCE_MAG = 0.5  # Pa

PML_SIZE = 20  # MATLAB default for kspaceFirstOrder2D

# ---------------------------------------------------------------------------
# Parity thresholds — homogeneous water + filtered continuous source +
# focused-arc geometry. PML-stripped p_max / p_rms / p_final fields should
# show high agreement; absorption is identically configured in both engines.
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r": 0.95,
    "rms_ratio_min": 0.85,
    "rms_ratio_max": 1.15,
    "psnr_db": 18.0,
}

FIGURE_PATH = DEFAULT_OUTPUT_DIR / "tvsp_transducer_field_patterns_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "tvsp_transducer_field_patterns_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_transducer_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_transducer_pykwavers_cache.npz"

REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _load_cache(path: os.PathLike) -> dict | None:
    if REFRESH_CACHE or not os.path.exists(os.fspath(path)):
        return None
    try:
        d = np.load(os.fspath(path), allow_pickle=False)
        if int(np.asarray(d["cache_version"]).reshape(())) != CACHE_VERSION:
            return None
        return {
            "p_final": np.asarray(d["p_final"], dtype=np.float64),
            "p_max": np.asarray(d["p_max"], dtype=np.float64),
            "p_rms": np.asarray(d["p_rms"], dtype=np.float64),
            "nt": int(d["nt"]),
            "dt": float(d["dt"]),
            "runtime_s": float(d["runtime_s"]),
        }
    except Exception:
        return None


def _save_cache(
    path: os.PathLike,
    p_final: np.ndarray,
    p_max: np.ndarray,
    p_rms: np.ndarray,
    nt: int,
    dt: float,
    runtime_s: float,
) -> None:
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    np.savez(
        os.fspath(path),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        p_final=np.asarray(p_final, dtype=np.float64),
        p_max=np.asarray(p_max, dtype=np.float64),
        p_rms=np.asarray(p_rms, dtype=np.float64),
        nt=np.array(nt, dtype=np.int64),
        dt=np.array(dt, dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )


def _resolve_field(
    arr_flat_or_2d: np.ndarray, shape: tuple[int, int], pml_size: int
) -> np.ndarray:
    """Embed a 1-D or interior-2-D recorded field into a full (NX, NY) canvas.

    Both engines may return either:
      - the flat full-grid field reshaped (NX*NY,)
      - the flat interior field (Nx-2P)*(Ny-2P,)
      - the 2-D interior field directly
      - the 2-D full-grid field

    Normalise to (NX, NY) with zeros in the PML annulus.
    """
    nx, ny = shape
    p = pml_size
    interior_shape = (nx - 2 * p, ny - 2 * p)

    if arr_flat_or_2d.ndim == 2:
        if arr_flat_or_2d.shape == (nx, ny):
            return arr_flat_or_2d.astype(np.float64)
        if arr_flat_or_2d.shape == interior_shape:
            full = np.zeros((nx, ny), dtype=np.float64)
            full[p:-p, p:-p] = arr_flat_or_2d
            return full
        if arr_flat_or_2d.shape == interior_shape[::-1]:
            full = np.zeros((nx, ny), dtype=np.float64)
            full[p:-p, p:-p] = arr_flat_or_2d.T
            return full
        if arr_flat_or_2d.shape == (ny, nx):
            return arr_flat_or_2d.T.astype(np.float64)
        raise AssertionError(
            f"Unexpected 2-D field shape {arr_flat_or_2d.shape}; "
            f"expected ({nx}, {ny}) or interior {interior_shape}"
        )

    # 1-D flat
    flat = arr_flat_or_2d.flatten()
    if flat.size == nx * ny:
        # Fortran order (pykwavers serialisation convention)
        return flat.reshape((ny, nx)).T.astype(np.float64)
    if flat.size == interior_shape[0] * interior_shape[1]:
        nxi, nyi = interior_shape
        full = np.zeros((nx, ny), dtype=np.float64)
        full[p:-p, p:-p] = flat.reshape((nyi, nxi)).T
        return full
    raise AssertionError(
        f"Unexpected 1-D field size {flat.size}; "
        f"expected {nx*ny} or {interior_shape[0]*interior_shape[1]}"
    )


# ---------------------------------------------------------------------------
# Shared inputs builder
# ---------------------------------------------------------------------------
def build_shared_inputs() -> dict:
    """Construct grid, arc transducer mask, filtered source signal."""
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.utils.filters import filter_time_series
    from kwave.utils.mapgen import make_arc

    kgrid = kWaveGrid(Vector([NX, NY]), Vector([DX, DY]))
    medium = kWaveMedium(
        sound_speed=C0, alpha_coeff=ALPHA_COEFF, alpha_power=ALPHA_POWER
    )
    kgrid.makeTime(medium.sound_speed)
    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)
    t_array = kgrid.t_array.flatten().astype(np.float64)

    # Arc transducer source mask (k-wave-python make_arc).
    arc_mask = np.asarray(
        make_arc(Vector([NX, NY]), Vector(list(ARC_POS)), ARC_RADIUS, ARC_DIAMETER, Vector(list(FOCUS_POS))),
        dtype=np.float64,
    )
    # make_arc returns a 0/1 grid; ensure non-zero count.
    n_active = int(np.count_nonzero(arc_mask))
    if n_active == 0:
        raise RuntimeError("make_arc produced an empty mask")

    # Driving signal: continuous sine, filtered.
    raw_signal = SOURCE_MAG * np.sin(2.0 * np.pi * SOURCE_FREQ * t_array)
    raw_signal_2d = raw_signal[None, :]
    filtered_signal = np.asarray(
        filter_time_series(kgrid, medium, raw_signal_2d), dtype=np.float64
    ).flatten()

    return {
        "kgrid": kgrid,
        "medium": medium,
        "arc_mask": arc_mask,
        "n_active": n_active,
        "p_signal_filtered": filtered_signal,
        "nt": nt,
        "dt": dt,
        "t_array": t_array,
    }


# ---------------------------------------------------------------------------
# k-wave-python run
# ---------------------------------------------------------------------------
def run_kwave(inputs: dict, *, no_cache: bool = False) -> dict:
    if not no_cache:
        cached = _load_cache(_KWAVE_CACHE)
        if cached is not None:
            print("  [k-wave] Loading from cache...")
            return cached

    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    kgrid = inputs["kgrid"]
    medium = inputs["medium"]
    arc_mask = inputs["arc_mask"]
    p_signal = inputs["p_signal_filtered"]
    nt = inputs["nt"]
    dt = inputs["dt"]

    source = kSource()
    source.p_mask = (arc_mask != 0).astype(bool)
    source.p = p_signal  # 1-D, broadcast across mask points

    sensor = kSensor(mask=np.ones((NX, NY), dtype=bool))
    sensor.record = ["p_final", "p_max", "p_rms"]

    print(f"  [k-wave] Running 2-D PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        smooth_p0=False,
        pml_inside=False,
        pml_size=PML_SIZE,
        backend="python",
        device="cpu",
        quiet=True,
    )
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    p_final = _resolve_field(np.asarray(result["p_final"]), (NX, NY), PML_SIZE)
    p_max = _resolve_field(np.asarray(result["p_max"]), (NX, NY), PML_SIZE)
    p_rms = _resolve_field(np.asarray(result["p_rms"]), (NX, NY), PML_SIZE)

    _save_cache(_KWAVE_CACHE, p_final, p_max, p_rms, nt, dt, runtime_s)
    return {
        "p_final": p_final,
        "p_max": p_max,
        "p_rms": p_rms,
        "nt": nt,
        "dt": dt,
        "runtime_s": runtime_s,
    }


# ---------------------------------------------------------------------------
# pykwavers run
# ---------------------------------------------------------------------------
def run_pykwavers(inputs: dict, *, no_cache: bool = False) -> dict:
    if not no_cache:
        cached = _load_cache(_PKWAV_CACHE)
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    arc_mask = inputs["arc_mask"]
    p_signal = inputs["p_signal_filtered"]
    nt = inputs["nt"]
    dt = inputs["dt"]

    grid = pkw.Grid(NX, NY, 1, DX, DY, DX)
    medium = pkw.Medium.homogeneous(
        sound_speed=C0,
        density=RHO0,
        absorption=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
    )

    # Pressure-mask source via Source.from_mask. The frequency parameter
    # is informational — the actual driving signal is the supplied 1-D
    # filtered sinusoid broadcast across the mask points.
    arc_mask_3d = arc_mask[:, :, None].astype(np.float64)
    # k-wave-python's kspaceFirstOrder applies pressure-mask sources in
    # `additive_no_correction` mode by default (the kspace source-correction
    # is disabled when source.p_mode is unset). pykwavers' default
    # `additive` mode applies a kspace correction that double-counts here
    # and produces ~3e8× amplitude blow-up; switch explicitly to match
    # k-wave-python.
    source = pkw.Source.from_mask(
        mask=arc_mask_3d,
        signal=p_signal,
        frequency=SOURCE_FREQ,
        mode="additive_no_correction",
    )

    sensor_mask = np.ones((NX, NY, 1), dtype=bool)
    sensor = pkw.Sensor.from_mask(sensor_mask)
    sensor.set_record(["p_final", "p_max", "p_rms"])

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(False)

    print(f"  [pykwavers] Running CPU PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=nt, dt=dt)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    p_final = _resolve_field(np.asarray(result.p_final), (NX, NY), PML_SIZE)
    p_max = _resolve_field(np.asarray(result.p_max), (NX, NY), PML_SIZE)
    p_rms = _resolve_field(np.asarray(result.p_rms), (NX, NY), PML_SIZE)

    _save_cache(_PKWAV_CACHE, p_final, p_max, p_rms, nt, dt, runtime_s)
    return {
        "p_final": p_final,
        "p_max": p_max,
        "p_rms": p_rms,
        "nt": nt,
        "dt": dt,
        "runtime_s": runtime_s,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(
    inputs: dict,
    kw: dict,
    pkw_res: dict,
    metrics: dict[str, dict[str, float]],
    *,
    status: str,
) -> None:
    arc_mask = inputs["arc_mask"]
    fields = [
        ("p_max", "Maximum Pressure"),
        ("p_rms", "RMS Pressure"),
        ("p_final", "Final Wavefield"),
    ]

    p = PML_SIZE
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))

    # Crop to interior view
    arc_view = arc_mask[p:-p, p:-p].astype(bool) if NX > 2 * p else arc_mask.astype(bool)

    for row, (key, title) in enumerate(fields):
        kw_arr = kw[key][p:-p, p:-p]
        py_arr = pkw_res[key][p:-p, p:-p]
        diff = py_arr - kw_arr
        vmax = float(max(np.abs(kw_arr).max(), np.abs(py_arr).max(), 1e-30))
        dmax = float(max(np.abs(diff).max(), 1e-30))
        m = metrics[key]

        # Overlay arc mask in red on each panel
        for col, (arr, panel_title) in enumerate(
            [
                (kw_arr, f"k-wave-python  {title}"),
                (py_arr, f"pykwavers  {title}"),
                (diff, f"diff  max|Δ|={dmax:.2e}"),
            ]
        ):
            ax = axes[row, col]
            cmap = "seismic"
            v = vmax if col != 2 else dmax
            im = ax.imshow(
                arr.T,
                origin="lower",
                cmap=cmap,
                vmin=-v,
                vmax=v,
                extent=[0, arr.shape[0] * DX * 1e3, 0, arr.shape[1] * DY * 1e3],
                aspect="equal",
            )
            # Arc overlay (red, 35% alpha)
            arc_overlay = np.zeros((arc_view.shape[0], arc_view.shape[1], 4),
                                   dtype=np.float32)
            arc_overlay[arc_view, 0] = 1.0
            arc_overlay[arc_view, 3] = 0.5
            ax.imshow(
                np.transpose(arc_overlay, (1, 0, 2)),
                origin="lower",
                extent=[0, arr.shape[0] * DX * 1e3, 0, arr.shape[1] * DY * 1e3],
            )
            ax.set_title(panel_title, fontsize=10)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate the row with metrics
        m_str = (
            f"r={m['pearson_r']:.4f}  "
            f"rms_ratio={m['rms_ratio']:.4f}  "
            f"PSNR={m['psnr_db']:.1f} dB"
        )
        axes[row, 0].text(
            0.02, 0.97, m_str, transform=axes[row, 0].transAxes,
            fontsize=8, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
        )

    fig.suptitle(
        f"tvsp_transducer_field_patterns: k-wave-python vs pykwavers  [{status}]\n"
        f"Curved-arc transducer in absorbing water, "
        f"f={SOURCE_FREQ/1e6:.2f} MHz, A={SOURCE_MAG} Pa",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(str(FIGURE_PATH), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Compare pykwavers with k-wave-python for "
                     "tvsp_transducer_field_patterns.")
    )
    parser.add_argument("--no-cache", action="store_true",
                        help="Force a fresh run (ignore cached NPZ files).")
    parser.add_argument("--allow-failure", action="store_true",
                        help="Exit 0 even when parity targets fail.")
    args = parser.parse_args()
    no_cache = args.no_cache

    print("=" * 78)
    print("tvsp_transducer_field_patterns: k-wave-python vs pykwavers")
    print(f"  Grid    : {NX}×{NY}   dx={DX*1e3:.4f} mm")
    print(f"  Medium  : c0={C0} m/s  ρ0={RHO0} kg/m³  α={ALPHA_COEFF} (y={ALPHA_POWER})")
    print(f"  Source  : arc_pos={ARC_POS}  radius={ARC_RADIUS}  diameter={ARC_DIAMETER}")
    print(f"            focus={FOCUS_POS}  f={SOURCE_FREQ/1e6:.2f} MHz  A={SOURCE_MAG} Pa")
    print(f"  PML     : {PML_SIZE} pts outside")
    print("=" * 78)

    print("\n[0/2] Building shared inputs (arc mask, filtered signal)...")
    inputs = build_shared_inputs()
    print(f"  Nt={inputs['nt']}  dt={inputs['dt']:.3e} s  arc points={inputs['n_active']}")

    print("\n[1/2] k-wave-python ...")
    kw = run_kwave(inputs, no_cache=no_cache)
    for fld in ("p_final", "p_max", "p_rms"):
        print(f"  {fld:8s}: shape={kw[fld].shape}  peak={float(np.abs(kw[fld]).max()):.3e} Pa")

    print("\n[2/2] pykwavers ...")
    pkw_res = run_pykwavers(inputs, no_cache=no_cache)
    for fld in ("p_final", "p_max", "p_rms"):
        print(f"  {fld:8s}: shape={pkw_res[fld].shape}  "
              f"peak={float(np.abs(pkw_res[fld]).max()):.3e} Pa")

    # Per-mode metrics on PML-stripped interior.
    p = PML_SIZE
    metrics: dict[str, dict[str, float]] = {}
    for fld in ("p_final", "p_max", "p_rms"):
        kw_int = kw[fld][p:-p, p:-p]
        py_int = pkw_res[fld][p:-p, p:-p]
        metrics[fld] = compute_image_metrics(kw_int, py_int)

    thr = PARITY_THRESHOLDS
    print("\n--- Parity evaluation (per recording mode, PML-stripped) ---")
    all_ok = True
    for fld in ("p_max", "p_rms", "p_final"):
        m = metrics[fld]
        ok = (
            m["pearson_r"] >= thr["pearson_r"]
            and thr["rms_ratio_min"] <= m["rms_ratio"] <= thr["rms_ratio_max"]
            and m["psnr_db"] >= thr["psnr_db"]
        )
        all_ok = all_ok and ok
        print(f"  [{('OK' if ok else 'FAIL'):4s}] {fld:8s}  "
              f"r={m['pearson_r']:.4f}  rms_ratio={m['rms_ratio']:.4f}  "
              f"PSNR={m['psnr_db']:.2f} dB  rmse={m['rmse']:.3e}")
    status = "PASS" if all_ok else "FAIL"
    print(f"  Overall: {status}")
    print(f"  runtime: k-wave={kw['runtime_s']:.1f}s  pykwavers={pkw_res['runtime_s']:.1f}s")

    plot_comparison(inputs, kw, pkw_res, metrics, status=status)

    header_lines = [
        "tvsp_transducer_field_patterns parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}  dx={DX:.6e} m",
        f"medium: c0={C0}  rho0={RHO0}  alpha={ALPHA_COEFF}  alpha_power={ALPHA_POWER}",
        f"source: arc_pos={ARC_POS} radius={ARC_RADIUS} diameter={ARC_DIAMETER} "
        f"focus={FOCUS_POS} f={SOURCE_FREQ:.3e} Hz A={SOURCE_MAG} Pa",
        f"pml_size: {PML_SIZE} (outside)",
        f"nt={inputs['nt']}  dt={inputs['dt']:.6e} s",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
    ]
    report_lines: list[str] = []
    for fld in ("p_max", "p_rms", "p_final"):
        m = metrics[fld]
        report_lines.append(
            f"{fld:8s}: r={m['pearson_r']:.6f}  rms_ratio={m['rms_ratio']:.6f}  "
            f"PSNR={m['psnr_db']:.2f} dB  rmse={m['rmse']:.3e}  "
            f"peak_ratio={m['peak_ratio']:.4f}"
        )
    save_text_report(METRICS_PATH, "\n".join(header_lines), report_lines)
    print(f"  Saved: {METRICS_PATH}")
    return 0 if (status == "PASS" or args.allow_failure) else 1


if __name__ == "__main__":
    raise SystemExit(main())
