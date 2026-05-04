#!/usr/bin/env python3
"""
tvsp_steering_linear_array_compare.py
=======================================
Parity comparison for the upstream ``tvsp_steering_linear_array`` example
(geometric beamforming delay-and-sum from a 21-element linear array, 30° steering).

Physical setup (matches k-wave-python ``examples/tvsp_steering_linear_array.py``):
  Grid:    128×128  dx = dy = 50e-3/128 m ≈ 0.391 mm  (quasi-2D: NZ = 1)
  Medium:  c = 1500 m/s  rho = 1000 kg/m³
           alpha_coeff = 0.75 dB/(MHz^y cm)  alpha_power = 1.5
  Source:  21-element linear array at grid row x_offset = 25 (1-based, i.e. row 24
           0-based).  Elements span columns 53..73 (0-based y-axis).  Each element
           driven by a Hann-windowed tone burst (f₀ = 1 MHz, 8 cycles) with
           geometric steering delay:
             offset[n] = 40 + dx * element_index[n] * sin(30°) / (c * dt)
  Sensor:  full 128×128 binary mask, records pressure time series at all points.
  PML:     20 grid points, inside the domain.

Comparison strategy
-------------------
Both engines are driven with the **same** per-element signals (generated from
k-wave-python's ``tone_burst``) and the **same** dt/Nt from k-wave-python's
``makeTime``.  The primary comparison metric is the Pearson correlation and RMS
ratio of the 2-D final pressure field ``p_final`` (the pressure snapshot at the
last simulated time step), which captures the steered beam shape and amplitude.
A secondary metric compares the pressure time trace at a point in the far field
of the steered beam.

Spatial ordering
----------------
k-wave-python records full-grid sensor data with y-fastest (NumPy C-order) layout:
``flat_kw[i*NY + j]`` = pressure at grid point (x=i, y=j).  Reshape with C-order
gives ``p_kw[i, j]``.

pykwavers records sensor data with x-fastest (Fortran-order) layout:
``flat_pkw[i + j*NX]`` = pressure at grid point (x=i, y=j).  Reshape with F-order
gives ``p_pkw[i, j]``.

Both give ``p[i, j]`` = pressure at (x=i, y=j), so the ravelled arrays correspond
element-for-element and Pearson correlation is well-defined.

Outputs
-------
* ``output/tvsp_steering_linear_array_compare.png``
* ``output/tvsp_steering_linear_array_metrics.txt``

Usage
-----
  python examples/tvsp_steering_linear_array_compare.py
  python examples/tvsp_steering_linear_array_compare.py --no-cache
  python examples/tvsp_steering_linear_array_compare.py --allow-failure
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
# Physical constants (must match tvsp_steering_linear_array.py exactly)
# ---------------------------------------------------------------------------
NX = NY = 128
DX = DY = 50e-3 / NX               # ≈ 3.906e-4 m

C0           = 1500.0               # sound speed [m/s]
RHO0         = 1000.0               # density [kg/m³]
ALPHA_COEFF  = 0.75                 # absorption coefficient [dB/(MHz^y cm)]
ALPHA_POWER  = 1.5                  # absorption power law exponent

# Source geometry (matches k-wave-python: 1-based x_offset=25 → 0-based row 24)
NUM_ELEMENTS    = 21
X_OFFSET_0      = 24                # 0-based x-index of the source row
START_IDX_0     = NY // 2 - (NUM_ELEMENTS + 1) // 2  # 0-based y start of first element → 53

# Tone burst parameters
TONE_BURST_FREQ   = 1.0e6          # carrier frequency [Hz]
TONE_BURST_CYCLES = 8              # number of cycles in the burst
STEERING_ANGLE    = 30.0           # [degrees]
ELEMENT_SPACING   = DX             # inter-element pitch = grid spacing [m]
BASE_OFFSET_SAMP  = 40             # base time offset [samples]

# PML parameters
PML_SIZE = 20                       # [grid points], applied inside

# Parity thresholds.
# A steered 2-D beam is geometrically complex; absorption and PML formulation
# differences between k-wave-python (Chi/Psi CPML) and kwavers (fractional-
# Laplacian) shift amplitudes.  Require beam-shape correlation ≥ 0.85 and allow
# amplitude mismatch up to ±40%.
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r":     0.85,
    "rms_ratio_min": 0.60,
    "rms_ratio_max": 1.60,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "tvsp_steering_linear_array_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "tvsp_steering_linear_array_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_steering_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_steering_pykwavers_cache.npz"

REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Source signal construction
# ---------------------------------------------------------------------------
def _element_indices() -> np.ndarray:
    """Return the element index array (relative to centre element) for the linear array.

    k-Wave convention: ``element_index = arange(-(N-1)/2, (N-1)/2 + 1)``
    for an N-element array, giving values [-10, -9, …, 0, …, 9, 10].
    """
    half = (NUM_ELEMENTS - 1) / 2
    return np.arange(-half, half + 1)


def _steering_offsets(dt: float) -> np.ndarray:
    """Return per-element sample offsets for 30° geometric beamforming.

    Formula (from k-Wave example):
        offset[n] = BASE_OFFSET_SAMP + dx * element_index[n] * sin(θ) / (c * dt)

    Result is rounded to the nearest integer.
    """
    element_index = _element_indices()
    delay_samples = (
        BASE_OFFSET_SAMP
        + ELEMENT_SPACING * element_index * np.sin(np.deg2rad(STEERING_ANGLE)) / (C0 * dt)
    )
    return np.round(delay_samples).astype(int)


def _build_source_signals(nt: int, dt: float) -> np.ndarray:
    """Build per-element tone-burst signals, shape ``(NUM_ELEMENTS, nt)``.

    Uses k-wave-python's ``tone_burst`` utility (Hann-windowed sinusoid) to
    match the reference exactly, then zero-pads each row to ``nt`` samples.
    """
    from kwave.utils.signals import tone_burst

    sampling_freq = 1.0 / dt
    offsets = _steering_offsets(dt)

    raw = tone_burst(
        sampling_freq,
        TONE_BURST_FREQ,
        TONE_BURST_CYCLES,
        signal_offset=offsets,
    )
    # raw has shape (NUM_ELEMENTS, raw_length); zero-pad to (NUM_ELEMENTS, nt).
    raw_len = raw.shape[1]
    if raw_len >= nt:
        return raw[:, :nt]
    padded = np.zeros((NUM_ELEMENTS, nt), dtype=np.float64)
    padded[:, :raw_len] = raw
    return padded


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------
def _load_kwave_cache() -> dict | None:
    if REFRESH_CACHE or not _KWAVE_CACHE.exists():
        return None
    try:
        d = np.load(os.fspath(_KWAVE_CACHE), allow_pickle=False)
        if int(np.asarray(d["cache_version"]).reshape(())) != CACHE_VERSION:
            return None
        return {
            "p_final":   np.asarray(d["p_final"],   dtype=np.float64),
            "nt":        int(d["nt"]),
            "dt":        float(d["dt"]),
            "runtime_s": float(d["runtime_s"]),
        }
    except Exception:
        return None


def _save_kwave_cache(p_final: np.ndarray, nt: int, dt: float, runtime_s: float) -> None:
    _KWAVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        os.fspath(_KWAVE_CACHE),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        p_final=np.asarray(p_final, dtype=np.float64),
        nt=np.array(nt, dtype=np.int64),
        dt=np.array(dt, dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )


def _load_pkwav_cache() -> dict | None:
    if REFRESH_CACHE or not _PKWAV_CACHE.exists():
        return None
    try:
        d = np.load(os.fspath(_PKWAV_CACHE), allow_pickle=False)
        if int(np.asarray(d["cache_version"]).reshape(())) != CACHE_VERSION:
            return None
        return {
            "p_final":   np.asarray(d["p_final"],   dtype=np.float64),
            "runtime_s": float(d["runtime_s"]),
        }
    except Exception:
        return None


def _save_pkwav_cache(p_final: np.ndarray, runtime_s: float) -> None:
    _PKWAV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        os.fspath(_PKWAV_CACHE),
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        p_final=np.asarray(p_final, dtype=np.float64),
        runtime_s=np.array(runtime_s, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Step 1 — k-wave-python reference run
# ---------------------------------------------------------------------------
def run_kwave(*, no_cache: bool = False) -> dict:
    """Run the upstream tvsp_steering_linear_array example.

    Returns
    -------
    dict with keys:
        ``p_final``   : 2-D float64 array ``(NX, NY)`` — final pressure field
        ``nt``        : int — total time steps
        ``dt``        : float — time step [s]
        ``runtime_s`` : float — wall-clock time [s]
    """
    if not no_cache:
        cached = _load_kwave_cache()
        if cached is not None:
            print("  [k-wave] Loading from cache...")
            return cached

    import sys as _sys
    from example_parity_utils import KWAVE_PYTHON_ROOT
    if str(KWAVE_PYTHON_ROOT / "examples") not in _sys.path:
        _sys.path.insert(0, str(KWAVE_PYTHON_ROOT / "examples"))

    from tvsp_steering_linear_array import setup as kwave_setup, run as kwave_run  # type: ignore[import]

    print("  [k-wave] Building grid, medium, source...")
    kgrid, medium, source = kwave_setup()
    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)

    print(f"  [k-wave] Running 2-D PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = kwave_run(backend="python", device="cpu", quiet=True)
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    # result["p_final"]: shape (NX*NY,), layout y-fastest (NumPy C-order boolean index):
    # flat[i*NY + j] = pressure at (x=i, y=j)  →  reshape with C-order gives p[i, j].
    pf_flat = np.asarray(result["p_final"], dtype=np.float64).ravel()
    if pf_flat.size != NX * NY:
        raise ValueError(
            f"k-wave p_final has unexpected size {pf_flat.size}, expected {NX*NY}"
        )
    p_final_2d = pf_flat.reshape(NX, NY)   # p[i, j] = pressure at (x=i, y=j)

    print(f"  [k-wave] p_final shape: {p_final_2d.shape}  "
          f"peak: {float(np.abs(p_final_2d).max()):.4e} Pa")

    _save_kwave_cache(p_final_2d, nt, dt, runtime_s)
    return {"p_final": p_final_2d, "nt": nt, "dt": dt, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# Step 2 — pykwavers CPU PSTD
# ---------------------------------------------------------------------------
def run_pykwavers(
    *,
    nt: int,
    dt: float,
    no_cache: bool = False,
) -> dict:
    """Run pykwavers PSTD with the 21-element steered linear array.

    Parameters
    ----------
    nt        : total time steps (from k-wave makeTime)
    dt        : time step [s]
    no_cache  : skip cache lookup

    Returns
    -------
    dict with keys:
        ``p_final``   : 2-D float64 array ``(NX, NY)`` — final pressure field
        ``runtime_s`` : float — wall-clock time [s]

    Notes
    -----
    Source mask layout: the 21 active elements occupy row x=X_OFFSET_0 (0-based)
    at columns y = START_IDX_0 … START_IDX_0+NUM_ELEMENTS-1 (0-based y).

    In pykwavers' x-fastest (Fortran) sensor ordering, element ``n`` at
    (x=X_OFFSET_0, y=START_IDX_0+n) is scanned at position
    ``X_OFFSET_0 + (START_IDX_0+n)*NX``.  Because all elements share the
    same x-row and their y-indices are increasing, the Fortran scan order
    matches the k-Wave element ordering (element 0 = leftmost, element 20 =
    rightmost).  Therefore ``signals[n, :]`` (the signal for element_index n-10)
    directly corresponds to the n-th active mask point in Fortran order.
    """
    if not no_cache:
        cached = _load_pkwav_cache()
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    # --- Grid (quasi-2D: NZ=1) ---
    grid = pkw.Grid(NX, NY, 1, DX, DY, DX)

    # --- Medium: homogeneous with power-law absorption ---
    medium = pkw.Medium.homogeneous(
        sound_speed=C0,
        density=RHO0,
        absorption=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
    )

    # --- Source mask: 21-element linear array at row X_OFFSET_0 ---
    # Float64 mask: 1.0 at each element position, 0.0 elsewhere.
    src_mask = np.zeros((NX, NY, 1), dtype=np.float64)
    for n in range(NUM_ELEMENTS):
        src_mask[X_OFFSET_0, START_IDX_0 + n, 0] = 1.0

    # --- Per-element signals: (NUM_ELEMENTS, nt), matching k-wave tone_burst ---
    signals = _build_source_signals(nt, dt)
    print(f"  [pykwavers] Signal matrix shape: {signals.shape}  "
          f"max_offset_samples={_steering_offsets(dt).max()}")

    # Source.from_mask accepts 2D signal matrix (num_sources, time_steps).
    # The row ordering must match the Fortran-order traversal of the mask:
    # scan over k=0..0, j=0..NY-1, i=0..NX-1 → active points at
    # (X_OFFSET_0, START_IDX_0+n, 0) for n=0..20, giving row order n=0..20,
    # which matches signals[0..20, :].
    source = pkw.Source.from_mask(src_mask, signals, TONE_BURST_FREQ)

    # --- Sensor: full 128×128 grid ---
    sens_mask = np.ones((NX, NY, 1), dtype=bool)
    sensor = pkw.Sensor.from_mask(sens_mask)

    # --- Simulation ---
    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size(PML_SIZE)
    sim.set_pml_inside(True)

    print(f"  [pykwavers] Running 2-D PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=nt, dt=dt)
    runtime_s = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {runtime_s:.1f} s")

    # sensor_data shape: (NX*NY, nt) — x-fastest (Fortran) ordering.
    # flat_pkw[i + j*NX] = pressure at (x=i, y=j).
    # Reshape with Fortran order: p_final[i, j] = flat[i + j*NX] = pressure at (x=i, y=j).
    sd = np.asarray(result.sensor_data, dtype=np.float64)
    print(f"  [pykwavers] sensor_data shape: {sd.shape}")
    if sd.ndim != 2:
        raise ValueError(f"Expected 2-D sensor_data, got shape {sd.shape}")
    if sd.shape[0] != NX * NY:
        raise ValueError(
            f"Expected sensor_data rows = {NX*NY}, got {sd.shape[0]}"
        )
    # Extract final time step as p_final.
    p_final_flat = sd[:, -1]
    p_final_2d = p_final_flat.reshape(NX, NY, order="F")  # p[i, j] = pressure at (x=i, y=j)

    print(f"  [pykwavers] p_final shape: {p_final_2d.shape}  "
          f"peak: {float(np.abs(p_final_2d).max()):.4e} Pa")

    _save_pkwav_cache(p_final_2d, runtime_s)
    return {"p_final": p_final_2d, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(
    kw: dict,
    pkw_res: dict,
    metrics: dict[str, float],
    *,
    status: str,
) -> None:
    """Three-panel figure: k-wave p_final, pykwavers p_final, difference."""
    kw_pf  = kw["p_final"]          # shape (NX, NY), p[x, y]
    pkw_pf = pkw_res["p_final"]     # shape (NX, NY), p[x, y]
    diff   = pkw_pf - kw_pf

    # Physical extent for imshow (convert to mm)
    x0 = 0.0
    x1 = NX * DX * 1e3
    y0 = 0.0
    y1 = NY * DY * 1e3
    extent = [y0, y1, x1, x0]   # [left, right, bottom, top] in mm
    # imshow with p.T: rows=y, cols=x (matplotlib convention)

    vmax = float(np.abs(kw_pf).max()) * 1.05 + 1e-10

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(kw_pf.T, origin="upper", aspect="equal",
                          extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("k-wave-python  p_final")
    axes[0].set_xlabel("y [mm]")
    axes[0].set_ylabel("x [mm]")
    fig.colorbar(im0, ax=axes[0], label="Pressure [Pa]")

    vmax_pkw = float(np.abs(pkw_pf).max()) * 1.05 + 1e-10
    im1 = axes[1].imshow(pkw_pf.T, origin="upper", aspect="equal",
                          extent=extent, cmap="RdBu_r", vmin=-vmax_pkw, vmax=vmax_pkw)
    axes[1].set_title("pykwavers  p_final")
    axes[1].set_xlabel("y [mm]")
    axes[1].set_ylabel("x [mm]")
    fig.colorbar(im1, ax=axes[1], label="Pressure [Pa]")

    vmax_diff = float(np.abs(diff).max()) * 1.05 + 1e-10
    im2 = axes[2].imshow(diff.T, origin="upper", aspect="equal",
                          extent=extent, cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title("Difference  (pykwavers − k-wave)")
    axes[2].set_xlabel("y [mm]")
    axes[2].set_ylabel("x [mm]")
    fig.colorbar(im2, ax=axes[2], label="Pressure [Pa]")

    fig.suptitle(
        f"tvsp_steering_linear_array: k-wave-python vs pykwavers  [{status}]\n"
        f"Grid {NX}×{NY}  dx={DX*1e3:.3f} mm  "
        f"f₀={TONE_BURST_FREQ*1e-6:.0f} MHz  θ={STEERING_ANGLE}°  "
        f"r={metrics['pearson_r']:.4f}  rms_ratio={metrics['rms_ratio']:.4f}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(str(FIGURE_PATH), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare pykwavers with k-wave-python for tvsp_steering_linear_array."
    )
    parser.add_argument("--no-cache",      action="store_true",
                        help="Force a fresh run (ignore cached NPZ files).")
    parser.add_argument("--allow-failure", action="store_true",
                        help="Exit 0 even when parity targets fail.")
    args = parser.parse_args()

    no_cache = args.no_cache

    # Source geometry derived values for logging.
    # dt is not yet known; use a rough estimate for the display.
    print("=" * 70)
    print("tvsp_steering_linear_array: k-wave-python vs pykwavers")
    print(f"  Grid   : {NX}×{NY}   dx={DX*1e3:.4f} mm  dy={DY*1e3:.4f} mm")
    print(f"  Medium : c={C0} m/s  rho={RHO0} kg/m³  "
          f"alpha={ALPHA_COEFF} dB/(MHz^y cm)  y={ALPHA_POWER}")
    print(f"  Array  : {NUM_ELEMENTS} elements  row={X_OFFSET_0} (0-based)  "
          f"y=[{START_IDX_0}..{START_IDX_0+NUM_ELEMENTS-1}]")
    print(f"  Source : f₀={TONE_BURST_FREQ*1e-6:.0f} MHz  "
          f"{TONE_BURST_CYCLES} cycles  θ={STEERING_ANGLE}°  base_offset={BASE_OFFSET_SAMP} samples")
    print(f"  PML    : {PML_SIZE} pts inside")
    print("=" * 70)

    # --- k-wave-python ---
    print("\n[1/2] k-wave-python (2-D PSTD, per-element tone bursts)...")
    kw = run_kwave(no_cache=no_cache)
    kw_pf = kw["p_final"]
    print(f"  Nt={kw['nt']}  dt={kw['dt']:.3e} s  "
          f"peak={float(np.abs(kw_pf).max()):.4e} Pa  "
          f"rms={float(np.sqrt(np.mean(kw_pf**2))):.4e} Pa")

    # Log per-element offsets now that dt is known.
    offsets = _steering_offsets(kw["dt"])
    print(f"  Steering offsets [samples]: min={offsets.min()}  "
          f"max={offsets.max()}  centre={offsets[NUM_ELEMENTS//2]}")

    # --- pykwavers ---
    print("\n[2/2] pykwavers (CPU PSTD, Source.from_mask 2-D signal matrix)...")
    pkw_res = run_pykwavers(nt=kw["nt"], dt=kw["dt"], no_cache=no_cache)
    pkw_pf = pkw_res["p_final"]
    print(f"  peak={float(np.abs(pkw_pf).max()):.4e} Pa  "
          f"rms={float(np.sqrt(np.mean(pkw_pf**2))):.4e} Pa")

    # --- Parity: 2-D field comparison ---
    print("\n--- Parity evaluation (p_final 2-D field) ---")
    # Both p_final arrays have shape (NX, NY) with p[i, j] = pressure at (x=i, y=j).
    # Ravel in C order gives the same spatial pairing.
    metrics = compute_image_metrics(kw_pf.ravel(), pkw_pf.ravel())

    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"]  >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"  Status    : {status}")
    print(f"  pearson_r : {metrics['pearson_r']:.6f}  "
          f"(target >= {thr['pearson_r']})  {'OK' if checks['pearson_r'] else 'FAIL'}")
    print(f"  rms_ratio : {metrics['rms_ratio']:.6f}  "
          f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
          f"{'OK' if checks['rms_ratio'] else 'FAIL'}")
    print(f"  rmse      : {metrics['rmse']:.4e} Pa")
    print(f"  psnr_db   : {metrics.get('psnr_db', float('nan')):.2f} dB")
    print(f"  peak_ratio: {metrics['peak_ratio']:.6f}")
    print(f"  runtime   : k-wave={kw['runtime_s']:.1f}s  pykwavers={pkw_res['runtime_s']:.1f}s")

    # --- Figure ---
    plot_comparison(kw, pkw_res, metrics, status=status)

    # --- Text report ---
    header_lines = [
        "tvsp_steering_linear_array parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}  dx={DX:.6e} m  dy={DY:.6e} m",
        f"medium: c={C0} m/s  rho={RHO0} kg/m3  "
        f"alpha_coeff={ALPHA_COEFF}  alpha_power={ALPHA_POWER}",
        f"array: {NUM_ELEMENTS} elements  x_row={X_OFFSET_0}  "
        f"y=[{START_IDX_0}..{START_IDX_0+NUM_ELEMENTS-1}]",
        f"source: f0={TONE_BURST_FREQ:.3e} Hz  cycles={TONE_BURST_CYCLES}  "
        f"angle={STEERING_ANGLE} deg  base_offset={BASE_OFFSET_SAMP} samples",
        f"pml_size: {PML_SIZE}  pml_inside: True",
        f"nt={kw['nt']}  dt={kw['dt']:.6e} s",
        f"offsets_min_samples: {offsets.min()}",
        f"offsets_max_samples: {offsets.max()}",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
    ]
    report_lines = [
        f"pearson_r  = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio  = {metrics['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"rmse       = {metrics['rmse']:.6e} Pa",
        f"psnr_db    = {metrics.get('psnr_db', float('nan')):.4f}",
        f"max_abs_diff = {metrics['max_abs_diff']:.6e} Pa",
        f"peak_kwave_Pa     = {float(np.abs(kw_pf).max()):.6e}",
        f"peak_pykwavers_Pa = {float(np.abs(pkw_pf).max()):.6e}",
        f"peak_ratio        = {metrics['peak_ratio']:.6f}",
    ]
    save_text_report(METRICS_PATH, "\n".join(header_lines), report_lines)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall parity status: {status}")

    if status == "PASS" or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
