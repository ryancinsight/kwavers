#!/usr/bin/env python3
"""
tvsp_snells_law_compare.py
===========================
Parity comparison for the upstream ``tvsp_snells_law`` example
(Snell's law / critical-angle reflection from a 61-element linear array
steered at 35° into a two-layer acoustic medium, c₁=1500 m/s / c₂=3000 m/s).

Physical setup (matches k-wave-python ``examples/tvsp_snells_law.py`` and
MATLAB ``example_tvsp_snells_law.m``):

  Grid:    128×128  dx = dy = 50e-3/128 m ≈ 0.391 mm  (quasi-2D: NZ = 1)
  Medium:  Two-layer heterogeneous
             Layer 1: x-rows 0..62 (0-based)  c₁ = 1500 m/s
             Layer 2: x-rows 63..127 (0-based) c₂ = 3000 m/s
           Interface at x-index 63 (MATLAB: Nx/2 = 64, 1-based)
           ρ₀ = 1000 kg/m³ (uniform)
           α₀ = 0.75 dB/(MHz^y cm)  (uniform; power law y=1.5 in k-wave-python)
  Source:  61-element linear array at grid row x_offset = 25 (1-based, i.e. row 24
           0-based).  Elements span columns 13..73 (0-based y-axis); y_offset = 20.
           Each element driven by a Hann-windowed tone burst (f₀ = 1 MHz, 8 cycles)
           with geometric steering delay:
             offset[n] = 200 + dx * element_index[n] * sin(35°) / (c₁ * dt)
           element_index ∈ {-30, -29, …, 0, …, 29, 30}
  PML:     20 grid points, inside the domain.

Critical angle and Snell's law
--------------------------------
**Theorem (Snell's law).**  At a planar interface between media with sound
speeds c₁ (incident) and c₂ (transmitted), the refraction angle θ₂ satisfies

    sin θ₂ / c₂ = sin θ₁ / c₁

**Critical angle.**  Total internal reflection occurs when c₂ > c₁ and
θ₁ ≥ θ_c = arcsin(c₁/c₂).  Here:

    θ_c = arcsin(1500/3000) = arcsin(0.5) = 30°

Since θ₁ = 35° > θ_c = 30°, the transmitted wave is evanescent.  The
simulation should show the reflected beam at 35° to the interface normal,
with no propagating transmitted beam into the c₂ layer.

Note on alpha_power
-------------------
k-wave-python applies power-law absorption with exponent y = 1.5.  The
pykwavers heterogeneous medium currently uses y = 1.0 internally (hardcoded
in ``HeterogeneousMedium::absorption_coefficient``).  This introduces a
secondary amplitude mismatch; the primary physics (beam shape, refraction) is
unaffected.  Parity thresholds are set accordingly.

Spatial ordering
-----------------
Identical to the steering example:
  k-wave-python: C-order reshape → p[i, j] = pressure at (x=i, y=j)
  pykwavers:     F-order reshape → p[i, j] = pressure at (x=i, y=j)
Both arrays are directly comparable element-for-element.

Outputs
-------
* ``output/tvsp_snells_law_compare.png``
* ``output/tvsp_snells_law_metrics.txt``

Usage
-----
  python examples/tvsp_snells_law_compare.py
  python examples/tvsp_snells_law_compare.py --no-cache
  python examples/tvsp_snells_law_compare.py --allow-failure
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
# Physical constants (must match k-wave-python example exactly)
# ---------------------------------------------------------------------------
NX = NY = 128
DX = DY = 50e-3 / NX               # ~3.906e-4 m

C1 = 1500.0                         # sound speed, layer 1 [m/s]
C2 = 3000.0                         # sound speed, layer 2 [m/s]
RHO0 = 1000.0                       # density [kg/m^3]
ALPHA_COEFF = 0.75                  # absorption coefficient [dB/(MHz^y cm)]
ALPHA_POWER = 1.5                   # absorption exponent (k-wave-python)

# Layer boundary (0-based row index where c2 begins)
# MATLAB: medium.sound_speed(Nx/2:end, :) = c2  -->  Nx/2 = 64 (1-based)
#         corresponds to rows 63..127 (0-based)
LAYER_SPLIT_0 = NX // 2 - 1        # first 0-based row with c2 = 63

# Source geometry
NUM_ELEMENTS = 61
X_OFFSET_0 = 24                     # 0-based x-index of the source row
# MATLAB: start_index = Ny/2 - round(num_elements/2) + 1 - y_offset
#       = 64 - round(30.5) + 1 - 20 = 64 - 31 + 1 - 20 = 14  (1-based)  →  13 (0-based)
# k-wave-python: start_index = Ny//2 - (num_elements+1)//2 + 1 - y_offset = 14 → 0-based 13
# NOTE: Python's round(30.5)=30 (banker's rounding) differs from MATLAB round(30.5)=31.
#       Use integer division (num_elements+1)//2 = 31 to match k-wave-python exactly.
Y_OFFSET = 20
START_IDX_0 = NY // 2 - (NUM_ELEMENTS + 1) // 2 + 1 - 1 - Y_OFFSET  # 0-based = 13
# Verification: 64 - 31 + 1 - 1 - 20 = 13. Elements span y=13..73 (61 points).

# Tone burst parameters
TONE_BURST_FREQ = 1.0e6             # carrier frequency [Hz]
TONE_BURST_CYCLES = 8
STEERING_ANGLE = 35.0               # [degrees]
ELEMENT_SPACING = DX                # inter-element pitch [m]
BASE_OFFSET_SAMP = 200              # base time offset [samples]

# Critical angle (Snell's law): arcsin(c1/c2) = arcsin(0.5) = 30 deg.
# At 35 deg > 30 deg, total internal reflection occurs (no propagating
# transmitted beam into c2).
CRITICAL_ANGLE_DEG = float(np.degrees(np.arcsin(C1 / C2)))  # 30.0 deg

# PML
PML_SIZE = 20                       # [grid points], applied inside

# Parity thresholds.
# Beam-shape parity: the reflected pattern is complex and k-wave-python's
# alpha_power=1.5 differs from pykwavers heterogeneous alpha_power=1.0, so
# we allow amplitude mismatch up to +/-50% while requiring beam-shape
# correlation >= 0.80.
PARITY_THRESHOLDS: dict[str, float] = {
    "pearson_r":     0.80,
    "rms_ratio_min": 0.50,
    "rms_ratio_max": 2.00,
}

# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------
FIGURE_PATH = DEFAULT_OUTPUT_DIR / "tvsp_snells_law_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "tvsp_snells_law_metrics.txt"
_KWAVE_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_snells_law_kwave_cache.npz"
_PKWAV_CACHE = DEFAULT_OUTPUT_DIR / "tvsp_snells_law_pykwavers_cache.npz"

REFRESH_CACHE = os.getenv("KWAVERS_REFRESH_CACHE", "0") == "1"
CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Source signal construction
# ---------------------------------------------------------------------------
def _element_indices() -> np.ndarray:
    """Return element index array relative to centre element.

    k-Wave convention: ``element_index = -(N-1)/2 : (N-1)/2``
    → for N=61: arange(-30, 31).
    """
    half = (NUM_ELEMENTS - 1) / 2
    return np.arange(-half, half + 1)


def _steering_offsets(dt: float) -> np.ndarray:
    """Return per-element sample offsets for 35° geometric beamforming.

    Formula (from k-Wave MATLAB and k-wave-python):
        offset[n] = BASE_OFFSET_SAMP + dx * element_index[n] * sin(35°) / (c1 * dt)

    Rounded to the nearest integer.
    """
    element_index = _element_indices()
    delay_samples = (
        BASE_OFFSET_SAMP
        + ELEMENT_SPACING * element_index * np.sin(np.deg2rad(STEERING_ANGLE)) / (C1 * dt)
    )
    return np.round(delay_samples).astype(int)


def _build_source_signals(nt: int, dt: float) -> np.ndarray:
    """Build per-element tone-burst signals, shape ``(NUM_ELEMENTS, nt)``.

    Uses k-wave-python's ``tone_burst`` to match the reference exactly.
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
    raw_len = raw.shape[1]
    if raw_len >= nt:
        return raw[:, :nt].astype(np.float64)
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
    """Run the upstream tvsp_snells_law example.

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

    from tvsp_snells_law import setup as kwave_setup, run as kwave_run  # type: ignore[import]

    print("  [k-wave] Building grid, medium, source...")
    kgrid, medium, source = kwave_setup()
    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)

    print(f"  [k-wave] Running 2-D PSTD  (Nt={nt}, dt={dt:.3e} s)...")
    t0 = time.perf_counter()
    result = kwave_run(backend="python", device="cpu", quiet=True)
    runtime_s = time.perf_counter() - t0
    print(f"  [k-wave] Done in {runtime_s:.1f} s")

    # result["p_final"]: shape (NX*NY,), y-fastest (NumPy C-order) layout.
    # flat[i*NY + j] = pressure at (x=i, y=j).  Reshape C-order → p[i, j].
    pf_flat = np.asarray(result["p_final"], dtype=np.float64).ravel()
    if pf_flat.size != NX * NY:
        raise ValueError(
            f"k-wave p_final has unexpected size {pf_flat.size}, expected {NX * NY}"
        )
    p_final_2d = pf_flat.reshape(NX, NY)   # p[i, j] = pressure at (x=i, y=j)

    print(
        f"  [k-wave] p_final shape: {p_final_2d.shape}  "
        f"peak: {float(np.abs(p_final_2d).max()):.4e} Pa"
    )

    _save_kwave_cache(p_final_2d, nt, dt, runtime_s)
    return {"p_final": p_final_2d, "nt": nt, "dt": dt, "runtime_s": runtime_s}


# ---------------------------------------------------------------------------
# Step 2 — pykwavers CPU PSTD
# ---------------------------------------------------------------------------
def _build_two_layer_medium() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the two-layer (NX, NY, 1) medium arrays.

    Sound speed
    -----------
    c[i, j, 0] = C1 = 1500 m/s  for i < LAYER_SPLIT_0  (rows 0..62)
    c[i, j, 0] = C2 = 3000 m/s  for i >= LAYER_SPLIT_0 (rows 63..127)

    Matches k-wave-python:
        c = c1 * np.ones((Nx, Ny))
        c[Nx // 2 - 1 :, :] = c2   # rows 63..127 (0-based)

    Density: uniform ρ₀ = 1000 kg/m³.
    Absorption: uniform α₀ = 0.75 dB/(MHz^y cm) as a 3D array.

    Note: pykwavers HeterogeneousMedium uses alpha_power = 1.0 internally;
    k-wave-python uses 1.5.  This produces secondary amplitude differences
    in heavily absorbed regions, but does not affect beam refraction physics.

    Returns
    -------
    (c_3d, rho_3d, abs_3d) : ndarray of shape (NX, NY, 1)
    """
    c_3d = np.full((NX, NY, 1), C1, dtype=np.float64)
    c_3d[LAYER_SPLIT_0:, :, :] = C2

    rho_3d = np.full((NX, NY, 1), RHO0, dtype=np.float64)
    abs_3d = np.full((NX, NY, 1), ALPHA_COEFF, dtype=np.float64)

    return c_3d, rho_3d, abs_3d


def run_pykwavers(
    *,
    nt: int,
    dt: float,
    no_cache: bool = False,
) -> dict:
    """Run pykwavers PSTD with a 61-element steered array in a two-layer medium.

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
    Source element ordering in pykwavers:
        Fortran-order scan of src_mask (NX, NY, 1) visits the 61 active
        elements at (X_OFFSET_0=24, START_IDX_0+n, 0) for n=0..60 in
        ascending n order, matching the k-Wave element_index ordering
        element_index[n] = -30 + n.  Therefore signals[n, :] directly
        corresponds to the n-th element in scan order.
    """
    if not no_cache:
        cached = _load_pkwav_cache()
        if cached is not None:
            print("  [pykwavers] Loading from cache...")
            return cached

    # --- Grid (quasi-2D: NZ=1) ---
    grid = pkw.Grid(NX, NY, 1, DX, DY, DX)

    # --- Medium: two-layer heterogeneous ---
    c_3d, rho_3d, abs_3d = _build_two_layer_medium()
    medium = pkw.Medium(
        sound_speed=c_3d,
        density=rho_3d,
        absorption=abs_3d,
    )
    print(
        f"  [pykwavers] Medium: c_max={float(c_3d.max()):.0f} m/s  "
        f"c_min={float(c_3d.min()):.0f} m/s  "
        f"layer_split_row={LAYER_SPLIT_0} (0-based)"
    )

    # --- Source mask: 61-element linear array ---
    # Elements at (X_OFFSET_0=24, START_IDX_0+n, 0) for n=0..60 (0-based).
    # START_IDX_0 = 13, so elements span y=13..73.
    src_mask = np.zeros((NX, NY, 1), dtype=np.float64)
    for n in range(NUM_ELEMENTS):
        src_mask[X_OFFSET_0, START_IDX_0 + n, 0] = 1.0

    active_count = int(np.count_nonzero(src_mask))
    if active_count != NUM_ELEMENTS:
        raise RuntimeError(
            f"Source mask has {active_count} active elements, expected {NUM_ELEMENTS}"
        )

    # --- Per-element signals: (NUM_ELEMENTS, nt) ---
    signals = _build_source_signals(nt, dt)
    offsets = _steering_offsets(dt)
    print(
        f"  [pykwavers] Signal matrix shape: {signals.shape}  "
        f"offset_range=[{offsets.min()}, {offsets.max()}] samples  "
        f"centre_offset={offsets[NUM_ELEMENTS // 2]} samples"
    )

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

    # sensor_data: (NX*NY, nt), x-fastest (Fortran) ordering.
    # flat[i + j*NX] = pressure at (x=i, y=j).
    # Reshape F-order: p[i, j] = flat[i + j*NX] = pressure at (x=i, y=j).
    sd = np.asarray(result.sensor_data, dtype=np.float64)
    print(f"  [pykwavers] sensor_data shape: {sd.shape}")
    if sd.ndim != 2:
        raise ValueError(f"Expected 2-D sensor_data, got shape {sd.shape}")
    if sd.shape[0] != NX * NY:
        raise ValueError(
            f"Expected sensor_data rows = {NX * NY}, got {sd.shape[0]}"
        )

    p_final_flat = sd[:, -1]
    p_final_2d = p_final_flat.reshape(NX, NY, order="F")  # p[i, j] = pressure at (x=i, y=j)

    print(
        f"  [pykwavers] p_final shape: {p_final_2d.shape}  "
        f"peak: {float(np.abs(p_final_2d).max()):.4e} Pa"
    )

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
    """Four-panel figure: k-wave p_final, pykwavers p_final, difference, sound speed."""
    kw_pf = kw["p_final"]        # (NX, NY), p[x, y]
    pkw_pf = pkw_res["p_final"]  # (NX, NY), p[x, y]
    diff = pkw_pf - kw_pf

    # Physical extent for imshow (mm)
    x1 = NX * DX * 1e3
    y1 = NY * DY * 1e3
    extent = [0.0, y1, x1, 0.0]  # [left, right, bottom, top] in mm

    vmax = float(np.abs(kw_pf).max()) * 1.05 + 1e-10

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # k-wave p_final
    im0 = axes[0].imshow(
        kw_pf.T, origin="upper", aspect="equal",
        extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    axes[0].axhline(y=LAYER_SPLIT_0 * DX * 1e3, color="k", ls="--", lw=0.8, label="interface")
    axes[0].set_title("k-wave-python  p_final")
    axes[0].set_xlabel("y [mm]")
    axes[0].set_ylabel("x [mm]")
    fig.colorbar(im0, ax=axes[0], label="Pressure [Pa]")

    # pykwavers p_final
    vmax_pkw = float(np.abs(pkw_pf).max()) * 1.05 + 1e-10
    im1 = axes[1].imshow(
        pkw_pf.T, origin="upper", aspect="equal",
        extent=extent, cmap="RdBu_r", vmin=-vmax_pkw, vmax=vmax_pkw,
    )
    axes[1].axhline(y=LAYER_SPLIT_0 * DX * 1e3, color="k", ls="--", lw=0.8)
    axes[1].set_title("pykwavers  p_final")
    axes[1].set_xlabel("y [mm]")
    axes[1].set_ylabel("x [mm]")
    fig.colorbar(im1, ax=axes[1], label="Pressure [Pa]")

    # Difference
    vmax_diff = float(np.abs(diff).max()) * 1.05 + 1e-10
    im2 = axes[2].imshow(
        diff.T, origin="upper", aspect="equal",
        extent=extent, cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff,
    )
    axes[2].axhline(y=LAYER_SPLIT_0 * DX * 1e3, color="k", ls="--", lw=0.8)
    axes[2].set_title("Difference  (pykwavers - k-wave)")
    axes[2].set_xlabel("y [mm]")
    axes[2].set_ylabel("x [mm]")
    fig.colorbar(im2, ax=axes[2], label="Pressure [Pa]")

    # Sound speed map
    c_2d = np.full((NX, NY), C1)
    c_2d[LAYER_SPLIT_0:, :] = C2
    im3 = axes[3].imshow(
        c_2d.T, origin="upper", aspect="equal",
        extent=extent, cmap="viridis",
    )
    axes[3].set_title("Sound speed map")
    axes[3].set_xlabel("y [mm]")
    axes[3].set_ylabel("x [mm]")
    fig.colorbar(im3, ax=axes[3], label="c [m/s]")

    theta_c_str = f"theta_c={CRITICAL_ANGLE_DEG:.1f} deg"
    fig.suptitle(
        f"tvsp_snells_law: k-wave-python vs pykwavers  [{status}]\n"
        f"Grid {NX}x{NY}  dx={DX * 1e3:.3f} mm  "
        f"c1={C1:.0f}/c2={C2:.0f} m/s  theta_steer={STEERING_ANGLE} deg  "
        f"{theta_c_str}  r={metrics['pearson_r']:.4f}  rms_ratio={metrics['rms_ratio']:.4f}",
        fontsize=8,
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
        description="Compare pykwavers with k-wave-python for tvsp_snells_law."
    )
    parser.add_argument("--no-cache", action="store_true",
                        help="Force a fresh run (ignore cached NPZ files).")
    parser.add_argument("--allow-failure", action="store_true",
                        help="Exit 0 even when parity targets fail.")
    args = parser.parse_args()

    no_cache = args.no_cache

    print("=" * 70)
    print("tvsp_snells_law: k-wave-python vs pykwavers")
    print(f"  Grid      : {NX}x{NY}  dx={DX * 1e3:.4f} mm  dy={DY * 1e3:.4f} mm")
    print(f"  Medium    : c1={C1:.0f} m/s (rows 0..{LAYER_SPLIT_0-1})  "
          f"c2={C2:.0f} m/s (rows {LAYER_SPLIT_0}..{NX-1})")
    print(f"             rho={RHO0:.0f} kg/m3  alpha={ALPHA_COEFF} dB/(MHz^y cm)  y={ALPHA_POWER}")
    print(f"  Array     : {NUM_ELEMENTS} elements  row={X_OFFSET_0} (0-based)  "
          f"y=[{START_IDX_0}..{START_IDX_0+NUM_ELEMENTS-1}]")
    print(f"  Source    : f0={TONE_BURST_FREQ * 1e-6:.0f} MHz  "
          f"{TONE_BURST_CYCLES} cycles  theta={STEERING_ANGLE} deg  "
          f"base_offset={BASE_OFFSET_SAMP} samples")
    print(f"  Physics   : theta_c={CRITICAL_ANGLE_DEG:.1f} deg  "
          f"({'total internal reflection' if STEERING_ANGLE > CRITICAL_ANGLE_DEG else 'partial refraction'})")
    print(f"  PML       : {PML_SIZE} pts inside")
    print("=" * 70)

    # --- k-wave-python ---
    print("\n[1/2] k-wave-python (2-D PSTD, two-layer medium)...")
    kw = run_kwave(no_cache=no_cache)
    kw_pf = kw["p_final"]
    print(
        f"  Nt={kw['nt']}  dt={kw['dt']:.3e} s  "
        f"peak={float(np.abs(kw_pf).max()):.4e} Pa  "
        f"rms={float(np.sqrt(np.mean(kw_pf ** 2))):.4e} Pa"
    )

    offsets = _steering_offsets(kw["dt"])
    print(
        f"  Steering offsets [samples]: min={offsets.min()}  "
        f"max={offsets.max()}  centre={offsets[NUM_ELEMENTS // 2]}"
    )

    # --- pykwavers ---
    print("\n[2/2] pykwavers (CPU PSTD, heterogeneous two-layer medium)...")
    pkw_res = run_pykwavers(nt=kw["nt"], dt=kw["dt"], no_cache=no_cache)
    pkw_pf = pkw_res["p_final"]
    print(
        f"  peak={float(np.abs(pkw_pf).max()):.4e} Pa  "
        f"rms={float(np.sqrt(np.mean(pkw_pf ** 2))):.4e} Pa"
    )

    # --- Parity ---
    print("\n--- Parity evaluation (p_final 2-D field) ---")
    metrics = compute_image_metrics(kw_pf.ravel(), pkw_pf.ravel())

    thr = PARITY_THRESHOLDS
    checks = {
        "pearson_r": metrics["pearson_r"] >= thr["pearson_r"],
        "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    print(f"  Status    : {status}")
    print(
        f"  pearson_r : {metrics['pearson_r']:.6f}  "
        f"(target >= {thr['pearson_r']})  {'OK' if checks['pearson_r'] else 'FAIL'}"
    )
    print(
        f"  rms_ratio : {metrics['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
        f"{'OK' if checks['rms_ratio'] else 'FAIL'}"
    )
    print(f"  rmse      : {metrics['rmse']:.4e} Pa")
    print(f"  psnr_db   : {metrics.get('psnr_db', float('nan')):.2f} dB")
    print(f"  peak_ratio: {metrics['peak_ratio']:.6f}")
    print(
        f"  runtime   : k-wave={kw['runtime_s']:.1f}s  "
        f"pykwavers={pkw_res['runtime_s']:.1f}s"
    )

    # --- Figure ---
    plot_comparison(kw, pkw_res, metrics, status=status)

    # --- Text report ---
    header_lines = [
        "tvsp_snells_law parity metrics",
        f"parity_status: {status}",
        f"grid: {NX}x{NY}  dx={DX:.6e} m  dy={DY:.6e} m",
        f"medium: c1={C1:.0f}/c2={C2:.0f} m/s  "
        f"layer_split_row={LAYER_SPLIT_0}  "
        f"rho={RHO0:.0f} kg/m3  "
        f"alpha_coeff={ALPHA_COEFF}  alpha_power_kwave={ALPHA_POWER}",
        f"snell_critical_angle_deg: {CRITICAL_ANGLE_DEG:.4f}",
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
        f"pearson_r     = {metrics['pearson_r']:.6f}  (target >= {thr['pearson_r']})",
        f"rms_ratio     = {metrics['rms_ratio']:.6f}  "
        f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
        f"rmse          = {metrics['rmse']:.6e} Pa",
        f"psnr_db       = {metrics.get('psnr_db', float('nan')):.4f}",
        f"max_abs_diff  = {metrics['max_abs_diff']:.6e} Pa",
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
