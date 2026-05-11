#!/usr/bin/env python3
"""
us_beam_patterns_compare.py
============================
Side-by-side comparison of k-wave-python vs pykwavers for the canonical
beam-patterns example: 32-element focused linear transducer on a homogeneous
medium, recording p_rms and p_max on the xy-midplane.

Physical setup (matches k-wave-python reference exactly):
  Grid:        88×44×44 active voxels (128×64×64 total), PML=[20,10,10]
  dx = dy = dz = 40e-3/88 ≈ 4.545e-4 m
  Medium:      homogeneous, c0=1540, ρ=1000, α=0.75 dB/MHz^1.5/cm, BonA=6
  Transducer:  32-element, width=1 pt, length=12 pts, kerf=0
  Focus:       20 mm (azimuth), 19 mm (elevation); steering=0°
  Apodization: Rectangular TX  (receive beamforming not used)
  Source:      velocity ux, 0.5 MHz 5-cycle tone burst,
               source_strength / (c0 * rho0)  [m/s]
  Sensor:      xy-midplane at z_active = Nz//2 = 22, records p_rms and p_max

k-wave sensor ordering:
  sensor_data["p_rms"] is flat in MATLAB Fortran order (x fastest):
    reshape([Ny, Nx]).T  →  (Nx, Ny) image where [ix, iy] = pressure.

pykwavers sensor ordering:
  Sensor mask set on active-domain xy-slice in the full grid.
  np.argwhere visits (ix, iy) in C order (y fastest for fixed iz):
    sensor_data_flat.reshape(Nx, Ny)  →  (Nx, Ny) image.  ✓ same layout.

Output:
  output/us_beam_patterns_compare.png   — 2×3 comparison figure
  output/us_beam_patterns_metrics.txt   — Pearson r / RMS ratio / PSNR

Usage:
  python examples/us_beam_patterns_compare.py            # CPU k-wave (default)
  python examples/us_beam_patterns_compare.py --gpu      # GPU k-wave binary
  python examples/us_beam_patterns_compare.py --no-cache # force re-run
"""

from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    compute_image_metrics,
    save_text_report,
)

bootstrap_example_paths()

import pykwavers as pkw
from kwave.data import Vector
from kwave.kWaveSimulation import SimulationOptions
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.dotdictionary import dotdict
from kwave.utils.signals import tone_burst

# ---------------------------------------------------------------------------
# Grid / medium / transducer constants  (match k-wave-python EXACTLY)
# ---------------------------------------------------------------------------
PML_X_SIZE = 20
PML_Y_SIZE = 10
PML_Z_SIZE = 10

# Active-domain grid points
NX = 128 - 2 * PML_X_SIZE   # 88
NY = 64  - 2 * PML_Y_SIZE   # 44
NZ = 64  - 2 * PML_Z_SIZE   # 44

# Full grid (active + PML on both sides)
TNX = NX + 2 * PML_X_SIZE   # 128
TNY = NY + 2 * PML_Y_SIZE   # 64
TNZ = NZ + 2 * PML_Z_SIZE   # 64

X_LEN = 40e-3                 # domain x-length [m]
DX    = X_LEN / NX            # ≈4.545e-4 m
DY    = DX
DZ    = DX

C0           = 1540.0         # sound speed [m/s]
RHO0         = 1000.0         # density [kg/m³]
ALPHA_COEFF  = 0.75           # absorption [dB/MHz^y/cm]
ALPHA_POWER  = 1.5
BON_A        = 6.0            # nonlinearity (B/A)

SOURCE_STRENGTH   = 1e6       # peak source pressure [Pa]
TONE_BURST_FREQ   = 0.5e6    # centre frequency [Hz]
TONE_BURST_CYCLES = 5
T_END             = 45e-6    # simulation end time [s]

# Transducer geometry
N_ELEMENTS    = 32
ELEM_WIDTH    = 1             # grid points per element
ELEM_LENGTH   = 12            # elevation extent [grid points]
ELEM_SPACING  = 0             # kerf [grid points]
FOCUS_DIST    = 20e-3         # azimuth focus [m]
ELEV_FOCUS    = 19e-3         # elevation focus [m]
STEERING_ANGLE = 0.0          # [degrees]

# Sensor plane: xy-midplane
SENSOR_IZ_ACTIVE = NZ // 2                        # 22  (active domain)
SENSOR_IZ_FULL   = PML_Z_SIZE + SENSOR_IZ_ACTIVE  # 32  (full grid)

# ---------------------------------------------------------------------------
# Parity targets (image-level; local to this example)
# ---------------------------------------------------------------------------
PARITY_THRESHOLDS: dict[str, dict] = {
    "p_rms": {
        "pearson_r": 0.99,
        "rms_ratio_min": 0.90,
        "rms_ratio_max": 1.10,
        "psnr_db": 26.0,
    },
    "p_max": {
        "pearson_r": 0.99,
        "rms_ratio_min": 0.85,
        "rms_ratio_max": 1.15,
        "psnr_db": 25.0,
    },
}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
FIGURE_PATH  = DEFAULT_OUTPUT_DIR / "us_beam_patterns_compare.png"
METRICS_PATH = DEFAULT_OUTPUT_DIR / "us_beam_patterns_metrics.txt"
KWAVE_CACHE  = DEFAULT_OUTPUT_DIR / "us_beam_patterns_kwave_cache.npz"
PKWAV_CACHE  = DEFAULT_OUTPUT_DIR / "us_beam_patterns_pykwavers_cache.npz"


# ---------------------------------------------------------------------------
# Step 1 — Build shared k-wave configuration
# ---------------------------------------------------------------------------
def build_kwave_config() -> tuple:
    """
    Return (kgrid, medium, not_transducer, input_signal_1d).

    The returned kgrid provides dt and Nt for both legs of the comparison.
    not_transducer.beamforming_delays is used in the pykwavers TX delay build.
    """
    kgrid = kWaveGrid([NX, NY, NZ], [DX, DY, DZ])
    medium = kWaveMedium(
        sound_speed=C0,
        density=RHO0,
        alpha_coeff=ALPHA_COEFF,
        alpha_power=ALPHA_POWER,
        BonA=BON_A,
    )
    kgrid.makeTime(medium.sound_speed, t_end=T_END)

    # Velocity source signal [m/s]:  (source_strength / (c0 * rho0)) * tone_burst
    input_signal = tone_burst(1.0 / kgrid.dt, TONE_BURST_FREQ, TONE_BURST_CYCLES)
    input_signal = (SOURCE_STRENGTH / (C0 * RHO0)) * input_signal

    # Transducer geometry
    transducer_width = (
        N_ELEMENTS * ELEM_WIDTH + (N_ELEMENTS - 1) * ELEM_SPACING
    )
    tr = dotdict()
    tr.number_elements = N_ELEMENTS
    tr.element_width   = ELEM_WIDTH
    tr.element_length  = ELEM_LENGTH
    tr.element_spacing = ELEM_SPACING
    tr.radius          = float("inf")
    tr.position = np.round(
        [1, NY / 2 - transducer_width / 2, NZ / 2 - ELEM_LENGTH / 2]
    )
    transducer = kWaveTransducerSimple(kgrid, **tr)

    # NotATransducer applies TX delays + apodization
    nt = dotdict()
    nt.sound_speed              = C0
    nt.focus_distance           = FOCUS_DIST
    nt.elevation_focus_distance = ELEV_FOCUS
    nt.steering_angle           = STEERING_ANGLE
    nt.transmit_apodization     = "Rectangular"
    nt.receive_apodization      = "Rectangular"
    nt.active_elements          = np.ones((N_ELEMENTS, 1))
    nt.input_signal             = input_signal
    not_transducer = NotATransducer(transducer, kgrid, **nt)

    return kgrid, medium, not_transducer, np.asarray(input_signal).ravel()


# ---------------------------------------------------------------------------
# Step 2 — k-wave-python reference (with caching)
# ---------------------------------------------------------------------------
def run_kwave(kgrid, medium, not_transducer, use_gpu: bool = False) -> dict:
    """
    Run k-wave-python and return p_rms / p_max as (NX, NY) images [Pa].

    k-wave sensor data ordering
    ---------------------------
    The sensor mask ``sensor_mask[:, :, Nz//2] = 1`` covers NX*NY points.
    k-wave returns a flat array in MATLAB Fortran column-major order where x
    (the first index) varies fastest:

        flat[ix + NX*iy]  →  reshape([NY, NX]).T  →  image[ix, iy]
    """
    if KWAVE_CACHE.exists():
        print("  [k-wave] Loading from cache...")
        d = np.load(KWAVE_CACHE)
        return {
            "p_rms":     d["p_rms"],      # (NX, NY)
            "p_max":     d["p_max"],      # (NX, NY)
            "dt":        float(d["dt"]),
            "runtime_s": float(d["runtime_s"]),
        }

    # Sensor mask: full xy-midplane in the active domain
    sensor_mask_kw = np.zeros((NX, NY, NZ))
    sensor_mask_kw[:, :, SENSOR_IZ_ACTIVE] = 1

    sensor = kSensor(sensor_mask_kw)
    sensor.record = ["p_rms", "p_max"]

    sim_opts = SimulationOptions(
        pml_inside=False,
        pml_size=Vector([PML_X_SIZE, PML_Y_SIZE, PML_Z_SIZE]),
        data_cast="single",
        save_to_disk=True,
    )
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=use_gpu)

    print("  [k-wave] Running simulation...")
    t0 = time.perf_counter()
    sensor_data = kspaceFirstOrder3D(
        medium=medium,
        kgrid=kgrid,
        source=not_transducer,
        sensor=sensor,
        simulation_options=sim_opts,
        execution_options=exec_opts,
    )
    elapsed = time.perf_counter() - t0
    print(f"  [k-wave] Done in {elapsed:.1f} s")

    # Reshape flat MATLAB Fortran-order → (NX, NY) image
    # flat[k] = sensor at x = k % NX, y = k // NX
    # reshape([NY, NX])  →  arr[iy, ix]
    # .T                 →  arr[ix, iy]  ✓
    p_rms = np.asarray(sensor_data["p_rms"], dtype=np.float64).reshape([NY, NX]).T
    p_max = np.asarray(sensor_data["p_max"], dtype=np.float64).reshape([NY, NX]).T

    result = {
        "p_rms":     p_rms,
        "p_max":     p_max,
        "dt":        float(kgrid.dt),
        "runtime_s": elapsed,
    }
    np.savez(KWAVE_CACHE, **result)
    return result


# ---------------------------------------------------------------------------
# Step 3 — Build pykwavers source inputs
# ---------------------------------------------------------------------------
def build_pkw_source(
    dt: float, Nt: int, input_signal: np.ndarray, not_transducer
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the 3D source mask and per-source-point velocity signal matrix
    for use with pkw.Source.from_velocity_mask_2d().

    TX delay model (Thomenius 1996)
    --------------------------------
    Azimuth delays are taken directly from not_transducer.beamforming_delays
    (integer sample offsets, may be negative; we shift so min → 0).

    Rectangular apodization → uniform weight 1.0 for all elements.

    k-Wave additive velocity source scaling
    ----------------------------------------
    kwavers applies 2·c₀·dt/d_axis internally for additive velocity sources
    (commit caabc640). Do NOT apply the factor here; doing so would double-count
    it and produce a ~0.6× amplitude deficit relative to k-Wave.

    Source mask layout (full grid, 128×64×64)
    -----------------------------------------
    Transducer is at x = PML_X_SIZE = 20 (first active voxel).
    k-wave 1-indexed position [1, 6, 16] → 0-indexed active-domain [0, 5, 15]:
        y0_active = round(NY/2 − transducer_width/2) − 1 = 5
        z0_active = round(NZ/2 − ELEM_LENGTH/2)      − 1 = 15
    Full-grid position:
        y0 = y0_active + PML_Y_SIZE = 15
        z0 = z0_active + PML_Z_SIZE = 25

    Returns
    -------
    source_mask : ndarray (TNX, TNY, TNZ) float64, 1 at source voxels
    ux_signals  : ndarray (n_src, Nt) float64  [m/s]
    """
    transducer_width = N_ELEMENTS * ELEM_WIDTH + (N_ELEMENTS - 1) * ELEM_SPACING

    # Active-domain corner (0-indexed).
    # kWaveTransducerSimple.position uses 1-indexed MATLAB convention:
    #   tr.position = [1, round(NY/2 - tw/2), round(NZ/2 - EL/2)]
    # The same arithmetic evaluated in Python yields 0-indexed values that are 1 too high.
    # Subtracting 1 converts the k-wave 1-indexed position to the correct 0-indexed cell.
    y0_act = int(np.round(NY / 2 - transducer_width / 2)) - 1
    z0_act = int(np.round(NZ / 2 - ELEM_LENGTH / 2)) - 1

    # Full-grid coordinates
    x_src = PML_X_SIZE
    y0    = y0_act + PML_Y_SIZE
    z0    = z0_act + PML_Z_SIZE

    # Source mask
    source_mask = np.zeros((TNX, TNY, TNZ), dtype=np.float64)
    for i in range(N_ELEMENTS):
        ys = y0 + i * (ELEM_WIDTH + ELEM_SPACING)
        source_mask[x_src, ys : ys + ELEM_WIDTH, z0 : z0 + ELEM_LENGTH] = 1.0
    n_src = int(source_mask.sum())   # 32 × 1 × 12 = 384

    # Azimuth TX delays (samples, non-negative)
    az_delays     = not_transducer.beamforming_delays   # (N_ELEMENTS,) int
    az_offset     = int(-az_delays.min())               # shift so min = 0
    az_delays_abs = az_delays + az_offset               # non-negative

    # Build per-source-point signals
    # Rectangular apodization: weight = 1.0 for all elements.
    # kwavers applies 2*c0*dt/dx internally; no manual scaling here.
    input_1d  = np.asarray(input_signal).ravel()
    L         = len(input_1d)
    max_delay = int(az_delays_abs.max())
    base_pad  = np.zeros(L + max_delay)
    base_pad[:L] = input_1d

    ux_signals = np.zeros((n_src, Nt), dtype=np.float64)
    p = 0
    for i in range(N_ELEMENTS):
        az_d = int(az_delays_abs[i])
        w = 1.0
        for _yw in range(ELEM_WIDTH):
            for _j in range(ELEM_LENGTH):
                end     = min(az_d + L, Nt)
                src_end = end - az_d
                if src_end > 0:
                    ux_signals[p, az_d:end] = base_pad[:src_end] * w
                p += 1

    return source_mask, ux_signals


# ---------------------------------------------------------------------------
# Step 4 — pykwavers simulation (CPU PSTD, with caching)
# ---------------------------------------------------------------------------
def run_pykwavers(dt: float, Nt: int, input_signal: np.ndarray, not_transducer) -> dict:
    """
    Run pykwavers CPU PSTD and return p_rms / p_max as (NX, NY) images [Pa].

    Sensor data ordering
    --------------------
    Sensor mask: sensor_mask[PML_X_SIZE:PML_X_SIZE+NX, PML_Y_SIZE:PML_Y_SIZE+NY, SENSOR_IZ_FULL] = True
    np.argwhere visits in C order → for fixed iz: x varies slowest, y fastest.
    Flat sensor point k  →  ix_active = k // NY,  iy_active = k % NY.
    Therefore: sensor_flat.reshape(NX, NY) → image[ix_active, iy_active].  ✓
    """
    if PKWAV_CACHE.exists():
        print("  [pykwavers] Loading from cache...")
        d = np.load(PKWAV_CACHE)
        return {
            "p_rms":     d["p_rms"],      # (NX, NY)
            "p_max":     d["p_max"],      # (NX, NY)
            "runtime_s": float(d["runtime_s"]),
        }

    print("  [pykwavers] Building source inputs...")
    source_mask, ux_signals = build_pkw_source(dt, Nt, input_signal, not_transducer)

    # Sensor: xy-midplane in the active domain (full-grid coordinates)
    sensor_mask_pkw = np.zeros((TNX, TNY, TNZ), dtype=bool)
    sensor_mask_pkw[
        PML_X_SIZE : PML_X_SIZE + NX,
        PML_Y_SIZE : PML_Y_SIZE + NY,
        SENSOR_IZ_FULL,
    ] = True
    # n_sensor = NX * NY = 88 * 44 = 3872

    # Full-grid homogeneous medium arrays (background + absorption + nonlinearity)
    ss_full  = np.full((TNX, TNY, TNZ), C0,          dtype=np.float64)
    rho_full = np.full((TNX, TNY, TNZ), RHO0,        dtype=np.float64)
    abs_full = np.full((TNX, TNY, TNZ), ALPHA_COEFF, dtype=np.float64)
    nl_full  = np.full((TNX, TNY, TNZ), BON_A,       dtype=np.float64)

    grid   = pkw.Grid(TNX, TNY, TNZ, DX, DY, DZ)
    medium = pkw.Medium(
        sound_speed=ss_full,
        density=rho_full,
        absorption=abs_full,
        nonlinearity=nl_full,
    )

    source = pkw.Source.from_velocity_mask_2d(source_mask, ux=ux_signals, mode="additive")
    sensor = pkw.Sensor.from_mask(sensor_mask_pkw)

    sim = pkw.Simulation(grid, medium, source, sensor, solver=pkw.SolverType.PSTD)
    sim.set_pml_size_xyz(PML_X_SIZE, PML_Y_SIZE, PML_Z_SIZE)
    sim.set_pml_inside(True)
    sim.set_nonlinear(True)
    sim.set_alpha_coeff(ALPHA_COEFF)
    sim.set_alpha_power(ALPHA_POWER)

    print("  [pykwavers] Running simulation...")
    t0 = time.perf_counter()
    result = sim.run(time_steps=Nt, dt=dt)
    elapsed = time.perf_counter() - t0
    print(f"  [pykwavers] Done in {elapsed:.1f} s")

    # sensor_data: (n_sensor_pts, Nt) — one row per sensor point (time series)
    sd = np.asarray(result.sensor_data, dtype=np.float64)   # (NX*NY, Nt)

    # Compute statistics over time axis
    p_rms_flat = np.sqrt(np.mean(sd ** 2, axis=1))   # (NX*NY,)
    p_max_flat = np.max(np.abs(sd), axis=1)           # (NX*NY,)

    # SensorRecorder iterates sensor voxels in Fortran (MATLAB-compatible) order
    # — x fastest, then y, then z. For a slab at fixed iz the flat index is
    # k = iy*NX + ix, so reshape(NY, NX).T yields image[ix, iy].
    p_rms = p_rms_flat.reshape(NY, NX).T
    p_max = p_max_flat.reshape(NY, NX).T

    output = {
        "p_rms":     p_rms,
        "p_max":     p_max,
        "runtime_s": elapsed,
    }
    np.savez(PKWAV_CACHE, **output)
    return output


# ---------------------------------------------------------------------------
# Step 5 — Plotting
# ---------------------------------------------------------------------------
def plot_comparison(kw: dict, pkw_res: dict) -> None:
    """
    2×3 comparison figure:
      Rows: p_rms, p_max
      Cols: k-wave-python | pykwavers | difference

    x-axis = y-position (lateral) in mm, y-axis = x-position (depth) in mm.
    The transducer face is at x=0; depth increases downward.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Axis vectors for the active domain
    y_mm_vec = (np.arange(NY) - NY / 2 + 0.5) * DY * 1e3   # lateral [mm]
    x_mm_vec = np.arange(NX) * DX * 1e3                      # depth   [mm]
    extent = [y_mm_vec[0], y_mm_vec[-1], x_mm_vec[-1], x_mm_vec[0]]

    measures = ["p_rms", "p_max"]
    titles   = ["RMS Pressure", "Peak Pressure"]

    for row, (measure, title) in enumerate(zip(measures, titles)):
        kw_img  = kw[measure]        # (NX, NY)  [Pa]
        pkw_img = pkw_res[measure]   # (NX, NY)  [Pa]
        vmax    = max(kw_img.max(), pkw_img.max())
        diff    = pkw_img - kw_img   # pykwavers − k-wave

        # Column 0: k-wave reference
        ax = axes[row, 0]
        im = ax.imshow(
            kw_img * 1e-6,
            extent=extent, aspect="auto", cmap="jet",
            vmin=0, vmax=vmax * 1e-6,
        )
        ax.set_title(f"k-wave-python — {title}")
        ax.set_xlabel("Lateral y [mm]")
        ax.set_ylabel("Depth x [mm]")
        fig.colorbar(im, ax=ax, label="Pressure [MPa]", fraction=0.046, pad=0.04)

        # Column 1: pykwavers
        ax = axes[row, 1]
        im = ax.imshow(
            pkw_img * 1e-6,
            extent=extent, aspect="auto", cmap="jet",
            vmin=0, vmax=vmax * 1e-6,
        )
        ax.set_title(f"pykwavers — {title}")
        ax.set_xlabel("Lateral y [mm]")
        ax.set_ylabel("Depth x [mm]")
        fig.colorbar(im, ax=ax, label="Pressure [MPa]", fraction=0.046, pad=0.04)

        # Column 2: signed difference
        diff_lim = np.abs(diff).max() * 1e-6 or 1e-9
        ax = axes[row, 2]
        im = ax.imshow(
            diff * 1e-6,
            extent=extent, aspect="auto", cmap="RdBu_r",
            vmin=-diff_lim, vmax=diff_lim,
        )
        ax.set_title(f"Difference (pykwavers − k-wave)")
        ax.set_xlabel("Lateral y [mm]")
        ax.set_ylabel("Depth x [mm]")
        fig.colorbar(im, ax=ax, label="ΔPressure [MPa]", fraction=0.046, pad=0.04)

    fig.suptitle(
        "us_beam_patterns: k-wave-python vs pykwavers\n"
        f"32-element focused array  ·  f0={TONE_BURST_FREQ * 1e-6:.2f} MHz  "
        f"·  focus={FOCUS_DIST * 1e3:.0f} mm  ·  grid {NX}×{NY}×{NZ} active",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(str(FIGURE_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURE_PATH}")


# ---------------------------------------------------------------------------
# Step 6 — Directivity comparison at focal depth
# ---------------------------------------------------------------------------
def plot_directivity(kw: dict, pkw_res: dict) -> None:
    """
    Normalized lateral directivity at the focal depth (ix ≈ round(FOCUS_DIST/DX)).
    Saved as a separate figure alongside the main comparison.
    """
    focus_ix = min(int(round(FOCUS_DIST / DX)), NX - 1)
    y_mm_vec = (np.arange(NY) - NY / 2 + 0.5) * DY * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, measure, title in zip(axes, ["p_rms", "p_max"], ["RMS Pressure", "Peak Pressure"]):
        kw_slice  = kw[measure][focus_ix, :]
        pkw_slice = pkw_res[measure][focus_ix, :]

        kw_peak  = kw_slice.max()  + 1e-30
        pkw_peak = pkw_slice.max() + 1e-30

        ax.plot(y_mm_vec, kw_slice  / kw_peak,  "k-",  linewidth=2, label="k-wave-python")
        ax.plot(y_mm_vec, pkw_slice / pkw_peak, "r--", linewidth=1.5, label="pykwavers")
        ax.set_title(f"{title} — directivity at x={focus_ix*DX*1e3:.1f} mm (focus)")
        ax.set_xlabel("Lateral y [mm]")
        ax.set_ylabel("Normalised amplitude")
        ax.set_xlim([-10, 10])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "us_beam_patterns: lateral directivity at focal depth\n"
        "(normalised; dashed = pykwavers, solid = k-wave)",
        fontsize=10,
    )
    fig.tight_layout()
    dir_path = DEFAULT_OUTPUT_DIR / "us_beam_patterns_directivity.png"
    fig.savefig(str(dir_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {dir_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pykwavers with k-wave-python for us_beam_patterns."
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run k-wave-python with GPU execution binary.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Delete cached results and force a fresh run.",
    )
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Exit 0 even when scientific parity targets fail (for diagnostics).",
    )
    args = parser.parse_args()

    if args.no_cache:
        for cache_path in [KWAVE_CACHE, PKWAV_CACHE]:
            if cache_path.exists():
                cache_path.unlink()
                print(f"  Removed cache: {cache_path}")

    print("=" * 62)
    print("us_beam_patterns: k-wave-python vs pykwavers")
    print(f"  Active grid : {NX}×{NY}×{NZ}   dx={DX * 1e3:.4f} mm")
    print(f"  Full  grid  : {TNX}×{TNY}×{TNZ}  PML=[{PML_X_SIZE},{PML_Y_SIZE},{PML_Z_SIZE}]")
    print(f"  Transducer  : {N_ELEMENTS} elements  f0={TONE_BURST_FREQ * 1e-6:.2f} MHz")
    print(f"  Focus       : az={FOCUS_DIST * 1e3:.0f} mm  el={ELEV_FOCUS * 1e3:.0f} mm")
    print(f"  Sensor      : xy-midplane  z_active={SENSOR_IZ_ACTIVE}")
    print(f"  k-wave exec : {'GPU' if args.gpu else 'CPU (OMP)'}")
    print("=" * 62)

    # --- Shared k-wave configuration ---
    print("\n[1/3] Building k-wave configuration...")
    kgrid, medium, not_transducer, input_signal = build_kwave_config()
    print(
        f"  dt={kgrid.dt:.4e} s   Nt={kgrid.Nt}   "
        f"t_end={kgrid.Nt * kgrid.dt * 1e6:.1f} µs"
    )

    # --- k-wave-python reference ---
    print("\n[2/3] k-wave-python reference...")
    kw = run_kwave(kgrid, medium, not_transducer, use_gpu=args.gpu)
    print(f"  p_rms peak  = {kw['p_rms'].max() * 1e-6:.4f} MPa")
    print(f"  p_max peak  = {kw['p_max'].max() * 1e-6:.4f} MPa")
    print(f"  runtime     = {kw['runtime_s']:.1f} s")

    # --- pykwavers ---
    print("\n[3/3] pykwavers (CPU PSTD)...")
    pkw_res = run_pykwavers(kgrid.dt, kgrid.Nt, input_signal, not_transducer)
    print(f"  p_rms peak  = {pkw_res['p_rms'].max() * 1e-6:.4f} MPa")
    print(f"  p_max peak  = {pkw_res['p_max'].max() * 1e-6:.4f} MPa")
    print(f"  runtime     = {pkw_res['runtime_s']:.1f} s")

    # --- Parity metrics ---
    print("\n--- Parity evaluation ---")
    all_pass = True
    report_sections: list[str] = []

    for measure in ["p_rms", "p_max"]:
        metrics = compute_image_metrics(kw[measure], pkw_res[measure])
        thr     = PARITY_THRESHOLDS[measure]
        checks  = {
            "pearson_r": metrics["pearson_r"] >= thr["pearson_r"],
            "rms_ratio": thr["rms_ratio_min"] <= metrics["rms_ratio"] <= thr["rms_ratio_max"],
            "psnr_db":   metrics["psnr_db"]   >= thr["psnr_db"],
        }
        status   = "PASS" if all(checks.values()) else "FAIL"
        all_pass = all_pass and (status == "PASS")

        print(f"  [{measure}]  {status}")
        print(f"    Pearson r  = {metrics['pearson_r']:.6f}  "
              f"(target >= {thr['pearson_r']})  {'[OK]' if checks['pearson_r'] else '[X]'}")
        print(f"    RMS ratio  = {metrics['rms_ratio']:.6f}  "
              f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])  "
              f"{'[OK]' if checks['rms_ratio'] else '[X]'}")
        print(f"    PSNR       = {metrics['psnr_db']:.2f} dB  "
              f"(target >= {thr['psnr_db']} dB)  {'[OK]' if checks['psnr_db'] else '[X]'}")

        report_sections.extend([
            f"{measure}: {status}",
            f"  pearson_r = {metrics['pearson_r']:.6f}  "
            f"(target >= {thr['pearson_r']})",
            f"  rms_ratio = {metrics['rms_ratio']:.6f}  "
            f"(target [{thr['rms_ratio_min']}, {thr['rms_ratio_max']}])",
            f"  psnr_db   = {metrics['psnr_db']:.2f}  "
            f"(target >= {thr['psnr_db']} dB)",
            "",
        ])

    overall = "PASS" if all_pass else "FAIL"

    # --- Figures ---
    plot_comparison(kw, pkw_res)
    plot_directivity(kw, pkw_res)

    # --- Metrics text report ---
    header = "\n".join([
        "us_beam_patterns parity metrics",
        f"parity_status: {overall}",
        f"grid_active: {NX}x{NY}x{NZ}   dx={DX:.6e} m",
        f"transducer: {N_ELEMENTS} elements, "
        f"f0={TONE_BURST_FREQ * 1e-6:.2f} MHz, "
        f"focus={FOCUS_DIST * 1e3:.0f} mm az / {ELEV_FOCUS * 1e3:.0f} mm el",
        f"kwave_runtime_s: {kw['runtime_s']:.3f}",
        f"pykwavers_runtime_s: {pkw_res['runtime_s']:.3f}",
        "",
    ])
    save_text_report(METRICS_PATH, header, report_sections)
    print(f"\n  Saved: {METRICS_PATH}")
    print(f"  Overall parity status: {overall}")

    if not all_pass and not args.allow_failure:
        raise SystemExit(
            "us_beam_patterns parity targets were not met. "
            "Run with --allow-failure to collect diagnostics without a non-zero exit."
        )

    print("\nDone.")
    print(f"  Figures: {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
