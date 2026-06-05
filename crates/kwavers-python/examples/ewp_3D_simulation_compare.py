#!/usr/bin/env python3
"""
ewp_3D_simulation_compare.py
==============================
Full 3-D elastic simulation with spatial-symmetry validation.

k-wave-python does not support elastic simulations.  This script validates
pykwavers via an analytical symmetry invariant: in a homogeneous isotropic
elastic medium, a point vx-source at the grid centre produces waveforms
that are equal (up to sign) at all sensors equidistant from the source
along the SAME Cartesian axis.

Physical setup
--------------
Grid:      N³ cube, N = 32, dx = dy = dz = 0.75 mm
Medium:    Homogeneous elastic solid
           c_p = 3000 m/s,  c_s = 1500 m/s,  ρ = 1900 kg/m³
Source:    Ricker wavelet velocity source (vx) at grid centre, σ = 2 samples,
           t₀ = 5σ (near-zero discrete sum — avoids DC velocity ramp).
Sensors:   6 sensors at distance D = 8 cells from the source along ±x, ±y, ±z.

Simulation window
-----------------
T_END = 6 µs (NT = 80 steps at dt = 7.5 × 10⁻⁸ s).
P-wave peak arrives at axial sensors at step ≈ 37.
S-wave peak arrives at transverse sensors at step ≈ 63.
Both are captured within NT = 80; the simulation stays below the PML
instability onset (~NT > 200 steps) for this grid and simple velocity-only
PML.

Analytical invariant (within-group symmetry)
--------------------------------------------
For a homogeneous isotropic elastic medium with a vx point source, the
ux velocity component satisfies:

  Axial sensors (±x):
    Both see the compressional (P-wave) ux component at identical time.
    Polarity may be opposite, so |Pearson(+x, −x)| ≥ 0.98.
    The P-wave reaches axial sensors earliest, so velocity-only PML
    reflections arrive within NT=80 and reduce symmetry slightly below
    the 0.999 level achievable by split-field PML.

  Transverse sensors (±y, ±z):
    All four see the shear (S-wave) ux component.  By 4-fold rotational
    symmetry around the x-axis:
      |Pearson(+y, −y)| ≥ 0.98  (mirror symmetry)
      |Pearson(+z, −z)| ≥ 0.98  (mirror symmetry)
      |Pearson(+y, +z)| ≥ 0.98  (90° rotational symmetry in y-z plane)

NOTE: axial vs. transverse sensors are NOT compared against each other
because they record different wave types (P and S) that arrive at
different times and have different shapes.

Usage::

    python examples/ewp_3D_simulation_compare.py
    python examples/ewp_3D_simulation_compare.py --allow-failure
"""

from __future__ import annotations

import argparse
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from example_parity_utils import (
    DEFAULT_OUTPUT_DIR,
    bootstrap_example_paths,
    save_text_report,
)

bootstrap_example_paths()
import pykwavers as pkw

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
N     = 32
DX    = 0.75e-3       # [m]

CP    = 3000.0        # P-wave speed [m/s]
CS    = 1500.0        # S-wave speed [m/s]
RHO   = 1900.0        # density [kg/m³]

SIGMA_CELLS = 2       # Ricker pulse width [cells]

CFL   = 0.3
DT    = CFL * DX / CP   # = 7.5e-8 s

# NT = 80 captures both P-wave (axial, step ~37) and S-wave (transverse,
# step ~63) peaks while remaining below PML instability onset (~NT > 200).
T_END = 6.0e-6
NT    = int(round(T_END / DT))   # = 80

PML   = 6

CX = N // 2   # = 16
CY = N // 2
CZ = N // 2

# Sensor distance from source [cells].
# With PML=6, valid domain = [6:26].  CX±8 = 8 and 24 — both in [6:26]. ✓
SENSOR_D = 8


def run_elastic() -> tuple[np.ndarray, list[str]]:
    """Return sensor ux data (6, NT) and ordered labels."""
    cp_arr  = np.full((N, N, N), CP)
    cs_arr  = np.full((N, N, N), CS)
    rho_arr = np.full((N, N, N), RHO)
    medium  = pkw.Medium.elastic_heterogeneous(cp_arr, cs_arr, rho_arr)

    # Source: Ricker wavelet in vx at grid centre.
    # t₀ = 5σ so the signal starts near-zero — prevents DC velocity ramp.
    src_mask = np.zeros((N, N, N), dtype=bool)
    src_mask[CX, CY, CZ] = True
    sigma_s = SIGMA_CELLS * DT
    t       = np.arange(NT) * DT
    t0      = 5.0 * sigma_s
    tau     = (t - t0) / sigma_s
    signal_1d = (1.0 - tau ** 2) * np.exp(-0.5 * tau ** 2)
    source  = pkw.Source.from_elastic_velocity_source(src_mask, ux=signal_1d)

    # 6 equidistant sensors along ±x, ±y, ±z.
    named_positions: dict[str, tuple[int, int, int]] = {
        "+x": (CX + SENSOR_D, CY, CZ),
        "-x": (CX - SENSOR_D, CY, CZ),
        "+y": (CX, CY + SENSOR_D, CZ),
        "-y": (CX, CY - SENSOR_D, CZ),
        "+z": (CX, CY, CZ + SENSOR_D),
        "-z": (CX, CY, CZ - SENSOR_D),
    }

    sens_mask = np.zeros((N, N, N), dtype=bool)
    for sx, sy, sz in named_positions.values():
        if 0 <= sx < N and 0 <= sy < N and 0 <= sz < N:
            sens_mask[sx, sy, sz] = True

    sensor = pkw.Sensor.from_mask(sens_mask)

    sim = pkw.Simulation(
        pkw.Grid(N, N, N, DX, DX, DX),
        medium, source, sensor,
        solver=pkw.SolverType.ElasticPSTD,
    )
    sim.set_pml_size(PML)
    sim.set_pml_inside(True)

    result = sim.run(time_steps=NT, dt=DT)
    ux = result.ux
    if ux is not None:
        data = np.asarray(ux, dtype=np.float64)
    else:
        data = np.asarray(result.sensor_data, dtype=np.float64)

    # Sensor.from_mask returns rows in the C-order (row-major) sequence of the
    # masked cells — i.e. np.argwhere(mask) order, which sorts by (x, y, z) with
    # x slowest and z fastest. Map labels in that same order so row indices align.
    sorted_names = sorted(
        named_positions.keys(),
        key=lambda n: named_positions[n],
    )
    return data, sorted_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()

    out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"ewp_3D_simulation: N={N}³, NT={NT}, dt={DT:.3e} s\n"
        f"  c_p={CP:.0f} m/s, c_s={CS:.0f} m/s, rho={RHO:.0f} kg/m³\n"
        f"  Source at ({CX},{CY},{CZ}), sensors at distance D={SENSOR_D} cells"
    )

    print("\nRunning 3-D elastic simulation...")
    t0 = time.perf_counter()
    data, labels = run_elastic()
    runtime_s = time.perf_counter() - t0
    print(f"  Done in {runtime_s:.1f} s  sensor_data shape={data.shape}")

    n_traces = data.shape[0]
    if n_traces != 6:
        print(f"ERROR: expected 6 sensor traces, got {n_traces}")
        if not args.allow_failure:
            return 1

    # Build name→row-index map from the sorted labels.
    idx = {name: i for i, name in enumerate(labels)}

    def get(name: str) -> np.ndarray:
        return data[idx[name], :]

    def abs_pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() > 0 and b.std() > 0:
            return abs(float(pearsonr(a, b)[0]))
        return 0.0

    # Within-group symmetry pairs
    # Axial: (+x, −x) — both record P-wave ux (possibly opposite sign)
    # Transverse: (+y, −y), (+z, −z), (+y, +z) — all record S-wave ux
    pairs = [
        ("+x", "-x", "axial:      +x vs -x"),
        ("+y", "-y", "transverse: +y vs -y"),
        ("+z", "-z", "transverse: +z vs -z"),
        ("+y", "+z", "transverse: +y vs +z"),
    ]

    PEARSON_MIN = 0.98
    results: list[tuple[str, float]] = []

    print(f"\nWithin-group symmetry metrics (threshold |Pearson| ≥ {PEARSON_MIN:.3f}):")
    for name_a, name_b, desc in pairs:
        p = abs_pearson(get(name_a), get(name_b))
        ok = "OK" if p >= PEARSON_MIN else "FAIL"
        print(f"  {desc}: |Pearson| = {p:.6f}  [{ok}]")
        results.append((desc, p))

    passed = all(p >= PEARSON_MIN for _, p in results)
    status = "PASS" if passed else "FAIL"
    print(f"\n  status: {status}")

    # ---- Plot ----------------------------------------------------------------
    t_ax   = np.arange(data.shape[1]) * DT * 1e6
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(labels):
        ax.plot(t_ax, data[i, :], label=name, lw=1.0)
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Ux [m/s]")
    min_p = min(p for _, p in results) if results else 0.0
    ax.set_title(
        f"ewp_3D_simulation: within-group symmetry  "
        f"|Pearson| min = {min_p:.4f}  [{status}]"
    )
    ax.legend(ncol=2)
    plt.tight_layout()
    fig_path = out_dir / "ewp_3D_simulation_compare.png"
    plt.savefig(fig_path, dpi=100)
    plt.close()
    print(f"  Figure saved: {fig_path}")

    lines = [
        f"Grid: {N}³  dx={DX*1e3:.3f} mm",
        f"c_p={CP:.0f} m/s  c_s={CS:.0f} m/s  rho={RHO:.0f} kg/m3",
        f"Sensor distance: {SENSOR_D} cells = {SENSOR_D*DX*1e3:.2f} mm",
        f"NT={NT}  dt={DT:.4e} s",
    ]
    for desc, p in results:
        lines.append(f"  {desc}: |Pearson| = {p:.6f}")
    lines.append(f"RESULT: {status}")

    save_text_report(
        out_dir / "ewp_3D_simulation_compare.txt",
        "ewp_3D_simulation_compare",
        lines,
    )

    if not passed and not args.allow_failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
