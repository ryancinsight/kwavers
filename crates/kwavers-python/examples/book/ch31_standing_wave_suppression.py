"""Chapter 31: Iterative Standing-Wave Suppression via Reconstruction and Phase Optimization.

Problem statement
-----------------
Standing waves arise in histotripsy and FWI/RTM imaging when forward-travelling
waves reflect off impedance discontinuities and constructively interfere with the
incident field at λ/2 spatial intervals.  The spatial periodicity:

  * redistributes peak pressure away from the intended focal target,
  * creates unintended hot spots in healthy tissue, and
  * corrupts FWI/RTM reconstructions via coherent λ/2 artefacts.

Suppression loop (all physics in Rust via kwavers)
---------------------------------------------------
1. Precompute  — one linearised 2D FDTD run per element (Rayon parallel) → G_i(x,y).
2. Reconstruct — p(x,y; φ) = Σ_i exp(iφ_i) G_i(x,y)  [Born/RTM superposition].
3. Analyse     — SWI from spectral λ/2 fringe; p_focal from peak |p| at target.
4. Optimise    — gradient-descent step on φ with Armijo backtracking.
5. Repeat steps 2–4 for n_opt_iter iterations, recording the full time series.

Environment variables
---------------------
KWAVERS_CH31_NX                 default 128
KWAVERS_CH31_NY                 default 64
KWAVERS_CH31_N_ELEMENTS         default 12
KWAVERS_CH31_N_ITER             default 25
KWAVERS_CH31_SWI_WEIGHT         default 0.70
KWAVERS_CH31_FOCAL_WEIGHT       default 0.30
KWAVERS_CH31_FREQUENCY_HZ       default 250000.0
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


BOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BOOK_DIR.parents[2]
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch31"
PY_PACKAGE = REPO_ROOT / "pykwavers" / "python"

if "PYKWAVERS_EXTENSION_PATH" not in os.environ:
    for candidate in (
        REPO_ROOT / "target" / "release" / "pykwavers.dll",
        REPO_ROOT / "target" / "maturin" / "pykwavers.dll",
        REPO_ROOT / "target" / "debug" / "pykwavers.dll",
    ):
        if candidate.exists():
            os.environ["PYKWAVERS_EXTENSION_PATH"] = str(candidate)
            break

if str(PY_PACKAGE) not in sys.path:
    sys.path.insert(0, str(PY_PACKAGE))
if str(BOOK_DIR) not in sys.path:
    sys.path.insert(0, str(BOOK_DIR))

import pykwavers as kw  # noqa: E402

from standing_wave_opt.figures import (  # noqa: E402
    plot_before_after,
    plot_convergence,
    plot_field_evolution,
    plot_geometry,
)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def run() -> dict[str, object]:
    """Generate Chapter 31 figures and return a metrics dictionary."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nx = _env_int("KWAVERS_CH31_NX", 128)
    ny = _env_int("KWAVERS_CH31_NY", 64)
    n_elements = _env_int("KWAVERS_CH31_N_ELEMENTS", 12)
    n_iter = _env_int("KWAVERS_CH31_N_ITER", 25)
    swi_weight = _env_float("KWAVERS_CH31_SWI_WEIGHT", 0.70)
    focal_weight = _env_float("KWAVERS_CH31_FOCAL_WEIGHT", 0.30)
    frequency_hz = _env_float("KWAVERS_CH31_FREQUENCY_HZ", 250_000.0)

    # Scale geometry proportionally when the grid is resized
    scale_x = nx / 128
    scale_y = ny / 64
    focus_x = max(12, int(68 * scale_x))
    focus_y = ny // 2
    layer_x_start = max(focus_x + 6, int(90 * scale_x))
    layer_x_end = min(nx - 11, int(96 * scale_x))
    element_y_min = max(11, int(12 * scale_y))
    element_y_max = min(ny - 12, int(52 * scale_y))

    print(
        f"[ch31] grid {nx}×{ny}, {n_elements} elements, "
        f"focus ({focus_x},{focus_y}), "
        f"layer x=[{layer_x_start},{layer_x_end}), "
        f"{n_iter} iterations, {frequency_hz/1e3:.0f} kHz"
    )
    print("[ch31] Phase 1 — Rust FDTD Green's function precomputation (Rayon parallel) …")

    result = kw.run_standing_wave_suppression(
        nx=nx,
        ny=ny,
        frequency_hz=frequency_hz,
        n_elements=n_elements,
        element_y_min=element_y_min,
        element_y_max=element_y_max,
        focus_x=focus_x,
        focus_y=focus_y,
        layer_x_start=layer_x_start,
        layer_x_end=layer_x_end,
        n_opt_iter=n_iter,
        swi_weight=swi_weight,
        focal_weight=focal_weight,
    )

    swi_hist = np.asarray(result["swi_history"], dtype=float)
    pf_hist = np.asarray(result["focal_pressure_history"], dtype=float)
    swi_reduction = 100.0 * (swi_hist[0] - swi_hist[-1]) / (swi_hist[0] + 1e-12)
    pf_ratio = float(pf_hist[-1]) / (float(pf_hist[0]) + 1e-30)

    print(
        f"[ch31] Phase 2 complete — "
        f"SWI {swi_hist[0]:.4f} → {swi_hist[-1]:.4f} "
        f"(−{swi_reduction:.1f}% reduction)   "
        f"p_focal × {pf_ratio:.4f}"
    )

    print("[ch31] Phase 3 — rendering figures …")
    figures = [
        plot_geometry(result, OUT_DIR),
        plot_field_evolution(result, OUT_DIR),
        plot_convergence(result, OUT_DIR),
        plot_before_after(result, OUT_DIR),
    ]

    metrics = {
        "chapter": 31,
        "analysis": (
            "Iterative standing-wave suppression via Born/RTM Green's function "
            "reconstruction and gradient-descent phase optimization of a linear "
            "transducer array through a bone-like reflective layer"
        ),
        "simulation_type": (
            "linearised 2D FDTD Green's function precomputation (Rayon parallel per element); "
            "closed-form Born superposition for optimization; "
            "SWI via spectral λ/2 fringe detection (windowed DFT)"
        ),
        "nx": int(result["nx"]),
        "ny": int(result["ny"]),
        "dx_m": float(result["dx_m"]),
        "frequency_hz": float(result["frequency_hz"]),
        "n_elements": int(result["n_elements"]),
        "focus": [int(result["focus_x"]), int(result["focus_y"])],
        "reflector_cells": [int(result["reflector_x_start"]), int(result["reflector_x_end"])],
        "n_iterations": len(swi_hist) - 1,
        "swi_initial": float(swi_hist[0]),
        "swi_final": float(swi_hist[-1]),
        "swi_reduction_fraction": float((swi_hist[0] - swi_hist[-1]) / (swi_hist[0] + 1e-12)),
        "focal_pressure_initial_pa": float(pf_hist[0]),
        "focal_pressure_final_pa": float(pf_hist[-1]),
        "focal_pressure_change_fraction": float((pf_hist[-1] - pf_hist[0]) / (pf_hist[0] + 1e-30)),
        "swi_weight": float(result["swi_weight"]),
        "focal_weight": float(result["focal_weight"]),
        "swi_history": [float(v) for v in swi_hist],
        "focal_pressure_history": [float(v) for v in pf_hist],
        "objective_history": [float(v) for v in np.asarray(result["objective_history"])],
        "figures": [str(p) for p in figures],
    }

    out_metrics = OUT_DIR / "metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[ch31] Metrics written to {out_metrics}")
    return metrics


if __name__ == "__main__" or __name__ == "ch31":
    run()
