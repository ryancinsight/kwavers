"""
Chapter 2: Numerical Methods — Figure Generation Script
=======================================================

Generates analytical numerical-method figures for docs/book/numerical_methods.md.

Figures produced:
  fig01: CFL stability region for the second-order acoustic FDTD scheme.
  fig02: Discrete phase-velocity error for second-, fourth-, and sixth-order stencils.
  fig03: k-space temporal correction sinc(c k dt / 2).
  fig04: PSTD spectral differentiation compared with finite-difference symbols.
  fig05: Sparse sensor recorder memory scaling for owned output vs borrowed views.

The figures use closed-form amplification factors and modified-wavenumber
symbols, so they are deterministic and do not depend on a solver runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / "docs" / "book" / "figures" / "ch02"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        plt.savefig(OUT_DIR / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch02/{name}.{{pdf,png}}")


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 1.7,
    }
)


def fd_symbol(theta: np.ndarray, order: int) -> np.ndarray:
    """Return first-derivative modified wavenumber k* dx for centered stencils."""
    if order == 2:
        return np.sin(theta)
    if order == 4:
        return (8.0 * np.sin(theta) - np.sin(2.0 * theta)) / 6.0
    if order == 6:
        return (45.0 * np.sin(theta) - 9.0 * np.sin(2.0 * theta) + np.sin(3.0 * theta)) / 30.0
    raise ValueError(f"unsupported order {order}")


# Figure 01: CFL stability region for 2-D explicit wave stepping.
print("[fig01] FDTD CFL stability region")
cfl_x = np.linspace(0.0, 1.2, 350)
cfl_z = np.linspace(0.0, 1.2, 350)
cx, cz = np.meshgrid(cfl_x, cfl_z, indexing="ij")
stable = cx * cx + cz * cz <= 1.0
plt.figure(figsize=(6.3, 5.2))
plt.contourf(cx, cz, stable.astype(float), levels=[-0.1, 0.5, 1.1], colors=["#f4b6b6", "#b9dfc5"])
plt.contour(cx, cz, cx * cx + cz * cz, levels=[1.0], colors="k", linewidths=1.8)
plt.xlabel("CFL_x = c dt / dx")
plt.ylabel("CFL_z = c dt / dz")
plt.title("2-D explicit acoustic CFL region: CFL_x^2 + CFL_z^2 <= 1")
plt.text(0.18, 0.20, "stable", fontsize=12)
plt.text(0.88, 0.88, "unstable", fontsize=12)
plt.grid(alpha=0.25)
savefig("fig01_fdtd_cfl_region")
plt.close()


# Figure 02: finite-difference phase-velocity error.
print("[fig02] Modified-wavenumber phase error")
theta = np.linspace(1.0e-6, np.pi, 700)
plt.figure(figsize=(7.2, 4.8))
for order in (2, 4, 6):
    ratio = fd_symbol(theta, order) / theta
    _ord = {2: "2nd", 4: "4th", 6: "6th"}.get(order, f"{order}th")
    plt.plot(theta / np.pi, ratio, label=f"{_ord}-order centered")
plt.axhline(1.0, color="k", linewidth=0.9, linestyle=":")
plt.xlabel("k dx / pi")
plt.ylabel("k* / k")
plt.title("Finite-difference dispersion from modified-wavenumber symbols")
plt.legend()
plt.grid(alpha=0.3)
savefig("fig02_modified_wavenumber_error")
plt.close()


# Figure 03: k-space temporal propagator correction.
print("[fig03] k-space correction factor")
cfl_values = (0.1, 0.3, 0.5, 0.7)
theta = np.linspace(0.0, np.pi, 700)
plt.figure(figsize=(7.2, 4.8))
for cfl in cfl_values:
    x = 0.5 * cfl * theta
    correction = np.ones_like(theta)
    mask = x != 0.0
    correction[mask] = np.sin(x[mask]) / x[mask]
    plt.plot(theta / np.pi, correction, label=f"CFL={cfl:.1f}")
plt.xlabel("k dx / pi")
plt.ylabel("sinc(c k dt / 2)")
plt.title("k-space temporal correction used by PSTD schemes")
plt.legend()
plt.grid(alpha=0.3)
savefig("fig03_kspace_temporal_correction")
plt.close()


# Figure 04: spectral derivative vs finite-difference symbols.
print("[fig04] Spectral and finite-difference derivative symbols")
theta = np.linspace(0.0, np.pi, 700)
plt.figure(figsize=(7.2, 4.8))
plt.plot(theta / np.pi, theta / np.pi, color="k", label="spectral derivative")
for order in (2, 4, 6):
    plt.plot(theta / np.pi, fd_symbol(theta, order) / np.pi, linestyle="--", label=f"FD {order}")
plt.xlabel("k dx / pi")
plt.ylabel("k* dx / pi")
plt.title("Derivative-symbol accuracy before Nyquist")
plt.legend()
plt.grid(alpha=0.3)
savefig("fig04_derivative_symbols")
plt.close()


# Figure 05: memory scaling for sparse recorder output.
print("[fig05] Recorder memory scaling")
n_steps = np.array([250, 500, 1_000, 2_000, 4_000])
n_sensors = 512
bytes_f64 = 8
owned_mb = n_sensors * n_steps * bytes_f64 / 2**20
clone_then_trim_mb = 2.0 * owned_mb
borrowed_view_mb = np.zeros_like(owned_mb)
plt.figure(figsize=(7.2, 4.8))
plt.plot(n_steps, clone_then_trim_mb, marker="o", label="clone + trim path")
plt.plot(n_steps, owned_mb, marker="s", label="single owned output")
plt.plot(n_steps, borrowed_view_mb, marker="^", label="borrowed view")
plt.xlabel("recorded time steps")
plt.ylabel("transient allocation [MiB]")
plt.title("Sparse recorder extraction memory invariant")
plt.legend()
plt.grid(alpha=0.3)
savefig("fig05_recorder_memory_scaling")
plt.close()


print(
    f"\nChapter 2 figures written to: {os.path.relpath(OUT_DIR)}\n"
    "  fig01_fdtd_cfl_region.*          - explicit 2-D CFL stability region\n"
    "  fig02_modified_wavenumber_error.* - finite-difference dispersion\n"
    "  fig03_kspace_temporal_correction.* - PSTD k-space correction\n"
    "  fig04_derivative_symbols.*        - spectral vs FD derivative symbols\n"
    "  fig05_recorder_memory_scaling.*   - sparse recorder allocation scaling\n"
)
