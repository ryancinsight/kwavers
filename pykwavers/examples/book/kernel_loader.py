"""
Thin Python shim: load PSTD-derived `.npz` kernels into
`pykwavers.KernelCube` for histotripsy treatment planners.

All resampling, kernel placement, and `(f0, pnp)` interpolation lives
in `kwavers::physics::field_surrogate` (Rust); this file is just I/O —
it reads the `.npz` files written by `cavitation_kernel.py`,
constructs `pykwavers.FocalKernel` instances from the numpy arrays,
and assembles a `pykwavers.KernelCube`.

Public API
----------
* `discover_kernels(roc, diameter)` — scan `data/kernels/` for matching
  `.npz` files and return a list of `(f0, pnp_realised)` pairs.
* `load_focal_kernel(f0, pnp, roc, diameter)` — load a single `.npz`
  into a `pykwavers.FocalKernel`. Falls back to the nearest available
  `pnp` and rescales linearly (water B/A=0).
* `build_cube(f0_values, pnp_values, roc, diameter)` — load every
  `(f0, pnp)` corner into a `pykwavers.KernelCube`.
* `kernel_focal_envelope(scenario_f0, target_shape, target_focus_idx,
  target_dx_m, ..., cache=...)` — drop-in replacement for the previous
  Python implementation; now delegates to a memoized `KernelCube`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

import pykwavers as kw

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
KERNEL_DIR = os.path.join(REPO_ROOT, "data", "kernels")

_KERNEL_FILE_RE = re.compile(
    r"kernel_(?P<f0>[0-9.]+)MHz_(?P<pnp>[0-9]+)MPa_"
    r"(?P<roc>[0-9]+)roc_(?P<diam>[0-9]+)diam\.npz"
)


@dataclass
class CachedKernel:
    """Backwards-compatible facade for callers that still expect the
    old per-kernel object. Wraps a `pykwavers.FocalKernel`."""
    inner: "kw.FocalKernel"
    source_npz: str

    @property
    def p_neg_field(self) -> np.ndarray:
        return np.asarray(self.inner.field())

    @property
    def focus_idx(self) -> tuple[int, int, int]:
        return tuple(self.inner.focus_idx)

    @property
    def dx_m(self) -> float:
        return self.inner.dx_m

    @property
    def f0(self) -> float:
        return self.inner.f0

    @property
    def pnp_realised(self) -> float:
        return self.inner.pnp_realised

    @property
    def source_pa(self) -> float:
        return self.inner.source_pa

    @property
    def fwhm_lat_m(self) -> float:
        return self.inner.fwhm_lat_m

    @property
    def fwhm_ax_m(self) -> float:
        return self.inner.fwhm_ax_m

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.inner.shape)


def _kernel_path(f0: float, pnp: float, roc: float, diameter: float) -> str:
    return os.path.join(
        KERNEL_DIR,
        f"kernel_{f0/1e6:.2f}MHz_{pnp/1e6:.0f}MPa_"
        f"{roc*1e3:.0f}roc_{diameter*1e3:.0f}diam.npz",
    )


def discover_kernels(roc: float = 30e-3, diameter: float = 30e-3) -> list[tuple[float, float]]:
    """Scan KERNEL_DIR for kernels matching the given (roc, diameter)
    geometry. Returns a sorted list of `(f0_Hz, pnp_Pa)` pairs found
    on disk."""
    pairs: list[tuple[float, float]] = []
    if not os.path.isdir(KERNEL_DIR):
        return pairs
    for name in sorted(os.listdir(KERNEL_DIR)):
        m = _KERNEL_FILE_RE.match(name)
        if m is None:
            continue
        if int(m.group("roc")) != int(roc * 1e3):
            continue
        if int(m.group("diam")) != int(diameter * 1e3):
            continue
        pairs.append((float(m.group("f0")) * 1e6, float(m.group("pnp")) * 1e6))
    return pairs


def load_focal_kernel(f0: float, pnp: float, roc: float = 30e-3,
                      diameter: float = 30e-3,
                      fallback_pnp: float | None = None) -> CachedKernel:
    """Load a `.npz` kernel and wrap it as a `pykwavers.FocalKernel`.

    If the exact `(f0, pnp)` is not on disk, fall back to the nearest
    available `pnp` and rescale the field linearly (water B/A = 0)."""
    path = _kernel_path(f0, pnp, roc, diameter)
    rescale = 1.0
    if not os.path.exists(path):
        for candidate in (30.0e6, 15.0e6) if fallback_pnp is None else (fallback_pnp,):
            alt = _kernel_path(f0, candidate, roc, diameter)
            if os.path.exists(alt):
                path = alt
                rescale = pnp / candidate
                break
        else:
            raise FileNotFoundError(
                f"No cached kernel for f0={f0/1e6:.2f}MHz pnp={pnp/1e6:.0f}MPa "
                f"roc={roc*1e3:.0f}mm diam={diameter*1e3:.0f}mm; tried {path} "
                f"and pnp fallbacks. Run cavitation_kernel.py --sweep first.")

    with np.load(path) as d:
        p_min = np.asarray(d["p_min"], dtype=np.float64)
        kernel_dx = float(d["dx"])
        focus = tuple(int(v) for v in d["focus_idx"])
        f0_in = float(d["f0"])
        pnp_realised = float(d["pnp_realised"]) * rescale
        source_pa = float(d["source_pa"]) * rescale
        fwhm_lat = float(d["fwhm_lat_m"])
        fwhm_ax = float(d["fwhm_ax_m"])

    p_neg = (-p_min) * rescale
    p_neg = np.maximum(p_neg, 0.0)  # peak rarefactional is non-negative

    inner = kw.FocalKernel(
        p_neg, kernel_dx, focus, f0_in, pnp_realised, source_pa,
        fwhm_lat, fwhm_ax,
    )
    return CachedKernel(inner=inner, source_npz=os.path.basename(path))


def build_cube(f0_values: Iterable[float] | None = None,
               pnp_values: Iterable[float] | None = None,
               roc: float = 30e-3, diameter: float = 30e-3) -> kw.KernelCube:
    """Build a `pykwavers.KernelCube` from the cached `.npz` kernels.

    If `f0_values` / `pnp_values` are not given, every kernel matching
    `(roc, diameter)` on disk is used. The (f0, pnp) pairs must form a
    Cartesian grid; missing corners cause `KernelCube.__init__` to
    raise."""
    if f0_values is None or pnp_values is None:
        discovered = discover_kernels(roc, diameter)
        if not discovered:
            raise FileNotFoundError(
                f"No kernels found in {KERNEL_DIR} matching "
                f"roc={roc*1e3:.0f}mm diam={diameter*1e3:.0f}mm. "
                f"Run cavitation_kernel.py --sweep first.")
        if f0_values is None:
            f0_values = sorted({pair[0] for pair in discovered})
        if pnp_values is None:
            pnp_values = sorted({pair[1] for pair in discovered})

    kernels: list[kw.FocalKernel] = []
    for f0 in f0_values:
        for pnp in pnp_values:
            ck = load_focal_kernel(f0, pnp, roc=roc, diameter=diameter)
            kernels.append(ck.inner)
    return kw.KernelCube(kernels)


# ───────────────────────────────────────────────────────────────────────
# Backward-compat helpers — drop-in replacements for the prior Python
# implementation so callers (ch21e) need only swap the import.
# ───────────────────────────────────────────────────────────────────────


def kernel_focal_envelope(scenario_f0: float,
                          target_shape: tuple[int, int, int],
                          target_focus_idx: tuple[int, int, int],
                          target_dx_m: float,
                          roc: float = 30e-3, diameter: float = 30e-3,
                          cache: dict | None = None) -> np.ndarray:
    """Per-voxel **normalized** focal envelope (`env.max() == 1`) on
    the planner grid. Wraps `pykwavers.KernelCube.query`.

    `cache` is a caller-owned dict; keys are
    `("__cube__", roc, diameter)` for the cube itself and
    `(f0, target_shape, target_focus_idx, target_dx_m)` for query
    results. Both are memoized to amortize cube construction (~1 s)
    and per-query resampling (~50 ms) across many calls.
    """
    cache_key = (scenario_f0, target_shape, target_focus_idx,
                 target_dx_m, roc, diameter)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    cube_key = ("__cube__", roc, diameter)
    if cache is not None and cube_key in cache:
        cube = cache[cube_key]
    else:
        cube = build_cube(roc=roc, diameter=diameter)
        if cache is not None:
            cache[cube_key] = cube

    # `pnp` is degenerate in the linear-water regime — pass any value
    # in the sweep; the cube ignores it for shape selection.
    pnp_axis = list(cube.pnp_axis)
    pnp_dummy = pnp_axis[0] if pnp_axis else 30.0e6
    env = np.asarray(cube.query(scenario_f0, pnp_dummy, target_shape,
                                  target_focus_idx, target_dx_m))
    if cache is not None:
        cache[cache_key] = env
    return env


def load_kernel(f0: float, pnp: float, roc: float = 30e-3, diameter: float = 30e-3,
                target_dx_m: float | None = None,
                fallback_pnp: float | None = None) -> CachedKernel:
    """Backwards-compatible alias retained for the unit-test suite.
    Resampling now happens inside `pykwavers.KernelCube.query`, so
    `target_dx_m` is no longer used; if you need a kernel at a
    different dx, build a `KernelCube` and call `query`."""
    if target_dx_m is not None:
        # Honour the resample request by routing through the cube.
        ck = load_focal_kernel(f0, pnp, roc=roc, diameter=diameter,
                                fallback_pnp=fallback_pnp)
        cube = kw.KernelCube([ck.inner])
        # Synthesise a kernel at target_dx_m by using the cube's
        # internal trilinear resample; we wrap the result back into
        # CachedKernel for compat. We do this by querying a grid
        # centred on the resampled focal voxel.
        # Rather than reverse-engineering the resampled kernel from
        # query output, perform the resample-and-place directly:
        # query returns a normalized envelope, so multiply by the
        # source kernel's focal pressure to recover absolute Pa.
        target_shape = tuple(
            max(1, int(round(s * ck.dx_m / target_dx_m))) for s in ck.shape
        )
        new_focus = tuple(int(round(idx * ck.dx_m / target_dx_m))
                           for idx in ck.focus_idx)
        env = np.asarray(cube.query(ck.f0, ck.pnp_realised, target_shape,
                                      new_focus, target_dx_m))
        peak_pa = ck.inner.focal_pressure()
        field = env * peak_pa
        new_kernel = kw.FocalKernel(
            field, target_dx_m, new_focus, ck.f0, ck.pnp_realised,
            ck.source_pa, ck.fwhm_lat_m, ck.fwhm_ax_m,
        )
        return CachedKernel(inner=new_kernel, source_npz=ck.source_npz)
    return load_focal_kernel(f0, pnp, roc=roc, diameter=diameter,
                              fallback_pnp=fallback_pnp)


def place_kernel_at_focus(kernel: CachedKernel,
                          target_shape: tuple[int, int, int],
                          target_focus_idx: tuple[int, int, int]) -> np.ndarray:
    """Backwards-compatible placement helper. Delegates to the Rust
    `pykwavers.place_kernel_at_focus` binding which preserves absolute
    Pa amplitude (no normalization)."""
    return np.asarray(kw.place_kernel_at_focus(
        kernel.inner, target_shape, target_focus_idx,
    ))
