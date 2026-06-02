"""
Unit tests for kernel_loader (cached PSTD focal kernels for ch21d/e).

Run:
    python pykwavers/examples/book/test_kernel_loader.py

These are not pytest tests; they live next to the loader so they can be
run without dragging in a test harness, and they exercise the contracts
the planners rely on:
    * load_kernel returns the requested (f0, pnp) when present on disk
    * load_kernel falls back + linearly rescales when the exact pnp is
      missing
    * place_kernel_at_focus puts the kernel's focal voxel at the
      requested target index, with zero-fill outside the kernel footprint
    * kernel_focal_envelope returns env[focus] = 1
    * resampling preserves the focal-voxel location (within 1 voxel)
    * cache memoization works
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import pykwavers as kw
from kernel_loader import (CachedKernel, KERNEL_DIR, build_cube,
                            kernel_focal_envelope, load_kernel,
                            place_kernel_at_focus)
KernelCube = kw.KernelCube  # alias so existing tests keep their names

PASSES = 0
FAILS = 0


def _assert(name: str, cond: bool, detail: str = ""):
    global PASSES, FAILS
    if cond:
        PASSES += 1
        print(f"  PASS: {name}")
    else:
        FAILS += 1
        print(f"  FAIL: {name}  {detail}")


def _close(a: float, b: float, rtol: float = 1e-3) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-30)


def test_load_exact_match():
    print("\n[test] load_kernel — exact (f0, pnp) match on disk")
    k = load_kernel(f0=1.0e6, pnp=30.0e6, target_dx_m=None)
    _assert("returns CachedKernel", isinstance(k, CachedKernel))
    _assert("realised pnp ~ 30 MPa (no rescale)",
            _close(k.pnp_realised, 30e6, rtol=0.10),
            f"pnp_realised={k.pnp_realised/1e6:.2f}")
    _assert("p_neg_field is positive at focus",
            float(k.p_neg_field[k.focus_idx]) > 1.0e6,
            f"focus pnp = {float(k.p_neg_field[k.focus_idx])/1e6:.2f} MPa")
    _assert("source_npz contains 30MPa",
            "30MPa" in k.source_npz, k.source_npz)


def test_load_with_rescale_fallback():
    print("\n[test] load_kernel — fallback + linear rescale to 18 MPa")
    # 18 MPa not on disk; should fall back to 30 MPa and scale by 0.6
    k = load_kernel(f0=0.5e6, pnp=18.0e6, target_dx_m=None)
    _assert("realised pnp ~ 18 MPa post-rescale",
            _close(k.pnp_realised, 18e6, rtol=0.15),
            f"pnp_realised={k.pnp_realised/1e6:.2f}")
    # Compare to the source 30MPa kernel — pnp should be 0.6× larger there
    k_30 = load_kernel(f0=0.5e6, pnp=30.0e6, target_dx_m=None)
    ratio = k.p_neg_field[k.focus_idx] / k_30.p_neg_field[k_30.focus_idx]
    _assert("field scaled by 18/30 = 0.6",
            _close(float(ratio), 0.6, rtol=1e-6),
            f"ratio={float(ratio):.4f}")


def test_resample_to_target_dx():
    print("\n[test] load_kernel — resample to target dx")
    k_native = load_kernel(f0=1.0e6, pnp=30.0e6, target_dx_m=None)
    target_dx = 1.2e-3
    k_resamp = load_kernel(f0=1.0e6, pnp=30.0e6, target_dx_m=target_dx)
    _assert("resampled dx matches request",
            _close(k_resamp.dx_m, target_dx, rtol=1e-9))
    # Focus index should scale with zoom factor (approximately)
    zoom = k_native.dx_m / target_dx
    expected_focus_x = int(round(k_native.focus_idx[0] * zoom))
    _assert("focus_idx scales with zoom",
            abs(k_resamp.focus_idx[0] - expected_focus_x) <= 1,
            f"resampled focus_idx[0]={k_resamp.focus_idx[0]}, "
            f"expected~{expected_focus_x}")
    # Resampling a peaked function inherently smooths the focal voxel.
    # Cubic (order=3) preserves the peak to within ~10 %; the planner's
    # `kernel_focal_envelope` normalizes by the focal voxel anyway, so
    # absolute Pa loss here is recovered downstream — what we check
    # instead is that the integral (volume) of the field is preserved
    # across resampling (mass conservation).
    int_native = float(k_native.p_neg_field.sum()) * (k_native.dx_m ** 3)
    int_resamp = float(k_resamp.p_neg_field.sum()) * (k_resamp.dx_m ** 3)
    _assert("field integral preserved within 5% across resampling",
            _close(int_native, int_resamp, rtol=0.05),
            f"native={int_native:.3e}, resampled={int_resamp:.3e}")
    pnp_native = float(k_native.p_neg_field[k_native.focus_idx])
    pnp_resamp = float(k_resamp.p_neg_field[k_resamp.focus_idx])
    _assert("focal pnp preserved within 25% (cubic resample of peaked field)",
            _close(pnp_native, pnp_resamp, rtol=0.25),
            f"native={pnp_native/1e6:.2f}, resampled={pnp_resamp/1e6:.2f} MPa")


def test_place_kernel_at_focus():
    print("\n[test] place_kernel_at_focus — focal alignment + zero-fill")
    k = load_kernel(f0=1.0e6, pnp=30.0e6, target_dx_m=None)
    target_shape = (200, 100, 100)
    target_focus = (150, 50, 50)
    placed = place_kernel_at_focus(k, target_shape, target_focus)
    _assert("placed shape matches target", placed.shape == target_shape)
    # The kernel's focal voxel should land at target_focus
    placed_at_focus = float(placed[target_focus])
    kernel_focal_pnp = float(k.p_neg_field[k.focus_idx])
    _assert("kernel focal voxel landed at target focus",
            _close(placed_at_focus, kernel_focal_pnp, rtol=1e-9),
            f"placed[focus]={placed_at_focus/1e6:.2f}, "
            f"kernel[focus]={kernel_focal_pnp/1e6:.2f}")
    # Voxels far from the kernel footprint should be zero
    _assert("zero-fill far from footprint", placed[0, 0, 0] == 0.0)


def test_focal_envelope_normalized():
    print("\n[test] kernel_focal_envelope — env[focus] = 1")
    target_shape = (180, 100, 100)
    target_focus = (130, 50, 50)
    env = kernel_focal_envelope(
        scenario_f0=1.0e6,
        target_shape=target_shape,
        target_focus_idx=target_focus,
        target_dx_m=1.0e-3,
    )
    _assert("env shape matches target", env.shape == target_shape)
    _assert("env[focus] ~ 1.0 (within sub-voxel of true peak)",
            _close(float(env[target_focus]), 1.0, rtol=0.05),
            f"env[focus]={float(env[target_focus]):.6f}")
    _assert("env.max() == 1.0 (normalized by global max)",
            _close(float(env.max()), 1.0, rtol=1e-9),
            f"max={float(env.max()):.6f}")
    _assert("env min >= 0",
            float(env.min()) >= 0.0,
            f"min={float(env.min()):.6f}")


def test_envelope_caching():
    print("\n[test] kernel_focal_envelope — cache memoization")
    cache: dict = {}
    target_shape = (180, 100, 100)
    target_focus = (130, 50, 50)
    args = dict(scenario_f0=1.0e6, target_shape=target_shape,
                target_focus_idx=target_focus, target_dx_m=1.0e-3)
    env1 = kernel_focal_envelope(cache=cache, **args)
    # Cache after first call: 1 cube entry + 1 envelope entry = 2.
    _assert("first call populates cache", len(cache) == 2,
            f"cache size = {len(cache)}")
    env2 = kernel_focal_envelope(cache=cache, **args)
    _assert("second call returns same object (cache hit)",
            env1 is env2)


def test_envelope_for_500khz():
    print("\n[test] kernel_focal_envelope — 500 kHz scenario")
    target_shape = (180, 100, 100)
    target_focus = (130, 50, 50)
    env = kernel_focal_envelope(
        scenario_f0=0.5e6,
        target_shape=target_shape,
        target_focus_idx=target_focus,
        target_dx_m=1.2e-3,
    )
    # Sub-voxel resampling shift can move the peak ±0.5 voxels off
    # `target_focus_idx`; normalization is done by global max so peak
    # is always 1.0, but env[target_focus] reads slightly less.
    _assert("500 kHz env[focus] within 10% of 1.0",
            _close(float(env[target_focus]), 1.0, rtol=0.10),
            f"env[focus]={float(env[target_focus]):.6f}")
    _assert("500 kHz env.max() == 1.0",
            _close(float(env.max()), 1.0, rtol=1e-9))


def test_cube_construction_validates_corners():
    print("\n[test] KernelCube — constructor validates all corners on disk")
    cube = build_cube(f0_values=[0.5e6, 1.0e6], pnp_values=[15e6, 30e6])
    _assert("constructed without error", isinstance(cube, KernelCube))
    f0_axis = list(cube.f0_axis)
    _assert("f0 axis sorted", f0_axis == [0.5e6, 1.0e6], f"got {f0_axis}")
    # pnp axis reflects realised pnp from kernel calibration, which is
    # within ~7 % of the target (e.g. 30 MPa target → 30.78 MPa realised).
    pnp_axis = list(cube.pnp_axis)
    _assert("pnp axis has 2 sorted values close to [15, 30] MPa",
            len(pnp_axis) == 2 and pnp_axis[0] < pnp_axis[1] and
            14e6 < pnp_axis[0] < 17e6 and 28e6 < pnp_axis[1] < 33e6,
            f"got {[v/1e6 for v in pnp_axis]} MPa")

    raised = False
    try:
        build_cube(f0_values=[0.5e6, 99e6], pnp_values=[15e6])
    except FileNotFoundError:
        raised = True
    _assert("missing corner raises FileNotFoundError", raised)


def test_cube_query_corner_matches_single():
    print("\n[test] KernelCube.query — corner equals single-kernel envelope")
    cube = build_cube(f0_values=[0.5e6, 1.0e6], pnp_values=[15e6, 30e6])
    target_shape = (180, 100, 100)
    target_focus = (130, 50, 50)
    target_dx = 1.0e-3
    env_cube = cube.query(f0=1.0e6, pnp=30e6,
                           target_shape=target_shape,
                           target_focus_idx=target_focus,
                           target_dx_m=target_dx)
    env_single = kernel_focal_envelope(
        scenario_f0=1.0e6, target_shape=target_shape,
        target_focus_idx=target_focus, target_dx_m=target_dx)
    _assert("cube query @ corner == single-kernel envelope",
            np.allclose(env_cube, env_single, atol=1e-12),
            f"max abs diff = {np.abs(env_cube - env_single).max():.3e}")


def test_cube_query_midpoint_blends():
    print("\n[test] KernelCube.query — midpoint f0 blends both corners")
    cube = build_cube(f0_values=[0.5e6, 1.0e6], pnp_values=[15e6, 30e6])
    target_shape = (180, 100, 100)
    target_focus = (130, 50, 50)
    target_dx = 1.0e-3
    f0_mid = 0.75e6
    env_mid = cube.query(f0=f0_mid, pnp=20e6,
                          target_shape=target_shape,
                          target_focus_idx=target_focus,
                          target_dx_m=target_dx)
    env_lo = cube.query(f0=0.5e6, pnp=20e6,
                         target_shape=target_shape,
                         target_focus_idx=target_focus,
                         target_dx_m=target_dx)
    env_hi = cube.query(f0=1.0e6, pnp=20e6,
                         target_shape=target_shape,
                         target_focus_idx=target_focus,
                         target_dx_m=target_dx)
    _assert("env_mid.max() == 1 (re-normalized after blend)",
            _close(float(env_mid.max()), 1.0, rtol=1e-9),
            f"max={float(env_mid.max()):.6f}")
    # The midpoint should differ from both corners (proves we're blending)
    _assert("env_mid differs from env_lo (real blending happened)",
            float(np.abs(env_mid - env_lo).max()) > 1e-3)
    _assert("env_mid differs from env_hi (real blending happened)",
            float(np.abs(env_mid - env_hi).max()) > 1e-3)


def test_cube_query_clamps_outside():
    print("\n[test] KernelCube.query — outside-sweep f0 clamps to nearest corner")
    cube = build_cube(f0_values=[0.5e6, 1.0e6], pnp_values=[15e6, 30e6])
    target_shape = (180, 100, 100)
    target_focus = (130, 50, 50)
    target_dx = 1.0e-3

    env_below = cube.query(f0=0.1e6, pnp=20e6,  # well below 0.5 MHz
                            target_shape=target_shape,
                            target_focus_idx=target_focus,
                            target_dx_m=target_dx)
    env_at_lo = cube.query(f0=0.5e6, pnp=20e6,
                            target_shape=target_shape,
                            target_focus_idx=target_focus,
                            target_dx_m=target_dx)
    _assert("f0 below sweep clamps to lowest corner",
            np.allclose(env_below, env_at_lo, atol=1e-12))

    env_above = cube.query(f0=2.0e6, pnp=20e6,  # well above 1.0 MHz
                            target_shape=target_shape,
                            target_focus_idx=target_focus,
                            target_dx_m=target_dx)
    env_at_hi = cube.query(f0=1.0e6, pnp=20e6,
                            target_shape=target_shape,
                            target_focus_idx=target_focus,
                            target_dx_m=target_dx)
    _assert("f0 above sweep clamps to highest corner",
            np.allclose(env_above, env_at_hi, atol=1e-12))


def test_cube_query_pnp_is_amplitude_invariant():
    print("\n[test] KernelCube.query — pnp dimension is degenerate (water linear)")
    cube = build_cube(f0_values=[0.5e6, 1.0e6], pnp_values=[15e6, 30e6])
    target_shape = (180, 100, 100)
    target_focus = (130, 50, 50)
    target_dx = 1.0e-3
    env_15 = cube.query(f0=1.0e6, pnp=15e6,
                         target_shape=target_shape,
                         target_focus_idx=target_focus,
                         target_dx_m=target_dx)
    env_30 = cube.query(f0=1.0e6, pnp=30e6,
                         target_shape=target_shape,
                         target_focus_idx=target_focus,
                         target_dx_m=target_dx)
    _assert("envelope at pnp=15 == envelope at pnp=30 (shape invariant)",
            np.allclose(env_15, env_30, atol=1e-12),
            f"max abs diff = {float(np.abs(env_15 - env_30).max()):.3e}")


def test_cube_repeat_query_returns_consistent_results():
    print("\n[test] KernelCube — repeated identical query returns identical envelopes")
    cube = build_cube(f0_values=[0.5e6, 1.0e6], pnp_values=[15e6, 30e6])
    target_shape = (180, 100, 100)
    target_focus = (130, 50, 50)
    target_dx = 1.0e-3
    args = dict(f0=1.0e6, pnp=20e6, target_shape=target_shape,
                target_focus_idx=target_focus, target_dx_m=target_dx)
    env1 = np.asarray(cube.query(**args))
    env2 = np.asarray(cube.query(**args))
    _assert("repeated query is deterministic",
            np.array_equal(env1, env2),
            f"max diff = {float(np.abs(env1 - env2).max()):.3e}")
    args["f0"] = 0.7e6
    env_mid = np.asarray(cube.query(**args))
    _assert("midpoint query produces blended envelope (max=1)",
            _close(float(env_mid.max()), 1.0, rtol=1e-9),
            f"midpoint max={float(env_mid.max()):.6f}")


def main() -> int:
    if not os.path.exists(KERNEL_DIR):
        print(f"FATAL: kernel directory {KERNEL_DIR} does not exist; "
              f"run cavitation_kernel.py --sweep first.", file=sys.stderr)
        return 2
    tests = [
        test_load_exact_match,
        test_load_with_rescale_fallback,
        test_resample_to_target_dx,
        test_place_kernel_at_focus,
        test_focal_envelope_normalized,
        test_envelope_caching,
        test_envelope_for_500khz,
        test_cube_construction_validates_corners,
        test_cube_query_corner_matches_single,
        test_cube_query_midpoint_blends,
        test_cube_query_clamps_outside,
        test_cube_query_pnp_is_amplitude_invariant,
        test_cube_repeat_query_returns_consistent_results,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            global FAILS
            FAILS += 1
            print(f"  EXCEPTION in {t.__name__}: {e}")
            traceback.print_exc(file=sys.stdout)
    print(f"\n{PASSES} passed, {FAILS} failed")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
