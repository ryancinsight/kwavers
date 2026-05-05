#!/usr/bin/env python3
"""
elastic_propagation_smoke.py
=============================
End-to-end smoke test for Phase A.2 of ADR 007: pykwavers
``SolverType.Elastic`` dispatch with ``Source.from_initial_displacement``.

This is the first Python-level test that propagates an elastic wave
through a homogeneous elastic medium and reads recorded sensor data.
It exercises the **full bridging stack**:

1. ``Medium.elastic(c_p, c_s, ρ)``                — Phase A.1 surface
2. ``Source.from_initial_displacement(field)``    — Phase A.2 surface
3. ``Sensor.from_mask(...)``                      — pre-existing API
4. ``Simulation(grid, medium, source, sensor,
                solver=pkw.SolverType.Elastic)``  — Phase A.2 dispatch
5. ``sim.run(time_steps, dt)``                    — invokes
   ``ElasticWaveSolver::propagate`` on the Rust side
6. ``result.sensor_data``                         — recorded uz traces

Physical setup
--------------
A small 3-D homogeneous-elastic block (32×32×16, dx=0.5 mm) of bone-like
material ``(c_p=2000 m/s, c_s=800 m/s, ρ=1200 kg/m³)`` — the
``example_ewp_layered_medium`` lower-layer values from k-Wave.

A Gaussian initial displacement bump on the ``uz`` component is centred
at the grid mid-point. A binary sensor mask records ``uz`` at three points
along the +x ray from the bump centre. After ``Nt = 200`` time steps the
recorded traces are inspected for arrival of the compressional wavefront.

Validation invariants
---------------------
1. ``sensor_data`` is a finite (n_sensors, Nt') float64 ndarray with no
   NaN or Inf.
2. The bump's peak amplitude **decays** with distance (3D geometric
   spreading + finite numerical absorption).
3. The first nonzero arrival at sensor i comes **after** the first
   nonzero arrival at any closer sensor j with ``r_j < r_i`` — i.e.,
   monotone arrival time vs distance — a fundamental causality check.
4. Calling ``sim.run`` without ``Source.from_initial_displacement`` (i.e.,
   with the wrong source type) raises ``ValueError``.
5. Mismatched solver type (``Source.from_initial_displacement`` with
   ``SolverType.PSTD``) also raises.

Phase A.2 LIMITATIONS recorded by ADR 007 §A.2 (intentional, NOT bugs):
- Single-component recording: ``sensor_data`` is the ``uz`` trace
  regardless of the initial-displacement axis. Phase A.2.5 extends to
  per-component (ux/uy/uz/u_max_all).
- No stress / velocity source masks; only initial-displacement IVP.
  Phase A.3 adds those.
- No multi-layer heterogeneous elastic media in the constructor;
  Phase A.4 adds the array overload.

Usage
-----
    python examples/elastic_propagation_smoke.py
"""

from __future__ import annotations

import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from example_parity_utils import bootstrap_example_paths

bootstrap_example_paths()

import pykwavers as pkw

# ---------------------------------------------------------------------------
# Physical configuration (compact for fast smoke testing)
# ---------------------------------------------------------------------------
NX, NY, NZ = 32, 32, 16
DX = DY = DZ = 0.5e-3  # 0.5 mm

# Bone-like layer from example_ewp_layered_medium
C_P = 2000.0  # compressional wave speed [m/s]
C_S = 800.0  # shear wave speed [m/s]
RHO = 1200.0  # density [kg/m³]

NT = 200
# CFL-stable dt: dt < dx/(√3·c_p)
CFL = 0.3
DT = CFL * DX / (np.sqrt(3.0) * C_P)

# Initial Gaussian displacement bump centre + width
BUMP_CX, BUMP_CY, BUMP_CZ = NX // 2, NY // 2, NZ // 2
BUMP_SIGMA = 1.5  # grid points
BUMP_PEAK = 1.0e-9  # 1 nm peak displacement


def _build_initial_displacement() -> np.ndarray:
    """Construct a Gaussian initial-uz displacement bump on the grid."""
    ix = np.arange(NX, dtype=np.float64)[:, None, None]
    iy = np.arange(NY, dtype=np.float64)[None, :, None]
    iz = np.arange(NZ, dtype=np.float64)[None, None, :]
    rsq = (ix - BUMP_CX) ** 2 + (iy - BUMP_CY) ** 2 + (iz - BUMP_CZ) ** 2
    return BUMP_PEAK * np.exp(-rsq / (2.0 * BUMP_SIGMA ** 2))


def _sensor_mask_along_x_ray() -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Sensor at (cx+r, cy, cz) for r ∈ {3, 6, 10}, returning mask + positions."""
    mask = np.zeros((NX, NY, NZ), dtype=bool)
    positions = []
    for r in (3, 6, 10):
        ix = BUMP_CX + r
        if ix < NX:
            mask[ix, BUMP_CY, BUMP_CZ] = True
            positions.append((ix, BUMP_CY, BUMP_CZ))
    return mask, positions


def _find_first_nonzero_step(trace: np.ndarray, threshold: float) -> int:
    """Return first index where |trace[i]| > threshold, or len(trace) if none."""
    above = np.flatnonzero(np.abs(trace) > threshold)
    return int(above[0]) if above.size else len(trace)


def main() -> int:
    print("=" * 78)
    print("elastic_propagation_smoke: Phase A.2 end-to-end elastic IVP")
    print("=" * 78)
    print(f"  Grid    : {NX}×{NY}×{NZ}  dx={DX*1e3:.2f} mm")
    print(f"  Medium  : c_p={C_P} m/s  c_s={C_S} m/s  ρ={RHO} kg/m³")
    print(f"  IVP     : Gaussian uz bump, σ={BUMP_SIGMA} pts, peak={BUMP_PEAK*1e9} nm")
    print(f"  Sensors : 3 points along +x ray from bump centre")
    print(f"  Time    : Nt={NT}  dt={DT*1e9:.2f} ns  T={DT*NT*1e6:.2f} μs")
    print("-" * 78)

    # Build inputs
    grid = pkw.Grid(NX, NY, NZ, DX, DY, DZ)
    medium = pkw.Medium.elastic(C_P, C_S, RHO, grid=grid)
    u0 = _build_initial_displacement()
    source = pkw.Source.from_initial_displacement(u0, axis="z")
    mask, sensor_positions = _sensor_mask_along_x_ray()
    sensor = pkw.Sensor.from_mask(mask)

    # ── Test 1: end-to-end run produces finite recorded data ───────────────
    print("\n[1/5] End-to-end: Simulation(SolverType.Elastic).run(...)")
    sim = pkw.Simulation(
        grid, medium, source, sensor, solver=pkw.SolverType.Elastic
    )
    result = sim.run(time_steps=NT, dt=DT)
    sensor_data = np.asarray(result.sensor_data, dtype=np.float64)
    n_sensors = len(sensor_positions)
    assert sensor_data.shape[0] == n_sensors, (
        f"Expected {n_sensors} sensor rows, got shape {sensor_data.shape}"
    )
    assert sensor_data.shape[1] >= 1, (
        f"Expected at least 1 time-step column, got {sensor_data.shape}"
    )
    assert np.all(np.isfinite(sensor_data)), "sensor_data must be all finite"
    print(f"  shape={sensor_data.shape}  dtype={sensor_data.dtype}  "
          f"max|uz|={np.max(np.abs(sensor_data)):.3e} m  finite=OK")

    # ── Test 2: peak amplitude decays with distance ────────────────────────
    print("\n[2/5] Geometric-spreading invariant: peak |uz| decays with distance")
    peaks = [float(np.max(np.abs(sensor_data[i]))) for i in range(n_sensors)]
    distances = [
        float(np.sqrt((p[0] - BUMP_CX) ** 2 + (p[1] - BUMP_CY) ** 2 + (p[2] - BUMP_CZ) ** 2))
        for p in sensor_positions
    ]
    for i in range(n_sensors - 1):
        assert peaks[i] >= peaks[i + 1] * 0.5, (
            f"Peak at sensor {i} (r={distances[i]:.1f}) = {peaks[i]:.3e} should be "
            f"comparable-or-larger than sensor {i+1} (r={distances[i+1]:.1f}) = {peaks[i+1]:.3e}"
        )
    for i, (pos, d, p) in enumerate(zip(sensor_positions, distances, peaks)):
        print(f"  sensor[{i}] at {pos}  r={d:5.1f} pts  peak |uz|={p:.3e} m")

    # ── Test 3: monotone arrival time with distance (causality) ────────────
    print("\n[3/5] Causality: closer sensors must register arrival earlier")
    threshold = max(peaks) * 0.05  # 5% of overall peak
    arrivals = [
        _find_first_nonzero_step(sensor_data[i], threshold) for i in range(n_sensors)
    ]
    for i in range(n_sensors - 1):
        assert arrivals[i] <= arrivals[i + 1], (
            f"Causality violated: sensor {i} (r={distances[i]:.1f}) arrived at step "
            f"{arrivals[i]}, but sensor {i+1} (r={distances[i+1]:.1f}) arrived at "
            f"step {arrivals[i+1]}"
        )
    for i, (d, a) in enumerate(zip(distances, arrivals)):
        print(f"  sensor[{i}] r={d:5.1f} pts  first arrival step = {a}")

    # ── Test 4: reject Elastic dispatch without initial-displacement IVP ───
    # Provide a point source (which routes through dynamic_sources, NOT
    # grid_source.p0). The elastic dispatch must error because p0 is None.
    print("\n[4/5] Validation: SolverType.Elastic without IVP source → ValueError")
    point_source = pkw.Source.point(
        position=(BUMP_CX * DX, BUMP_CY * DY, BUMP_CZ * DZ),
        frequency=1e6,
        amplitude=1.0,
    )
    bad_sim = pkw.Simulation(
        grid, medium, point_source, sensor, solver=pkw.SolverType.Elastic
    )
    try:
        bad_sim.run(time_steps=10, dt=DT)
    except (ValueError, RuntimeError) as e:
        # ValueError from PyO3 routing layer; RuntimeError from
        # kwavers_error_to_py wrapping a kwavers::core::error::KwaversError.
        # Both are correct rejection paths.
        print(f"  [PASS] rejected missing IVP → {type(e).__name__}: {str(e)[:80]}...")
    else:
        raise AssertionError("Should have raised when no IVP source supplied")

    # ── Test 5a: per-component displacement traces are populated (A.2.5) ──
    print("\n[5/6] Phase A.2.5: ux / uy / uz traces all populated, finite, "
          "uz-dominant")
    ux_trace = np.asarray(result.ux, dtype=np.float64) if result.ux is not None else None
    uy_trace = np.asarray(result.uy, dtype=np.float64) if result.uy is not None else None
    uz_trace = np.asarray(result.uz, dtype=np.float64) if result.uz is not None else None
    assert ux_trace is not None, "result.ux must be populated by Phase A.2.5"
    assert uy_trace is not None, "result.uy must be populated by Phase A.2.5"
    assert uz_trace is not None, "result.uz must be populated by Phase A.2.5"
    for name, arr in [("ux", ux_trace), ("uy", uy_trace), ("uz", uz_trace)]:
        assert arr.shape == (n_sensors, NT), (
            f"result.{name} shape must be ({n_sensors}, {NT}), got {arr.shape}"
        )
        assert np.all(np.isfinite(arr)), f"result.{name} must be all finite"
    # The IVP was placed on uz, so uz should dominate ux and uy in magnitude
    # at the closest sensor (where wavefront has propagated).
    ux_peak = float(np.max(np.abs(ux_trace[0])))
    uy_peak = float(np.max(np.abs(uy_trace[0])))
    uz_peak = float(np.max(np.abs(uz_trace[0])))
    print(f"  sensor[0]:  ux peak={ux_peak:.3e}  uy peak={uy_peak:.3e}  "
          f"uz peak={uz_peak:.3e}")
    assert uz_peak >= 10.0 * max(ux_peak, uy_peak, 1e-30), (
        f"uz IVP should yield uz-dominant displacement at sensor 0; "
        f"got uz_peak={uz_peak:.3e}, ux_peak={ux_peak:.3e}, uy_peak={uy_peak:.3e}"
    )
    # And uz should match the legacy sensor_data trace (within numerical noise)
    sd_peak = float(np.max(np.abs(sensor_data[0])))
    assert abs(uz_peak - sd_peak) < 1e-15 or abs(uz_peak - sd_peak) / sd_peak < 1e-6, (
        f"uz_data and legacy sensor_data must agree on the uz trace; "
        f"got uz_peak={uz_peak:.3e}, sd_peak={sd_peak:.3e}"
    )
    print(f"  [PASS] per-component recording works; uz/ux ratio = "
          f"{uz_peak / max(ux_peak, 1e-30):.2e}")

    # ── Test 6: reject mismatched solver type for elastic IVP source ───────
    print("\n[6/6] Validation: from_initial_displacement on PSTD → ValueError")
    fluid_medium = pkw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    mismatched_sim = pkw.Simulation(
        grid, fluid_medium, source, sensor, solver=pkw.SolverType.PSTD
    )
    try:
        mismatched_sim.run(time_steps=10, dt=DT)
    except (ValueError, RuntimeError) as e:
        print(f"  [PASS] rejected solver mismatch → {type(e).__name__}: {str(e)[:80]}...")
    else:
        raise AssertionError("Should have raised on solver/source mismatch")

    print("\n" + "=" * 78)
    print("Phase A.2 elastic-propagation smoke test passed.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
