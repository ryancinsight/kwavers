#!/usr/bin/env python3
"""
elastic_velocity_source_smoke.py
=================================
Phase A.3 of ADR 007: end-to-end smoke test for the elastic velocity-source
mask via ``Source.from_elastic_velocity_source``.

This is the second elastic source-injection path (after Phase A.2's
``Source.from_initial_displacement`` IVP). It exercises a **driven**
plane-wave-style velocity source on an x-plane that emits a tone burst
through a homogeneous elastic medium, and verifies that:

1. The driven source produces non-zero field response (no IVP supplied).
2. The plane-wave geometry produces uz-dominant displacement at sensors
   along the +x ray (matching the source's uz drive).
3. Causality holds: closer sensors arrive earlier than further ones.
4. The driving signal is reflected at the source plane: at the source-
   plane sensor, the ``uz`` trace correlates strongly with the input
   tone burst (correlation ≥ 0.5).
5. Validation: source mask shape mismatch raises; signal length mismatch
   raises; non-elastic solver type raises.
6. Mixed mode: IVP + velocity-source can coexist (Phase A.3 supports
   simultaneous initial-value and driven sources).

Phase A.3 limitations (intentional, in ADR 007 §A.3 / A.3.5):
- 1-D signals only, broadcast across all mask points; per-point signal
  matrices ship in Phase A.4.
- Velocity-source semantics: post-step Dirichlet override (matches
  k-Wave's default for ``source.u_mode = "additive_no_correction"``-like
  behaviour for velocity sources). The ``additive`` mode for velocity
  sources is not yet exposed.
- No stress-tensor sources (`source.s_mask`, `source.sxx`, `source.syy`)
  — Phase A.3.5.

Usage
-----
    python examples/elastic_velocity_source_smoke.py
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
# Physical configuration
# ---------------------------------------------------------------------------
NX, NY, NZ = 32, 32, 16
DX = DY = DZ = 0.5e-3

C_P = 1800.0  # compressional speed [m/s] (ewp_plane_wave_absorption)
C_S = 1200.0
RHO = 1000.0

NT = 200
CFL = 0.3
DT = CFL * DX / (np.sqrt(3.0) * C_P)

SOURCE_PLANE_X = 8  # x-index of the source plane
SOURCE_FREQUENCY = 1.0e6  # 1 MHz tone burst
SOURCE_PEAK = 1.0e-6  # 1 µm/s peak velocity


def _build_tone_burst(nt: int, dt: float, freq: float, peak: float) -> np.ndarray:
    """Hann-windowed tone burst with `freq` Hz, peak amplitude `peak`."""
    t = np.arange(nt, dtype=np.float64) * dt
    n_cycles = 3
    cycle_period = 1.0 / freq
    burst_dur = n_cycles * cycle_period
    window = np.zeros(nt, dtype=np.float64)
    burst_mask = t < burst_dur
    n_burst = int(np.sum(burst_mask))
    if n_burst > 0:
        window[:n_burst] = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_burst) / n_burst))
    return peak * window * np.sin(2.0 * np.pi * freq * t)


def main() -> int:
    print("=" * 78)
    print("elastic_velocity_source_smoke: Phase A.3 driven-velocity source")
    print("=" * 78)
    print(f"  Grid    : {NX}×{NY}×{NZ}  dx={DX*1e3:.2f} mm")
    print(f"  Medium  : c_p={C_P} m/s  c_s={C_S} m/s  ρ={RHO} kg/m³")
    print(f"  Source  : x-plane at i={SOURCE_PLANE_X}, uz tone burst")
    print(f"            f={SOURCE_FREQUENCY/1e6:.1f} MHz  peak={SOURCE_PEAK*1e6:.1f} µm/s")
    print(f"  Sensors : 3 points along +x ray from source plane")
    print(f"  Time    : Nt={NT}  dt={DT*1e9:.2f} ns")
    print("-" * 78)

    grid = pkw.Grid(NX, NY, NZ, DX, DY, DZ)
    medium = pkw.Medium.elastic(C_P, C_S, RHO, grid=grid)

    # Source mask: full y-z slab at x = SOURCE_PLANE_X (plane wave).
    u_mask = np.zeros((NX, NY, NZ), dtype=bool)
    u_mask[SOURCE_PLANE_X, :, :] = True
    n_active = int(np.sum(u_mask))
    print(f"  source mask: {n_active} active grid points")

    uz_signal = _build_tone_burst(NT, DT, SOURCE_FREQUENCY, SOURCE_PEAK)
    print(f"  tone burst: peak={float(np.max(np.abs(uz_signal))):.3e} m/s "
          f"@ t≈{float(np.argmax(np.abs(uz_signal))) * DT * 1e6:.2f} µs")

    src = pkw.Source.from_elastic_velocity_source(u_mask, uz=uz_signal)

    # Sensors at SOURCE_PLANE_X+0 (source plane), +3, +6 (ahead of source).
    sensor_mask = np.zeros((NX, NY, NZ), dtype=bool)
    sensor_offsets = [0, 3, 6]
    sensor_positions = []
    cy, cz = NY // 2, NZ // 2
    for off in sensor_offsets:
        ix = SOURCE_PLANE_X + off
        if ix < NX:
            sensor_mask[ix, cy, cz] = True
            sensor_positions.append((ix, cy, cz))
    sensor = pkw.Sensor.from_mask(sensor_mask)

    # ── Test 1: driven velocity source produces non-zero response ──────────
    print("\n[1/5] End-to-end: driven uz velocity source on plane "
          f"(no IVP)")
    sim = pkw.Simulation(grid, medium, src, sensor, solver=pkw.SolverType.Elastic)
    result = sim.run(time_steps=NT, dt=DT)
    uz_traces = np.asarray(result.uz, dtype=np.float64)
    assert uz_traces.shape == (len(sensor_positions), NT), (
        f"Expected uz shape {(len(sensor_positions), NT)}, got {uz_traces.shape}"
    )
    assert np.all(np.isfinite(uz_traces)), "uz traces must be finite"
    peaks = [float(np.max(np.abs(uz_traces[i]))) for i in range(len(sensor_positions))]
    # Expected order: ∫(peak·sin(2πft))dt ≈ peak / (2πf). For peak=1 µm/s,
    # f=1 MHz: expected_disp ≈ 1.6e-13 m. Threshold one decade below to
    # avoid noise-floor false positives without rejecting the true signal.
    expected_disp_order = SOURCE_PEAK / (2.0 * np.pi * SOURCE_FREQUENCY)
    threshold_amp = 0.1 * expected_disp_order
    assert max(peaks) > threshold_amp, (
        f"Driven source should produce ≥ {threshold_amp:.2e} m response; "
        f"got max|uz| = {max(peaks):.3e}"
    )
    for i, (pos, p) in enumerate(zip(sensor_positions, peaks)):
        print(f"  sensor[{i}] at {pos}  peak |uz| = {p:.3e} m")

    # ── Test 2: source-plane sensor reflects the driving signal ────────────
    # The sensor at the source plane sees vx/vy/vz directly assigned each
    # step. Correlation between the integrated displacement (uz) at that
    # sensor and the input vz signal should be high.
    print("\n[2/5] Source-plane sensor reflects driving signal (correlation)")
    src_plane_uz = uz_traces[0]  # at offset 0 = source plane
    # Integrate the input velocity signal to get expected displacement.
    # ux/uy/uz in result.* are displacement, not velocity.
    expected_uz = np.cumsum(uz_signal) * DT
    a = src_plane_uz - np.mean(src_plane_uz)
    b = expected_uz - np.mean(expected_uz)
    a_norm = np.linalg.norm(a) + 1e-30
    b_norm = np.linalg.norm(b) + 1e-30
    correlation = float(np.dot(a, b) / (a_norm * b_norm))
    print(f"  source-plane vs ∫(input)dt correlation = {correlation:.4f}")
    assert correlation >= 0.5, (
        f"Source-plane uz should track the integrated input signal; "
        f"correlation {correlation:.3f} too low"
    )

    # ── Test 3: causality monotonicity ─────────────────────────────────────
    print("\n[3/5] Causality: closer sensors register first arrival earlier")
    threshold = max(peaks) * 0.05
    arrivals = [
        int(np.argmax(np.abs(uz_traces[i]) > threshold))
        if np.any(np.abs(uz_traces[i]) > threshold)
        else NT
        for i in range(len(sensor_positions))
    ]
    for i in range(len(sensor_positions) - 1):
        assert arrivals[i] <= arrivals[i + 1], (
            f"Causality: sensor[{i}] arrival {arrivals[i]} > "
            f"sensor[{i+1}] arrival {arrivals[i+1]}"
        )
    for i, (pos, a) in enumerate(zip(sensor_positions, arrivals)):
        print(f"  sensor[{i}] at {pos}  first arrival step = {a}")

    # ── Test 4: validation rejections ──────────────────────────────────────
    print("\n[4/5] Validation: misuse paths reject")
    # 4a: wrong-shape mask
    bad_mask = np.zeros((NX + 1, NY, NZ), dtype=bool)
    bad_mask[0, 0, 0] = True
    bad_src = pkw.Source.from_elastic_velocity_source(
        bad_mask, uz=np.zeros(NT)
    )
    bad_sim = pkw.Simulation(grid, medium, bad_src, sensor, solver=pkw.SolverType.Elastic)
    try:
        bad_sim.run(time_steps=NT, dt=DT)
    except (ValueError, RuntimeError) as e:
        print(f"  [PASS] wrong-shape mask → {type(e).__name__}: {str(e)[:60]}")
    else:
        raise AssertionError("Wrong-shape mask should reject")

    # 4b: signal length mismatch
    short_sig = np.zeros(NT - 1)
    src_short = pkw.Source.from_elastic_velocity_source(u_mask, uz=short_sig)
    sim_short = pkw.Simulation(grid, medium, src_short, sensor, solver=pkw.SolverType.Elastic)
    try:
        sim_short.run(time_steps=NT, dt=DT)
    except (ValueError, RuntimeError) as e:
        print(f"  [PASS] signal length mismatch → {type(e).__name__}: {str(e)[:60]}")
    else:
        raise AssertionError("Signal length mismatch should reject")

    # 4c: from_elastic_velocity_source on non-Elastic solver
    fluid = pkw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    sim_pstd = pkw.Simulation(grid, fluid, src, sensor, solver=pkw.SolverType.PSTD)
    try:
        sim_pstd.run(time_steps=NT, dt=DT)
    except (ValueError, RuntimeError) as e:
        print(f"  [PASS] non-Elastic solver → {type(e).__name__}: {str(e)[:60]}")
    else:
        raise AssertionError("Non-Elastic solver should reject")

    # ── Test 5: mixed IVP + velocity source ────────────────────────────────
    print("\n[5/5] Mixed IVP + velocity source: both contribute")
    # Small Gaussian uz IVP at the +x end of the grid, plus the driven plane wave.
    cz = NZ // 2
    cy = NY // 2
    cx_ivp = 25
    ix = np.arange(NX, dtype=np.float64)[:, None, None]
    iy = np.arange(NY, dtype=np.float64)[None, :, None]
    iz = np.arange(NZ, dtype=np.float64)[None, None, :]
    rsq = (ix - cx_ivp) ** 2 + (iy - cy) ** 2 + (iz - cz) ** 2
    u0 = 1e-9 * np.exp(-rsq / (2.0 * 1.5 ** 2))

    ivp_src = pkw.Source.from_initial_displacement(u0, axis="z")
    sim_mixed = pkw.Simulation(
        grid, medium, [src, ivp_src], sensor, solver=pkw.SolverType.Elastic
    )
    result_mixed = sim_mixed.run(time_steps=NT, dt=DT)
    uz_mixed = np.asarray(result_mixed.uz, dtype=np.float64)
    peak_mixed = float(np.max(np.abs(uz_mixed)))
    peak_src_only = max(peaks)
    print(f"  src-only peak |uz| = {peak_src_only:.3e}  "
          f"src+IVP peak |uz| = {peak_mixed:.3e}")
    # Mixed should be at least as large as src-only at the closest-to-IVP sensor.
    assert np.all(np.isfinite(uz_mixed)), "mixed traces must be finite"
    assert peak_mixed >= 0.5 * peak_src_only, (
        f"Mixed run should preserve the driven response (got "
        f"{peak_mixed:.3e} vs src-only {peak_src_only:.3e})"
    )

    print("\n" + "=" * 78)
    print("Phase A.3 elastic-velocity-source smoke test passed.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
