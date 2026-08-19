# ADR 020 — Bent-Ray Traveltime Tomography (Shortest-Path / Fermat)

**Status:** Accepted
**Change class:** [major] (new reconstruction kernel; follow-on to ADR 013)
**Date:** 2026-06-09

## Context

ADR 013 delivered straight-ray Radon/FBP and noted that *"bent-ray correction (SIRT/ART with
ray tracing) is an explicit follow-on."* The iterative algebraic reconstructors already exist
(`kwavers_solver::inverse::reconstruction::unified_sirt::{SirtAlgorithm::{Sirt,Art,Osem},
SirtReconstructor}`) and a streaming pipeline (`reconstruction::real_time_sirt`). What is
**missing** is the *bent-ray forward operator*: every projection path in the codebase is
straight-line — `acoustic_projection::project_acoustic` uses the Euclidean source→voxel
distance, and `sound_speed_shift::propagation::ShiftPropagation` has only a `StraightRay`
variant. In heterogeneous media (skull, tissue contrast) rays **refract**: the true path obeys
Fermat's principle (minimum traveltime), bending toward fast (low-slowness) regions. Straight-ray
tomography therefore mismodels the geometry and the bias grows with `|∇c|/c` (ADR 013
"Consequences"). This is the documented-but-unimplemented gap (backlog #4).

## Decision

Implement a **shortest-path (Dijkstra) bent-ray tracer** over a 2-D slowness field in a new
`kwavers_diagnostics::reconstruction::bent_ray` module:

- `bent_ray_path(slowness, dx, source, receiver) -> BentRay` — Dijkstra over an 8-connected grid
  graph whose edge cost between adjacent voxels `u,v` is `½(s_u + s_v)·L_{uv}` (trapezoidal
  slowness × Euclidean edge length, `L ∈ {dx, dx√2}`). The minimum-cost path **is** the
  Fermat (least-traveltime) ray; `BentRay` carries its `traveltime`, the voxel `path`, and the
  **per-voxel accumulated path length** `row` (sparse `(voxel_index, length)`).
- The `row` is exactly the system-matrix row of the discretized line integral
  `t = ∫ s\,d\ell = Σ_v s_v · row_v`, so it plugs directly into the existing
  `SirtReconstructor` (SIRT/ART/OSEM) as the bent-ray forward operator — closing the
  straight-ray-only gap without re-implementing the iterative solver.
- `bent_ray_traveltime(...) -> f64` convenience wrapper.

The path-length bookkeeping splits each traversed edge's length equally between its two
endpoint voxels, so `Σ_v s_v·row_v` reproduces the Dijkstra cost exactly (linear in slowness).

## Alternatives considered

- **Eikonal / fast-marching** — solves the traveltime *field* in `O(N log N)`, but recovering
  the *ray path* (needed for the system-matrix row) requires back-tracing the traveltime
  gradient, adding complexity and interpolation error. Shortest-path yields the path directly.
- **Analytic circular-arc (constant velocity gradient, Slotnick 1959)** — exact but only for a
  linear `v(z)`; not general heterogeneous media. Used here only as a *verification oracle*.
- **Linearized straight-ray perturbation (bent-ray approximated as straight)** — that is the
  status quo; it does not capture refraction. Rejected.

8-connectivity is chosen for a clean, fast, exactly-grid-representable tracer; its angular
metric error (≤ ~8 % for off-axis directions, 0 for axis/45° rays) is bounded and documented.
A denser stencil (16/32-connected) is a future refinement if angular accuracy demands it.

## Verification (value-semantic)

- **Homogeneous exactness** — for axis-aligned and 45° source/receiver pairs the bent path is
  the straight grid path and `traveltime == s·distance` to machine precision.
- **Graph-metric bound** — for a general pair the bent-ray traveltime is `≥ s·euclidean`
  (the graph geodesic is never shorter than the straight line) and `≤ (1+ε)·s·euclidean` with
  the documented 8-connectivity bound.
- **Fermat / refraction** — inserting a fast (low-slowness) channel off the straight line
  *lowers* the traveltime below the straight-line value and the recovered path *enters* the
  channel — the defining bent-ray behaviour.
- **Row consistency** — `Σ_v slowness_v · row_v == traveltime` exactly (the matrix row
  reproduces the cost).
- **Tomographic recovery** — the iterative trace↔solve reconstruction over bent-ray rows recovers a
  known slowness anomaly with high correlation, and a wrong uniform guess converges to the true
  uniform slowness. **Implemented** in `reconstruction::bent_ray_tomography`
  (`reconstruct_bent_ray_tomography`): the nonlinear fixed point re-traces rays through the evolving
  model and refines it by sparse-ART sweeps over the path-length rows; `rms_misfit` gives the data
  residual. Tests: a slow disk is recovered (correlation > 0.5, anomaly reconstructed slower than
  background, misfit falls across outer iterations); a 7%-wrong uniform guess recovers the true
  uniform slowness (mean interior error ~1%).

## Consequences

- Closes the ADR-013 follow-on: heterogeneous-media (refracting) acoustic CT now has a real
  forward operator **and an end-to-end inversion** (`bent_ray_tomography`). The inner solver is a
  sparse ART (Kaczmarz) over the bent-ray rows — the appropriate algorithm for the sparse
  path-length structure (the dense `SirtReconstructor` would require densifying the system).
- Straight-ray remains the fast default; bent-ray is selected when `|∇c|/c` is large.
- Reflection-CT geometry (distinct from this transmission tracer) remains a separate follow-on.
