# ADR 007: pykwavers Elastic-Wave Bindings — Phased Roadmap

**Status**: 🟡 In progress — Phase A.1 landed
**Date**: 2026-05-05
**Context**: Replicating k-Wave EWP example family in pykwavers
**Deciders**: Ryan Clanton

---

## Context

The k-Wave MATLAB toolbox ships four EWP (Elastic Wave Propagation) examples
that exercise `pstdElastic2D` / `pstdElastic3D`:

- `example_ewp_layered_medium.m` — disc IVP in two-layer compressional+shear medium
- `example_ewp_plane_wave_absorption.m` — Kelvin-Voigt α₀·f² verification on
  particle-velocity traces with two-sensor attenuation extraction
- `example_ewp_shear_wave_snells_law.m` — fluid/elastic Snell's-law dual run
  with focused-arc stress source
- `example_ewp_3D_simulation.m` — 3-D layered medium, focused velocity-source
  piston, cuboid-corner sensor

The user requested side-by-side k-wave-python ↔ pykwavers replication of
these examples. An audit established that:

- **Rust core has elastic SWE capability**: `kwavers/src/solver/forward/elastic/swe/`
  exposes `ElasticWaveSolver`, `ElasticWaveConfig`, `ElasticWaveField`,
  velocity-Verlet `TimeIntegrator`, PML boundary, and stress-tensor derivative
  computation. The Lamé-parameter trait `ElasticProperties` is implemented for
  `HomogeneousMedium`.
- **PyO3 bindings expose zero of this**: `pykwavers/src/lib.rs` had no elastic
  surface at all. `SolverType` enum had only `FDTD`, `PSTD`, `Hybrid`, `PstdGpu`.
  No `Medium.elastic`, no stress source, no velocity source, no displacement /
  velocity sensor records.

Replicating any of the four EWP examples requires bridging this gap. Doing it
in one cycle is too large for safe delivery; this ADR phases the work.

## Decision: 4-phase delivery

### Phase A.1 — Medium bridging (this ADR; landed)

**Scope**: expose the elastic medium constructor surface to Python with full
input validation and value-semantic tests. **No solver dispatch yet.**

Delivered in commit `<this commit>`:

1. `HomogeneousMedium::elastic_homogeneous(density, c_compression, c_shear, grid)`
   — closed-form Lamé-from-wave-speeds inversion:
   ```
   μ = ρ · c_s²
   λ = ρ · (c_p² − 2 · c_s²)
   ```
   Returns `Option<Self>`; rejects non-finite, non-positive, and the
   thermodynamic-stability-violating `2·c_s² > c_p²` (auxetic / ν<0).
2. `HomogeneousMedium::set_lame_parameters` / `lame_lambda_value` /
   `lame_mu_value` — public setter and accessors for the Lamé fields.
3. PyO3 `Medium.elastic(c_compression, c_shear, density, grid=None)` static
   method — wraps the Rust constructor, raising `ValueError` on rejection.
4. PyO3 getters: `Medium.c_compression`, `Medium.c_shear`,
   `Medium.lame_lambda`, `Medium.lame_mu`.
5. Rust tests (4): dispersion-relation round trip, fluid limit, stability
   rejection, setter validation.
6. Python smoke test (`pykwavers/examples/elastic_medium_smoke.py`):
   round-trip on the four canonical k-Wave EWP cases plus all rejection
   paths.

This phase is **independently usable** — Python code can construct elastic
media and inspect Lamé parameters even before the solver dispatch lands.

### Phase A.2 — SolverType.Elastic + Source/Sensor surface (`[minor]`, **landed**)

**Scope**: wire `SolverType::Elastic` into `Simulation::run` dispatch; add
`Source.from_initial_displacement(field, axis)` (uses
`ElasticWaveSolver::propagate` driving the velocity-Verlet integrator on
an initial-uz / -ux / -uy displacement IVP); extend `SimulationResult`
to expose displacement traces. End-to-end Python smoke test propagating
a Gaussian IVP through a 32×32×16 elastic medium with three sensors
along the +x ray, asserting causality and geometric spreading.

**Status**: landed in commit `590a52b7`.

### Phase A.2.5 — Multi-component sensor recording (`[patch]`, **landed**)

**Scope**: Rust feature addition — extend `ElasticWaveSolver::propagate`
to record `ux`, `uy`, `uz` displacement components separately into the
`SensorRecorder`'s ux/uy/uz buffers via `record_velocity_step` (in
addition to the legacy `record_step(&uz)` that preserves
`extract_recorded_data` back-compat). Use `SensorRecorder::with_spec`
with a `SensorRecordSpec` covering Pressure + VelocityX + VelocityY +
VelocityZ so all four buffers are allocated. Add public method
`ElasticWaveSolver::extract_recorded_displacement_components()` returning
`(Option<Array2<f64>>, Option<Array2<f64>>, Option<Array2<f64>>)`. PyO3
side: `run_elastic_impl` now populates `SimulationRunResult.{ux_data,
uy_data, uz_data}`, and routes the IVP per-axis ("x" / "y" / "z") via
an `elastic_ivp_axis: Option<String>` carried out-of-band from the
source-routing layer to the dispatch.

**Status**: landed alongside Phase A.2.

**Verification**:
- `result.ux`, `result.uy`, `result.uz` populated as `(n_sensors, NT)`
  ndarrays.
- For uz IVP: uz_peak >> ux_peak / uy_peak at the +x-ray sensor (uz
  peak = 1.35e-10 m vs. ux/uy peaks = 0 exactly at sensor 0 — symmetry
  is exact within numerical precision before P-to-S conversion).
- Legacy `sensor_data` trace agrees with `result.uz` (back-compat
  invariant).

### Phase A.3 — `Source.from_stress` + `Source.from_velocity` (`[minor]`, queued)

**Scope**: add stress-tensor source (`s_mask`, `sxx`, `syy [, sxy, szz, sxz, syz]`)
and particle-velocity source (`u_mask`, `ux`, `uy [, uz]`) bindings.
Required for `ewp_shear_wave_snells_law` (stress source) and
`ewp_3D_simulation` (velocity source with focusing).

**Blockers identified**:
- Current Rust `ElasticWaveSolver::propagate` accepts `Option<&ElasticBodyForceConfig>`
  for body forces but has no stress-mask or velocity-mask source-injection
  path. New Rust API surface required.

### Phase A.4 — Heterogeneous elastic medium + EWP example replications (`[minor]`, queued)

**Scope**:
- `Medium.elastic(c_p_array, c_s_array, density_array)` overload accepting
  3-D ndarrays for layered media (required by all four EWP examples).
- `ewp_layered_medium_compare.py` — simplest case once Phase A.2 + heterogeneous
  medium land.
- `ewp_plane_wave_absorption_compare.py` — adds Kelvin-Voigt absorption parity
  + analytical α₀·f² validation.
- `ewp_shear_wave_snells_law_compare.py` — depends on Phase A.3 stress source
  + heterogeneous medium.
- `ewp_3D_simulation_compare.py` — depends on Phase A.3 velocity source + 3-D.

**Blockers identified**:
- `HeterogeneousMedium` already supports per-voxel Lamé parameters (see
  `kwavers/src/domain/medium/heterogeneous/tissue/implementation/elastic.rs`).
  The PyO3 `Medium(...)` constructor that accepts arrays needs an elastic
  overload exposing them. Adding optional `c_compression_arr`, `c_shear_arr`
  parameters to the existing constructor is the cleanest extension.

## Consequences

### Positive

- Phase A.1 establishes the medium-construction surface needed by every
  later phase. Once it lands, each subsequent phase becomes a focused,
  bounded increment.
- The 4-phase split honors the user's `prefer the smallest reversible change`
  policy and the sprint-cycle phase-exit-criteria policy.
- Python users get immediate value from Phase A.1: they can construct
  elastic media, inspect Lamé parameters, and validate physics setups
  before any solver dispatch is wired.

### Negative

- Phase A.1 alone does not let the user replicate *any* of the four EWP
  examples end-to-end. The user explicitly asked for those replications;
  this ADR re-scopes the request as a 4-phase initiative because the gap
  is genuinely deeper than a bindings-only task.
- Rust feature additions (multi-component sensor recording, stress/velocity
  source masks) are required by Phase A.2 and A.3 — not pure bindings work.

### Neutral

- No physics or numerics changes — Phase A.1 is purely API exposure.
- No public API contraction. Existing fluid-acoustic surface unchanged.

## Verification (Phase A.1)

| Gate | Outcome |
|---|---|
| `cargo check --package kwavers` | clean |
| `cargo check --package pykwavers` | clean |
| 4 new Rust tests (`test_elastic_homogeneous_*`, `test_set_lame_parameters_*`) | 4/4 pass |
| Python `elastic_medium_smoke.py` (4 sections, 14 sub-cases) | all pass |
| Full nextest regression | (recorded in commit message) |

## References

- k-Wave MATLAB EWP examples in `external/k-wave/k-Wave/examples/`.
- Rust elastic solver in `kwavers/src/solver/forward/elastic/swe/`.
- ElasticProperties trait at `kwavers/src/domain/medium/elastic.rs`.
- ADR 005 — additive-deprecation precedent for staged migrations.
- Standards rule `compute_backend_trait` — abstraction-first delivery.
