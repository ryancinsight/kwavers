# ADR 021 — `MechanicalStress` (elastic) capability in the declarative `PhysicsCatalog`

**Status:** Accepted (design); implementation staged
**Change class:** [major] (new public `PhysicsModelType` variant + a genuine plugin adapter; touches the serde-stable capability enum)
**Date:** 2026-06-09

## Context

`kwavers_physics::factory::models::PhysicsModelType` is the serde-stable SSOT enumerating which
physics a declarative `PhysicsConfig` can switch on. `kwavers_solver::plugin::catalog::PhysicsCatalog::build_plugin`
is the *exhaustive* match that turns each variant into a concrete `Box<dyn Plugin>` (Theorem 21.1
in Chapter 21: every variant resolves to a plugin or a structured `ConfigError`, no silent fallback).

The catalog has **five** variants (`LinearAcoustics`, `NonlinearAcoustics`, `BubbleDynamics`,
`ThermalDiffusion`, `OpticalPropagation`). Elastic-wave propagation is **implemented** but is
**not reachable through the declarative catalog**:

- The genuine elastic time-stepper is
  `kwavers_solver::forward::pstd::extensions::elastic_orchestrator::ElasticPstdOrchestrator`
  (leapfrog stress–velocity PSTD, k-space corrected, with scalar and Bérenger split-field PML),
  driven by `pstd::extensions::PstdElasticPlugin`'s spectral stress/velocity primitives.
- A *previous* `PhysicsModelType::MechanicalStress` variant and its wrapping `ElasticWavePlugin`
  were **deliberately deleted** during the elastic-as-PSTD-plugin consolidation (documented in
  `crates/kwavers-solver/src/forward/mod.rs`, "Architecture: elastic-as-PSTD-plugin"): they were a
  μ ≡ 0 *duplicate* of the acoustic PSTD stepper. The genuinely useful spectral primitives survived
  under `pstd::extensions`; the duplicate adapter did not.

Backlog item #14 ("wire elastic / `MechanicalStress` into the `PhysicsCatalog`") was filed as
`[minor]` — "add a variant + one `build_plugin` match arm". **That premise is stale.** There is no
longer a drop-in elastic `Plugin` to point a one-line arm at, and re-creating the deleted thin
wrapper would either (a) reintroduce the μ = 0 acoustic duplicate, or (b) be a *mock* — a `Plugin`
whose `update` ignores the unified field cube — both prohibited (CLAUDE.md integrity, HARD tier).

## Problem statement (the real impedance mismatch)

The `Plugin` contract is **scalar-acoustic-pipeline shaped**:

```rust
fn update(&mut self, fields: &mut Array4<f64>, grid, medium, dt, t, ctx) -> KwaversResult<()>;
```

`fields` is the unified scalar field cube (pressure, etc.); the solver loop owns time, source
injection routes through `ctx.sources`, and plugins compose by declaring `required_fields` /
`provided_fields` over `UnifiedFieldType`.

The elastic orchestrator is a **vector-velocity + rank-2-stress** system with:
- its own persistent state (`VelocityFields`, `StressFields`, spectral mirrors, PML sub-fields)
  that lives **outside** the `Array4<f64>` cube;
- a batch `propagate(n_steps, source, sensor_mask)` API (no public single-step entry point);
- its own velocity-source injection (`ElasticPstdVelocitySource`), not `ctx.sources`;
- sensor output as trace matrices, not an in-place field write.

So "wiring" is **not** a match arm. A genuine adapter must (1) expose a single-step orchestrator
primitive, (2) define how vector/tensor elastic state maps to/from the unified pipeline
(`provided_fields`: the isotropic pressure `p = -⅓ tr(σ)` and/or new vector/tensor
`UnifiedFieldType`s), and (3) route `ctx.sources` into the elastic velocity source. These are
**design decisions with public-API consequences**, which is why this is `[major]` and gated on an
ADR rather than an autonomous `[minor]` edit.

## Decision

1. **Reclassify backlog #14 `[minor] → [major]`** and correct its premise (done in `backlog.md`):
   elastic propagation *is* implemented (orchestrator); the gap is *catalog exposure*, and it costs
   a genuine adapter, not a match arm.

2. **Add `PhysicsModelType::MechanicalStress { wave_kind: ElasticWaveKind }`** (serde-stable,
   additive variant) where `ElasticWaveKind ∈ { Isotropic }` initially (room for `Anisotropic`,
   `Nonlinear` later without a breaking change). Additive enum growth is `[minor]` for
   `cargo-semver-checks`, but the *exhaustive* `build_plugin` match makes wiring mandatory in the
   same change (Theorem 21.1), so the unit of delivery is `[major]`.

3. **Implement a genuine `MechanicalStressPlugin`** (new module
   `forward/pstd/extensions/elastic_plugin.rs`) that:
   - **owns** an `ElasticPstdOrchestrator` (constructed from `grid`, the `medium`'s Lamé/density
     fields, `dt`);
   - implements `Plugin::update` by performing **exactly one** leapfrog stress–velocity step
     (a new `ElasticPstdOrchestrator::step(source_slice)` extracted from the `propagate` loop body —
     the loop is refactored to call it, keeping a single SSOT stepper, no duplication);
   - declares `provided_fields = [Pressure]` and writes the isotropic stress trace
     `p = -⅓(σxx+σyy+σzz)` into the unified pressure plane each step (real coupling: the value
     depends on the full elastic state, so the mock-detection heuristic fails — replacing the body
     with `Default::default()` changes the pressure field);
   - adapts `ctx.sources` velocity injection into the orchestrator's `inject_velocity_source` path.

4. **Verification (value-semantic, before the item is closed):**
   - **μ = 0 reduction.** With shear modulus ≡ 0 the plugin's per-step pressure output matches the
     baseline acoustic PSTD plugin to a bounded epsilon on a shared initial condition (the elastic
     stress update constant-folds to the acoustic one — the theorem in `pstd::extensions::elastic`).
   - **Shear support.** With μ > 0 a transverse velocity perturbation propagates at
     `c_s = √(μ/ρ)` (measured arrival within one grid cell of the analytic shear traveltime) —
     behaviour the acoustic path *cannot* produce, proving the variant is not an acoustic alias.
   - **Catalog round-trip.** `PhysicsCatalog::build` with a `MechanicalStress` config yields exactly
     one plugin; a `LinearAcoustics{PSTD} + MechanicalStress` config builds two and resolves a valid
     schedule (Theorem 22.2 scheduling soundness, mirroring the existing PSTD+Bubble test).
   - **`step` ≡ `propagate`.** The extracted single-step primitive, iterated `n` times, reproduces
     `propagate(n, …)` bit-for-bit (guards the refactor against behavioural drift).

## Alternatives considered

- **Thin re-created `ElasticWavePlugin` (the deleted design).** Rejected: it hard-coded μ = 0 and
  duplicated the acoustic stepper — exactly what the consolidation removed. Re-adding it is a
  regression and, without real shear physics, a mock.
- **Leave elastic out of the catalog; document direct orchestrator use only.** Tenable (the
  orchestrator is already a complete, tested public entry point), but it permanently excludes
  elastic from the declarative/serde config surface and from multi-physics composition (e.g.
  acoustic + elastic + thermal) that the catalog exists to provide. Deferred-not-rejected: this ADR
  chooses exposure because composition is the catalog's whole value proposition.
- **A separate elastic field cube / second pipeline.** Larger blast radius; revisit only if the
  isotropic-pressure projection proves insufficient for downstream consumers.

## Consequences

- Closes the real capability gap: elastic propagation becomes selectable from a `PhysicsConfig`
  and composable with the other capabilities, without reintroducing the μ = 0 duplicate.
- `ElasticPstdOrchestrator` gains a public single-step `step(...)` primitive (SSOT; `propagate`
  delegates to it) — a generally useful API for embedding elastic stepping in other loops.
- `PhysicsModelType` grows one additive, serde-stable variant; `ElasticWaveKind` leaves room for
  anisotropic/nonlinear elastic modes as future additive variants.
- The exhaustive `build_plugin` match (Theorem 21.1) is preserved: the new variant has a real arm,
  not an `unsupported(...)` stub.

## Status / staging

Design **Accepted**. Implementation is the next execution increment (single-step `step` extraction →
`MechanicalStressPlugin` → `MechanicalStress` variant + catalog arm → the four verification tests →
update Chapter 21 §21.3 to mark the variant wired). Tracked as backlog #14 `[major]`.
