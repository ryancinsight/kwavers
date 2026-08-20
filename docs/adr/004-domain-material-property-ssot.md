# ADR 004 — Domain material-property SSOT (composition pattern)

- **Status:** Accepted
- **Date:** 2026-01-12 · **Audited:** 2026-06-03
- **Change class:** [arch]
- **Relates:** paths relocated by the crate split ([ADR 011](011-workspace-crate-split.md));
  typed skull configuration consolidation tracked by
  [KW-EXAMPLES-115](../../backlog.md#kw-examples-115--type-and-partition-the-seismic-example-workflows-major-arch--in-progress-2026-08-20)
- **Revision 2026-08-20:** The skull workflow's existing
  `AcousticSkullProperties` configuration now stores Aequitas quantities and is
  consumed directly by both CT mappings. A proposed parallel bone-material
  struct was rejected because it would duplicate this accepted SSOT boundary.

## Context

Material properties (acoustic, elastic, thermal, optical, electromagnetic) were
defined by duplicate structs across the physics, clinical, and solver layers.
Different modules used different constants for the same material, derived
quantities (wave speeds, impedances) were re-implemented, validation was
inconsistent or absent, and a change required edits in 6+ places.

## Decision

Adopt a **composition pattern with one canonical definition per property type**
in the domain layer:

- The domain owns point-wise, validated property structs and their derived
  quantities (the SSOT).
- Physics/clinical layers do not redefine properties; they *compose* the domain
  structs through array bridges (`uniform()`, `at()`, `from_domain()`), keeping
  computational layouts where they belong without duplicating the source of truth.

## Current state (audited 2026-06-03)

Done. The SSOT is `crates/kwavers-medium/src/properties/` — note this is now
a **directory**, not the single `properties.rs` file the original ADR cited. Its
`mod.rs` header still assigns material-property ownership to the medium domain.
Canonical structs:

- `AcousticPropertyData` — `properties/acoustic.rs:57`
- `ElasticPropertyData` — `properties/elastic/mod.rs:48`
- `ThermalPropertyData` — `properties/thermal.rs:53`
- `OpticalPropertyData` — `properties/optical/mod.rs:47`
- `ElectromagneticPropertyData` — `properties/electromagnetic.rs:40` (the original
  ADR called this `EMMaterialProperties`; renamed)
- plus `StrengthPropertyData` and the composite `AcousticMaterialProperties`
  (`properties/material.rs:55`)

Composition boundaries:
- `OpticalPropertyData` retains tissue presets, refractive index, and spatial
  material aggregation while Hyperion owns its validated interaction
  coefficients and every derived optical-transport law. The former
  `DiffusionOpticalProperties` bridge is deleted by
  [ADR 046](046-hyperion-optical-transport-ownership.md).
- `TissuePropertyMap::{uniform,water,liver}` over `AcousticPropertyData` —
  `crates/kwavers-therapy/src/therapy/therapy_integration/tissue/mod.rs:62`

No genuine duplicate property structs remain. Domain-config structs that *consume*
the SSOT (e.g. `AcousticSkullProperties`, `crates/kwavers-physics/src/acoustics/skull/properties.rs:29`)
are distinct concerns, not violations.

## Consequences

- One validated definition per property; derived quantities computed once.
- Layer separation preserved: domain semantics vs. layer-specific array layouts.
- The original `domain/medium/properties.rs` and `clinical/` paths are pre-split;
  current locations are under `crates/kwavers-medium`, `crates/kwavers-physics`,
  and `crates/kwavers-therapy` (the top-level `clinical/` no longer exists — see
  [ADR 011](011-workspace-crate-split.md)).
