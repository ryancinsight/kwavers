# ADR 040: Adopt Aequitas for bubble-energy quantities

- Status: Accepted
- Date: 2026-07-19
- Class: [arch] [major]

## Context

`kwavers-physics` used `uom` only in the bubble-energy subsystem. That local
dependency duplicated Atlas scalar ownership and exposed storage-specific
`uom::si::f64` types in public methods.

The migration also exposed a dimensional defect in
`update_temperature_from_energy`: `c_v` has units J/(kg·K), but the parameter
was declared as heat capacity in J/K. The implementation escaped to raw
scalars before evaluating `ΔE / (m c_v)`, so the type system could not reject
the mismatch.

Aequitas owns the Atlas physical-quantity law over Eunomia scalar types. Its
provider decision and comparison against `uom` 0.38.0 are recorded in
[Aequitas ADR 0001](https://github.com/ryancinsight/aequitas/blob/49ee8004e008a480ac871c3782d00d921ba41c01/docs/adr/0001-aequitas-quantity-law.md).

## Decision

- Pin Aequitas revision
  `49ee8004e008a480ac871c3782d00d921ba41c01` as the quantity provider for
  `kwavers-physics`.
- Resolve Eunomia through the same unqualified Git source as the workspace;
  the lockfile remains the exact revision pin. This prevents duplicate package
  identities and makes `--locked` resolution invariant across local patches
  and hosted CI.
- Remove the direct `uom` dependency and all `uom` call sites from that crate.
- Express heat-transfer and temperature-update equations with Aequitas
  quantity arithmetic. Unit extraction occurs only at the `BubbleState` raw-SI
  boundary.
- Use `SpecificHeatCapacity` in J/(kg·K), so
  `Energy / (Mass * SpecificHeatCapacity)` resolves to thermodynamic
  temperature at compile time.

## Public migration

The public quantity types change from `uom::si::f64::*` to
`aequitas::systems::si::quantities::*`.

```rust
use aequitas::systems::si::{
    quantities::Power,
    units::Watt,
};

let power = Power::from_unit::<Watt>(12.0_f64);
assert_eq!(power.in_unit::<Watt>(), 12.0);
```

Callers of `update_temperature_from_energy` pass
`SpecificHeatCapacity` constructed with `JoulePerKilogramKelvin`, not
`HeatCapacity`.

No compatibility adapter is retained.

## Rejected alternatives

### Keep `uom` for bubble dynamics

Rejected because it preserves two scalar and dimensional-law providers in
Atlas and prevents generic use of Eunomia wrapper types.

### Wrap `uom` behind Kwavers-local aliases

Rejected because aliases hide rather than resolve provider ownership and
retain the incorrect heat-capacity contract.

### Convert Aequitas quantities to raw scalars inside each equation

Rejected because it discards the compile-time dimension proof and recreates
the defect channel this migration closes.

## Consequences

- The affected public methods require a major-version migration.
- Aequitas is the single quantity-law dependency for this subsystem.
- The heat-transfer path retains no intermediate allocation, dynamic dispatch,
  or runtime dimension metadata.
- Future missing quantity laws are implemented in Aequitas, not locally in
  Kwavers.

## Verification

- `cargo check --locked -p kwavers-physics`
- `cargo clippy --locked -p kwavers-physics --all-targets -- -D warnings`
- `cargo nextest run --locked -p kwavers-physics`: 1,554 passed, 1 skipped
- `cargo test --locked -p kwavers-physics --doc`: 8 passed, 6 ignored
- residue scan: no `uom`, `.get::<...>`, or `::new::<...>` remains in
  `kwavers-physics`

`cargo semver-checks` cannot complete against this repository topology. The
all-feature audit is blocked by a pre-existing duplicate-Leto type split in
`kwavers-imaging`; the no-default-feature audit builds the current crate, then
cannot resolve the baseline clone's `../apollo` path dependency. The public
parameter and return-type replacements above establish the major
classification directly.
