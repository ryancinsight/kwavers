# Gap Audit (Single Source)

Status: validated
Last Updated: 2025-11-06

## Summary
- Duplicate, unused `HeterogeneousMedium` implementation existed at `src/medium/heterogeneous/implementation.rs` with overlapping trait implementations.
- Canonical `HeterogeneousMedium` is defined in `src/medium/heterogeneous/core/structure.rs` and re-exported via `src/medium/heterogeneous/mod.rs`.
- Viscous trait for heterogeneous medium only implemented `viscosity`, leaving `shear_viscosity` and `bulk_viscosity` semantics inconsistent with available coefficient arrays.

## Actions
- Removed dead file `src/medium/heterogeneous/implementation.rs` to enforce one canonical definition.
- Enhanced `src/medium/heterogeneous/traits/viscous/properties.rs`:
  - Implemented `shear_viscosity` via `shear_viscosity_coeff`.
  - Implemented `bulk_viscosity` via `bulk_viscosity_coeff`.
  - Overrode `kinematic_viscosity` to use continuous interpolation of `density`.
- Added unit tests `tests/heterogeneous_viscous_properties.rs` validating dynamic, shear, bulk, and kinematic viscosity for both trilinear and nearest-neighbor modes.

## Evidence & Rationale
- Architectural consistency: Traits for heterogeneous media uniformly use `TrilinearInterpolator::get_field_value` across domains (acoustic, thermal, optical), now extended to viscous.
- Mathematical correctness: `ν = μ / ρ` implemented continuously; density clamped to `MIN_PHYSICAL_DENSITY` to maintain physical constraints.
- Code quality: Dead code removed; single canonical path reduces ambiguity and risk.

## Category, Severity, Status
- Category: Code Quality Issues, Algorithm Issues
- Severity: Major (API ambiguity + duplication)
- Status: resolved → validated

## Validation
- `cargo test` target includes `tests/heterogeneous_viscous_properties.rs` verifying expected values.

## Follow-ups
- Consider property map initialization paths to populate `shear_viscosity_coeff` and `bulk_viscosity_coeff` from materials databases.
- Align photoacoustic and reconstruction modules to reuse centralized interpolation utility when applicable.
