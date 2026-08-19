# 113. Type sonoluminescence emission components at the Aequitas boundary

- Status: Accepted (retroactive)
- Date: 2026-08-19
- Item: `backlog.md#kwavers-sono-113--type-sonoluminescence-emission-and-close-the-examplebook-slice-major-arch--in-progress-2026-08-19`
- Records: the implementation and verification commits for the slice

## Context

The sonoluminescence field calculator combined blackbody, bremsstrahlung, and
Cherenkov values in one raw `leto::Array3<f64>`. The first two models return
volumetric power density, while the Cherenkov helper returns a phenomenological
threshold yield. Adding the latter to the former made the field's `W/m³`
contract false. The calculator also allocated intermediate fields and cloned
charge-density arrays on every update, and the integrated dynamics path did not
refresh the emission field after updating bubble state.

`aequitas::systems::si::quantities::VolumetricPowerDensity` is the provider-owned
dimensioned boundary. `leto::Array3` remains the storage substrate; it carries
only the base scalar at the array boundary and does not own the physical unit.

## Decision

`EmissionComponents` owns the dimensioned contributions as private
`VolumetricPowerDensity<f64>` values. A scalar point calculation and the single
field traversal use the same model formulas, so point and field results share
one source of truth. The field traversal writes the sum of enabled dimensioned
components directly into the preallocated output and applies the temperature
cutoff and opacity once.

Cherenkov output remains on the spectral/arbitrary-unit path. It is never added
to `EmissionComponents::total()` or the dimensioned volumetric field. The
Cherenkov threshold and Frank–Tamm factor remain authoritative in
`CherenkovModel`; a field helper may expose threshold yield only when its return
type and documentation state that it is not power density.

`IntegratedSonoluminescence::new` copies `BubbleParameters::t0` and
`initial_gas_pressure` into its initial fields. `simulate_step` refreshes the
emission field after storing the updated state. The emission API takes only the
fields used by the dimensioned update; unused pressure and time parameters are
removed rather than retained as compatibility shims.

## Alternatives rejected

**Keep a raw combined field and relabel it.** A label cannot make a
dimensionally heterogeneous sum valid; this would preserve a false physical
contract.

**Keep per-mechanism temporary arrays.** This preserves avoidable allocation and
copy traffic in the per-step path and creates multiple formula sites.

**Add an adapter around the old signature.** The removed parameters are not
part of the computation. A forwarding API would retain dead surface and violate
the repository's replacement-without-shims rule.

## Consequences

The change is a breaking public-surface change. Callers use the typed component
accessors and the integrated calculator's field accessor; no old names or
forwarding wrappers remain. Spectral Cherenkov values are comparable only as
the documented arbitrary-unit yield, while blackbody and bremsstrahlung totals
retain their `W/m³` contract. Focused value-semantic tests cover constructor
initialization, scalar/field equality, cutoff behavior, and post-step refresh;
the examples and book use the same public path.
