# ADR-036: Clinical-imaging feature boundary

## Status

Accepted for the 4.0.0 workspace API.

## Context

`kwavers-physics` and `kwavers-solver` unconditionally depend on
`kwavers-imaging`; Physics also unconditionally enables
`kwavers-core/registration`. Those edges reach RITK image I/O and registration,
including `ritk-filter`, even when a consumer uses only forward PSTD. LeoNeuro's
simulation crate uses the latter path and neither imports nor configures the
clinical imaging modules.

The unconditional edge makes an unrelated clinical-image build failure prevent
forward-wave verification. It also makes the dependency graph contradict the
layered API: `acoustics::imaging`, photoacoustic workflow types, clinical FWI,
and elastography inversion are optional clinical capabilities rather than
requirements of acoustic propagation. The pure temperature-dependent
`GrueneisenModel` remains available without image I/O because electromagnetic
photoacoustic simulation consumes that material law independently of clinical
image types.

## Decision

`kwavers-physics` exposes a `clinical-imaging` feature that owns its complete
imaging and image-type-dependent photoacoustics workflow surfaces. Its
`kwavers-imaging` and `ritk-registration` edges are optional and activated only
by that feature. The default physics surface contains acoustic, thermal,
optical, chemistry, mechanics, therapy, transcranial kernels, and pure
thermoelastic material laws that do not require medical image I/O.

`kwavers-solver` exposes the matching `clinical-imaging` feature. It owns the
complete inverse elastography and frequency-domain clinical FWI surfaces and
activates the corresponding physics feature. Forward PSTD and the remaining
non-clinical inverse kernels do not activate this feature.

HU-to-sound-speed evaluation in transcranial phase correction uses
`kwavers_core::constants::hu_mapping::HuAcousticModel`, the provider already
used by `kwavers-medium::CtMediumBuilder`, rather than a loader-owned wrapper.

In-workspace clinical consumers opt in explicitly. LeoNeuro's forward simulation
does not, so its active graph excludes clinical image I/O and registration.
The KWaveArray BLI mapper rejects source samples only once their finite
interpolation window cannot overlap the grid; this preserves valid nearby
off-grid source support without converting arbitrary distant sinc tails into
boundary sources.

## Consequences

- This is a [major] feature-surface change: consumers of the gated public modules
  add `features = ["clinical-imaging"]` to their Kwavers dependency.
- There is no fallback implementation. A consumer that needs clinical imaging
  must enable the complete provider path; a forward-only consumer cannot
  accidentally compile it.
- Examples importing a gated clinical surface declare
  `required-features = ["clinical-imaging"]`, so default example builds do not
  expose an unavailable API.
- `kwavers-physics::acoustics::imaging::fusion` remains functionally complete
  behind the feature in this cut. Its ownership conflicts with the documented
  Diagnostics layer and is tracked for native promotion after the dependency
  boundary is verified; no compatibility re-export will be retained during that
  promotion.

## Rejected alternatives

- Keep unconditional dependencies and fix RITK downstream: makes forward
  simulation availability depend on unrelated clinical-image code.
- Add a silent RITK-free registration fallback: changes clinical semantics and
  conceals provider failures.
- Gate all inverse solvers: removes non-clinical reconstruction and seismic
  capability from the default solver without a dependency reason.

## Verification

The dependency proof is `cargo tree -p leoneuro-sim -i ritk-filter`, which must
report no matching active package after the cut. The forward-PSTD regression
remains the LeoNeuro finite-aperture boundary test. Locked offline execution
passes 1,554/1,554 `kwavers-physics` tests without the feature and 1,710/1,710
with `clinical-imaging`; the corresponding Leo package passes 29/29 tests.
