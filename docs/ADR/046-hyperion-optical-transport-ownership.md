# ADR 046 — Hyperion optical-transport ownership

- **Status:** Implemented
- **Date:** 2026-07-21
- **Change class:** [major] [arch]
- **Relates:** [ADR 004](004-domain-material-property-ssot.md), Atlas ADR 0030

## Context

Kwavers implemented the same optical laws in four layers. `kwavers-optics`
exported raw scalar functions; `kwavers-medium` recomputed derived material
properties; `kwavers-physics` introduced `DiffusionOpticalProperties` and
`OpticalAbsorption`; and `kwavers-solver` repeated diffusion and effective-
attenuation formulas. Validation, units, zero-medium behavior, and diagnostics
therefore depended on the caller.

Hyperion now supplies the lower common owner for validated photon-interaction
coefficients, Aequitas quantities, reduced scattering, optical depth,
transmission, and diffusion-derived coefficients. Keeping the Kwavers copies
would add a package without consolidating code.

## Decision

Depend directly on published Hyperion `064a189` from every Kwavers crate that
uses its contract.

- `kwavers-medium::OpticalPropertyData` remains the material aggregate. It
  privately stores Hyperion coefficient values and anisotropy, retains
  refractive index and tissue presets, and exposes explicit raw-SI projections
  only at consumer boundaries.
- `kwavers-physics` and `kwavers-solver` consume
  `DiffusionCoefficients<f64>`, typed paths, optical depth, and transmission.
  Spatial Green functions, Monte Carlo transport policy, fields, and
  photoacoustic source coupling remain local.
- `kwavers-optics` retains chromophore spectra. Its optical-transport formula
  module is deleted.

Dependency direction is one way:

```text
Aequitas + Eunomia -> Hyperion -> kwavers-medium -> kwavers-physics
                                      |                    |
                                      +----------------> kwavers-solver
```

## Deletion accounting

The migration deletes the complete `kwavers-optics::optical_transport` module,
`DiffusionOpticalProperties`, `OpticalAbsorption`, the tissue wrapper attached
to that parallel model, the default `mu_s' = 10 mu_a` heuristic, and consumer
copies of reduced-scattering, diffusion, effective-attenuation, albedo,
penetration-depth, optical-depth, and transmission formulas.

Photoacoustic initial-pressure and fluence-compensation behavior stays in
Kwavers because it couples optical deposition to acoustic source policy rather
than defining photon transport. Refractive-index handling and diffusion-
approximation applicability also stay local.

## Rejected alternatives

- Re-exporting Hyperion from `kwavers-optics` preserves the old ownership
  fiction and creates a compatibility facade.
- Keeping raw structs and delegating only selected formulas leaves invalid
  coefficient states and parallel validation in memory.
- Moving Monte Carlo, photoacoustic, or spatial solver code to Hyperion widens
  the provider beyond the shared interaction-law boundary.

## Verification

The consumer gate covers Hyperion/aggregate coefficient equivalence, uniform-
fluence decay, photoacoustic pressure and diffusion behavior, invalid and
degenerate coefficient rejection, package-scoped Nextest, warning-denied
Clippy, doctests, Rustdoc, residue scans, and public SemVer classification.
