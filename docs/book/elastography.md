# Elastography

## Scope

Elastography covers shear-wave speed, strain imaging, ARFI, time-of-flight inversion, phase-gradient inversion, nonlinear elastography, and uncertainty. Code ownership maps to `kwavers::solver::inverse::elastography`, `kwavers::physics::acoustics::mechanics`, and `kwavers::analysis`.

## Theorem: Shear Modulus from Shear-Wave Speed

For an isotropic linear elastic medium with density `rho` and shear-wave speed `c_s`,

```text
mu = rho c_s^2.
```

### Proof Sketch

The transverse wave equation in a homogeneous isotropic elastic solid has speed `c_s = sqrt(mu/rho)`. Squaring and rearranging gives the modulus identity.

## Algorithm: Elastography Validation

1. Validate displacement or velocity tracking.
2. Estimate shear-wave speed from time-of-flight, phase gradient, or inversion.
3. Convert speed to modulus only after density is known.
4. Validate stiffness maps against analytical phantoms and literature ranges.

## Implementation Targets

- Keep displacement tracking separate from mechanical inversion.
- Preserve density dependence in all modulus conversions.
- Test nonlinear parameter recovery with value-semantic synthetic data.

## Research Anchors

- 2024 elastography review: https://doi.org/10.3390/app14104308
- Surface acoustic wave elastography review: https://pubmed.ncbi.nlm.nih.gov/38597908/
