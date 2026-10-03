<a id="kw-aperture-001"></a>

## KW-APERTURE-001 — Own finite circular-piston propagation [minor] — review

priority: feature; needs: none; scope: `kwavers-transducer::transducers::physics`, its public exports and analytical/differential tests

- **Driver:** a private downstream consumer duplicated and double-counted finite-aperture diffraction.
- **Acceptance:** the provider evaluates the baffled Rayleigh first integral with the `k/(2*pi)` surface-pressure prefactor, area-consistent disk quadrature, oriented half-space suppression and complex coherent summation; analytical and far-field reference tests plus package gates pass.
- **Evidence:** exact on-axis and disk-area oracles, far-field Bessel differential oracle, rotation and baffle invariants, warning-denied Clippy/docs, Nextest 209/209. Registry-baseline semver analysis was externally unavailable.
- **Next step:** re-validate at the default branch; close if present.
