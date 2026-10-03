<a id="kw-aperture-002"></a>

## KW-APERTURE-002 — General planar aperture propagation [major] — review

priority: feature; needs: none; scope: `kwavers-transducer` Rayleigh aperture types and kernel, ADR-035

- **Driver:** hybrid Fresnel-zone pMUT cells require independently driven central and annular electrode sectors without circular-piston tessellation.
- **Acceptance:** one bounded provider kernel integrates disks and oriented annular sectors, preserves the existing circular-piston oracles, and proves coherent sector superposition before the consumer adds electrode control topology.
- **Evidence:** warning-denied all-target/all-feature Clippy; Nextest 214/214 (one existing skip); doctests 1/1; warning-clean package documentation.
- **Next step:** re-validate at the default branch (`transducers/physics/rayleigh.rs`); close if present.
