# Media and Tissue Models

## Scope

Media chapters cover density, sound speed, attenuation, nonlinearity, anisotropy, skull layers, implants, thermal dependence, optical absorption, and heterogeneous maps. Code ownership maps to `kwavers::domain::medium`, `kwavers::physics::thermal`, `kwavers::physics::acoustics::skull`, and `kwavers::clinical`.

## Theorem: Impedance Reflection at Normal Incidence

For two lossless media with impedances `Z1 = rho1 c1` and `Z2 = rho2 c2`, the pressure reflection coefficient at normal incidence is

```text
R = (Z2 - Z1) / (Z2 + Z1).
```

### Proof Sketch

Continuity of pressure and normal particle velocity at the interface gives `p_i + p_r = p_t` and `(p_i - p_r)/Z1 = p_t/Z2`. Solving for `p_r/p_i` gives the stated coefficient.

## Algorithm: Tissue Model Acceptance

1. Load or synthesize material maps with explicit units.
2. Validate finite positive `rho`, `c`, and attenuation values.
3. Validate interface behavior with impedance identities.
4. Preserve metadata for CT/HU, segmentation labels, and interpolation provenance.

## Implementation Targets

- Keep tissue property maps as authoritative domain data, not solver-specific arrays.
- Route DICOM-derived media through the RITK adapter.
- Test medium iterators and interpolation with value-based invariants.

## Research Anchors

- Photoacoustic sound-speed estimation and aberration-correction context: https://doi.org/10.1016/j.pacs.2024.100621
- k-Wave heterogeneous medium examples: https://k-wave-python.readthedocs.io/
