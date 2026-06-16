# ADR 025 — Optoacoustic SOAP / OFUS emitter modeling

- Status: Accepted
- Change class: [major] (additive: new public material type + presets in
  `kwavers-medium`; the design analytics in `kwavers-physics` are a sibling
  addition tracked here for context)
- Date: 2026-06-16
- Reference: Li et al., "Optically-generated focused ultrasound for noninvasive
  brain stimulation with ultrahigh precision," *Light: Sci. Appl.* **11**, 321
  (2022). https://doi.org/10.1038/s41377-022-01004-2

## Context

A **soft optoacoustic pad (SOAP)** generates **optically-generated focused
ultrasound (OFUS)**: a nanosecond laser pulse illuminates a thin light-absorbing
nanocomposite layer coated on a *curved* PDMS surface; thermoelastic expansion
launches an ultrasound pulse from every surface element, and the spherical-cap
geometry focuses the wavefronts at the geometric centre *by geometry alone* — no
electronic delays. A high numerical aperture (NA ≈ 0.95), reachable with a soft
crack-free absorber but not with single-crystal PZT, yields an ultrahigh focal
gain and a lateral focus of ≈83 µm at 15 MHz — two orders of magnitude tighter
than conventional transcranial focused ultrasound.

The paper's quantitative results we target for replication:

- Lateral resolution (Eq. 1): `R_L = 0.71·ν/(NA·f)`. At the device point
  (ν = 1500 m/s, f = 15 MHz) this is exactly the reported empirical fit
  `R[µm] = 71.51/NA` (`0.71·1500/15e6 = 71.0 µm`), giving 75 µm at NA 0.95.
- Focal pressure gain (Eq. 2): `G = (2πf/c₀)·r·(1 − √(1 − 1/(4 f_N²)))`,
  `f_N = r/D_t = 1/(2·NA)`. For the device (r = 6.35 mm, D_t = 12.1 mm,
  f_N ≈ 0.52) the lossless value is ≈289; with water attenuation over the path
  the paper reports `G_max ≈ 280`.
- CS-PDMS focal pressure 48 MPa at 0.62 mJ/cm² (NA 0.95). Four absorbers
  (CS, CNT, CNP-PDMS, HSM) with focal-pressure ratio 1 : 1/6 : 1/6 : 1/30 and
  centre frequencies 15 / 5 / 5 / 3 MHz.

## Decision

Model OFUS/SOAP across three crates along their existing domain boundaries, so
that no crate gains a dependency it should not have:

1. **`kwavers-physics::analytical::transducer::optoacoustic`** — the closed-form
   *design analytics*: spherical-cap geometry ⇄ NA ⇄ f-number conversions
   (`numerical_aperture_from_geometry`, `f_number_from_na`, `na_from_f_number`),
   the focal gain `G` (`soap_focal_gain`, Eq. 2), and the acoustic-resolution
   lateral resolution (`acoustic_resolution_lateral`, Eq. 1). `soap_focal_gain`
   is proven equal to the O'Neil focused-bowl focal-gain limit `k·h` already
   implemented in `focused_bowl_onaxis`, so the two derivations cross-check.

2. **`kwavers-medium::properties::optoacoustic`** — the *absorber materials*: a
   validating `OptoacousticEmitter` carrying host-dominated acoustic properties
   (ρ, c, α) for placing the layer in a heterogeneous medium, plus the
   optoacoustic-conversion properties (Grüneisen Γ, optical absorption μ_a,
   measured optoacoustic sensitivity S, pulse centre frequency, pulse FWHM).
   Presets `PDMS`, `CS_PDMS`, `CNT_PDMS`, `CNP_PDMS`, `HSM`.

3. **`kwavers-optics`** — the *source amplitude* `p₀ = Γ·μ_a·F` already lives in
   `optical_transport::initial_pressure`; the emitter's measured sensitivity
   `S` is the load-bearing surface-pressure-per-fluence used for the SOAP source.

The composition law is the single contract that joins the three:

```
p_surface = S · F                          (kwavers-medium: emitter)
p_focus   = G · p_surface                   (× kwavers-physics: soap_focal_gain)
R_lateral = 0.71·ν/(NA·f)                   (kwavers-physics: Eq. 1)
```

### Why a *measured* sensitivity S, not first-principles Γ·μ_a·F

For a *thin* surface absorber with an acoustically soft (air) backing, the
radiated surface pressure is not the naive volumetric initial pressure
`Γ·μ_a·F`: stress buildup over the absorption depth, the air-backing reflection,
and acoustic out-coupling all rescale it (here by ≈6×). The paper reports the
*measured* focal pressures, not the per-absorber μ_a needed to derive them. We
therefore make the measured optoacoustic sensitivity `S` (back-calculated from
48 MPa / (G_max · F)) the load-bearing quantity — **empirical-tier evidence,
honestly labelled** — and carry Γ and μ_a as documented physical properties for
first-principles light-transport workflows, *without* asserting they reproduce
`S` (which would imply an unphysical conversion efficiency > 1).

## Alternatives considered

- **Put everything in one crate.** Rejected: the absorber materials belong with
  the other materials (`kwavers-medium`), the geometry/gain analytics belong with
  the other transducer analytics (`kwavers-physics`), and forcing one crate to
  own all three would create a cross-domain dependency.
- **Derive 48 MPa from Γ·μ_a·F.** Rejected: it requires fabricating per-absorber
  μ_a / confinement data the paper does not provide, and the thin-layer physics
  genuinely exceeds the volumetric value — a fabricated derivation would violate
  the evidence-tier rule.

## Consequences

- A user can place a `CS_PDMS` (etc.) layer in a heterogeneous medium, compute
  the SOAP focal gain from its geometry, and obtain the focal pressure — matching
  the paper to within rounding (verified by tests in both crates).
- The book gains a chapter on OFUS/SOAP with the replicated closed-form results
  (focal gain vs f-number → G_max ≈ 280; lateral resolution vs NA → 71.5/NA).
- Follow-on (not in this slice): a full-wave 2-D arc-source focusing example that
  measures the simulated focal FWHM, and a `kwavers-transducer` SOAP source type
  that drives the curved emitter with the broadband photoacoustic pulse.

## Verification

- `kwavers-medium`: `CS_PDMS.focal_pressure(6.2, 280) ≈ 48 MPa`; sensitivity
  ratios 1 : 1/6 : 1/6 : 1/30; centre-frequency ordering; bare-PDMS inactive.
- `kwavers-physics`: `numerical_aperture_from_geometry(6.35e-3, 12.1e-3) ≈ 0.953`;
  `soap_focal_gain ≈ 289` and equal to `focused_bowl_onaxis` focal value;
  `acoustic_resolution_lateral ≈ 74.5 µm` and the `71.0/NA` fit.
