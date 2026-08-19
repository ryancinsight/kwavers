# ADR 015 — CMUT vs PMUT MEMS Transducer Models (Flexible / IVUS)

**Status:** Accepted
**Change class:** [major] (new electromechanical element models in `kwavers-transducer`)
**Date:** 2026-06-06
**Supersedes backlog item:** "Piezoelectric (Mason) + CMUT transducer models" — folded into a
CMUT-vs-PMUT comparison framed for flexible transducers and IVUS.

## Context

The book has no chapter on micromachined transducers, and `kwavers-transducer` injects only a
*prescribed kinematic* source — there is no CMUT (capacitive) or PMUT (piezoelectric)
electromechanical element model. CMUT vs PMUT is the central design question for
catheter-integrated, flexible, high-frequency arrays (notably IVUS at 20–60 MHz). We want
first-principles, testable models so the new chapter's comparisons ("which is better for IVUS")
are backed by real computation, not assertion.

## Decision

Add a `kwavers_transducer::mems` module:

- `mems::plate` — shared clamped-circular-plate physics: flexural rigidity
  `D = E h³/(12(1−ν²))`, vacuum fundamental `f = (10.216/2π)(h/a²)√(E/(12ρ(1−ν²)))`, and Lamb
  fluid-loading downshift `f_imm = f_vac/√(1 + 0.6689 ρ_f a/(ρ_m h))`.
- `mems::cmut::CmutCell` — vacuum-gap electrostatic cell: capacitance, effective modal
  stiffness/mass (self-consistent with `f_vac`), parallel-plate **collapse voltage**
  `V_c = √(8 k g₀³/(27 ε₀ A))`, bias-dependent coupling `k²=(V_dc/V_c)²` (capped), dielectric
  self-heating `P = π f C V² tanδ`, and radiation-damping-limited fractional bandwidth.
- `mems::pmut::PmutCell` — piezoelectric unimorph plate: composite resonance, film capacitance,
  effective coupling from `e₃₁,f`, dielectric self-heating, moderate fractional bandwidth,
  transmit sensitivity.
- `mems::comparison` — `IvusComparison`: scores CMUT vs PMUT at an IVUS target frequency on
  bandwidth (axial resolution), self-heating, drive voltage, and integration; returns the
  weighted figure of merit and the verdict.

Closed-form lumped/analytic models (not FEM). Material presets: CMUT (Si membrane), PMUT-AlN,
PMUT-PZT.

## Alternatives

- Full Mason 1-D thickness-mode circuit — appropriate for bulk piezo, not flexural MEMS plates;
  the plate/membrane lumped model is the correct abstraction for CMUT/PMUT.
- FEM coupled-field — out of scope; analytic scaling laws suffice for design comparison and are
  verifiable in closed form.

## Verification

- Plate: `f ∝ h/a²`; immersion `f_imm < f_vac`; known reference value.
- CMUT: `V_c ∝ g₀^{1.5}`; `k²` rises with bias and is bounded; capacitance formula.
- PMUT: composite resonance ordering; capacitance; PZT `tanδ` ⇒ higher self-heating than AlN/CMUT.
- IVUS: for representative 40 MHz designs the framework returns CMUT-favoured on
  bandwidth + thermal FoM (with PMUT's drive-voltage advantage reported) — the chapter's claim,
  made reproducible.

## Consequences

- Real, testable design models behind the new "CMUT vs PMUT" chapter and its figures.
- Bulk-piezo Mason circuit and FEM remain future items if a coupled forward field is needed.
