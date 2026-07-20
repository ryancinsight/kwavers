# ADR 042: Proteus temperature-response ownership

- Status: Accepted
- Date: 2026-07-20
- Class: [arch] [major]

## Context

Kwavers owned two independent implementations of the same relative
thermophysical temperature response:

- `kwavers-medium` evaluated quadratic conductivity and linear specific heat
  around a material reference temperature.
- `kwavers-physics` evaluated separate linear conductivity and specific-heat
  functions around body temperature.

Both implementations operated on raw coefficients despite the coefficients
having K⁻¹ or K⁻² dimensions. The physics update also assumed every evaluated
property remained valid and panicked when construction failed.

Proteus now owns dimensionally typed constant, linear, and quadratic response
strategies plus a statically dispatched temperature constitutive law. Aequitas
owns the coefficient and temperature dimensions.

## Decision

- Store one `TemperatureLaw` inside `TemperatureDependentThermal`.
- Use a zero-sized constant density response, a linear specific-heat response,
  and a quadratic conductivity response.
- Evaluate all thermal properties once when producing a combined
  temperature-dependent material.
- Compose the `kwavers-physics` body-temperature update from the same Proteus
  law while retaining Kwavers-owned perfusion, absorption, and sound-speed
  behavior.
- Delete the standalone conductivity and specific-heat temperature helpers.
- Return errors for non-finite temperatures and non-physical evaluated
  properties instead of panicking.

## Proof obligations

At the reference temperature, `ΔT = 0`, so the linear and quadratic factors
equal one. The evaluated density, heat capacity, and conductivity therefore
equal the reference bundle exactly.

The Aequitas coefficient dimensions establish
`[β₁ΔT] = Θ⁻¹Θ = 1` and `[β₂ΔT²] = Θ⁻²Θ² = 1`. Proteus revalidates the complete
evaluated property bundle, preventing a negative or non-finite response from
entering a solver.

The response strategies are generic fields of `TemperatureLaw`; dispatch is
monomorphized and has no vtable. The invariant density policy is zero-sized.

## Alternatives rejected

- Keep the scalar helpers as wrappers: rejected because compatibility aliases
  would preserve two public ownership models for the same law.
- Move tissue coefficients into Proteus: rejected because coefficient catalogs
  and perfusion remain tissue-domain concerns.
- Evaluate conductivity and heat capacity separately: rejected because it
  repeats validation and constitutive evaluation in combined-material paths.
- Use a runtime polynomial enum: rejected because every catalog fixes its
  response order at construction.

## Consequences

The result-bearing evaluation methods are an intentional public migration.
In-repository callers migrate in this change. Acoustic density remains
Kwavers-owned; combined thermal diffusivity reuses that evaluated density
through the canonical Proteus thermophysical bundle.

CFDrs can adopt the same density response after advancing its Proteus pin.

## Verification

- reference and elevated-temperature value regressions;
- non-finite coefficient and temperature rejection;
- preservation of acoustic-density coupling in combined diffusivity;
- warning-denied Clippy for `kwavers-medium` and `kwavers-physics`;
- package-scoped Nextest, doctests, Rustdoc, dependency policy, and SemVer
  checks.
