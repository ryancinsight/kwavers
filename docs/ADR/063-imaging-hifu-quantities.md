# ADR 063: Type HIFU imaging quantities

- Status: Accepted
- Date: 2026-07-27
- Scope: `kwavers-imaging` HIFU planning and the focused HIFU field caller

## Decision

Use Aequitas SI quantities at the public HIFU planning boundary:

- `Frequency` for transducer and pulse-repetition frequency;
- `Power` for transducer and phase acoustic power;
- `Length` for focal geometry, target geometry, offsets, and monitoring points;
- `Time` for treatment, pulse, cooling, and phase durations;
- `Intensity` for acoustic safety limits;
- `ThermodynamicTemperature` for absolute temperature limits; and
- `TemperatureDifference` for allowed temperature rises.

`max_thermal_dose` remains a scalar because CEM43 is a consumer dose model
unit, not an SI base quantity. Duty cycle, target shape, feedback channels,
and other protocol controls remain dimensionless or categorical. The focused
field kernel extracts SI base values only at its numerical formula boundary.

Absolute temperature is stored in kelvin. The default 85 °C limit is therefore
`358.15 K`, and the 100 °C validation ceiling is `373.15 K`. The legacy
1000 W/cm² intensity ceiling is stored as its SI equivalent, `10^7 W/m²`.

## Alternatives rejected

1. Keep raw `f64` fields with unit-suffixed comments. Rejected because callers
   can pass metres, centimetres, Celsius, or W/cm² without a type error.
2. Add local HIFU wrapper types. Rejected because Aequitas already owns the
   canonical dimensional vocabulary and another wrapper would duplicate it.
3. Convert all dense field arrays to quantity elements. Rejected because those
   arrays are numerical storage and formula boundaries; typing their public
   configuration and result contracts supplies the dimensional guarantee without
   changing the solver's storage representation.

## Verification

The HIFU focused-field tests remain value-semantic: geometric focus,
lateral symmetry, and the peak-pressure intensity identity. Treatment-plan
tests additionally verify that absolute-temperature and SI-intensity limits
reject values above their typed ceilings. Package check, focused Nextest,
warning-denied Clippy, doctests, Rustdoc, rustfmt, and `git diff --check` are
the acceptance gates.
