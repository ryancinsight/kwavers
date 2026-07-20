# ADR 041: Proteus thermophysical properties

- Status: Accepted
- Date: 2026-07-20
- Class: [arch] [major]

## Context

`ThermalPropertyData` stored and validated raw density, specific heat capacity,
and thermal conductivity, then repeated `alpha = k / (rho c_p)`. The same
material-property contract also exists in fluid and radiation consumers.
Kwavers additionally owns tissue perfusion, which is not a shared property in
the current consumer set.

## Decision

- Compose Proteus `ThermophysicalProperties<f64>` inside
  `ThermalPropertyData`.
- Expose scalar accessors at the Kwavers solver boundary while storing each
  dimensional property once.
- Construct static water, soft-tissue, and bone catalog entries through the
  same validated path as runtime material data.
- Keep perfusion and blood heat capacity in Kwavers.
- Remove the public raw thermophysical fields and update every in-repository
  caller in this change.

## Alternatives rejected

- Call Proteus only from `thermal_diffusivity` while retaining raw fields:
  rejected because validation and storage would still have two owners.
- Move perfusion into Proteus: rejected because Helios and CFDrs do not share
  that law in the present scope.
- Preserve raw fields alongside the typed bundle: rejected as dual storage
  with synchronization risk.

## Consequences

The field-to-accessor migration is intentionally breaking. Proteus newtypes and
the cohesive bundle remain transparent, statically dispatched values; there is
no allocation or runtime unit metadata. Catalog constants retain their
Kwavers-owned cited values rather than being conflated with CFDrs references.
