# Registration Ownership Inventory

## Canonical owner

- `ritk`

## `kwavers` responsibilities

- workflow orchestration
- modality integration
- no generic reusable registration implementation as canonical ownership

## Current tranche truth

- `kwavers` already conditionally re-exports `ritk` registration facilities for functional ultrasound workflows
- a `RegistrationEngine` trait now exists in the solver interface layer for future direct dependency inversion
