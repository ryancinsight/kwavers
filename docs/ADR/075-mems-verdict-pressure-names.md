# ADR 075: Remove Unit Suffixes from MEMS Verdict Pressure Fields

- Status: Accepted
- Date: 2026-07-31
- Driver: `KWAVERS-AEQ-MET-37`

## Decision

The public `TherapyVerdict` pressure fields are named `cmut_output` and
`pmut_output`. Both retain Aequitas `Pressure` types, so the unit contract is
carried by the type rather than a `Pa` suffix. Existing comparison formulas
extract canonical SI scalars only at the dimensionless scoring boundary.

All in-repository callers and tests migrate in the same change. No alias,
forwarding field, or deprecated compatibility name remains.

## Rejected alternative

Keeping `cmut_output_pa` and `pmut_output_pa` would preserve a redundant unit
encoding after the values became typed and would leave a second public naming
convention for the same Aequitas pressure contract.

## Verification

The transducer package check, full Nextest (219/219 with one skip),
warning-denied Clippy, doctests, rustfmt, and diff checks pass. The change is
real-valued and does not alter the Eunomia complex-quantity boundary.
