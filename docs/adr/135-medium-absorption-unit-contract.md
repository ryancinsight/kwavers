# ADR 135: Medium absorption stays dB/cm in storage, Np/m at the boundary

Status: Accepted

## Context

Every medium stores the power-law prefactor α₀ in dB/(MHz^y·cm) at the
1 MHz reference (water: `WATER_ABSORPTION_ALPHA_0 = 0.0022`). Every
solver-facing getter returns Np/m (`ReciprocalLength`), converting through
the canonical `DB_TO_NP × 100` ([Np/dB] × [cm/m]):

- homogeneous: `kwavers-medium/src/acoustic.rs:41`,
- heterogeneous tissue: `heterogeneous/tissue/implementation/acoustic.rs:25,31`,
- default trait implementor: `heterogeneous/traits/acoustic/properties.rs:132`
  (Szabo 1994 cited at each site).

The raw view `absorption_array` exposes the stored prefactor unconverted.
Its only writer, `apply_residual_gas_shielding_in_place`
(`kwavers-solver/src/multiphysics/residual_gas_coupling.rs:235`), converts
its Np/m excess through `np_m_to_power_law_db_cm`, which routes through the
same canonical constants and round-trips against its inverse in test. The
stored field therefore cannot silently become Np/m: the writer would break.

`validate_absorption_model`
(`kwavers-analysis/src/validation/mod.rs:118`) pins the boundary: it
compares the getter against `WATER_ABSORPTION_ALPHA_0·DB_TO_NP·100` with a
5% threshold. An unconverted getter is low by 100·ln(10)/20 = 11.5129×, a
relative error of ≈0.91, so the validator detects the skew rather than
absorbing it. Verified Np/m consumers of the getters are the time-reversal
amplitude path, the optical and Kuznetsov diffusion operators, and the
elastic-nonlinear path.

This record is about units; coefficient ownership and precedence are
ADR 120 and unchanged here.

## Decision

1. The stored field stays α₀ at 1 MHz in dB/(MHz^y·cm), documented as such.
2. Every Np/m getter converts through `DB_TO_NP × 100`, matching the Szabo
   tissue implementor.
3. `PowerLawAbsorption` is the reference evaluation; getters must agree
   with it per medium.
4. The water acceptance test below is the regression oracle: a getter that
   stops converting fails it, and a dB-valued measurement fails its
   skew leg.

## Rejected alternatives

- **Store Np/m at the boundary:** breaks the raw-view writer contract and
  every serialized medium; migration cost with no physical gain.
- **Widen the validator threshold:** masks an 11.5× physical error as
  noise; the 5% threshold covers model spread, not unit skew.
- **Per-crate conversion factors:** recreates the `0.1151`/`8.686`
  duplicate family; one canonical path through `DB_TO_NP`.

## Consequences

- The dB/Np duplicate family stays closed: every conversion routes through
  the canonical constants, enforced by the census of `DB_TO_NP` users.
- Consumer tests that pinned dB values must be re-derived in Np/m, never
  re-toleranced.

## Overturning evidence

- A getter returning values inconsistent with `PowerLawAbsorption`
  beyond 5% for any shipped medium.
- A consumer proving Np/m-at-the-boundary breaks a validated pipeline.
