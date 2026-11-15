# ADR 0138: Advance the Eunomia and Moirai provider contracts

Status: Accepted  \
Date: 2026-11-15  \
Change class: [arch] [major]  \
Board item: [ATLAS-VERSION-COHERENCE-SWEEP-2026-09-30](../../backlog.md)

## Context

`kwavers-math` re-exports Eunomia's complex types (`Complex32`,
`Complex64`) and the workspace resolves its data-parallel dispatch
through Moirai's parallel executor. The stack's provider cohort has
moved past both lines:

- Eunomia 0.12.0 is the current provider head. Rust treats the
  `Complex64` type of Eunomia 0.11 and the same-named type of Eunomia
  0.12 as distinct nominal types, so a consumer that names the
  re-exported type while depending on Eunomia 0.11 directly resolves two
  distinct types for the same name, and the mixture stops compiling.
- Moirai 0.12.0 removes the required `StateWord::get_mut` method.
  Kwavers implements no `StateWord` and calls no removed method, so no
  Kwavers source changes with the move.

The requirement also gates the rest of the stack. The providers' current
default heads require Moirai 0.12 (`apollo-fft` 0.29.5 and
`coeus-core` 0.12.0 name `moirai-runtime ^0.12.0`), so every consumer
that resolves those providers with a `^0.11.0` requirement cannot
re-resolve onto the current provider heads at all.

Two alternatives were considered:

1. Stay on Moirai 0.11 and keep the current lock. The provider cohort
   then splits across two Moirai identities inside one workspace, and
   the consumers that resolve Kwavers inherit the same split.
2. Require `>=0.10.0, <0.13.0` for Eunomia and let the lock choose. The
   range admits both provider lines, but the lock selects the newest
   compatible revision anyway, so the re-exported types move exactly as
   they do under the selected option. The wider range only hides the
   move from the requirement.

## Decision

Move the Moirai-parallel requirement to 0.12.0 and widen the Leto and
Leto-ops requirements to `>=0.47.1, <0.49.0` so the 0.48 line resolves.
The Eunomia requirement already admits the 0.12 line; the lock selects
it. Regenerate the lock outside the Atlas dependency overlay, so every
provider row names one revision per provider repository and the
`source = "git+..."` rows survive.

No Kwavers source changes with the move. The removed provider method is
unused in this repository.

## Consequences and migration

- Code generic over the Eunomia 0.11 `Complex64` type no longer
  satisfies Kwavers's public bounds. A consumer updates its Eunomia
  imports to 0.12 and selects the same provider cohort before it uses
  Kwavers's public generic types. Consumers already using concrete
  scalar types keep the same Kwavers API after their dependency graph
  selects the new cohort.
- The public contract change is a [major] change class; the release
  tooling assigns the version at release.
- No compatibility alias or duplicate complex type is added: two Eunomia
  identities under one name would keep the split the provider owns.
