# ADR 146: Qualify the Driver API independently of the simulation facade

- Status: Proposed
- Date: 2026-11-22
- Class: [major]

## Context

The published 5.0.0 facade, 5.1.0 Physics and Solver, 6.x layers and 7.x GPU crate have independent major versions. The published Driver is at 0.5.0. Its default feature is `io`; its optional `kwavers` feature selects the Transducer 6.x dependency. [ADR 138](138-release-scope-4-0.md) scopes an earlier simulation release. The [Driver architecture](../../crates/kwavers-driver/docs/ARCHITECTURE.md) and its crate checklist govern its implementation.

## Proposed decision

Select Driver's first stable surface from its current Rust exports, public error contracts, documented file formats and feature combinations. Inventory each surface against a real external caller or documented use. Changes needed before the freeze require the ordinary migration and review process. Preserve the other packages' released majors and compare each changed package against its own baseline.

## Alternatives and verification

A workspace-wide 1.0 stamp would move released public packages backward. Counting the facade's version alone would leave Driver's compatibility promises unspecified. Verify a fresh caller of the normalized Driver archive with default features off, default I/O, the Transducer feature, and both features together. Check documented outputs and errors, archive contents and the locked provider identity. Run API compatibility and applicable performance checks on the selected candidate. Obtain the Driver owner's scope approval before its stable publication.
