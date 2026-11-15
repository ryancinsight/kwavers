# ADR 138: Kwavers release scope for 4.0

## Status

Accepted (2026-11-12)

## Context

Kwavers (workspace crates mostly 3.0.0: solvers, imaging, medium, GPU PSTD,
diagnostics, and Python bindings over the Atlas provider stack) is the
stack's wave-simulation and imaging provider. 63 open items remain
across solver correctness, GPU PSTD parity, aperture and imaging capability,
medium assembly, and CI/book hygiene (the `backlog/*.md` files are the
source of truth; the `backlog.md` index is stale and tracked by
KW-BOARD-OFFSET-STATUSES), and no milestone ladder exists to the
next stable release.

## Decision

Adopt a three-milestone ladder to 4.0 recorded in `backlog/milestones.md`:

- **M-solvers-core** (3.1.0): solver and GPU PSTD core verify — PSTD
  output/dispatch honesty, FFT lattice, adjoint-reciprocity, unit-exponent
  parity, host-run GPU tests, `kwavers/src` teardown, manifest reduction,
  FWI L-BFGS ownership, Apollo axis FFT, FDTD layout, and provider floors.
- **M-imaging-capability** (3.2.0): imaging capability closes — aperture
  propagation (001/002/003), clinical-imaging boundary, Asclepius response,
  multimodal fusion, Doppler, I/Q ensemble, CT medium assembly, coupling
  contract, focus path, and the Apollo provider lock.
- **M-4.0** (4.0.0): stable simulation release (authority-gated) with
  book/oracle gates, clippy/edition/docs hygiene, and CI matrix health.

In scope: solver kernels, GPU PSTD, aperture/imaging/diagnostic capability,
medium assembly, book chapters, lint and CI hygiene. Out of scope: the
permanently-red recurseml check (KW-CI-094, owned by tooling, non-ladder);
prose-only PR plumbing, crate publication, and thermal-diffusion typing
(blocked items, outside the ladder); any `publish = true` without release
authority (veto point: M-4.0 never forces publication).

## Alternatives

- Plan 4.0 as a single release: rejected because solver-parity and
  imaging-capability slices remain open across many crates.
- Keep roadmap implicit in backlog items: rejected because stack policy
  requires a runnable capability-oriented milestone ladder.

## Consequences

- Open items carry `milestone:` assignments in their `backlog/*.md` files
  and name the exit check they serve.
- Milestones close only when exit checks pass on fetched default commits;
  4.0 release/publication remains authority-gated.
- This ADR is rewritten in place as scope evidence changes.
