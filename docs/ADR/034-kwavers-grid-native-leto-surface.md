# ADR 034 — Kwavers-grid native Leto surface

- **Status:** Accepted
- **Date:** 2026-07-10
- **Change class:** [arch] / [major]

## Context

`kwavers-grid` now uses Leto for its owned arrays and views, but retains a
`compat` module that re-exports Leto under both `ndarray` and `leto` namespaces.
It also publishes `_leto` forwarding methods beside canonical operations that
already accept or return Leto types. These parallel surfaces duplicate one
implementation contract, preserve obsolete provider terminology, and allow
future call sites to diverge.

Workspace source inspection finds no consumers of the operator, wave-number,
or field-construction `_leto` methods. The canonical view-based operators are
the zero-copy boundary: callers borrow an owned Leto array with `.view()` and
the operation allocates only its declared output.

## Decision

Delete the compatibility module and every redundant `_leto` forwarding API in
`kwavers-grid`. Import Leto types directly at their canonical operation homes.
Retain one name per operation, update all in-workspace callers in the same
change, and remove identity conversions left by the completed provider swap.

No compatibility alias or deprecation bridge is retained. The working branch
is the migration boundary, and every workspace consumer is converted before
the change is committed.

## Consequences

- Leto ownership is explicit in imports and public signatures.
- Each grid operation has one implementation and one public name.
- Array views remain borrowed and allocation-free; owned outputs retain their
  existing allocation contract.
- External callers of the deleted transitional names must use the canonical
  operation and pass `.view()` where the signature requires a view.

## Verification

- Static source audit: no `kwavers-grid::compat`, `crate::compat`, or redundant
  `_leto` grid API remains.
- `cargo clippy -p kwavers-grid --all-targets -- -D warnings`.
- `cargo nextest run -p kwavers-grid` and `cargo test --doc -p kwavers-grid`.
- `cargo doc -p kwavers-grid --no-deps` with warnings denied by the workspace.
