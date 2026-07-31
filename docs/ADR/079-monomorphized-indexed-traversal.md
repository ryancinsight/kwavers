# ADR 079 — Monomorphized indexed traversal and gradient entrypoints

- Status: Accepted
- Date: 2026-07-31
- Board item: `KWAVERS-GEN-01`

## Context

The workspace carried several indexed traversal functions whose bodies differed
only by the number of immutable source views: one-source, pair, three-source,
and four-source variants. The grid operators also carried two copies of the
same centered finite-difference gradient kernel, with the optimized copy
adding only optional coefficient and spacing caches.

These variants duplicate shape/storage handling and allow arity-specific APIs
to drift. The provider already exposes `ZipSources<N>`, whose tuple
implementations preserve heterogeneous source types and compile-time dispatch.

## Decision

Use one generic `for_each_indexed_mut_with<T, S, F>` over `S: ZipSources<3>`
for indexed mutable traversals. Keep the dense Moirai traversal and strided
fallback in the canonical core implementation. Migrate all workspace callers
to source values or source tuples and remove the pair, three-source, and
four-source entrypoints.

Use the cache-aware `gradient_optimized` kernel as the sole centered-stencil
implementation. The public uncached `gradient` entrypoint delegates with no
cache, while cached callers retain the same optional-cache path.

The type parameters and source tuple arity are resolved at each callsite, so
the abstraction has no runtime type erasure or per-element dynamic dispatch.

## Consequences

- In-repository callers use one API for all indexed source arities.
- The public `kwavers-core` pair helper is removed; external consumers must
  migrate to `for_each_indexed_mut_with` in the next major API release.
- The gradient implementation has one source of truth while preserving both
  the uncached convenience and cache-aware operator contracts.
- Generic instantiations remain limited to the source tuple shapes actually
  used by the workspace.

## Verification

The generic traversal is covered by value-semantic tests for one source and a
heterogeneous two-source tuple. Package format, compile, lint, and focused
nextest gates are required for the affected core, grid, physics, analysis,
solver, and therapy packages.
