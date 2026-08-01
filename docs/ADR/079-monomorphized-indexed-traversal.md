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

Use Leto's `indexed_map_inplace` for indexed mutation of one output and
`indexed_zip_mut_with` for indexed mutation with read-only sources or multiple
mutable outputs. Leto owns view validation, logical row-major traversal, and
the `ZipSources<N>` tuple family. Moirai remains the execution provider for
operations that explicitly select a parallel policy; this zip cutover preserves
Leto's validated traversal semantics and makes no unmeasured runtime claim.
Kwavers does not retain a local indexed zip adapter.

Use the cache-aware `gradient_optimized` kernel as the sole centered-stencil
implementation. The public uncached `gradient` entrypoint delegates with no
cache, while cached callers retain the same optional-cache path.

The type parameters and source tuple arity are resolved at each callsite, so
the abstraction has no runtime type erasure or per-element dynamic dispatch.

## Consequences

- In-repository callers use one API for all indexed source arities.
- The public `kwavers-core` indexed zip helper is removed; consumers use the
  provider-owned `indexed_map_inplace` or `indexed_zip_mut_with` contracts.
- The gradient implementation has one source of truth while preserving both
  the uncached convenience and cache-aware operator contracts.
- Generic instantiations remain limited to the source tuple shapes actually
  used by the workspace.

## Verification

The Leto provider is covered by value-semantic tests for one, two, and three
outputs, zero sources, heterogeneous source tuples, dense and strided views,
and logical indices. Package format, compile, lint, and focused nextest gates
are required for the affected core, grid, physics, analysis, solver, and
therapy packages.
