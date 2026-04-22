# Scientific Acceptance Contract

This repository uses one scientific completion contract across `kwavers`, `pykwavers`,
`apollo`, `gaia`, and `ritk`.

No retained production-facing algorithm, workflow, or public API is complete until every
required item below exists and passes review.

## Canonical Rule

A canonical implementation may not be:

- a placeholder
- a mock or stub
- a silent approximation without documented validity limits
- a heuristic fallback without explicit regime checks
- a TODO-marked incomplete physics path
- a duplicate owner of functionality already canonical elsewhere

If an implementation cannot satisfy this contract yet, it must remain outside the canonical
public surface and be treated as experimental.

## Required Evidence

1. Governing specification
   State the governing equations, transform definition, optimization problem, or constitutive
   model precisely.

2. Algorithm specification
   Document the implemented algorithm, numerical realization, stopping criteria, normalization
   conventions, and update equations.

3. Assumptions and validity regime
   Record boundary assumptions, medium assumptions, sampling constraints, stability restrictions,
   confinement conditions, observability limits, and breakdown cases.

4. References
   Cite primary literature, authoritative external references, or official reference
   implementations used to derive the method.

5. Failure modes and invariants
   Record unit constraints, shape constraints, admissible parameter ranges, positivity or
   conservation requirements, and expected failure cases.

6. Deterministic tests
   Add unit and integration tests covering nominal, edge, and failure paths.

7. Numerical validation
   Add analytical checks, convergence studies, conservation checks, or property-based invariants
   where appropriate.

8. External validation
   Define the authoritative baseline:
   - `kwavers`: analytical solutions, published studies, or `external/k-wave-python`
   - `pykwavers`: parity against canonical `kwavers` Rust APIs and `external/k-wave-python`
   - `apollo`: parity against RustFFT, NumPy, or backend cross-checks
   - `gaia`: geometric validity and mesh-orientation correctness
   - `ritk`: literature, public datasets, and multimodal registration baselines

9. Performance evidence
   Add or update benchmarks for hot paths and record throughput targets and comparison context.

10. Memory evidence
    State workspace ownership, scratch-buffer reuse policy, and expected steady-state allocation
    behavior.

## Completion Gates

`Gate 0`: Inventory and source-of-truth audit  
`Gate 1`: Architecture and public interface definition  
`Gate 2`: Implementation completion  
`Gate 3`: Scientific validation against literature, parity, or datasets  
`Gate 4`: Performance and memory closure

Anything that fails a gate must not be represented as complete in docs, release notes, or public
API summaries.

## Public API Rule

Public APIs may remain only if they satisfy this contract or are clearly isolated as experimental.
Silent placeholders, undocumented approximations, and misleading completion claims are forbidden.

## Evidence Packaging

For each retained solver, optimizer, transport model, reconstruction workflow, or binding
surface, provide the following as applicable:

- `Config`
- `Scenario`
- `State`
- `Workspace`
- `ValidationCase`
- `BenchmarkCase`
- scientific metadata or references
- parity references when external baselines exist

This is the minimum contract for future development, review, and release claims.
