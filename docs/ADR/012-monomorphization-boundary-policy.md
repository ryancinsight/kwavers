# ADR 012 — Monomorphization & dynamic-dispatch boundary policy

- **Status:** Accepted (investigation complete; no code conversion warranted).
- **Date:** 2026-06-05
- **Change class:** [arch] decision (no API change)
- **Relates:** follows [ADR 011](011-workspace-crate-split.md) (the crate split that
  fixes the implementor distributions analysed here)

## Context

After the `kwavers-domain` decomposition we audited the workspace for
monomorphization / zero-cost-abstraction improvements (eliminating `dyn`/vtable
dispatch in throughput-critical paths, per the project standards). The audit was
evidence-driven; the findings rule out every candidate conversion.

### Findings

1. **The codebase is `f64`-concrete.** There is no `Scalar` trait, hence no
   fake-generics and no type-suffix identifier violations (the lone `from_u32` is
   a bit-unpack constructor, not a generic). The "generic over scalar precision"
   class of monomorphization work does not apply.

2. **The per-cell hot paths are already monomorphic.** The FDTD/PSTD time loops
   update `ndarray::Array3<f64>` fields with concrete `Zip` kernels. Material
   properties are **sampled into `Array3` once at construction** (`density_at`,
   `sound_speed_at`, … in a setup loop) — the time loop reads arrays, never
   `dyn Medium`. Boundaries are the concrete `CPMLBoundary`, not `dyn`. There is
   **no per-cell vtable dispatch anywhere.**

3. **Remaining `dyn` dispatch is O(1)/step or setup-time, not per-cell.**
   - `dyn Medium` (~169 sites): consumed at setup (sampling) via DIP; the solver
     depends on the `Medium` abstraction, not a concrete type. Per-step cost: none.
   - `dyn Source` / `dyn Signal`: sampled **once per source per step** (the scalar
     `amplitude(t)`), then applied to a concrete mask `Array3` per cell. Per-step
     vtable cost is O(num_sources), dwarfed by the O(cells) concrete kernel.
   - `dyn Plugin` (plugin-based solver): the explicit extension boundary.

### Candidate conversions, and why each is rejected

| Candidate | Verdict | Reason |
|---|---|---|
| `Medium` → `<M: Medium + ?Sized>` generics | **Net-zero, rejected** | Genericizable surface is **setup-only** (media are pre-sampled), so monomorphization yields no measurable runtime gain. ~169 signatures across 10 crates would gain a generic param purely for form. The per-step `dyn` it would *not* reach (see below). |
| `Source` → closed `SourceKind` enum | **Infeasible** | `Source` has 14 implementors split across `kwavers-source` (primitives) **and** `kwavers-transducer` (devices). A closed enum in `kwavers-source` cannot name `kwavers-transducer` types without a circular dependency — i.e. it would collapse the crate split from ADR 011. `Source` is a genuinely **open, cross-crate-extensible** set. |
| `Signal` → closed `SignalKind` enum | **Infeasible** | `Signal` has 22 implementors across **6 crates** (`kwavers-signal`, `-source`, `-transducer`, `-solver`, `-python`, facade). Same circular-dependency blocker; also open and cross-crate-extensible. |
| Convert the per-step dispatch (`Plugin::update(&dyn Medium)`, `Source`/`Boundary` trait methods) to generics | **Infeasible** | These are **object-safe trait methods** required for `Box<dyn Plugin>` / `Vec<Box<dyn Source>>`. Adding generic type parameters to a trait method makes the trait non-object-safe, breaking the heterogeneous collections that are the whole point. |

## Decision

**Retain the existing `dyn` boundaries.** They are the correct tool, matching the
standards' own sanctioned exceptions:

- `dyn Medium` — dependency inversion (solver depends on an abstraction); dispatch
  is setup-time, not throughput-critical.
- `dyn Source` / `dyn Signal` — **open, cross-crate-extensible** implementor sets;
  stored in heterogeneous collections (`Vec<Box<dyn Source>>`, `Arc<dyn Signal>`);
  type genuinely unknown at compile time at the storage site.
- `dyn Plugin` — plugin/extension boundary on a non-per-cell path.
- `dyn Boundary` — boundary-strategy abstraction; one dispatch per step.

No conversion is performed: every option is either net-zero (Medium), infeasible
without re-introducing circular crate dependencies (Source/Signal enums), or
blocked by object-safety (the per-step trait methods). Manufacturing the change
anyway would satisfy the rule's form while regressing the architecture or adding
churn for no benefit — which the standards explicitly prohibit.

The canonical `dyn` sites carry inline `// dyn:` justification comments (trait
definitions and trait-object storage fields) recording which sanctioned condition
applies; per-call `&dyn Trait` parameters inherit that rationale and are not
individually annotated.

## Consequences

- The hot paths remain zero-cost (concrete `Array3<f64>` kernels) — verified.
- Future genuinely-per-cell dynamic dispatch (if ever introduced) **is** a defect
  to convert; this ADR scopes the decision to the current architecture.
- If scalar-precision genericity (e.g. `f32`/`f16` solves) is ever required, that
  introduces a `Scalar` trait and re-opens monomorphization work under a new ADR.

## Verification

- Per-cell kernels confirmed to operate on concrete `Array3<f64>` (no `dyn` in
  `forward/{fdtd,pstd}` per-cell `Zip` bodies).
- Implementor-distribution evidence: `impl Signal for` spans 6 crates;
  `impl Source for` spans `kwavers-source` + `kwavers-transducer`.
- `cargo test --workspace` green (unchanged behaviour; documentation-only change).
