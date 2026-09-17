# ADR 132: One lockstep field traversal, in kwavers-core

- **Status:** Accepted
- **Date:** 2026-09-16
- **Board item:** [KW-LOCKSTEP-TRAVERSAL](../../backlog.md#kw-lockstep-traversal)

## Context

Element-wise kernels that write one to three fields from other fields of the
same shape went through three independent adapter families:

| Family | Surface | Pairing | Fallback | Task width |
|---|---|---|---|---|
| `kwavers-core::utils::iterators` | public, 1 output, indexed | logical (`as_slice`) | logical loops | per element |
| `kwavers-physics::parallel` | crate-private, 1–2 outputs, 0–4 inputs, indexed and not | **memory order** | `if let Ok(..)` | 1024 elements |
| `kwavers-therapy::parallel` | crate-private, 1–3 outputs, 1–5 inputs, any rank | logical (`as_slice`) | logical walk | 4096 elements |

The physics family paired `as_slice_memory_order` slices. Leto returns that
slice for a C-dense *or* F-dense layout, so a C-ordered output zipped with a
transposed input paired the wrong elements, and its indexed forms decoded a
memory position as a C-order index, handing the closure wrong coordinates for
any F-dense output. Its fallback discarded the `indexed_iter_mut` error, so a
failure there wrote nothing and said nothing. The shape precondition of the
public core pair traversal was a `debug_assert`, absent in release.

Therapy depends on physics, and both depend on core; each family re-derived
the same traversal with a different arity set and a hand-picked chunk size.

## Decision

`kwavers_core::traversal` owns the traversal: `zip_mut`, `zip_mut_pair` and
`zip_mut_triple`, each with an `_indexed` form, generic over rank `N`, taking the
read-only views as a `ZipInputs` value — `()`, one view, or a tuple of two to
five views — so one function serves every input arity and the closure
destructures what it reads.

- Pairing is by logical row-major position. The dense path requires every
  field to be C-dense (`as_slice`); any other layout takes the logical walk.
- Dense work runs as moirai unit tasks whose width follows the bytes one
  element moves (outputs plus `ZipInputs::UNIT_BYTES`), as ADR 0059 in moirai
  prescribes, replacing the 1024- and 4096-element chunk constants.
- Each output count has one implementation, the indexed form; the plain
  form drops the index in a wrapping closure, which is dead arithmetic once
  the closure inlines.
- Shape agreement is an `assert!` with `#[track_caller]`.
- A mutable view that cannot enumerate its elements is an invariant
  violation (`expect`), never a silent skip.

The three families and `utils::iterators::{for_each_indexed_mut,
for_each_indexed_pair_mut, apply_inplace}` are deleted and every caller moves
to the new module.

## Alternatives

- **Keep per-arity functions** (`zip_mut_two_refs`, `zip_mut_three_refs`, …):
  the arity is a variation dimension, and ten named functions per crate is how
  three families drifted apart. Rejected.
- **A generic associated type for the input references**
  (`type Refs<'r>` with a higher-ranked closure bound): the `where Self: 'r`
  bound forces `'static` inputs under the current trait solver. The inputs
  borrow for one fixed `'a`, so a plain associated type expresses it.
- **Upstream in leto:** leto owns array traversal, and a parallel lockstep
  zip is plausibly its capability. The unit-task sizing policy and the
  moirai dependency are already leto's, so this remains the natural next
  home; it is not done here because no other stack member needs the
  multi-output form yet. Re-open when a second member writes one.
- **Memory-order pairing with a layout-equality check:** would keep F-dense
  fields on the dense path, but no caller passes one; the logical walk is
  correct for them and the check would be a second path to test.

## Consequences

- [major] for `kwavers-core`: `utils::iterators::{for_each_indexed_mut,
  for_each_indexed_pair_mut, apply_inplace}` are removed; the CHANGELOG
  names their replacements. Indexed closures receive `[usize; N]` instead of
  `(usize, usize, usize)`.
- F-dense or strided fields run serially. Every current caller passes
  C-dense fields, so no kernel changes path.

## Verification

`kwavers-core/src/traversal/tests.rs`: a transposed input and transposed
outputs pair by logical index, every input arity and output count against an
index-coded oracle, the dense indexed paths against the logical index, the
shape assertion, and the empty field. The same two transposed cases, run as a
probe against the physics family before its removal, misplaced 13 248 of
13 824 elements and wrote wrong coordinates. Injecting an off-by-one into the
dense path, and a one-element skew into the logical walk, each fails a test.
The migrated kernels' own suites run unchanged.

Adapter A/B, old physics family against the new functions, alternating in one
loop, fastest repeat of each, host load 10%: 16 cubed 7.1-8.2 us to
1.0-1.9 us for every form (a 4K-element field no longer splits into 1024-element
tasks); indexed 3.7-4.2x faster at 64 cubed and 2.8-3.5x at 128 cubed;
three inputs 11-14% and 5-17% faster; one input equal at 64 cubed and 5-9%
faster at 128 cubed.
