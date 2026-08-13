# 106. Rigid walls by even reflection, with the divergence defined as −Gᵀ

- Status: Accepted
- Date: 2026-08-13
- Item: `backlog.md#kw-sol-085`
- Supersedes the boundary decision recorded under `backlog.md#kw-sol-074`

## Context

`StaggeredLeapfrogOperator` closed its stencils by **zero-extension**: a tap
falling outside the grid contributed nothing. That was chosen under KW-SOL-074
because it makes `D = −Gᵀ` provable by a clean re-indexing argument — the sums
run over all integers and out-of-range taps drop from both sides — which is the
condition for a conserved discrete energy and was the fix for KW-SOL-081.

The argument is correct. What it did not ask is what wall it implies.

Zero-extension means `p = 0` half a cell outside the domain: a **pressure-release**
wall. So a transversely uniform field does not have a zero transverse gradient,
because the stencil sees the step down to zero at the wall. Measured on the
order-4 operator with `dy = 1e-4`, for a field of ones:

| `ny` | transverse gradient |
|---|---|
| 1 | `[-11250]` |
| 2 | `[0, -10833]` |
| 4 | `[-417, 0, 417, -10833]` |
| 8 | `[-417, 0, 0, 0, 0, 0, 417, -10833]` |

A singleton axis is not exempt. The consequence is that an `N × 4 × 4` slab —
the dominant quasi-1-D idiom in this crate's tests and models — stopped being a
1-D line and became a four-cell-wide soft waveguide. A purely axial packet
launched into one (transverse energy exactly zero at step 0) had **more energy
in transverse velocity than axial by step 150**, ran at roughly half speed, and
never coherently arrived: the far sensor of the KW-SOL-084 cross-path test read
`0.0000` where PSTD, periodic transversely, read `1.0008`.

Nothing regressed against what KW-SOL-074 verified. Energy conservation and
adjointness both held throughout. The gap was that its verification never asked
whether a thin slab stays inert.

## Decision

Close the gradient by **even reflection** — `p[−1] = p[0]`, `p[nx] = p[nx−1]`,
mirroring about the wall — and **define the divergence as `−Gᵀ`** rather than
writing it as its own stencil.

Only the gradient has a stencil. The divergence is the transpose applied
directly, which is why it scatters where the gradient gathers: each face sends
`∓cₙ` of its value to the two cells the gradient drew from, reflected indices
included.

## Consequences

- A uniform field has **exactly** zero gradient, at every order and every
  extent down to one cell. Thin slabs are inert again.
- The wall is `∂p/∂n = 0` — rigid, the conventional acoustic default and what
  every pre-KW-SOL-074 test assumed.
- `D = −Gᵀ` holds *identically* rather than because a closure argument works
  out, so energy conservation no longer depends on getting a boundary case
  right. This is strictly stronger than what zero-extension gave.
- The far velocity face vanishes on its own: at `i = nx−1` every tap pair
  becomes `p[nx+n−1] − p[nx−n]`, which reflection maps to zero. Where
  `StaggeredGridOperator` forced that face to zero as a separate step, it is now
  a consequence, so there is nothing to forget.
- The divergence scatters rather than gathers, which is the same asymptotic work
  but less friendly to vectorization. Not measured as a bottleneck; if it
  becomes one, the interior can gather on the shifted stencil and only the halo
  needs the scatter.
- Verification cost fell sharply. With slabs inert, the cross-path test runs at
  a transverse extent of 1 instead of 4 and shares one lossless reference across
  cases: **57 s → 2.1 s**, assertions unchanged. That the 1-cell grid reproduces
  the 4-cell answer is itself evidence the slab is inert.
- KW-SOL-084 unblocks, and lands tighter than it was before the regression: 3 %
  against the prescribed law and 4 % between the two paths, against 9.1 %.

## Alternatives rejected

**Keep zero-extension and reach for summation-by-parts operators.** SBP-SAT is
the general answer to high-order rigid walls and would work. It is a large
investment, and it buys physics we have no requirement for — nothing in scope
needs a pressure-release wall. Rejected on cost.

**Rigid walls by omitting the boundary face**, so `u` lives only on interior
faces and `u = 0` at the wall is enforced by having no degree of freedom there.
Exactly adjoint by construction at order 2, and it keeps thin slabs inert. But
at order 4 and above the near-wall rows still reach outside the grid and need a
reduced-order one-sided closure, which reintroduces the non-zero diagonal that
broke adjointness in the first place. Rejected: it solves the problem only at
the one order that was never in question.

**Fix the affected tests instead.** The failing measurement was correct; the
wall was wrong. Reshaping tests around it would have left every quasi-1-D model
in the crate silently mis-simulating.

## Not covered

The collocated path (`ConservativeCentralDifference`) has the same
pressure-release wall and **cannot take this fix**. Reflection folds `f[−1] =
f[0]` onto row 0, putting a non-zero entry on the diagonal, and a skew-symmetric
matrix has a zero diagonal by definition — so on a collocated grid reflection
and conservation are in direct conflict. Recovering a rigid wall there needs SBP
operators; filed as KW-SOL-086. The staggered path is the default and is the one
to use for quasi-1-D work.
