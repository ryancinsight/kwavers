# 107. Summation by parts for the collocated rigid wall

- Status: Accepted
- Date: 2026-08-13
- Item: `backlog.md#kw-sol-086`
- Follows [106](106-rigid-walls-by-even-reflection.md), which fixed the staggered path and recorded this as explicitly not covered

## Context

ADR 106 gave the staggered path rigid walls by reflecting taps, so a uniform
field has zero gradient and a thin slab stays a 1-D line rather than becoming a
soft-walled waveguide. It recorded that the collocated path could not take the
same fix, and why.

The obstruction is structural. A collocated leapfrog uses **one** operator for
both the gradient and the divergence, so its energy behaviour is governed
entirely by that operator's symmetry: it conserves exactly when `Dᵀ = −D`. A
skew-symmetric matrix has a zero diagonal by definition. Reflection folds
`f[−1] = f[0]` back onto row zero, which puts a non-zero entry *on* the
diagonal. Reflection and conservation are therefore in direct conflict on a
collocated grid — the same obstruction one-sided closures hit, and the reason
the previous closure was zero-extension, which is conservative but is a
*pressure-release* wall.

## Decision

Replace the skew-symmetric operator with a **summation-by-parts** operator:
a positive diagonal norm `H` and a `Q` with

```text
  D = H⁻¹Q     and     Q + Qᵀ = B = diag(−1, 0, …, 0, +1)
```

and impose `u = 0` on the wall-normal component at each pair of outer faces.

The energy identity becomes

```text
  d/dt ‖E‖_H = −pᵀ(Q + Qᵀ)u = −pᵀB u = −( p_{N−1}u_{N−1} − p₀u₀ )
```

which vanishes under that wall condition. The demand that `D` be skew-symmetric
in the *plain* inner product is dropped; conservation moves into the weighted
norm the operator carries.

The boundary blocks are **derived at construction** from the accuracy and
symmetry conditions, not transcribed from the literature.

## Consequences

- A uniform field has zero derivative at every order and every extent, so a thin
  transverse axis is inert. This is the property the whole exercise is for, and
  it now holds on both discretisation paths.
- The wall is rigid, matching the staggered path. The two are no longer
  physically different domains.
- **The conserved energy must be measured in the `H` norm.** `H` is the
  trapezoidal weight — half at the end points, one inside — so this is a
  *better* discretisation of the energy integral than the unweighted sum it
  replaces, but code that sums energy without weights will see drift and be
  wrong about why. `norm_weight` exists for exactly this.
- The rigid wall is not decoration on a conservative scheme; it is the half of
  it that lives at the boundary. Removing it removes conservation.
- Diagonal-norm SBP caps boundary accuracy at half the interior order, so global
  order is `m + 1` rather than `2m`. This is the known cost of keeping `H`
  diagonal, which is what keeps the energy a weighted sum instead of requiring a
  norm solve every step. Interior accuracy is unchanged.
- An axis shorter than its boundary block falls back to the highest order that
  fits, down to an inert zero operator for a single point. That is an accuracy
  reduction confined to an axis that cannot resolve a wave anyway, and it is
  what lets legitimate quasi-1-D grids work at all.

## Derived, not transcribed

The standard blocks are published (Strand 1994), and copying them would have
been less work. It was rejected because a transcribed table is unverifiable at
the call site: a mistyped rational is silently wrong, and nothing downstream can
tell. Deriving from the defining conditions means the result can be checked
against them, and it is — `Q + Qᵀ = B` on the assembled matrix, exactness on the
polynomials each row claims, positivity of the norm, and the order-2 case
against its textbook closed form. Construction fails loudly if the derived block
does not satisfy the conditions it was solved from, so a bad solve cannot ship
as an operator.

This also matches how every other stencil in this codebase is obtained: the
central and staggered coefficients are Vandermonde solves, not tables.

The one subtlety is that the conditions are over-determined but consistent, so
the solve is least-squares and would happily return a plausible vector for an
*in*consistent system. That is why the residual is checked rather than the solve
trusted, and why `dense_solve` has a test asserting that an inconsistent system
does not look solved.

## Alternatives rejected

**Keep zero-extension and accept a pressure-release wall on this path.** It
leaves the two discretisations modelling different domains, and leaves every
quasi-1-D collocated grid silently wrong. Rejected: this is the defect, not a
trade-off.

**Delete the collocated path.** It is not the default, the staggered path
dominates it on Courant limit and accuracy, and removing it would have been the
cheapest way to close this item. Rejected because it is used, and because "we
could not give it a correct wall" is not a reason to remove a discretisation —
it is a reason to give it one.

**Non-diagonal (block) norms**, which reach full `2m` boundary accuracy.
Rejected: the energy stops being a weighted sum, every conservation check needs
a norm application, and the accuracy gain is at the boundary of a domain whose
wall is the thing being modelled rather than resolved.
