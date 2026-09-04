# ADR 128: First-derivative stencils belong to Leto

- Status: Accepted
- Date: 2026-09-04
- Item: `backlog.md#kw-leto-fd-ssot`
- Related: [ADR 106](106-rigid-walls-by-even-reflection.md) (the wall closure this
  operator carries), atlas `docs/adr/0039-compute-substrate-topology.md`,
  atlas `docs/audit/math-ssot-ledger.md` rows 137-140

## Context

Kwavers carried its own first-derivative stencils in
`kwavers-math::numerics::operators::differential`: `CentralDifference2/4/6`,
the second-order Yee `StaggeredGridOperator`, and the arbitrary-even-order
`StaggeredLeapfrogOperator` with its own Fornberg coefficient derivation. Leto
carried the same central 2/4/6 kernels and a fixed-order staggered pair; CFDrs
had carried a third copy until it deleted `cfd-math/src/differentiation` in
favour of Leto.

The duplication persisted for a specific reason rather than by neglect. The
provider covered only fixed orders, and the FDTD solver runs
`config.spatial_order` up to 8. The one operator that could not be deleted was
the one the solver actually used, so nothing below it could collapse either.

## Decision

First-derivative stencils are Leto's. Kwavers holds none.

Leto gained the general case first: `leto_ops::StaggeredLeapfrog3D`, generic
over the scalar type, deriving its taps at any even order `2N`, `N = 1..=8`,
with the same reflection wall closure and the same scatter-the-transpose
divergence that make `D = -Gᵀ` true by construction (leto PR #169, merged at
`6548a00`). Only then could this deletion follow.

Consumers now bind:

| Was | Is |
| --- | --- |
| `CentralDifference2/4/6` | `leto_ops::FiniteDifference3D` + `FiniteDifference3DScheme` |
| `StaggeredGridOperator` | `leto_ops::FiniteDifference3D` staggered schemes |
| `StaggeredLeapfrogOperator` | `leto_ops::StaggeredLeapfrog3D` |
| `differential::Axis` | `leto_ops::Axis` |
| `staggered_first_derivative_coefficients` | `leto_ops::staggered_first_derivative_coefficients` |
| `central_first_derivative_coefficients` | `leto_ops::central_first_derivative_coefficients` |
| `DifferentialOperator` | deleted with its implementors |

`SummationByPartsOperator` stays in kwavers. It is not a stencil with a
boundary fall-back: its closure blocks are derived per axis against a norm, and
its contract is the discrete energy estimate rather than a pointwise truncation
order. Leto does not own that family, and inventing a seam for a single
consumer would be speculative generality. It now draws its interior
coefficients and its `Axis` from the provider.

## Consequences

Kwavers sheds 4,862 lines net. The three stencil families and their tests are
gone; what replaces them is a handful of import lines, because the provider
surface is one type parameterized by scheme rather than one struct per order.

Two contracts tightened at the boundary rather than passing through unchanged:

- `gradient_into` and `divergence_into` return `Result<()>`. The kwavers
  versions checked the destination shape with `debug_assert_eq!`, which is
  absent from a release build — precisely where a shape mismatch would read
  and write past the intended cells silently. The FDTD sweeps propagate the
  error with `?`.
- The operator is generic over the scalar type. Nothing in kwavers yet
  instantiates it at anything but `f64`, but the precision is no longer welded
  into the kernel, which is what a future mixed-precision FDTD would need.

A test that duplicated provider coverage was deleted rather than rerouted:
`fdtd::tests::test_finite_difference_coefficients` measured the 4th-order
stencil's RMS error against a sinusoid. Leto proves the same stencils *exact*
on polynomials of the matching degree, which is the stronger oracle, and
measures the order of accuracy at 2/4/6/8 besides.

The critical-path benchmark's per-order dispatch enum went with the per-order
structs it dispatched over. Its recorded baselines do not carry across: the
kernel under measurement is a different one.

## What this does not yet do

The call sites now reach one CPU implementation. They do not yet reach a GPU
one. Hephaestus owns device stencils and today exposes only a 2-D Laplacian
seam (`StencilOps`); Coeus binds Leto and Hephaestus as backends of one
`ComputeBackend` but has no finite-difference op at all.

The remaining path, in order: a `FiniteDifference3DOps<T>: ComputeBackend`
trait in `coeus-ops` with a `coeus-leto` implementation delegating to the
surface this ADR adopts; the 3-D axis-derivative family added to the
Hephaestus device seam and its per-vendor kernels; a `coeus-hephaestus`
implementation. Kwavers then binds `<B: ComputeBackend>` instead of the Leto
type directly, and the same sweeps monomorphize to either backend. That last
step is what makes this ADR's deletion worth more than a line count: with one
implementation there is one place to add the device path, and the third copy
in `kwavers-gpu`'s FDTD shaders becomes deletable too.
