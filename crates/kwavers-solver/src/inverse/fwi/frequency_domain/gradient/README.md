# Prepared Born objective

An inversion fixes acquisition geometry, voxel shape/spacing, frequencies and
reference sound speed while changing slowness. `Objective` owns the immutable
Born coefficients and evaluates every model through shared native-precision
weighted integration and adjoint kernels. Public operators remain stateless.

## Mathematical and rounding contracts

Incident fields I, receiver Green fields G and direct source-to-receiver
pressures depend on the fixed experiment. Preparation evaluates their original
equations with the original arguments and summation order. Receiver sharing
requires equality of all three SI-coordinate bit patterns; each transmit keeps
its own receiver mapping, including rotating acquisitions.

The weighted product remains `(I * G) * weight`, with the original complex
expression tree. A derivative uses `(source_scale * I) * G` before weighting.
The projection remains
`p = derivative.re * residual.re - (-derivative.im) * residual.im`.
Its const policy either assigns p or performs exactly one `old + p` addition.
Per-voxel receiver order and the objective's left fold remain unchanged.
All arithmetic executes in the generic scalar precision, without conversions.

The shared cubic voxel measure is `(h*h)*h`. A repeatability regression
reproduces the original four-ULP discrepancy from general `powi(3)` under Miri.
Two IEEE products give every model/Hessian probe the same spatial measure;
existing numerical tolerances remain unchanged.

## Bounded coefficient representations

The sealed `FieldProducts<T>` trait selects factored incident/Green slices or
prepared spatial products through static dispatch. Both use the same weighting,
integration, projection, source-fitting and residual reductions. Preparation
calls the same canonical product expression as the factored representation.
The immutable products are spatial coefficients; model weights, predictions,
source fits, residuals, gradients and Hessian differences remain recomputed.

For a fixed unit source scale, preparation retains both `I*G` and `(1*I)*G`.
Keeping both expression trees preserves signed-zero behavior. Source fitting
uses the complete factored calculation because its scale changes with the
model. Products that are nonfinite or exceed the budget also retain the complete
factored calculation, preserving IEEE propagation and every selected voxel.

The per-inversion budget B is 8 MiB of requested expanded vector allocations.
For F validated nonempty frequency observations, each receives floor(B/F)
bytes. Therefore their sum cannot exceed B. A frequency requests
`T*sizeof(ProductRow) + 2*V*sum(R_t)*sizeof(Complex64)` bytes: row metadata plus
the two complete product arrays. Checked arithmetic rejects an unrepresentable
layout. Vectors reserve exactly their complete row sizes; three fixed complex
blocks (12 KiB) serve preparation without voxel-sized temporary allocations.
Allocator bookkeeping and existing per-frequency collection metadata are
excluded. Repreparation drops the previous expanded allocation first.
Changing B changes representation only; bitwise equivalence defines acceptance.

The factored geometry stores `(T+U)V + sum(R_t)` complex values and sum(R_t)
receiver indices, where U counts distinct receiver positions. For the unchanged
two-frequency 12×12×1, 16-transmit/16-receiver ring, those payloads occupy
155,648 and 4,096 bytes. Expanded products request 2,361,088 bytes on the tested
64-bit layout, including row metadata. Model-dependent scratch remains O(V+R).
No model-dependent Jacobian or Hessian is retained. Larger experiments keep
the factored representation under the same memory budget.

Alternatives are repeating all spatial products for every gradient/Hessian
probe, or allocating an unbounded expanded operator. Bounded preparation removes
only invariant work and retains the existing factored algorithm for all sizes.
The trait and row callbacks keep representation separate from model evaluation;
no algorithm body is cloned per layout or scalar precision.

## Native scalar arithmetic lowering

One Rust operation specification supplies product, weighting, complex accumulation
and real projection bodies for each supported scalar. The dependency-free build
script places those bodies inside concrete scalar implementations before Rust
typechecks them. Required trait methods prevent an implicit generic operator
implementation. Rust 1.97 MIR inspection confirms primitive arithmetic for the
double-precision operations, with the original multiply/subtract/negate/add order.
This changes operator forwarding; every floating-point operation still executes.
The implementation adds no casts, unsafe code, compiler pin, optimization flag
or interpreter exception. The generated source is a normal Cargo build output;
source/lock/compiler/profile provenance and the wheel build-script handoff apply.

The adjoint weights each spatial coefficient and projects it directly into its
existing accumulator in one traversal. Concrete methods own the complete checked
weighting, accumulation and projection block loops: one call per block replaces
per-voxel arithmetic dispatch. Each loop visits exactly the valid prefix, retaining
the original intermediate values, scalar expression trees and addition order.
The adjoint does not write and reread an intermediate weighted complex buffer.
Rust 1.97 MIR confirms the primitive operations inside these block bodies.

## Checked traversal and exact projected probes

Checked valid-prefix blocks reuse caller-owned storage. Equal input lengths,
range bounds and output capacity are checked before arithmetic. Only overwritten
lanes are exposed; spare capacity never contributes. The 32 KiB active-working-
set budget counts nine native FD scalars per voxel; its largest fitting power
of two is 256. Correctness is independent of cache size and block capacity.
For block b, `b*K <= V` and `b*K + len <= V`, so matching full field/model
lengths bound every projection, including the final partial block.

Each loop retains its immutable length and visits exactly `[0, len)` with a
counter. A 180-second full-fixture diagnostic at `0250d5af` attributes 16.499
inclusive seconds to 4,558,590 repeated slice-length queries. Hoisting metadata
and combining projection accumulation removes dispatch, a temporary buffer and
a second traversal while preserving each scalar expression and addition.

If a Hessian probe projects onto the exact current Born model, repeatable
polynomial arithmetic gives `(g(m)-g(m))/epsilon = +0` for finite gradients and
finite positive epsilon. Bitwise model equality permits that identity; changed
models, other operators and nonfinite inputs retain full evaluation. The
current model/gradient pair is a private consumer invariant. Bound-directed and
inward probes compare every result bit against the full difference formula.

## Verification and remaining gates

Exact dyadic/native-precision contracts cover four scalar precisions, capacities
1/4/8/16/256, nonzero offsets, empty/partial/full blocks and untouched spare
storage. Bitwise comparisons include non-dyadic values, signed zero, subnormals,
assignment and accumulation. Real ring/rotating coefficients and heterogeneous
pressures match uncached equations. Budget tests cover zero, one byte below,
exact capacity and nonfinite propagation. Prepared/factored objectives and every
gradient value agree across model changes and both source-fitting modes;
shape drift is rejected. Existing FD and full reconstruction fixtures retain
all inputs, stopping rules and tolerances. Isolated/full Miri retains default
safety checks and the existing 15-minute execution cap. Required final-head CI
and actual installed-wheel/k-Wave/benchmark checks remain mandatory.
