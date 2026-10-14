# Prepared Born objective

The inversion holds acquisition geometry, voxel shape/spacing, observation
frequencies and reference sound speed fixed. Its model and finite-difference
Hessian probes change slowness. Therefore the outgoing reference Green fields,
incident fields and direct source-to-receiver pressures are invariant during
one solve. `Objective` owns those coefficients and evaluates every model through
the same forward/adjoint kernels. Public operator interfaces remain stateless.

Preparation calls the original Green and incident equations with their original
arguments and summation order. Receivers share a field only when all three SI
coordinate bit patterns match. Each transmit retains its receiver mapping, so
rotating acquisitions do not inherit ring symmetry. Reusing these coefficients
preserves their bits. Model-dependent scattering potentials, source fitting,
predictions, residuals, gradients and Hessian differences remain recomputed.
Scalar volume factors and source-scaled incident fields are hoisted only across
loops in which their operands are invariant; multiplication grouping remains.
Forward pressure and the adjoint use one native-precision weighted Born kernel.
It reads Copy scalar fields directly and computes only the required real
adjoint projection, preserving the original complex expression tree, signed
zero and left-fold order. It introduces no precision-changing conversions.
The cubic voxel measure is the shared native expression `(h*h)*h`. Rust's
general integer-power operation has unspecified precision; an isolated-Miri
regression reproduces a four-ULP difference for the existing 1 mm spacing.
Two IEEE products give each model/Hessian probe the same spatial measure,
without disabling Miri's extra-rounding checks or changing numerical tolerances.
Checked const-sized blocks copy scalar field arrays once per block; one block
covers a 64-byte native FD scalar row. Block arity only partitions traversal.
The complete voxel sequence and addition order remain, with the same kernel
handling the tail. Contracts exercise arities 1, 4, 8 and 16 and mixed full/tail
shapes. Forward prediction prepares the same geometry within one call;
its public operator remains stateless. For T transmits, R receivers per row
and U distinct receiver positions, receiver Green evaluation falls from TRV
to UV without changing any source or receiver selection.
The adjoint keeps each gradient block local while sweeping the receivers in
their original order. Independent voxels interleave differently, but each
scalar expression and per-voxel receiver reduction remains bitwise identical.
Batch-versus-sequential differential tests cover full blocks and tails.

For each frequency, V voxels, T transmit rows, U distinct receiver positions
and R receivers per row require `(T + U)V + TR` complex coefficients and TR
receiver indices. This immutable workspace replaces repeated Green evaluation;
model-dependent scratch remains O(V). For the unchanged two-frequency 12×12×1,
16-transmit/16-receiver ring, coefficients occupy 155,648 bytes and receiver
indices occupy 4,096 bytes, excluding collection metadata. No Jacobian is
assembled. CBS and finite-window operators use their existing volume adjoints.

The alternatives are repeated trigonometric evaluation on every Hessian probe
or storing a full transmit×receiver×voxel Jacobian. Preparation stores separate
incident and receiver fields and leaves the model-dependent action matrix-free.
It neither changes the finite-difference step nor replaces the Hessian formula.
The objective rejects changes to its prepared shape and validates each model.
If a Hessian probe projects onto the exact current Born model, the prepared
coefficients and IEEE polynomial arithmetic make its gradient repeatable.
For finite current gradients and finite positive epsilon,
`(g(m) - g(m)) / epsilon` is exactly positive zero. A bitwise model comparison
permits that identity; other operators, changed models and nonfinite inputs
retain the full gradient evaluation. The current model/gradient pair is a
private consumer invariant. A bound-directed probe and an inward probe compare
every result bit against the complete finite-difference computation.

Verification uses bitwise Green/incident/direct comparisons for real ring and
rotating acquisitions, an exact dyadic adjoint oracle, and model updates compared
against stateless forward pressures with both source-fitting modes. Separate
pressure/geometry regressions compare against the uncached equations. Existing FD,
inversion and full reconstruction fixtures retain their inputs and tolerances.
The generic kernel has exact dyadic contracts in four native precisions and
bitwise complex-operation comparisons for non-dyadic values, subnormals and
signed zero.
Full Miri cases retain isolation, safety checks and the 15-minute execution cap.
