# ADR 031: Strong-regime inter-bubble coupling solver (direct + under-relaxed)

- Status: Accepted
- Date: 2026-06-19
- Change class: [major] (numerical robustness; opt-in coupling-scheme upgrade)
- Builds on: ADR 028 (explicit coupling), ADR 030 (self-consistent fixed point).

## Context

The self-consistent coupling (ADR 030) uses plain fixed-point (Jacobi) iteration on
the coupling field. Because the Keller-Miksis acceleration is **affine in the
driving pressure**, the self-consistent source strengths solve the linear system

```text
(I − D·G)·S = e ,   d_j = R_j²·∂R̈_j/∂p ,   G_jk = ρ/d_jk (k≠j) ,
                    e_j = c_j + d_j·p_ext,j ,  c_j = R_j²·R̈_j(0) + 2 R_j Ṙ_j² .
```

Plain fixed-point is just Jacobi on this system; it **converges only when the
spectral radius `ρ(D·G) < 1`** (weak/moderate coupling). For strong coupling (dense,
close, violently-pulsating bubbles) it diverges, and ADR 030 only caps the
iterations (best effort).

## Decision

Replace the `implicit_coupling: bool` (boolean-blindness) with a
**`CouplingScheme`** enum and add two robust strong-regime options:

```rust
pub enum CouplingScheme {
    Explicit,                                   // ADR 028 (lagged single pass)
    ImplicitFixedPoint { under_relaxation: f64 },// ADR 030 + damping ω ∈ (0,1]
    ImplicitDirect,                             // ADR 031 (this) — exact linear solve
}
```

1. **`ImplicitDirect`** — extract each cell's affine coefficients `(c_j, d_j)` with
   two Keller-Miksis acceleration evaluations (`R̈(0)`, `R̈(P_ref)`; exact since the
   forcing is linear), assemble `M = I − D·G` and `e`, and solve `M·S = e` with the
   validated `kwavers_math::linear_algebra::LinearAlgebra::solve_linear_system`
   (Gaussian elimination). The result is the **exact** self-consistent solution,
   robust regardless of `ρ(D·G)` (provided `M` is non-singular). The coupling field
   is `p_couple = G·S`. On a singular/ill-posed `M` it falls back to an under-relaxed
   fixed-point (surfaced, not silent).
2. **Under-relaxation** in `ImplicitFixedPoint`: `p_couple ← (1−ω)·p_couple_old +
   ω·p_couple_new`, extending the fixed point's convergence radius (`ω<1` damps the
   divergent oscillation) at `O(iter·n²)` cost without forming the matrix.

`CloudParameters.coupling_scheme` defaults to `Explicit`. `Explicit`,
coupling-disabled, or a single active cell reduce **exactly** to ADR 027/028.

## Why this is correct (validation plan)

- **Direct solve is machine-exact self-consistent**: at the returned field, the
  fixed-point equation `field_i = (ρ/d)·S_j(p_ext+field_j)` holds to ~`1e-9`
  (far tighter than the fixed-point tolerance) — value-semantic.
- **Direct == fixed-point in the convergent (weak) regime**: both schemes agree
  where Jacobi converges — differential check.
- **Direct succeeds where fixed-point diverges**: a strong-coupling configuration in
  which the under-relaxed fixed point fails to converge still yields a self-consistent
  direct solution.
- **Under-relaxation extends convergence**: an `ω<1` fixed point converges (residual
  decreasing) on a configuration where `ω=1` does not.
- **Affine extraction is exact**: `R̈(p)` reconstructed from the two-point fit equals
  the solver's `R̈` at a third pressure.

## Consequences

- The implicit coupling is now robust in the strong regime (the regime ADR 030 left
  open). Reuses the validated dense solver (SSOT). `O(n³)` for the direct solve
  (opt-in, for accuracy-critical / tractable clouds); the under-relaxed fixed point
  is the `O(iter·n²)` middle ground.

## Residual (still open — CLD-1)

`dp/dt` coupling (only the instantaneous pressure is self-consistent); `R(t)`-dependent
shielding; cloud-interface instabilities; sparse/iterative direct solve for very large
active counts; experimental/k-Wave erosion comparison.
