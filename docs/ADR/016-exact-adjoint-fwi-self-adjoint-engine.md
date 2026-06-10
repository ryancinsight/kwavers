# ADR 016 — Exact discrete-adjoint FWI via a self-adjoint second-order acoustic engine

**Status:** Accepted
**Change class:** [major] (new forward/adjoint engine for time-domain FWI; additive — existing FDTD/PSTD paths retained)
**Date:** 2026-06-07
**Supersedes/relates:** backlog Future-enhancement #15; depends on the gradient-test infrastructure
added in `inverse::fwi::time_domain::tests::gradient`.

## Context

The time-domain FWI gradient was found to be an **approximate** adjoint, not the exact discrete
adjoint the module docstrings claimed. The finite-difference gradient test
`test_fwi_adjoint_gradient_is_valid_descent_direction` measures, at a homogeneous `c0 = 1500 m/s`
model, `κ = (g·δm) / (dJ/ds)` where `dJ/ds` is a Richardson-extrapolated central finite difference
of the *actual* misfit. It found `κ ≈ 238` for one perturbation direction and `≈ 191` for another,
both stable under step-size refinement. An exact discrete adjoint gives `κ ≡ 1` for every direction.

Two independent, root-caused defects in the FDTD/PSTD-driven adjoint path:

1. **Global ~200× scale offset.** The adjoint re-injects the data residual through the *scaled
   additive-source* path used by the forward solve (`p_scale = 2·dt·c₀/(N·dx)`,
   `forward/fdtd/source_handler/scaling.rs:138`), while the forward *receiver* operator samples
   pressure directly. Source-injection is therefore not the transpose of receiver-sampling.
2. **~20% direction-dependent shape error.** For heterogeneous `c`, the first-order staggered
   leapfrog's forward operator is `K·D` (mass-injection `K = ρc²` left-multiplying the spatial
   operator `D`) whereas its true adjoint involves `D·K`; these commute only for constant `K`.
   The CPML convolutional memory variables are likewise not self-adjoint, and the plain
   time-reversal used is not the exact transpose of the leapfrog time integrator.

Impact: the gradient is a valid **descent direction** (sign-correct; the Armijo line search
absorbs both errors, so existing steepest-descent / L-BFGS FWI convergence is unaffected), but its
**absolute magnitude and fine shape are wrong**. That is acceptable for line-searched first-order
methods and fatal for Gauss-Newton, fixed-step updates, gradient-norm stopping criteria, and
cross-shot gradient weighting — the capabilities this ADR unblocks.

## Decision

Add a **dedicated self-adjoint second-order acoustic engine** for exact-gradient time-domain FWI,
selected via a new `FwiEngine::SecondOrderSelfAdjoint` on `FwiProcessor`. The existing
`FwiEngine::Solver` path (FDTD/PSTD `Box<dyn Solver>`) is retained unchanged as the default.

### Forward scheme (provably self-adjoint)

Variable-density acoustic wave equation in energy form, `W = diag(1/(ρc²))`,
`D = ∇·(1/ρ ∇)` discretised as a **symmetric** heterogeneous Dirichlet Laplacian (arithmetic
face-averaged `1/ρ`, zeros outside the domain):

```text
W (p^{n+1} − 2p^n + p^{n−1})/dt² = D p^n + s^n
⇒ p^{n+1} = 2p^n − p^{n−1} + dt² W⁻¹ (D p^n + s^n),   p^{-1}=p^0=0 (rest)
d^n = R p^n                      (receiver sampling: read p at receiver voxels)
J   = (dt/2) Σ_n ‖R p^n − d_obs^n‖²
```

`D = Dᵀ` (symmetric stencil) and the 3-point time operator is self-adjoint under time reversal, so
the discrete adjoint of this scheme is the **same scheme run backward**:

```text
ξ^{m−1} = 2ξ^m − ξ^{m+1} + dt² W⁻¹ D ξ^m − dt W⁻¹ Rᵀ r^m,   ξ^{N−1}=ξ^N=0,  m=N−1…1
r^m = R p^m − d_obs^m
```

The operator `W⁻¹D` is identical in forward and adjoint (no `K·D` vs `D·K` asymmetry), and the
adjoint source `−dt W⁻¹ Rᵀ r^m` is the exact transpose of receiver sampling injected through the
same `W⁻¹` path as the forward source (no spurious source-injection scaling).

### Exact gradient

`c` enters only through `W = 1/(ρc²)`; `∂W/∂c = −2/(ρc³)`. The reduced gradient is

```text
g_x = (−2/(ρ_x c_x³)) Σ_{n=0}^{N−2} ξ_x^n · (p^{n+1} − 2p^n + p^{n−1})_x
```

This is the literal algebraic gradient of the discrete `J`, so the finite-difference test must
return `κ ≈ 1` (to FD-truncation/round-off).

### Boundaries

`κ = 1` requires only self-adjointness, **not** absorption: zero-Dirichlet boundaries make `D`
symmetric, and any wall reflections are handled exactly by the adjoint (they are part of `F` and
`dF`).

**Update (2026-06-08): self-adjoint absorbing layer implemented.** An optional symmetric diagonal
sponge `B = diag(b) ≥ 0` is now supported via the damped leapfrog `W p̈ + B ṗ = D p + s`
(`coeffs`, `build_edge_sponge`, `FwiProcessor::with_self_adjoint_damping`):
`a⁺ = W/dt² + b/(2dt)`, `a⁻ = W/dt² − b/(2dt)`, `m = 2W/dt²`,
`p^{n+1} = (1/a⁺)[m p^n − a⁻ p^{n−1} + D p^n + s^n]`; `b = 0` recovers the lossless scheme exactly.
Because `B` is diagonal (symmetric), the discrete adjoint is the same damped scheme run backward
(`ξ^{m-1} = (1/a⁺)[(m+D)ξ^m − a⁻ξ^{m+1} − dt Rᵀr^m]`), so `κ ≈ 1` is preserved with absorption
(`self_adjoint_gradient_kappa_one_with_sponge`), and the sponge removes >70% of outgoing energy
(`self_adjoint_sponge_absorbs_outgoing_waves`).

## Alternatives considered

- **Path A — literal discrete transpose of the existing first-order staggered CPML FDTD/PSTD.**
  Exact for the deployed forward operator, but requires transposing every update including the CPML
  convolutional memory variables and staggered/leapfrog half-steps — a large, high-risk change to
  the shared solver that currently passes 66/66 k-wave-python and 5/5 KWave.jl parity. Rejected as
  the first step; can be revisited if FWI must invert with the exact CPML operator.
- **Fix only the source-injection scale.** Removes the ~200× offset but leaves the ~20% `K·D`/CPML
  shape error, so `κ ≠ 1`. Insufficient for the acceptance criterion.
- **Reuse the FDTD `Box<dyn Solver>` and post-scale the gradient by a fitted constant.** A constant
  cannot correct a direction-dependent shape error; this would be cosmetic.

## Verification

- `kappa ≈ 1` (|κ−1| < 1e-2; observed ~1e-4) for ≥3 independent perturbation directions on the SA
  engine, via Richardson-extrapolated central finite differences of the actual `J` — exact-adjoint
  (dot-product) test.
- Forward/adjoint round-trip self-consistency (`J(self-data)=0`; adjoint source = `R p − d_obs`).
- SA-engine FWI convergence smoke test (recovers a localized velocity anomaly; misfit decreases).
- Existing FDTD-engine FWI/L-BFGS tests remain green (default path untouched).

## Consequences

- New module `inverse::fwi::time_domain::self_adjoint` (engine + exact gradient) and a
  `FwiEngine` selector on `FwiProcessor`.
- The SA engine's synthetic amplitudes differ from the FDTD/PSTD engine (different discrete scheme);
  observed data for an SA inversion must be generated by the SA forward (self-consistent).
- The FDTD/PSTD engine remains the documented **approximate**-adjoint path (correct descent
  direction, line-search-only). Docstrings in `mod.rs`/`adjoint.rs` retain the "approximate" wording.
- Enables Gauss-Newton / Hessian-vector products, fixed-step updates, and gradient-norm stopping for
  FWI built on the SA engine.
