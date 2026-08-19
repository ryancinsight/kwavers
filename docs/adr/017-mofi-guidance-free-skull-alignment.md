# ADR 017 — MOFI: guidance-free rigid skull-template alignment for FWI

**Status:** Accepted
**Change class:** [minor] (additive: new module + public API; no breaking changes)
**Date:** 2026-06-07
**Depends on:** ADR 016 (self-adjoint exact-gradient engine)
**Reference:** Bates, Cueto, Coleman, Smith, Guasch, Calderon Agudo (2026), "Automatic
Skull-Template Alignment Without a Guidance Image", *Ultrasound in Medicine & Biology*.
https://doi.org/10.1016/j.ultrasmedbio.2026.05.003

## Context

Transcranial ultrasound FWI needs a CT-derived skull sound-speed template aligned to the patient's
anatomy. Existing alignment relies on a concurrent MRI guidance image (cost/logistics/throughput
barriers). The paper introduces **MOFI** (Manifold Optimisation for FWI): align the template using
only the recorded acoustic data, by minimising the FWI misfit over a low-dimensional rigid-body
(SE(2)) reparametrisation of the template rather than over the full pixel grid.

## Decision

Implement MOFI in `inverse::fwi::time_domain::mofi`:

- **Reparametrisation** (`transform.rs`): `c_φ(x) = c(T_φ⁻¹ x)` for `φ = {θ, δ₁, δ₂} ∈ SE(2)`,
  via bilinear back-mapping about the grid centre. The analytic per-voxel Jacobian
  `∂c_φ/∂φ` is derived from the bilinear interpolant's spatial gradient (the exact quantity the
  paper obtains by reverse-mode autodiff), verified voxel-for-voxel against finite differences.
- **Chained gradient**: `∂f/∂φ = (∂c_φ/∂φ)ᵀ ∂f/∂c`, where `∂f/∂c` is the **exact** FWI gradient
  from the self-adjoint engine (ADR 016). MOFI requires
  `FwiProcessor::with_engine(FwiEngine::SecondOrderSelfAdjoint)`.
- **Manifold optimisation** (`mod.rs`, paper Appendix A): SE(2) updates via the SO(2) log/exp maps
  (rotation kept on the shortest geodesic, `θ ∈ [−π, π]`) and `δ^{k+1} = δ^k + R_{θ^k} Δ_δ`, with
  gradient normalisation and an Armijo line search. θ and δ are balanced by optimising in the
  scaled space `(L·θ, δ₁, δ₂)` (`L` = domain half-width), the principled form of the paper's
  gradient normalisation that makes a unit rotation step comparable to a unit translation step.

## Alternatives considered

- **Flat-space (Euclidean) parameter updates** — the paper's baseline; the manifold (Lie-group)
  update is preferred for bounded rotation range and shortest-path behaviour under large
  misalignment (paper §Discussion, Appendix A).
- **Autodiff Jacobian (PyTorch/Stride, as in the paper)** — kwavers has no autodiff; the analytic
  bilinear-interpolant Jacobian is exact and dependency-free, and is FD-verified.
- **Pixel-wise FWI for alignment** — high-dimensional, ill-posed, and prone to cycle-skipping for a
  pure geometric misalignment (the motivating problem); MOFI's 3-parameter reduction is the point.

## Verification

- `mofi::transform::transform_tests::jacobian_matches_finite_difference` — analytic `∂c_φ/∂φ` vs FD.
- `mofi::tests::mofi_recovers_known_rigid_misalignment` — recovers a known `(θ=6°, δ=(2,−1.5) mm)`
  misalignment of an asymmetric 2-D phantom from ring-array acoustic data to **<1° and <1 mm**
  (the paper's in-silico tolerance), with the misfit collapsing >10×.
- `mofi::tests::mofi_is_stationary_when_already_aligned` — no spurious offset when starting aligned.

## Consequences / scope

- 2-D SE(2) only (matches the paper). 3-D SE(3) alignment and non-rigid/diffeomorphic extensions
  (paper §"Looking ahead") are future work.
- Single-acquisition; the driver consumes one `FwiGeometry` (which may carry multiple simultaneous
  source voxels). Multi-shot stacking is a straightforward extension.
- Public API: `mofi::{align, MofiConfig, MofiResult, RigidTransform}`, re-exported from
  `inverse::fwi::time_domain` as `mofi_align`, `MofiConfig`, `MofiResult`, `RigidTransform`.
