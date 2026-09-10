//! Self-adjoint second-order acoustic engine for **exact-gradient** FWI.
//!
//! See ADR 016. The existing FDTD/PSTD-driven FWI path produces only an
//! *approximate* adjoint (correct descent direction, wrong absolute magnitude
//! and ~20% direction-dependent shape error). This engine is a self-contained,
//! provably self-adjoint discretisation whose discrete adjoint is the same
//! scheme run backward in time, so the finite-difference gradient test returns
//! `κ = (g·δm)/(dJ/ds) ≈ 1` for every direction.
//!
//! # Forward scheme
//! Energy-form variable-density acoustic wave equation, `W = diag(1/(ρc²))`,
//! `D = ∇·(1/ρ ∇)` a **symmetric** heterogeneous Dirichlet Laplacian:
//! ```text
//! p^{n+1} = 2p^n − p^{n−1} + dt² W⁻¹ (D p^n + s^n),   p^{-1}=p^0=0
//! d^n = R p^n,   J = (dt/2) Σ_n ‖R p^n − d_obs^n‖²
//! ```
//!
//! # Exact adjoint and gradient (ADR 016)
//! ```text
//! ξ^{m−1} = 2ξ^m − ξ^{m+1} + dt² W⁻¹ D ξ^m − dt W⁻¹ Rᵀ r^m,  ξ^{N−1}=ξ^N=0
//! g_x = (−2/(ρ_x c_x³)) Σ_{n=0}^{N−2} ξ_x^n (p^{n+1} − 2p^n + p^{n−1})_x
//! ```
//! Because `D = Dᵀ`, the 3-point time operator is self-adjoint under reversal,
//! and the adjoint source `−dt W⁻¹ Rᵀ r^m` is the exact transpose of receiver
//! sampling injected through the same `W⁻¹` path as the forward source, `g` is
//! the literal algebraic gradient of the discrete `J`.

mod forward;
mod gradient;
mod operators;
mod types;

#[cfg(test)]
mod tests;

pub(crate) use forward::{forward, forward_sensor_only, forward_tail};
pub(crate) use gradient::{gradient, gradient_reconstructed};
pub(crate) use types::{Acquisition, SelfAdjointConfig};

#[cfg(test)]
pub(crate) use operators::build_edge_sponge;
