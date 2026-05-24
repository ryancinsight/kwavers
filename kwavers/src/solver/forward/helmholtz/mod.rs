//! Helmholtz equation solvers.
//!
//! The acoustic Helmholtz equation for heterogeneous media:
//! ```text
//! ∇²ψ + k²(1 + V)ψ = S
//! ```
//! where ψ is the acoustic field, k = ω/c₀ is the reference wavenumber,
//! V is the heterogeneity potential, and S is the source term.
//!
//! The canonical iterative frequency-domain solver is the Convergent Born
//! Series (CBS) implementation under
//! `solver::inverse::fwi::frequency_domain::cbs`, which provides the
//! Osnabrugge–Leedumrongwatthanakun–Vellekoop 2016 preconditioned fixed-point
//! iteration with spectral and PSTD Green operators plus exact discrete
//! adjoint-gradient support.
//!
//! This module retains the finite-element Helmholtz solver for mesh-based
//! domains and FDTD/FEM hybrid coupling.

pub mod fem;

// ── Re-exports ────────────────────────────────────────────────────────────────
pub use fem::{FemHelmholtzConfig, FemHelmholtzSolver, FemPreconditionerType};
