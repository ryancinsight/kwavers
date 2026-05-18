//! FEM Helmholtz Solver — P1 Linear Tetrahedral Elements
//!
//! ## Mathematical Foundation
//!
//! The Helmholtz equation `∇²u + k²u = −f` is discretised by the Galerkin method on a
//! tetrahedral mesh with P1 (linear) basis functions (Ihlenburg 1998, §2.1):
//! ```text
//! a(u,v) = ∫_Ω (∇u·∇v − k²uv) dΩ = ∫_Ω fv dΩ + ∫_Γ (∂u/∂n)v dΓ
//! ```
//!
//! ## Module Layout
//!
//! | Submodule | Contents                                                   |
//! |-----------|------------------------------------------------------------|
//! | `config`  | `FemHelmholtzConfig`, `FemPreconditionerType`                |
//! | `core`    | `FemHelmholtzSolver` struct + assembly, solve, interpolate |
//!
//! ## References
//! - Ihlenburg F (1998). *Finite Element Analysis of Acoustic Scattering*. Springer.

mod config;
mod core;
#[cfg(test)]
mod tests;

pub use config::{FemHelmholtzConfig, FemPreconditionerType};
pub use core::FemHelmholtzSolver;
