//! FDTD-FEM Coupling Implementation
//!
//! Coupling between Finite-Difference Time-Domain (FDTD) and Finite Element Method (FEM)
//! solvers for multi-scale acoustic simulations via the Schwarz alternating method.
//!
//! ## Mathematical Foundation
//!
//! ```text
//! FDTD Domain:     ∂u/∂t = c²∇²u + f     in Ω₁
//! FEM Domain:      ∇²u + k²u = f          in Ω₂
//! Interface:       u₁ = u₂, ∂u₁/∂n = ∂u₂/∂n   on Γ = Ω₁ ∩ Ω₂
//! ```
//!
//! ## References
//!
//! - Farhat & Lesoinne (2000): "Two-level FETI methods for stationary Stokes problems"
//! - Berenger (2002): "Application of the CFS PML to the absorption of evanescent waves"
//! - Kopriva (2009): "Implementing spectral methods for partial differential equations"

mod config;
mod coupler;
mod interface;
mod solver;
#[cfg(test)]
mod tests;

pub use config::FdtdFemCouplingConfig;
pub use coupler::FdtdFemCoupler;
pub use interface::CouplingInterface;
pub use solver::FdtdFemSolver;
