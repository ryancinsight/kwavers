//! Finite Element Method Helmholtz Solver
//!
//! Implements FEM discretization of the Helmholtz equation for complex geometries:
//!
//! ∇²u + k²u = f    in Ω
//! u = g_D         on Γ_D (Dirichlet boundary)
//! ∂u/∂n = g_N     on Γ_N (Neumann boundary)
//!
//! ## Features
//!
//! - Tetrahedral element discretization
//! - Higher-order polynomial basis functions
//! - Sparse matrix assembly with CSR format
//! - ILU preconditioned iterative solvers
//! - Radiation boundary conditions
//!
//! ## Applications
//!
//! - Complex anatomical geometries (skull, joints)
//! - Heterogeneous tissue modeling
//! - Implant and device scattering
//! - Transcranial ultrasound aberration correction
//!
//! ## References
//
//! - Wu (1997): "Pre-asymptotic error analysis of FEM for Helmholtz equation"
//! - Wu (2006): "Pre-asymptotic error analysis of CIP-FEM and DG methods"

pub mod assembly;
pub mod basis;
pub mod solver;

pub use assembly::*;
pub use basis::*;
pub use solver::*;
