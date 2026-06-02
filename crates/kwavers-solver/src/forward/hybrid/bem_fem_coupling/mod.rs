//! BEM-FEM Coupling Implementation
//!
//! This module implements coupling between Boundary Element Method (BEM)
//! and Finite Element Method (FEM) for problems with complex interior geometries
//! and unbounded exterior domains.
//!
//! ## Mathematical Foundation
//!
//! BEM-FEM coupling is ideal for problems with:
//! - Complex interior geometries requiring FEM discretization
//! - Unbounded exterior domains naturally handled by BEM
//! - Radiation conditions at infinity automatically satisfied by BEM
//!
//! ## Coupling Strategy
//!
//! The coupling enforces continuity across the interface Γ:
//!
//! ```text
//! Interior Domain (FEM):   ∇²u - k²u = f    in Ω₁
//! Exterior Domain (BEM):   ∫_Γ G(x,y) ∂u/∂n(y) ds(y) = u(x)    on Γ
//! Interface Conditions:    u₁ = u₂, ∂u₁/∂n = ∂u₂/∂n    on Γ
//! ```
//!
//! ## Implementation Features
//!
//! - Interface continuity enforcement between FEM and BEM meshes
//! - Conservative field transfer across structured/unstructured interfaces
//! - Automatic radiation boundary conditions through BEM
//! - Support for complex geometries in FEM domain
//!
//! ## Literature References
//!
//! - Wu, T. (2000). "Pre-asymptotic error analysis of BEM and FEM coupling"
//! - Costabel, M. (1987). "Boundary integral operators for the heat equation"
//! - Johnson, C. & Nédélec, J. C. (1980). "On the coupling of boundary integral
//!   and finite element methods"

pub mod config;
pub mod coupler;
pub mod interface;
pub mod solver;

pub use config::BemFemCouplingConfig;
pub use coupler::BemFemCoupler;
pub use interface::BemFemInterface;
pub use solver::BemFemSolver;
