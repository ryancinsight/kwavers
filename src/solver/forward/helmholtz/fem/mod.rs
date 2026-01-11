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
//! - Boundary conditions via [`crate::domain::boundary::FemBoundaryManager`]
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
//! ## Boundary Conditions
//!
//! FEM boundary conditions are managed through the domain boundary system:
//!
//! ```rust,ignore
//! use kwavers::domain::boundary::FemBoundaryManager;
//!
//! let mut boundary_manager = FemBoundaryManager::new();
//! boundary_manager.add_dirichlet(vec![(node_id, Complex64::new(0.0, 0.0))]);
//! boundary_manager.add_radiation(vec![node_id]); // Sommerfeld ABC
//!
//! // Apply to FEM system during assembly
//! boundary_manager.apply_all(&mut stiffness, &mut mass, &mut rhs, wavenumber)?;
//! ```
//!
//! ## References
//!
//! - Wu (1997): "Pre-asymptotic error analysis of FEM for Helmholtz equation"
//! - Wu (2006): "Pre-asymptotic error analysis of CIP-FEM and DG methods"

pub mod assembly;
pub mod basis;
pub mod solver;

pub use assembly::FemAssembly;
pub use basis::{BasisFunction, GaussPoint, GaussQuadrature};
pub use solver::{FemHelmholtzConfig, FemHelmholtzSolver, PreconditionerType};
