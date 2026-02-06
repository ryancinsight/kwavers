//! Nonlinear Solvers for Implicit Multiphysics Coupling
//!
//! Provides advanced iterative solvers for nonlinear systems arising in
//! implicit multiphysics simulations:
//!
//! - **GMRES**: Krylov subspace method for linear systems
//! - **JFNK**: Jacobian-Free Newton-Krylov for nonlinear systems
//! - **Preconditioners**: Physics-based block preconditioners
//!
//! # Applications
//!
//! - Monolithic multiphysics coupling (acoustic-optical-thermal)
//! - Implicit time integration for stiff systems
//! - Nonlinear acoustic propagation (Westervelt, KZK equations)
//! - HIFU therapy planning with temperature feedback
//!
//! # Architecture
//!
//! ```text
//! NonlinearSolver
//! ├── NewtonMethod (outer iteration)
//! │   └── JFNK (Jacobian-free)
//! └── LinearSolver (inner iteration)
//!     ├── GMRES (restarted Krylov)
//!     └── Preconditioner (optional)
//!         ├── BlockJacobi
//!         ├── ILU(k)
//!         └── PhysicsBased
//! ```

pub mod gmres;

pub use gmres::{ConvergenceInfo, GMRESConfig, GMRESSolver};
