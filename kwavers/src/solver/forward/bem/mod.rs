//! Boundary Element Method (BEM) Solver
//!
//! Implements boundary element methods for acoustic wave problems.
//! BEM only discretizes the boundary of the domain, making it efficient
//! for problems with infinite or large domains.
//!
//! ## Features
//!
//! - **Boundary-only discretization**: No volume meshing required
//! - **Exact radiation conditions**: Natural treatment of unbounded domains
//! - **High accuracy**: Spectral accuracy for smooth boundaries
//! - **Memory efficient**: O(N²) complexity where N is boundary nodes
//!
//! ## Applications
//!
//! - Acoustic scattering from complex objects
//! - Radiation and diffraction problems
//! - Underwater acoustics
//! - Medical ultrasound transducer modeling
//!
//! ## Mathematical Foundation
//!
//! BEM solves the boundary integral equation:
//!
//! c(p) + ∫_Γ G*∂p/∂n dΓ = ∫_Γ p*∂G/∂n dΓ
//!
//! where:
//! - G is the Green's function (fundamental solution)
//! - Γ is the boundary surface
//! - c is the solid angle factor
//!
//! ## Usage
//!
//! ```rust,ignore
//! use kwavers::domain::boundary::BemBoundaryManager;
//! use kwavers::solver::forward::BemSolver;
//!
//! // Create BEM solver
//! let mut solver = BemSolver::new(config, boundary_mesh)?;
//!
//! // Configure boundary conditions
//! let mut boundary_manager = solver.boundary_manager();
//! boundary_manager.add_dirichlet(vec![(node_id, pressure_value)]);
//! boundary_manager.add_radiation(vec![outer_boundary_nodes]);
//!
//! // Solve for surface pressure/velocity
//! let solution = solver.solve(wavenumber, source_terms)?;
//! ```

pub mod burton_miller;
pub mod solver;

pub use burton_miller::{BurtonMillerAssembler, BurtonMillerConfig};
pub use solver::{BemConfig, BemSolver};
