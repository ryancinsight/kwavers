//! Boundary Element Method (BEM) Solver
//!
//! Implements boundary element methods for acoustic wave problems.
//! BEM only discretizes the boundary of the domain, making it efficient
//! for problems with infinite or large domains.
//!
//! ## Module Structure
//!
//! - [`solver`] — Core `BemSolver` struct, system assembly, and solve logic
//! - [`field`] — Post-processing: vertex normals, incident wave fields, `BemSolution`
//! - [`geometry`] — Triangle geometry utilities (distance, barycentric coords)
//! - [`integrals`] — Numerical quadrature for boundary integrals (singular, near-field, far-field)
//! - [`gmres`] — Preconditioned GMRES iterative solver (Saad & Schultz 1986)
//! - [`green`] — Helmholtz Green's function and derivatives (SSOT)
//! - [`burton_miller`] — Burton-Miller CFIE formulation
//! - [`quadrature`] — Gaussian quadrature rules for triangles
//! - [`singular`] — Duffy transformation for singular integrals
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
pub mod field;
pub mod geometry;
pub mod gmres;
pub mod green;
pub mod integrals;
pub mod quadrature;
pub mod singular;
pub mod solver;

pub use burton_miller::{BurtonMillerAssembler, BurtonMillerConfig};
pub use field::{compute_vertex_normals, plane_wave_incident, BemSolution};
pub use gmres::solve_gmres;
pub use solver::{BemConfig, BemSolver};
