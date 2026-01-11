//! Spectral Element Method (SEM) Solver
//!
//! Implements the Spectral Element Method for high-fidelity wave propagation
//! in complex geometries. SEM combines the geometric flexibility of finite
//! elements with the spectral accuracy of pseudospectral methods.
//!
//! ## Key Features
//!
//! - **Exponential Convergence**: Unlike traditional FEM's algebraic convergence,
//!   SEM converges exponentially with increasing polynomial degree
//! - **Diagonal Mass Matrix**: Enables explicit time integration without solving
//!   linear systems at each time step
//! - **High-Order Accuracy**: Uses high-degree polynomial basis functions
//!   (typically degree 3-8) for superior accuracy
//! - **Complex Geometries**: Handles arbitrarily complex geometries via
//!   hexahedral element meshing
//!
//! ## Mathematical Foundation
//!
//! SEM approximates the solution using high-order polynomials:
//!
//! ```text
//! u(x,t) ≈ ∑_{i=1}^{N+1} ∑_{j=1}^{N+1} ∑_{k=1}^{N+1} u_{ijk}(t) · ℓ_i(ξ) · ℓ_j(η) · ℓ_k(ζ)
//! ```
//!
//! where ℓ_i are Lagrange polynomials and (ξ,η,ζ) are local element coordinates.
//!
//! ## Implementation Details
//!
//! - **Basis Functions**: Lagrange polynomials on Gauss-Lobatto-Legendre points
//! - **Elements**: Hexahedral elements with N×N×N GLL points per element
//! - **Time Integration**: Newmark method for second-order accuracy
//! - **Boundary Conditions**: Integration with domain boundary system
//!
//! ## Usage
//!
//! ```rust,ignore
//! use kwavers::solver::forward::SemSolver;
//! use kwavers::domain::boundary::FemBoundaryManager;
//!
//! // Create SEM solver
//! let mut solver = SemSolver::new(config, mesh)?;
//!
//! // Configure boundary conditions
//! let mut bc_manager = solver.boundary_manager();
//! bc_manager.add_dirichlet(vec![(node_id, pressure_value)]);
//!
//! // Solve wave propagation
//! let solution = solver.solve(wavenumber, time_steps, sources)?;
//! ```
//!
//! ## References
//!
//! - Komatitsch & Tromp (1999): "Introduction to the spectral element method
//!   for three-dimensional seismic wave propagation"
//! - Patera (1984): "A spectral element method for fluid dynamics"
//! - García et al. (2025): "Feasibility of spectral-element modeling of wave
//!   propagation through the anatomy of marine mammals"

pub mod basis;
pub mod elements;
pub mod integration;
pub mod mesh;
pub mod solver;

pub use solver::{SemConfig, SemSolver};
