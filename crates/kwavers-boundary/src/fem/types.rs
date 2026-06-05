//! Types for FEM boundary conditions.

use num_complex::Complex64;

/// Boundary condition types for FEM/variational methods
#[derive(Debug, Clone)]
pub enum FemBoundaryCondition {
    /// Dirichlet: u = g on boundary nodes
    Dirichlet(Vec<(usize, Complex64)>),
    /// Neumann: ∂u/∂n = g on boundary nodes
    Neumann(Vec<(usize, Complex64)>),
    /// Robin: ∂u/∂n + αu = g on boundary nodes
    Robin(Vec<(usize, f64, Complex64)>), // (node, alpha, g)
    /// Radiation: Sommerfeld ABC (∂u/∂n - iku = 0)
    Radiation(Vec<usize>),
}
