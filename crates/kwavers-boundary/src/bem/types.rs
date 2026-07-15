use kwavers_math::fft::Complex64;

/// Boundary condition types for BEM methods.
#[derive(Debug, Clone)]
pub enum BemBoundaryCondition {
    /// Dirichlet: p = g on boundary nodes.
    Dirichlet(Vec<(usize, Complex64)>),
    /// Neumann: ∂p/∂n = g on boundary nodes.
    Neumann(Vec<(usize, Complex64)>),
    /// Robin: ∂p/∂n + αp = g on boundary nodes (node, alpha, g).
    Robin(Vec<(usize, f64, Complex64)>),
    /// Radiation: Sommerfeld ABC for unbounded domains.
    Radiation(Vec<usize>),
}
