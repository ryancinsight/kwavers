//! FEM Helmholtz solver configuration types.

/// Preconditioner selection for the BiCGSTAB iterative solve.
///
/// # Status
/// - `None` — no preconditioning.
/// - `Diagonal` — Jacobi (diagonal) preconditioning (implemented).
/// - `ILU` / `AMG` — accepted but delegate to `Preconditioner::None` until implemented.
#[derive(Debug, Clone, Copy)]
pub enum PreconditionerType {
    /// No preconditioning.
    None,
    /// Diagonal (Jacobi) preconditioning.
    Diagonal,
    /// Incomplete LU factorization (not yet implemented).
    ILU,
    /// Algebraic multigrid (not yet implemented).
    AMG,
}

/// Configuration for the P1 tetrahedral FEM Helmholtz solver.
#[derive(Debug, Clone)]
pub struct FemHelmholtzConfig {
    /// Polynomial degree for basis functions (currently P1 is always used).
    pub polynomial_degree: usize,
    /// Wavenumber k for the Helmholtz equation ∇²u + k²u = −f.
    pub wavenumber: f64,
    /// BiCGSTAB convergence tolerance.
    pub tolerance: f64,
    /// Maximum BiCGSTAB iterations.
    pub max_iterations: usize,
    /// Preconditioner applied to the linear system.
    pub preconditioner: PreconditionerType,
    /// Enable radiation (absorbing) boundary conditions.
    pub radiation_boundary: bool,
}

impl Default for FemHelmholtzConfig {
    fn default() -> Self {
        Self {
            polynomial_degree: 1,
            wavenumber: 1.0,
            tolerance: 1e-8,
            max_iterations: 1000,
            preconditioner: PreconditionerType::Diagonal,
            radiation_boundary: true,
        }
    }
}
