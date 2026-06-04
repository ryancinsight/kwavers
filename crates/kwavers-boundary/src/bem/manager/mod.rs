//! `BemBoundaryManager` — applies Dirichlet, Neumann, Robin, and radiation BCs
//! to the boundary integral equations.
//!
//! ## Mathematical Formulation
//!
//! ```text
//! BIE: c(p) + ∫_Γ G·∂p/∂n dΓ = ∫_Γ p·∂G/∂n dΓ
//! ```
//!
//! - **Dirichlet (Γ_D)**: p = g_D
//! - **Neumann (Γ_N)**: ∂p/∂n = g_N
//! - **Robin (Γ_R)**: ∂p/∂n + αp = g_R
//! - **Radiation (Γ_∞)**: Sommerfeld — ∂p/∂n − ikp ≈ 0

mod applicators;
mod assembly;

use super::types::BemBoundaryCondition;
use kwavers_core::error::KwaversResult;
use kwavers_math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::Array1;
use num_complex::Complex64;

/// BEM boundary condition manager for boundary element solvers.
#[derive(Debug)]
pub struct BemBoundaryManager {
    pub(self) conditions: Vec<BemBoundaryCondition>,
}

impl BemBoundaryManager {
    /// Create new boundary manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    /// Clear all boundary conditions.
    pub fn clear(&mut self) {
        self.conditions.clear();
    }

    /// Add Dirichlet boundary condition.
    pub fn add_dirichlet(&mut self, node_values: Vec<(usize, Complex64)>) {
        self.conditions
            .push(BemBoundaryCondition::Dirichlet(node_values));
    }

    /// Add Neumann boundary condition.
    pub fn add_neumann(&mut self, node_derivatives: Vec<(usize, Complex64)>) {
        self.conditions
            .push(BemBoundaryCondition::Neumann(node_derivatives));
    }

    /// Add Robin boundary condition.
    pub fn add_robin(&mut self, node_conditions: Vec<(usize, f64, Complex64)>) {
        self.conditions
            .push(BemBoundaryCondition::Robin(node_conditions));
    }

    /// Add radiation boundary condition.
    pub fn add_radiation(&mut self, nodes: Vec<usize>) {
        self.conditions.push(BemBoundaryCondition::Radiation(nodes));
    }

    /// Apply all boundary conditions to the BEM system.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn apply_all(
        &self,
        h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        boundary_values: &mut Array1<Complex64>,
        wavenumber: f64,
    ) -> KwaversResult<()> {
        for condition in &self.conditions {
            match condition {
                BemBoundaryCondition::Dirichlet(nv) => {
                    self.apply_dirichlet(h_matrix, g_matrix, boundary_values, nv)?;
                }
                BemBoundaryCondition::Neumann(nd) => {
                    self.apply_neumann(h_matrix, g_matrix, boundary_values, nd)?;
                }
                BemBoundaryCondition::Robin(nc) => {
                    self.apply_robin(h_matrix, g_matrix, boundary_values, nc)?;
                }
                BemBoundaryCondition::Radiation(nodes) => {
                    self.apply_radiation(h_matrix, g_matrix, wavenumber, nodes)?;
                }
            }
        }
        Ok(())
    }

    /// Number of boundary conditions registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    /// True if no boundary conditions are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }
}

impl Default for BemBoundaryManager {
    fn default() -> Self {
        Self::new()
    }
}
