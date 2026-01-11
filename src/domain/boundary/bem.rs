//! BEM Boundary Conditions for Boundary Element Methods
//!
//! Handles boundary conditions for boundary element methods (BEM).
//! Unlike FEM which discretizes the entire volume, BEM only discretizes the boundary,
//! so boundary conditions are applied to the boundary integral formulation.
//!
//! ## Supported Boundary Conditions
//!
//! - **Dirichlet**: Prescribed pressure values (p = g)
//! - **Neumann**: Prescribed normal velocity/pressure gradient (∂p/∂n = g)
//! - **Robin**: Mixed conditions (∂p/∂n + αp = g)
//! - **Radiation**: Absorbing boundary conditions for unbounded domains
//!
//! ## Usage in BEM Assembly
//!
//! BEM boundary conditions modify the boundary integral equations:
//!
//! ```rust,ignore
//! // 1. Assemble boundary element matrices (no boundaries yet)
//! let (h_matrix, g_matrix) = assemble_boundary_elements(boundary_elements);
//!
//! // 2. Apply boundary conditions to BEM system
//! boundary_manager.apply_to_bem_system(
//!     &mut h_matrix,
//!     &mut g_matrix,
//!     &mut boundary_values,
//!     wavenumber
//! );
//! ```
//!
//! ## Mathematical Formulation
//!
//! For acoustic BEM, the boundary integral equation is:
//!
//! c(p) + ∫_Γ G*∂p/∂n dΓ = ∫_Γ p*∂G/∂n dΓ
//!
//! Boundary conditions are applied by:
//!
//! - **Dirichlet (Γ_D)**: p = g_D (known pressure)
//! - **Neumann (Γ_N)**: ∂p/∂n = g_N (known normal derivative)
//! - **Robin (Γ_R)**: ∂p/∂n + αp = g_R (mixed condition)
//! - **Radiation (Γ_∞)**: Sommerfeld radiation condition

use crate::core::error::KwaversResult;
use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::Array1;
use num_complex::Complex64;

/// Boundary condition types for BEM methods
#[derive(Debug, Clone)]
pub enum BemBoundaryCondition {
    /// Dirichlet: p = g on boundary nodes
    Dirichlet(Vec<(usize, Complex64)>),
    /// Neumann: ∂p/∂n = g on boundary nodes
    Neumann(Vec<(usize, Complex64)>),
    /// Robin: ∂p/∂n + αp = g on boundary nodes
    Robin(Vec<(usize, f64, Complex64)>), // (node, alpha, g)
    /// Radiation: Sommerfeld ABC for unbounded domains
    Radiation(Vec<usize>),
}

/// BEM boundary condition manager for boundary element solvers
#[derive(Debug)]
pub struct BemBoundaryManager {
    conditions: Vec<BemBoundaryCondition>,
}

impl BemBoundaryManager {
    /// Create new boundary manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    /// Add Dirichlet boundary condition
    pub fn add_dirichlet(&mut self, node_values: Vec<(usize, Complex64)>) {
        self.conditions
            .push(BemBoundaryCondition::Dirichlet(node_values));
    }

    /// Add Neumann boundary condition
    pub fn add_neumann(&mut self, node_derivatives: Vec<(usize, Complex64)>) {
        self.conditions
            .push(BemBoundaryCondition::Neumann(node_derivatives));
    }

    /// Add Robin boundary condition
    pub fn add_robin(&mut self, node_conditions: Vec<(usize, f64, Complex64)>) {
        self.conditions
            .push(BemBoundaryCondition::Robin(node_conditions));
    }

    /// Add radiation boundary condition
    pub fn add_radiation(&mut self, nodes: Vec<usize>) {
        self.conditions.push(BemBoundaryCondition::Radiation(nodes));
    }

    /// Apply all boundary conditions to the BEM system
    ///
    /// In BEM, boundary conditions modify the boundary integral formulation
    /// by moving known values to the right-hand side and modifying the system.
    ///
    /// # Arguments
    /// * `h_matrix` - H matrix from BEM (modified in-place)
    /// * `g_matrix` - G matrix from BEM (modified in-place)
    /// * `boundary_values` - Known boundary values vector (modified in-place)
    /// * `wavenumber` - Wavenumber k for radiation conditions
    pub fn apply_all(
        &self,
        h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        boundary_values: &mut Array1<Complex64>,
        wavenumber: f64,
    ) -> KwaversResult<()> {
        for condition in &self.conditions {
            match condition {
                BemBoundaryCondition::Dirichlet(node_values) => {
                    self.apply_dirichlet(h_matrix, g_matrix, boundary_values, node_values)?;
                }
                BemBoundaryCondition::Neumann(node_derivatives) => {
                    self.apply_neumann(h_matrix, g_matrix, boundary_values, node_derivatives)?;
                }
                BemBoundaryCondition::Robin(node_conditions) => {
                    self.apply_robin(h_matrix, g_matrix, boundary_values, node_conditions)?;
                }
                BemBoundaryCondition::Radiation(nodes) => {
                    self.apply_radiation(h_matrix, g_matrix, wavenumber, nodes)?;
                }
            }
        }

        Ok(())
    }

    /// Apply Dirichlet boundary conditions (p = g)
    ///
    /// For known pressure values, we modify the BEM system to move
    /// the known values to the RHS and enforce the boundary condition.
    fn apply_dirichlet(
        &self,
        h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        boundary_values: &mut Array1<Complex64>,
        node_values: &[(usize, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, bc_value) in node_values {
            // For Dirichlet BC p_i = g_i:
            // Move known pressure to RHS: boundary_values[i] = g_i
            // Zero the row in H and G matrices (unknown becomes known)
            h_matrix.zero_row(node_idx);
            g_matrix.zero_row(node_idx);
            boundary_values[node_idx] = bc_value;
        }

        Ok(())
    }

    /// Apply Neumann boundary conditions (∂p/∂n = g)
    ///
    /// For known normal derivatives, we modify the system to account
    /// for the known derivative values in the boundary integral equation.
    fn apply_neumann(
        &self,
        h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        boundary_values: &mut Array1<Complex64>,
        node_derivatives: &[(usize, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, deriv_value) in node_derivatives {
            // For Neumann BC ∂p/∂n_i = g_i:
            // This affects the boundary integral equation differently
            // The derivative is related to the normal component
            boundary_values[node_idx] = deriv_value;
        }

        Ok(())
    }

    /// Apply Robin boundary conditions (∂p/∂n + αp = g)
    ///
    /// Mixed boundary conditions combine both pressure and its derivative.
    fn apply_robin(
        &self,
        h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        boundary_values: &mut Array1<Complex64>,
        node_conditions: &[(usize, f64, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, alpha, g_value) in node_conditions {
            // For Robin BC ∂p/∂n + αp = g:
            // Modify the H matrix diagonal: H_ii += α
            let current_h_diag = h_matrix.get_diagonal(node_idx);
            h_matrix.set_diagonal(node_idx, current_h_diag + Complex64::new(alpha, 0.0));

            // Set boundary value
            boundary_values[node_idx] = g_value;
        }

        Ok(())
    }

    /// Apply radiation boundary conditions (Sommerfeld ABC)
    ///
    /// For unbounded domains, the Sommerfeld radiation condition ensures
    /// that outgoing waves are not reflected back into the domain.
    fn apply_radiation(
        &self,
        h_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        g_matrix: &mut CompressedSparseRowMatrix<Complex64>,
        wavenumber: f64,
        nodes: &[usize],
    ) -> KwaversResult<()> {
        // Sommerfeld ABC: ∂p/∂n - ikp ≈ 0 for large |r|
        // In BEM, this is approximated by modifying boundary integrals
        let radiation_term = Complex64::new(0.0, -wavenumber);

        for &node_idx in nodes {
            // Modify H matrix diagonal to account for radiation condition
            let current_h_diag = h_matrix.get_diagonal(node_idx);
            h_matrix.set_diagonal(node_idx, current_h_diag + radiation_term);
        }

        Ok(())
    }

    /// Get the number of boundary conditions
    #[must_use]
    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    /// Check if no boundary conditions are defined
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
    use ndarray::Array1;

    #[test]
    fn test_bem_boundary_manager_creation() {
        let manager = BemBoundaryManager::new();
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
    }

    #[test]
    fn test_dirichlet_boundary_condition() {
        let mut manager = BemBoundaryManager::new();

        // Add Dirichlet BC: p[0] = 1.0 + 0.5i
        manager.add_dirichlet(vec![(0, Complex64::new(1.0, 0.5))]);

        // Create dummy BEM matrices
        let mut h_matrix = CompressedSparseRowMatrix::create(3, 3);
        let mut g_matrix = CompressedSparseRowMatrix::create(3, 3);
        let mut boundary_values = Array1::zeros(3);

        // Apply boundary conditions
        manager
            .apply_all(&mut h_matrix, &mut g_matrix, &mut boundary_values, 1.0)
            .unwrap();

        // Check Dirichlet enforcement
        assert_eq!(boundary_values[0], Complex64::new(1.0, 0.5));

        // Verify boundary manager has 1 condition
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_neumann_boundary_condition() {
        let mut manager = BemBoundaryManager::new();

        // Add Neumann BC: ∂p/∂n[1] = 2.0
        manager.add_neumann(vec![(1, Complex64::new(2.0, 0.0))]);

        // Create dummy BEM matrices
        let mut h_matrix = CompressedSparseRowMatrix::create(3, 3);
        let mut g_matrix = CompressedSparseRowMatrix::create(3, 3);
        let mut boundary_values = Array1::zeros(3);

        // Apply boundary conditions
        manager
            .apply_all(&mut h_matrix, &mut g_matrix, &mut boundary_values, 1.0)
            .unwrap();

        // Check Neumann boundary value
        assert_eq!(boundary_values[1], Complex64::new(2.0, 0.0));

        // Verify boundary manager has 1 condition
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_robin_boundary_condition() {
        let mut manager = BemBoundaryManager::new();

        // Add Robin BC: ∂p/∂n + 0.5*p[2] = 3.0
        manager.add_robin(vec![(2, 0.5, Complex64::new(3.0, 0.0))]);

        // Create dummy BEM matrices
        let mut h_matrix = CompressedSparseRowMatrix::create(3, 3);
        let mut g_matrix = CompressedSparseRowMatrix::create(3, 3);
        let mut boundary_values = Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);

        // Initialize H matrix diagonal
        h_matrix.set_diagonal(2, Complex64::new(2.0, 0.0));

        // Apply boundary conditions
        manager
            .apply_all(&mut h_matrix, &mut g_matrix, &mut boundary_values, 1.0)
            .unwrap();

        // Check Robin modifications
        assert_eq!(h_matrix.get_diagonal(2), Complex64::new(2.5, 0.0)); // 2.0 + 0.5
        assert_eq!(boundary_values[2], Complex64::new(3.0, 0.0));
    }

    #[test]
    fn test_radiation_boundary_condition() {
        let mut manager = BemBoundaryManager::new();

        // Add radiation BC on node 0
        manager.add_radiation(vec![0]);

        // Create dummy BEM matrices
        let mut h_matrix = CompressedSparseRowMatrix::create(3, 3);
        let mut g_matrix = CompressedSparseRowMatrix::create(3, 3);
        let mut boundary_values = Array1::zeros(3);

        // Initialize H matrix diagonal
        h_matrix.set_diagonal(0, Complex64::new(1.0, 0.0));

        // Apply boundary conditions with k = 2.0
        manager
            .apply_all(&mut h_matrix, &mut g_matrix, &mut boundary_values, 2.0)
            .unwrap();

        // Check radiation term: H_ii += -i*k = -i*2.0 = (0.0, -2.0)
        let expected = Complex64::new(1.0, -2.0);
        assert_eq!(h_matrix.get_diagonal(0), expected);
    }
}
