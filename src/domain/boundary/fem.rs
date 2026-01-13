//! FEM Boundary Conditions for Variational Methods
//!
//! Handles boundary conditions for finite element and other variational methods.
//! These boundary conditions modify system matrices and load vectors during assembly,
//! unlike time-domain boundaries that modify field arrays during propagation.
//!
//! ## Supported Boundary Conditions
//!
//! - **Dirichlet**: Prescribed field values (u = g)
//! - **Neumann**: Prescribed normal derivatives (∂u/∂n = g)
//! - **Robin**: Mixed conditions (∂u/∂n + αu = g)
//! - **Radiation**: Sommerfeld absorbing boundary conditions
//!
//! ## Usage in FEM Assembly
//!
//! FEM boundary conditions are applied during matrix assembly:
//!
//! ```rust,ignore
//! // 1. Assemble element matrices (no boundaries)
//! let (k_elem, f_elem) = assemble_element(element, basis);
//!
//! // 2. Apply boundary conditions to global system
//! boundary_manager.apply_to_global_system(
//!     &mut global_stiffness,
//!     &mut global_mass,
//!     &mut global_load,
//!     wavenumber
//! );
//! ```
//!
//! ## Mathematical Formulation
//!
//! For the Helmholtz equation -∇²u + k²u = f in Ω:
//!
//! - **Dirichlet (Γ_D)**: u = g_D
//! - **Neumann (Γ_N)**: ∂u/∂n = g_N
//! - **Robin (Γ_R)**: ∂u/∂n + αu = g_R
//! - **Radiation (Γ_∞)**: ∂u/∂n - iku ≈ 0 (Sommerfeld ABC)

use crate::core::error::KwaversResult;
use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::Array1;
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

/// FEM boundary condition manager for variational solvers
#[derive(Debug)]
pub struct FemBoundaryManager {
    conditions: Vec<FemBoundaryCondition>,
}

impl FemBoundaryManager {
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
            .push(FemBoundaryCondition::Dirichlet(node_values));
    }

    /// Add Neumann boundary condition
    pub fn add_neumann(&mut self, node_fluxes: Vec<(usize, Complex64)>) {
        self.conditions
            .push(FemBoundaryCondition::Neumann(node_fluxes));
    }

    /// Add Robin boundary condition
    pub fn add_robin(&mut self, node_conditions: Vec<(usize, f64, Complex64)>) {
        self.conditions
            .push(FemBoundaryCondition::Robin(node_conditions));
    }

    /// Add radiation boundary condition
    pub fn add_radiation(&mut self, nodes: Vec<usize>) {
        self.conditions.push(FemBoundaryCondition::Radiation(nodes));
    }

    /// Apply all boundary conditions to the FEM system
    ///
    /// # Arguments
    /// * `stiffness` - Global stiffness matrix (modified in-place)
    /// * `mass` - Global mass matrix (modified in-place)
    /// * `rhs` - Right-hand side load vector (modified in-place)
    /// * `wavenumber` - Wavenumber k for radiation conditions
    pub fn apply_all(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        mass: &mut CompressedSparseRowMatrix<Complex64>,
        rhs: &mut Array1<Complex64>,
        wavenumber: f64,
    ) -> KwaversResult<()> {
        for condition in &self.conditions {
            match condition {
                FemBoundaryCondition::Dirichlet(node_values) => {
                    self.apply_dirichlet(stiffness, mass, rhs, node_values)?;
                }
                FemBoundaryCondition::Neumann(node_fluxes) => {
                    self.apply_neumann(rhs, node_fluxes)?;
                }
                FemBoundaryCondition::Robin(node_conditions) => {
                    self.apply_robin(stiffness, rhs, node_conditions)?;
                }
                FemBoundaryCondition::Radiation(nodes) => {
                    self.apply_radiation(stiffness, wavenumber, nodes)?;
                }
            }
        }

        Ok(())
    }

    /// Apply Dirichlet boundary conditions
    ///
    /// For Dirichlet conditions u_i = g_i, we modify the system:
    /// - Set diagonal K_ii = 1, zero off-diagonal elements in row i
    /// - Zero mass matrix row i
    /// - Set RHS[i] = g_i
    fn apply_dirichlet(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        mass: &mut CompressedSparseRowMatrix<Complex64>,
        rhs: &mut Array1<Complex64>,
        node_values: &[(usize, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, bc_value) in node_values {
            // Set diagonal to 1 and zero off-diagonal elements in row
            stiffness.set_diagonal(node_idx, Complex64::new(1.0, 0.0));
            stiffness.zero_row_off_diagonals(node_idx);
            mass.zero_row(node_idx);
            rhs[node_idx] = bc_value;
        }

        Ok(())
    }

    /// Apply Neumann boundary conditions
    ///
    /// For Neumann conditions ∂u/∂n = g, the contribution appears
    /// in the load vector: f_i += ∫_∂Ω g φ_i dS
    fn apply_neumann(
        &self,
        rhs: &mut Array1<Complex64>,
        node_fluxes: &[(usize, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, flux_value) in node_fluxes {
            // Add flux contribution to load vector
            rhs[node_idx] += flux_value;
        }

        Ok(())
    }

    /// Apply Robin boundary conditions
    ///
    /// For Robin conditions ∂u/∂n + αu = g, we modify:
    /// - Stiffness diagonal: K_ii += α
    /// - Load vector: f_i += g
    fn apply_robin(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        rhs: &mut Array1<Complex64>,
        node_conditions: &[(usize, f64, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, alpha, g_value) in node_conditions {
            // Robin: ∂u/∂n + αu = g
            // Modify diagonal: K_ii += α
            // Note: set_diagonal actually adds to existing value (via add_value),
            // so we just pass alpha directly
            stiffness.set_diagonal(node_idx, Complex64::new(alpha, 0.0));

            // Modify RHS: f_i += g
            rhs[node_idx] += g_value;
        }

        Ok(())
    }

    /// Apply radiation boundary conditions (Sommerfeld ABC)
    ///
    /// The Sommerfeld radiation condition: ∂u/∂n - iku ≈ 0
    /// For FEM, this is approximated by modifying the boundary terms.
    fn apply_radiation(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        wavenumber: f64,
        nodes: &[usize],
    ) -> KwaversResult<()> {
        let radiation_term = Complex64::new(0.0, -wavenumber);

        for &node_idx in nodes {
            // Sommerfeld ABC: ∂u/∂n - iku ≈ 0
            // Approximate by modifying diagonal: K_ii -= ik
            // Note: set_diagonal actually adds to existing value (via add_value),
            // so we just pass the radiation term directly
            stiffness.set_diagonal(node_idx, radiation_term);
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

impl Default for FemBoundaryManager {
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
    fn test_fem_boundary_manager_creation() {
        let manager = FemBoundaryManager::new();
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
    }

    #[test]
    fn test_dirichlet_boundary_condition() {
        let mut manager = FemBoundaryManager::new();

        // Add Dirichlet BC: u[0] = 1.0 + 0.5i
        manager.add_dirichlet(vec![(0, Complex64::new(1.0, 0.5))]);

        // Create dummy matrices/vectors
        let mut stiffness = CompressedSparseRowMatrix::create(3, 3);
        let mut mass = CompressedSparseRowMatrix::create(3, 3);
        let mut rhs = Array1::zeros(3);

        // Apply boundary conditions
        manager
            .apply_all(&mut stiffness, &mut mass, &mut rhs, 1.0)
            .unwrap();

        // Check Dirichlet enforcement
        assert_eq!(rhs[0], Complex64::new(1.0, 0.5));
        // Row should be identity (diagonal = 1)
        assert_eq!(stiffness.get_diagonal(0), Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_neumann_boundary_condition() {
        let mut manager = FemBoundaryManager::new();

        // Add Neumann BC: ∂u/∂n[1] = 2.0
        manager.add_neumann(vec![(1, Complex64::new(2.0, 0.0))]);

        // Create dummy matrices/vectors
        let mut stiffness = CompressedSparseRowMatrix::create(3, 3);
        let mut mass = CompressedSparseRowMatrix::create(3, 3);
        let mut rhs = Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);

        // Apply boundary conditions
        manager
            .apply_all(&mut stiffness, &mut mass, &mut rhs, 1.0)
            .unwrap();

        // Check Neumann contribution to RHS
        assert_eq!(rhs[1], Complex64::new(3.0, 0.0));
    }

    #[test]
    fn test_robin_boundary_condition() {
        let mut manager = FemBoundaryManager::new();

        // Add Robin BC: ∂u/∂n + 0.5*u[2] = 3.0
        manager.add_robin(vec![(2, 0.5, Complex64::new(3.0, 0.0))]);

        // Create dummy matrices/vectors
        let mut stiffness = CompressedSparseRowMatrix::create(3, 3);
        let mut mass = CompressedSparseRowMatrix::create(3, 3);
        let mut rhs = Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);

        // Initialize stiffness diagonal
        stiffness.set_diagonal(2, Complex64::new(2.0, 0.0));

        // Apply boundary conditions
        manager
            .apply_all(&mut stiffness, &mut mass, &mut rhs, 1.0)
            .unwrap();

        // Check Robin modifications
        assert_eq!(stiffness.get_diagonal(2), Complex64::new(2.5, 0.0)); // 2.0 + 0.5
        assert_eq!(rhs[2], Complex64::new(3.0, 0.0)); // 0.0 + 3.0
    }

    #[test]
    fn test_radiation_boundary_condition() {
        let mut manager = FemBoundaryManager::new();

        // Add radiation BC on node 0
        manager.add_radiation(vec![0]);

        // Create dummy matrices/vectors
        let mut stiffness = CompressedSparseRowMatrix::create(3, 3);
        let mut mass = CompressedSparseRowMatrix::create(3, 3);
        let mut rhs = Array1::zeros(3);

        // Initialize stiffness diagonal
        stiffness.set_diagonal(0, Complex64::new(1.0, 0.0));

        // Apply boundary conditions with k = 2.0
        manager
            .apply_all(&mut stiffness, &mut mass, &mut rhs, 2.0)
            .unwrap();

        // Check radiation term: K_ii += -i*k = -i*2.0 = (0.0, -2.0)
        let expected = Complex64::new(1.0, -2.0);
        assert_eq!(stiffness.get_diagonal(0), expected);
    }
}
