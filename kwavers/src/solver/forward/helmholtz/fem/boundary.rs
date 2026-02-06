//! FEM Boundary Condition Implementation
//!
//! Handles the application of various boundary conditions for Helmholtz FEM:
//! - Dirichlet (prescribed field)
//! - Neumann (prescribed normal derivative)
//! - Robin (mixed conditions)
//! - Radiation (Sommerfeld/ABC)
//! - Periodic boundaries

use crate::core::error::KwaversResult;
use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::Array1;
use num_complex::Complex64;

/// Boundary condition types for FEM
#[derive(Debug, Clone)]
pub enum FemBoundaryCondition {
    /// Dirichlet: u = g on boundary
    Dirichlet(Vec<(usize, Complex64)>),
    /// Neumann: ∂u/∂n = g on boundary
    Neumann(Vec<(usize, Complex64)>),
    /// Robin: ∂u/∂n + αu = g on boundary
    Robin(Vec<(usize, f64, Complex64)>), // (node, alpha, g)
    /// Radiation (Sommerfeld): ∂u/∂n - iku = 0
    Radiation(Vec<usize>),
}

/// Boundary condition manager for FEM systems
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
        self.conditions.push(FemBoundaryCondition::Dirichlet(node_values));
    }

    /// Add Neumann boundary condition
    pub fn add_neumann(&mut self, node_fluxes: Vec<(usize, Complex64)>) {
        self.conditions.push(FemBoundaryCondition::Neumann(node_fluxes));
    }

    /// Add Robin boundary condition
    pub fn add_robin(&mut self, node_conditions: Vec<(usize, f64, Complex64)>) {
        self.conditions.push(FemBoundaryCondition::Robin(node_conditions));
    }

    /// Add radiation boundary condition
    pub fn add_radiation(&mut self, nodes: Vec<usize>) {
        self.conditions.push(FemBoundaryCondition::Radiation(nodes));
    }

    /// Apply all boundary conditions to the system
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
    fn apply_dirichlet(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        mass: &mut CompressedSparseRowMatrix<Complex64>,
        rhs: &mut Array1<Complex64>,
        node_values: &[(usize, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, bc_value) in node_values {
            // Set row to identity: 1*u_i = g_i
            stiffness.set_row_to_identity(node_idx);
            mass.zero_row(node_idx);
            rhs[node_idx] = bc_value;

            // Zero column to maintain symmetry (optional for iterative solvers)
            // stiffness.zero_column(node_idx);
            // mass.zero_column(node_idx);
        }

        Ok(())
    }

    /// Apply Neumann boundary conditions
    fn apply_neumann(
        &self,
        rhs: &mut Array1<Complex64>,
        node_fluxes: &[(usize, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, flux_value) in node_fluxes {
            // Add flux to RHS: ∫_∂Ω g ∂φ/∂n dS contributes to load vector
            rhs[node_idx] += flux_value;
        }

        Ok(())
    }

    /// Apply Robin boundary conditions
    fn apply_robin(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        rhs: &mut Array1<Complex64>,
        node_conditions: &[(usize, f64, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, alpha, g_value) in node_conditions {
            // Robin: ∂u/∂n + αu = g
            // Modify diagonal: K_ii += α
            let current_diag = stiffness.get_diagonal(node_idx);
            stiffness.set_diagonal(node_idx, current_diag + Complex64::new(alpha, 0.0));

            // Modify RHS: f_i += g
            rhs[node_idx] += g_value;
        }

        Ok(())
    }

    /// Apply radiation boundary conditions (Sommerfeld ABC)
    fn apply_radiation(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        wavenumber: f64,
        nodes: &[usize],
    ) -> KwaversResult<()> {
        let radiation_term = Complex64::new(0.0, -wavenumber);

        for &node_idx in nodes {
            // Sommerfeld ABC: ∂u/∂n - iku ≈ 0
            // Approximate as: K_ii -= ik (for normal derivative term)
            let current_diag = stiffness.get_diagonal(node_idx);
            stiffness.set_diagonal(node_idx, current_diag + radiation_term);
        }

        Ok(())
    }
}

impl Default for FemBoundaryManager {
    fn default() -> Self {
        Self::new()
    }
}