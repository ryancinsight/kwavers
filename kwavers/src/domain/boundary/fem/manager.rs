//! FEM boundary condition manager.

use super::types::FemBoundaryCondition;
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
use ndarray::Array1;
use num_complex::Complex64;

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

    /// Clear all boundary conditions
    pub fn clear(&mut self) {
        self.conditions.clear();
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
    /// # Theorem: boundary-form domain preservation
    ///
    /// Let `K, M ∈ C^(n×n)` and `f ∈ C^n` be the assembled FEM system.
    /// Boundary functionals are well-defined only when every boundary degree of
    /// freedom is in `[0, n)`. Dirichlet replacement sets the constrained row to
    /// the identity equation `u_i = g_i`; Neumann adds the boundary load
    /// contribution to `f_i`; Robin adds the diagonal trace-mass contribution
    /// `αu_i`; and the first-order Sommerfeld term contributes `-ik u_i`.
    ///
    /// # Proof sketch
    ///
    /// Each operation modifies only rows or entries indexed by the boundary
    /// degree of freedom. If the index domain and matrix/vector dimensions are
    /// validated first, every CSR row access and RHS write is defined. The
    /// resulting algebraic equations are the nodal collocation of the weak
    /// boundary forms listed above.
    ///
    /// # Arguments
    /// * `stiffness` - Global stiffness matrix (modified in-place)
    /// * `mass` - Global mass matrix (modified in-place)
    /// * `rhs` - Right-hand side load vector (modified in-place)
    /// * `wavenumber` - Wavenumber k for radiation conditions
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn apply_all(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        mass: &mut CompressedSparseRowMatrix<Complex64>,
        rhs: &mut Array1<Complex64>,
        wavenumber: f64,
    ) -> KwaversResult<()> {
        Self::validate_system(stiffness, mass, rhs)?;
        if !wavenumber.is_finite() {
            return Err(Self::invalid_value(
                "wavenumber",
                wavenumber,
                "Radiation boundary wavenumber must be finite",
            ));
        }

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

    fn invalid_value(parameter: &str, value: f64, reason: &str) -> KwaversError {
        KwaversError::Validation(ValidationError::InvalidValue {
            parameter: parameter.to_owned(),
            value,
            reason: reason.to_owned(),
        })
    }

    fn invalid_parameter(parameter: &str, reason: impl Into<String>) -> KwaversError {
        KwaversError::Validation(ValidationError::InvalidParameter {
            parameter: parameter.to_owned(),
            reason: reason.into(),
        })
    }

    fn validate_system(
        stiffness: &CompressedSparseRowMatrix<Complex64>,
        mass: &CompressedSparseRowMatrix<Complex64>,
        rhs: &Array1<Complex64>,
    ) -> KwaversResult<()> {
        if stiffness.rows != stiffness.cols {
            return Err(Self::invalid_parameter(
                "stiffness",
                format!(
                    "Stiffness matrix must be square, got {}x{}",
                    stiffness.rows, stiffness.cols
                ),
            ));
        }

        if mass.rows != stiffness.rows || mass.cols != stiffness.cols {
            return Err(Self::invalid_parameter(
                "mass",
                format!(
                    "Mass matrix shape {}x{} must match stiffness shape {}x{}",
                    mass.rows, mass.cols, stiffness.rows, stiffness.cols
                ),
            ));
        }

        if rhs.len() != stiffness.rows {
            return Err(Self::invalid_parameter(
                "rhs",
                format!(
                    "RHS length {} must match FEM system dimension {}",
                    rhs.len(),
                    stiffness.rows
                ),
            ));
        }

        Ok(())
    }

    fn validate_node(node_idx: usize, system_size: usize) -> KwaversResult<()> {
        if node_idx >= system_size {
            return Err(Self::invalid_parameter(
                "node_idx",
                format!(
                    "Boundary node index {node_idx} is outside FEM system dimension {system_size}"
                ),
            ));
        }

        Ok(())
    }

    /// Apply Dirichlet boundary conditions
    ///
    /// For Dirichlet conditions u_i = g_i, we modify the system:
    /// - Set diagonal K_ii = 1, zero off-diagonal elements in row i
    /// - Zero mass matrix row i
    /// - Set RHS[i] = g_i
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn apply_dirichlet(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        mass: &mut CompressedSparseRowMatrix<Complex64>,
        rhs: &mut Array1<Complex64>,
        node_values: &[(usize, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, bc_value) in node_values {
            Self::validate_node(node_idx, rhs.len())?;
            stiffness.zero_row(node_idx);
            stiffness.add_value(node_idx, node_idx, Complex64::new(1.0, 0.0));
            mass.zero_row(node_idx);
            rhs[node_idx] = bc_value;
        }

        Ok(())
    }

    /// Apply Neumann boundary conditions
    ///
    /// For Neumann conditions ∂u/∂n = g, the contribution appears
    /// in the load vector: f_i += ∫_∂Ω g φ_i dS
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn apply_neumann(
        &self,
        rhs: &mut Array1<Complex64>,
        node_fluxes: &[(usize, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, flux_value) in node_fluxes {
            Self::validate_node(node_idx, rhs.len())?;
            rhs[node_idx] += flux_value;
        }

        Ok(())
    }

    /// Apply Robin boundary conditions
    ///
    /// For Robin conditions ∂u/∂n + αu = g, we modify:
    /// - Stiffness diagonal: K_ii += α
    /// - Load vector: f_i += g
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn apply_robin(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        rhs: &mut Array1<Complex64>,
        node_conditions: &[(usize, f64, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, alpha, g_value) in node_conditions {
            Self::validate_node(node_idx, rhs.len())?;
            if !alpha.is_finite() {
                return Err(Self::invalid_value(
                    "alpha",
                    alpha,
                    "Robin coefficient must be finite",
                ));
            }
            stiffness.set_diagonal(node_idx, Complex64::new(alpha, 0.0));
            rhs[node_idx] += g_value;
        }

        Ok(())
    }

    /// Apply radiation boundary conditions (Sommerfeld ABC)
    ///
    /// The Sommerfeld radiation condition: ∂u/∂n - iku ≈ 0
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn apply_radiation(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        wavenumber: f64,
        nodes: &[usize],
    ) -> KwaversResult<()> {
        let radiation_term = Complex64::new(0.0, -wavenumber);

        for &node_idx in nodes {
            Self::validate_node(node_idx, stiffness.rows)?;
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
