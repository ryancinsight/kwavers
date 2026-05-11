use super::FemHelmholtzSolver;
use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use crate::domain::medium::Medium;
use crate::domain::mesh::BoundaryType;
use crate::math::linear_algebra::sparse::csr::CompressedSparseRowMatrix;
use crate::solver::forward::helmholtz::fem::assembly::FemAssembly;
use ndarray::Array1;
use num_complex::Complex64;

impl FemHelmholtzSolver {
    /// Assemble the global system matrix A = K − k²M and apply boundary conditions.
    ///
    /// The `medium` parameter is reserved for future heterogeneous-k support;
    /// currently the uniform `config.wavenumber` is used.
    /// # Errors
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn assemble_system<M: Medium + ?Sized>(&mut self, _medium: &M) -> KwaversResult<()> {
        if self.mesh.elements.is_empty() {
            let num_nodes = self.mesh.nodes.len();
            let mut k_global = CompressedSparseRowMatrix::create(num_nodes, num_nodes);
            let mut m_global = CompressedSparseRowMatrix::create(num_nodes, num_nodes);
            let mut rhs_global = Array1::<Complex64>::zeros(num_nodes);

            self.boundary_manager.apply_all(
                &mut k_global,
                &mut m_global,
                &mut rhs_global,
                self.config.wavenumber,
            )?;

            self.system_matrix = k_global;
            self.rhs = rhs_global;
            return Ok(());
        }

        let (stiffness_vec, mass_vec, rhs_vec) = self.compute_element_matrices()?;

        let assembler = FemAssembly::new();
        let (mut k_global, mut m_global, rhs_global) = assembler
            .assemble_global_matrices_parallel(
                &self.mesh.elements,
                &stiffness_vec,
                &mass_vec,
                &rhs_vec,
            )?;

        let k_sq = Complex64::from(self.config.wavenumber.powi(2));

        if k_global.values.len() != m_global.values.len() {
            return Err(KwaversError::Numerical(NumericalError::Instability {
                operation: "matrix_combination".to_owned(),
                condition: 0.0,
            }));
        }

        for (val_k, val_m) in k_global.values.iter_mut().zip(m_global.values.iter()) {
            *val_k -= k_sq * val_m;
        }

        self.system_matrix = k_global;
        self.rhs = rhs_global;

        self.boundary_manager.apply_all(
            &mut self.system_matrix,
            &mut m_global,
            &mut self.rhs,
            self.config.wavenumber,
        )?;

        Ok(())
    }

    /// Add an exact nodal Helmholtz load to the assembled right-hand side.
    ///
    /// # Contract
    ///
    /// This method represents the discrete functional `F_i += value` for the
    /// nodal basis function `phi_i`. It does not project an arbitrary physical
    /// point to the nearest node; callers must provide the FEM degree of freedom
    /// explicitly so no hidden spatial approximation is introduced.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn add_nodal_load(&mut self, node_idx: usize, value: Complex64) -> KwaversResult<()> {
        self.validate_node_index(node_idx)?;
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "FEM nodal load must be finite, got {value}"
            )));
        }

        self.rhs[node_idx] += value;
        Ok(())
    }

    /// Apply homogeneous or inhomogeneous Dirichlet conditions to nodes tagged
    /// with the requested mesh boundary type.
    ///
    /// Boundary conditions are queued in the solver's boundary manager and take
    /// effect on the next `assemble_system` call.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn add_dirichlet_on_boundary_type(
        &mut self,
        boundary_type: BoundaryType,
        value: Complex64,
    ) -> KwaversResult<usize> {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "FEM Dirichlet value must be finite, got {value}"
            )));
        }

        let nodes: Vec<_> = self
            .mesh
            .nodes
            .iter()
            .filter(|node| node.boundary_type == boundary_type)
            .map(|node| (node.index, value))
            .collect();

        if nodes.is_empty() {
            return Ok(0);
        }

        let count = nodes.len();
        self.boundary_manager.add_dirichlet(nodes);
        Ok(count)
    }

    pub(super) fn validate_node_index(&self, node_idx: usize) -> KwaversResult<()> {
        if node_idx >= self.rhs.len() {
            return Err(KwaversError::InvalidInput(format!(
                "FEM node index {node_idx} is outside system dimension {}",
                self.rhs.len()
            )));
        }
        Ok(())
    }
}
