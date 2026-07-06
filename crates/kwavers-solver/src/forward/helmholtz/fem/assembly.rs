//! FEM Matrix Assembly for Helmholtz Elements
//!
//! Handles the assembly of global stiffness and mass matrices from element contributions.
//! Provides efficient sparse matrix construction and boundary condition application.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::linear_algebra::sparse::CompressedSparseRowMatrix;
use kwavers_mesh::Tetrahedron;
use moirai_parallel::{map_collect_index_with, Adaptive};
use ndarray::Array1;
use num_complex::Complex64;

/// FEM matrix assembly utilities
#[derive(Debug)]
pub struct FemAssembly {
    // Assembly configuration and optimization parameters
}

impl FemAssembly {
    /// Create new assembly helper
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    /// Estimate sparsity pattern for efficient memory allocation
    #[must_use]
    pub fn estimate_sparsity_pattern(&self, num_nodes: usize, _elements: &[Tetrahedron]) -> usize {
        // Estimate number of non-zeros based on tetrahedral connectivity
        // Each node connects to ~20-30 other nodes in 3D tetrahedral mesh

        let avg_connections_per_node = 25; // Conservative estimate
        num_nodes * avg_connections_per_node
    }

    /// Pre-allocate sparse matrices with estimated sparsity
    #[must_use]
    pub fn preallocate_matrices(
        &self,
        num_nodes: usize,
        elements: &[Tetrahedron],
    ) -> (
        CompressedSparseRowMatrix<Complex64>,
        CompressedSparseRowMatrix<Complex64>,
    ) {
        let estimated_nnz = self.estimate_sparsity_pattern(num_nodes, elements);

        let stiffness =
            CompressedSparseRowMatrix::with_capacity(num_nodes, num_nodes, estimated_nnz);
        let mass = CompressedSparseRowMatrix::with_capacity(num_nodes, num_nodes, estimated_nnz);

        (stiffness, mass)
    }

    /// Assemble global matrices from element contributions (parallel version)
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn assemble_global_matrices_parallel(
        &self,
        elements: &[Tetrahedron],
        element_stiffness: &[ndarray::Array2<Complex64>],
        element_mass: &[ndarray::Array2<Complex64>],
        element_rhs: &[Array1<Complex64>],
    ) -> KwaversResult<(
        CompressedSparseRowMatrix<Complex64>,
        CompressedSparseRowMatrix<Complex64>,
        Array1<Complex64>,
    )> {
        let num_nodes = self.find_max_node_index(elements) + 1;

        // Pre-allocate matrices
        let (mut global_stiffness, mut global_mass) =
            self.preallocate_matrices(num_nodes, elements);
        let mut global_rhs = Array1::<Complex64>::zeros(num_nodes);

        self.validate_element_array_lengths(
            elements,
            element_stiffness,
            element_mass,
            element_rhs,
        )?;

        let contributions = map_collect_index_with::<Adaptive, _, _>(elements.len(), |idx| {
            self.assemble_single_element(
                &elements[idx],
                &element_stiffness[idx],
                &element_mass[idx],
                &element_rhs[idx],
            )
        })
        .into_iter()
        .collect::<KwaversResult<Vec<_>>>()?;

        // Accumulate contributions into global matrices
        for contribution in contributions {
            self.add_contribution_to_global(
                &mut global_stiffness,
                &mut global_mass,
                &mut global_rhs,
                contribution,
            )?;
        }

        Ok((global_stiffness, global_mass, global_rhs))
    }

    fn validate_element_array_lengths(
        &self,
        elements: &[Tetrahedron],
        element_stiffness: &[ndarray::Array2<Complex64>],
        element_mass: &[ndarray::Array2<Complex64>],
        element_rhs: &[Array1<Complex64>],
    ) -> KwaversResult<()> {
        let expected = elements.len();
        let actual = (
            element_stiffness.len(),
            element_mass.len(),
            element_rhs.len(),
        );
        if actual != (expected, expected, expected) {
            return Err(KwaversError::InvalidInput(format!(
                "FEM element contribution length mismatch: elements={expected}, stiffness={}, mass={}, rhs={}",
                actual.0, actual.1, actual.2
            )));
        }

        Ok(())
    }

    /// Assemble single element contribution
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn assemble_single_element(
        &self,
        element: &Tetrahedron,
        elem_stiffness: &ndarray::Array2<Complex64>,
        elem_mass: &ndarray::Array2<Complex64>,
        elem_rhs: &Array1<Complex64>,
    ) -> KwaversResult<ElementContribution> {
        Ok(ElementContribution {
            stiffness: elem_stiffness.clone(),
            mass: elem_mass.clone(),
            rhs: elem_rhs.clone(),
            node_indices: element.nodes,
        })
    }

    /// Add element contribution to global matrices
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn add_contribution_to_global(
        &self,
        global_stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        global_mass: &mut CompressedSparseRowMatrix<Complex64>,
        global_rhs: &mut Array1<Complex64>,
        contribution: ElementContribution,
    ) -> KwaversResult<()> {
        let ElementContribution {
            stiffness,
            mass,
            rhs,
            node_indices,
        } = contribution;

        for i in 0..node_indices.len() {
            let global_i = node_indices[i];

            // Add to RHS
            global_rhs[global_i] += rhs[i];

            for j in 0..node_indices.len() {
                let global_j = node_indices[j];

                // Add to stiffness matrix
                global_stiffness.add_value(global_i, global_j, stiffness[[i, j]]);

                // Add to mass matrix
                global_mass.add_value(global_i, global_j, mass[[i, j]]);
            }
        }

        Ok(())
    }

    /// Apply Dirichlet boundary conditions
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_dirichlet_bc(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        mass: &mut CompressedSparseRowMatrix<Complex64>,
        rhs: &mut Array1<Complex64>,
        dirichlet_nodes: &[(usize, Complex64)],
    ) -> KwaversResult<()> {
        for &(node_idx, bc_value) in dirichlet_nodes {
            // Set diagonal entry to 1
            stiffness.set_diagonal(node_idx, Complex64::new(1.0, 0.0));
            mass.set_diagonal(node_idx, Complex64::new(0.0, 0.0));

            // Zero out row and column (except diagonal)
            stiffness.zero_row_off_diagonals(node_idx);
            mass.zero_row_off_diagonals(node_idx);

            // Set RHS to boundary value
            rhs[node_idx] = bc_value;
        }

        Ok(())
    }

    /// Apply Neumann boundary conditions
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_neumann_bc(
        &self,
        rhs: &mut Array1<Complex64>,
        neumann_faces: &[(usize, usize, usize, Complex64)], // (node1, node2, node3, flux_value)
        face_areas: &[f64],
    ) -> KwaversResult<()> {
        for (face_idx, &(n1, n2, n3, flux)) in neumann_faces.iter().enumerate() {
            let area = face_areas.get(face_idx).copied().unwrap_or(1.0);

            // Distribute flux equally among face nodes
            let flux_per_node = flux * area / 3.0;

            rhs[n1] += flux_per_node;
            rhs[n2] += flux_per_node;
            rhs[n3] += flux_per_node;
        }

        Ok(())
    }

    /// Apply radiation (Sommerfeld) boundary conditions
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_radiation_bc(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        wavenumber: f64,
        radiation_nodes: &[usize],
    ) -> KwaversResult<()> {
        let radiation_term = Complex64::new(0.0, -wavenumber);

        for &node_idx in radiation_nodes {
            // Add -ik to diagonal for Sommerfeld radiation condition
            let current_diag = stiffness.get_diagonal(node_idx);
            stiffness.set_diagonal(node_idx, current_diag + radiation_term);
        }

        Ok(())
    }

    /// Find maximum node index in mesh
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn find_max_node_index(&self, elements: &[Tetrahedron]) -> usize {
        elements
            .iter()
            .flat_map(|elem| elem.nodes.iter())
            .max()
            .copied()
            .unwrap_or(0)
    }

    /// Optimize matrix storage by removing explicit zeros
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compress_matrices(
        &self,
        stiffness: &mut CompressedSparseRowMatrix<Complex64>,
        mass: &mut CompressedSparseRowMatrix<Complex64>,
    ) -> KwaversResult<()> {
        // Remove entries below tolerance
        let tolerance = 1e-12;
        stiffness.compress(tolerance);
        mass.compress(tolerance);

        Ok(())
    }
}

/// Element contribution to global matrices
#[derive(Debug)]
struct ElementContribution {
    stiffness: ndarray::Array2<Complex64>,
    mass: ndarray::Array2<Complex64>,
    rhs: Array1<Complex64>,
    node_indices: [usize; 4],
}

impl Default for FemAssembly {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn unit_tetrahedron() -> Tetrahedron {
        Tetrahedron {
            nodes: [0, 1, 2, 3],
            material_id: 0,
            volume: 1.0,
            quality: 1.0,
        }
    }

    #[test]
    fn assembly_rejects_mismatched_element_contribution_lengths() {
        let assembler = FemAssembly::new();
        let elements = [unit_tetrahedron()];
        let stiffness: [Array2<Complex64>; 0] = [];
        let mass = [Array2::zeros((4, 4))];
        let rhs = [Array1::zeros(4)];

        let error = assembler
            .assemble_global_matrices_parallel(&elements, &stiffness, &mass, &rhs)
            .expect_err("mismatched element contributions must be rejected");

        match error {
            KwaversError::InvalidInput(message) => {
                assert!(message.contains("elements=1"));
                assert!(message.contains("stiffness=0"));
                assert!(message.contains("mass=1"));
                assert!(message.contains("rhs=1"));
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }
}
