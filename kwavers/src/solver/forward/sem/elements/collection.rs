//! Spectral element mesh collection.
//!
//! # Mesh construction
//!
//! [`SemMesh::new`] accepts a connectivity table and global node coordinates,
//! validates the topology (manifold check via face-count histogram), and
//! constructs one [`SemElement`] per entry.
//!
//! ## Manifold check
//!
//! Every interior face must be shared by exactly two elements; boundary
//! faces are shared by exactly one.  The implementation counts occurrences
//! of each sorted 4-node face key.  A count > 2 indicates a non-manifold
//! mesh, which is rejected immediately.
//!
//! ## DOF count
//!
//! With polynomial degree `p`, each element contributes `(p+1)³` tensor-product
//! GLL DOFs.  The total DOF count is `n_elements × (p+1)³` (no shared-node
//! assembly; each element is treated independently for now).
//!
//! # References
//! - Komatitsch & Tromp (1999). GJI 139, §3.

use crate::core::error::KwaversResult;
use ndarray::Array2;
use std::sync::Arc;

use super::super::basis::SemBasis;
use super::element::SemElement;

/// Ordered collection of hexahedral spectral elements.
#[derive(Debug)]
pub struct SemMesh {
    /// All spectral elements (shared references for cheap cloning).
    pub elements: Vec<Arc<SemElement>>,
    /// Total number of degrees of freedom across all elements.
    pub n_dofs: usize,
    /// GLL basis shared by every element.
    pub basis: Arc<SemBasis>,
}

impl SemMesh {
    /// Build a [`SemMesh`] from element connectivity and global node coordinates.
    ///
    /// # Arguments
    /// * `element_connectivity` — one `Vec<usize>` of length 8 per hexahedral element.
    /// * `node_coordinates`     — global node positions, shape `(n_nodes, 3)`.
    /// * `basis`                — shared GLL basis (determines polynomial degree).
    ///
    /// # Errors
    /// - `InvalidInput` when `node_coordinates.ncols() != 3`.
    /// - `InvalidInput` when any element connectivity has ≠ 8 entries.
    /// - `InvalidInput` when a node index is out of bounds.
    /// - `InvalidInput` when duplicate node indices appear in one element.
    /// - `InvalidInput` on non-manifold meshes (face shared by > 2 elements).
    /// - Propagates [`SemElement::new`] errors (negative Jacobian, etc.).
    pub fn new(
        element_connectivity: &[Vec<usize>],
        node_coordinates: &Array2<f64>,
        basis: Arc<SemBasis>,
    ) -> KwaversResult<Self> {
        if node_coordinates.ncols() != 3 {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "node coordinates must have 3 columns, got {}",
                node_coordinates.ncols()
            )));
        }

        // ── Manifold check (face-count histogram) ────────────────────────────
        let mut face_counts: std::collections::HashMap<[usize; 4], usize> =
            std::collections::HashMap::new();

        for connectivity in element_connectivity {
            if connectivity.len() != 8 {
                return Err(crate::core::error::KwaversError::InvalidInput(format!(
                    "hexahedral element connectivity must have 8 nodes, got {}",
                    connectivity.len()
                )));
            }

            // The 6 faces of a hexahedron (local node pairs, zero-based).
            let faces: [[usize; 4]; 6] = [
                [
                    connectivity[0],
                    connectivity[1],
                    connectivity[2],
                    connectivity[3],
                ],
                [
                    connectivity[4],
                    connectivity[5],
                    connectivity[6],
                    connectivity[7],
                ],
                [
                    connectivity[0],
                    connectivity[1],
                    connectivity[5],
                    connectivity[4],
                ],
                [
                    connectivity[1],
                    connectivity[2],
                    connectivity[6],
                    connectivity[5],
                ],
                [
                    connectivity[2],
                    connectivity[3],
                    connectivity[7],
                    connectivity[6],
                ],
                [
                    connectivity[3],
                    connectivity[0],
                    connectivity[4],
                    connectivity[7],
                ],
            ];

            for mut face in faces {
                face.sort_unstable();
                let count = face_counts.entry(face).or_insert(0);
                *count += 1;
                if *count > 2 {
                    return Err(crate::core::error::KwaversError::InvalidInput(format!(
                        "non-manifold face encountered with nodes {face:?}"
                    )));
                }
            }
        }

        // ── Construct elements ───────────────────────────────────────────────
        let mut elements = Vec::with_capacity(element_connectivity.len());

        for (elem_id, connectivity) in element_connectivity.iter().enumerate() {
            if connectivity.len() != 8 {
                return Err(crate::core::error::KwaversError::InvalidInput(format!(
                    "hexahedral element connectivity must have 8 nodes, got {}",
                    connectivity.len()
                )));
            }

            // Validate node indices and uniqueness.
            let mut seen = std::collections::HashSet::with_capacity(8);
            for &global_node in connectivity {
                if global_node >= node_coordinates.nrows() {
                    return Err(crate::core::error::KwaversError::InvalidInput(format!(
                        "node index {global_node} out of bounds (n_nodes={})",
                        node_coordinates.nrows()
                    )));
                }
                if !seen.insert(global_node) {
                    return Err(crate::core::error::KwaversError::InvalidInput(
                        "hexahedral element connectivity contains duplicate node indices".to_owned(),
                    ));
                }
            }

            // Extract the 8 node rows for this element.
            let mut element_nodes = Array2::<f64>::zeros((8, 3));
            for (local_idx, &global_node) in connectivity.iter().enumerate() {
                for coord in 0..3 {
                    element_nodes[[local_idx, coord]] = node_coordinates[[global_node, coord]];
                }
            }

            let element = SemElement::new(elem_id, element_nodes, &basis)?;
            elements.push(Arc::new(element));
        }

        // DOF count: each element contributes (degree+1)³ independent GLL nodes.
        let n_gll = basis.n_points();
        let n_dofs = elements.len() * n_gll * n_gll * n_gll;

        Ok(Self {
            elements,
            n_dofs,
            basis,
        })
    }

    /// Get element by ID.  Returns `None` when `id >= n_elements`.
    #[must_use]
    pub fn element(&self, id: usize) -> Option<&Arc<SemElement>> {
        self.elements.get(id)
    }

    /// Total mesh volume (sum of per-element GLL quadrature volumes).
    #[must_use]
    pub fn volume(&self) -> f64 {
        self.elements.iter().map(|e| e.volume()).sum()
    }
}
