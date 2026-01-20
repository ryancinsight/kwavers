//! Spectral Element Implementation
//!
//! Implements hexahedral spectral elements for SEM discretization.
//! Each element uses high-order polynomial basis functions on GLL points.

use crate::core::error::{KwaversResult, NumericalError};
use ndarray::{Array1, Array2, Array3, Array5};
use std::sync::Arc;

use super::basis::SemBasis;

/// Hexahedral spectral element
#[derive(Debug, Clone)]
pub struct SemElement {
    /// Element ID
    pub id: usize,
    /// Node coordinates: shape (8, 3) for hexahedral element
    pub nodes: Array2<f64>,
    /// Element Jacobian matrix at each GLL point
    /// Shape: (n_gll, n_gll, n_gll, 3, 3)
    pub jacobian: Array5<f64>,
    /// Determinant of Jacobian at each GLL point
    /// Shape: (n_gll, n_gll, n_gll)
    pub jacobian_det: Array3<f64>,
    /// Inverse Jacobian at each GLL point
    /// Shape: (n_gll, n_gll, n_gll, 3, 3)
    pub jacobian_inv: Array5<f64>,
    pub gll_weights: Array1<f64>,
}

impl SemElement {
    /// Create spectral element from node coordinates
    pub fn new(id: usize, nodes: Array2<f64>, basis: &SemBasis) -> KwaversResult<Self> {
        if nodes.nrows() != 8 || nodes.ncols() != 3 {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Hexahedral element requires 8 nodes with 3 coordinates each, got {}×{}",
                nodes.nrows(),
                nodes.ncols()
            )));
        }

        let n_gll = basis.n_points();

        // Compute Jacobian matrices at all GLL points
        let mut jacobian = Array5::<f64>::zeros((n_gll, n_gll, n_gll, 3, 3));
        let mut jacobian_det = Array3::<f64>::zeros((n_gll, n_gll, n_gll));
        let mut jacobian_inv = Array5::<f64>::zeros((n_gll, n_gll, n_gll, 3, 3));

        if !nodes.iter().all(|v| v.is_finite()) {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Hexahedral element nodes contain NaN/Inf".to_string(),
            ));
        }

        for i in 0..n_gll {
            for j in 0..n_gll {
                for k in 0..n_gll {
                    let xi = basis.gll_points[i];
                    let eta = basis.gll_points[j];
                    let zeta = basis.gll_points[k];

                    let (jac, det, inv) = Self::compute_jacobian(&nodes, xi, eta, zeta)?;
                    if !det.is_finite() {
                        return Err(crate::core::error::KwaversError::InvalidInput(
                            "Non-finite Jacobian determinant".to_string(),
                        ));
                    }
                    if det <= 0.0 {
                        return Err(crate::core::error::KwaversError::InvalidInput(format!(
                            "Inverted hexahedral element: det(J) <= 0 at GLL index ({i},{j},{k})"
                        )));
                    }
                    jacobian
                        .slice_mut(ndarray::s![i, j, k, .., ..])
                        .assign(&jac);
                    jacobian_det[[i, j, k]] = det;
                    jacobian_inv
                        .slice_mut(ndarray::s![i, j, k, .., ..])
                        .assign(&inv);
                }
            }
        }

        Ok(Self {
            id,
            nodes,
            jacobian,
            jacobian_det,
            jacobian_inv,
            gll_weights: basis.gll_weights.clone(),
        })
    }

    /// Compute Jacobian matrix and its inverse at reference coordinates (ξ,η,ζ)
    fn compute_jacobian(
        nodes: &Array2<f64>,
        xi: f64,
        eta: f64,
        zeta: f64,
    ) -> KwaversResult<(Array2<f64>, f64, Array2<f64>)> {
        // Hexahedral element shape functions (linear interpolation)
        let n1 = (1.0 - xi) * (1.0 - eta) * (1.0 - zeta) / 8.0;
        let n2 = (1.0 + xi) * (1.0 - eta) * (1.0 - zeta) / 8.0;
        let n3 = (1.0 + xi) * (1.0 + eta) * (1.0 - zeta) / 8.0;
        let n4 = (1.0 - xi) * (1.0 + eta) * (1.0 - zeta) / 8.0;
        let n5 = (1.0 - xi) * (1.0 - eta) * (1.0 + zeta) / 8.0;
        let n6 = (1.0 + xi) * (1.0 - eta) * (1.0 + zeta) / 8.0;
        let n7 = (1.0 + xi) * (1.0 + eta) * (1.0 + zeta) / 8.0;
        let n8 = (1.0 - xi) * (1.0 + eta) * (1.0 + zeta) / 8.0;

        let _shape_functions = [n1, n2, n3, n4, n5, n6, n7, n8];

        // Derivatives of shape functions w.r.t. reference coordinates
        let dn_dxi = [
            -(1.0 - eta) * (1.0 - zeta) / 8.0, // dN1/dξ
            (1.0 - eta) * (1.0 - zeta) / 8.0,  // dN2/dξ
            (1.0 + eta) * (1.0 - zeta) / 8.0,  // dN3/dξ
            -(1.0 + eta) * (1.0 - zeta) / 8.0, // dN4/dξ
            -(1.0 - eta) * (1.0 + zeta) / 8.0, // dN5/dξ
            (1.0 - eta) * (1.0 + zeta) / 8.0,  // dN6/dξ
            (1.0 + eta) * (1.0 + zeta) / 8.0,  // dN7/dξ
            -(1.0 + eta) * (1.0 + zeta) / 8.0, // dN8/dξ
        ];

        let dn_deta = [
            -(1.0 - xi) * (1.0 - zeta) / 8.0, // dN1/dη
            -(1.0 + xi) * (1.0 - zeta) / 8.0, // dN2/dη
            (1.0 + xi) * (1.0 - zeta) / 8.0,  // dN3/dη
            (1.0 - xi) * (1.0 - zeta) / 8.0,  // dN4/dη
            -(1.0 - xi) * (1.0 + zeta) / 8.0, // dN5/dη
            -(1.0 + xi) * (1.0 + zeta) / 8.0, // dN6/dη
            (1.0 + xi) * (1.0 + zeta) / 8.0,  // dN7/dη
            (1.0 - xi) * (1.0 + zeta) / 8.0,  // dN8/dη
        ];

        let dn_dzeta = [
            -(1.0 - xi) * (1.0 - eta) / 8.0, // dN1/dζ
            -(1.0 + xi) * (1.0 - eta) / 8.0, // dN2/dζ
            -(1.0 + xi) * (1.0 + eta) / 8.0, // dN3/dζ
            -(1.0 - xi) * (1.0 + eta) / 8.0, // dN4/dζ
            (1.0 - xi) * (1.0 - eta) / 8.0,  // dN5/dζ
            (1.0 + xi) * (1.0 - eta) / 8.0,  // dN6/dζ
            (1.0 + xi) * (1.0 + eta) / 8.0,  // dN7/dζ
            (1.0 - xi) * (1.0 + eta) / 8.0,  // dN8/dζ
        ];

        // Compute Jacobian matrix J = [∂x/∂ξ, ∂y/∂ξ, ∂z/∂ξ;
        //                              ∂x/∂η, ∂y/∂η, ∂z/∂η;
        //                              ∂x/∂ζ, ∂y/∂ζ, ∂z/∂ζ]
        let mut jacobian = Array2::<f64>::zeros((3, 3));

        for a in 0..3 {
            for i in 0..3 {
                let mut sum = 0.0;
                for n in 0..8 {
                    let dndx = [dn_dxi[n], dn_deta[n], dn_dzeta[n]][i];
                    sum += dndx * nodes[[n, a]];
                }
                jacobian[[a, i]] = sum;
            }
        }

        // Compute determinant
        let det = jacobian[[0, 0]]
            * (jacobian[[1, 1]] * jacobian[[2, 2]] - jacobian[[1, 2]] * jacobian[[2, 1]])
            - jacobian[[0, 1]]
                * (jacobian[[1, 0]] * jacobian[[2, 2]] - jacobian[[1, 2]] * jacobian[[2, 0]])
            + jacobian[[0, 2]]
                * (jacobian[[1, 0]] * jacobian[[2, 1]] - jacobian[[1, 1]] * jacobian[[2, 0]]);

        if det.abs() < 1e-12 {
            return Err(NumericalError::SingularMatrix {
                operation: "SEM Jacobian computation".to_string(),
                condition_number: det.abs(),
            }
            .into());
        }

        // Compute inverse Jacobian
        let mut jacobian_inv = Array2::<f64>::zeros((3, 3));

        jacobian_inv[[0, 0]] =
            (jacobian[[1, 1]] * jacobian[[2, 2]] - jacobian[[1, 2]] * jacobian[[2, 1]]) / det;
        jacobian_inv[[0, 1]] =
            (jacobian[[0, 2]] * jacobian[[2, 1]] - jacobian[[0, 1]] * jacobian[[2, 2]]) / det;
        jacobian_inv[[0, 2]] =
            (jacobian[[0, 1]] * jacobian[[1, 2]] - jacobian[[0, 2]] * jacobian[[1, 1]]) / det;

        jacobian_inv[[1, 0]] =
            (jacobian[[1, 2]] * jacobian[[2, 0]] - jacobian[[1, 0]] * jacobian[[2, 2]]) / det;
        jacobian_inv[[1, 1]] =
            (jacobian[[0, 0]] * jacobian[[2, 2]] - jacobian[[0, 2]] * jacobian[[2, 0]]) / det;
        jacobian_inv[[1, 2]] =
            (jacobian[[0, 2]] * jacobian[[1, 0]] - jacobian[[0, 0]] * jacobian[[1, 2]]) / det;

        jacobian_inv[[2, 0]] =
            (jacobian[[1, 0]] * jacobian[[2, 1]] - jacobian[[1, 1]] * jacobian[[2, 0]]) / det;
        jacobian_inv[[2, 1]] =
            (jacobian[[0, 1]] * jacobian[[2, 0]] - jacobian[[0, 0]] * jacobian[[2, 1]]) / det;
        jacobian_inv[[2, 2]] =
            (jacobian[[0, 0]] * jacobian[[1, 1]] - jacobian[[0, 1]] * jacobian[[1, 0]]) / det;

        Ok((jacobian, det, jacobian_inv))
    }

    /// Map from reference coordinates to physical coordinates
    #[must_use]
    pub fn reference_to_physical(&self, xi: f64, eta: f64, zeta: f64) -> [f64; 3] {
        let mut x = [0.0; 3];

        // Hexahedral shape functions
        let n1 = (1.0 - xi) * (1.0 - eta) * (1.0 - zeta) / 8.0;
        let n2 = (1.0 + xi) * (1.0 - eta) * (1.0 - zeta) / 8.0;
        let n3 = (1.0 + xi) * (1.0 + eta) * (1.0 - zeta) / 8.0;
        let n4 = (1.0 - xi) * (1.0 + eta) * (1.0 - zeta) / 8.0;
        let n5 = (1.0 - xi) * (1.0 - eta) * (1.0 + zeta) / 8.0;
        let n6 = (1.0 + xi) * (1.0 - eta) * (1.0 + zeta) / 8.0;
        let n7 = (1.0 + xi) * (1.0 + eta) * (1.0 + zeta) / 8.0;
        let n8 = (1.0 - xi) * (1.0 + eta) * (1.0 + zeta) / 8.0;

        let shape_functions = [n1, n2, n3, n4, n5, n6, n7, n8];

        for (i, x_i) in x.iter_mut().enumerate() {
            for (j, n_j) in shape_functions.iter().enumerate() {
                *x_i += n_j * self.nodes[[j, i]];
            }
        }

        x
    }

    /// Get element volume
    #[must_use]
    pub fn volume(&self) -> f64 {
        let n_gll = self.gll_weights.len();
        let mut v = 0.0;
        for i in 0..n_gll {
            let wi = self.gll_weights[i];
            for j in 0..n_gll {
                let wij = wi * self.gll_weights[j];
                for k in 0..n_gll {
                    v += self.jacobian_det[[i, j, k]] * wij * self.gll_weights[k];
                }
            }
        }
        v
    }
}

/// Collection of spectral elements forming a mesh
#[derive(Debug)]
pub struct SemMesh {
    /// Spectral elements
    pub elements: Vec<Arc<SemElement>>,
    /// Total number of degrees of freedom
    pub n_dofs: usize,
    /// Basis functions used by all elements
    pub basis: Arc<SemBasis>,
}

impl SemMesh {
    /// Create SEM mesh from element connectivity and node coordinates
    pub fn new(
        element_connectivity: &[Vec<usize>],
        node_coordinates: &Array2<f64>,
        basis: Arc<SemBasis>,
    ) -> KwaversResult<Self> {
        if node_coordinates.ncols() != 3 {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Node coordinates must have 3 columns, got {}",
                node_coordinates.ncols()
            )));
        }

        let mut face_counts: std::collections::HashMap<[usize; 4], usize> =
            std::collections::HashMap::new();

        for connectivity in element_connectivity {
            if connectivity.len() != 8 {
                return Err(crate::core::error::KwaversError::InvalidInput(format!(
                    "Hexahedral element connectivity must have 8 nodes, got {}",
                    connectivity.len()
                )));
            }

            let faces = [
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
                let entry = face_counts.entry(face).or_insert(0);
                *entry += 1;
                if *entry > 2 {
                    return Err(crate::core::error::KwaversError::InvalidInput(format!(
                        "Non-manifold face encountered with nodes {:?}",
                        face
                    )));
                }
            }
        }

        let mut elements = Vec::new();

        for (elem_id, connectivity) in element_connectivity.iter().enumerate() {
            if connectivity.len() != 8 {
                return Err(crate::core::error::KwaversError::InvalidInput(format!(
                    "Hexahedral element connectivity must have 8 nodes, got {}",
                    connectivity.len()
                )));
            }

            let mut seen = std::collections::HashSet::with_capacity(8);
            for &global_node in connectivity {
                if global_node >= node_coordinates.nrows() {
                    return Err(crate::core::error::KwaversError::InvalidInput(format!(
                        "Node index {} out of bounds (n_nodes={})",
                        global_node,
                        node_coordinates.nrows()
                    )));
                }
                if !seen.insert(global_node) {
                    return Err(crate::core::error::KwaversError::InvalidInput(
                        "Hexahedral element connectivity contains duplicate node indices"
                            .to_string(),
                    ));
                }
            }

            // Extract node coordinates for this element
            let mut element_nodes = Array2::<f64>::zeros((8, 3));
            for (local_idx, &global_node) in connectivity.iter().enumerate() {
                for coord in 0..3 {
                    element_nodes[[local_idx, coord]] = node_coordinates[[global_node, coord]];
                }
            }

            let element = SemElement::new(elem_id, element_nodes, &basis)?;
            elements.push(Arc::new(element));
        }

        // Calculate total DOFs: each element has (degree+1)³ nodes
        let n_gll = basis.n_points();
        let dofs_per_element = n_gll * n_gll * n_gll;
        let n_dofs = elements.len() * dofs_per_element;

        Ok(Self {
            elements,
            n_dofs,
            basis,
        })
    }

    /// Get element by ID
    #[must_use]
    pub fn element(&self, id: usize) -> Option<&Arc<SemElement>> {
        self.elements.get(id)
    }

    /// Total mesh volume
    #[must_use]
    pub fn volume(&self) -> f64 {
        self.elements.iter().map(|elem| elem.volume()).sum()
    }
}
