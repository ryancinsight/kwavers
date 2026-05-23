use crate::core::error::KwaversResult;
use crate::domain::mesh::tetrahedral::TetrahedralMesh;
use nalgebra::Vector3;
use std::collections::HashMap;

#[cfg(test)]
mod tests;

/// Interface definition between FEM and BEM domains
#[derive(Debug, Clone)]
pub struct BemFemInterface {
    /// FEM mesh nodes at interface
    pub fem_interface_nodes: Vec<usize>,
    /// BEM boundary elements at interface
    pub bem_interface_elements: Vec<usize>,
    /// Interface mapping (FEM node index → BEM element index)
    pub node_element_mapping: HashMap<usize, usize>,
    /// Interface quadrature points and weights
    pub quadrature_points: Vec<(f64, f64, f64)>,
    pub quadrature_weights: Vec<f64>,
    /// Interface normals (pointing from FEM to BEM domain)
    pub interface_normals: Vec<(f64, f64, f64)>,
}

impl BemFemInterface {
    /// Create BEM-FEM interface from FEM mesh and BEM boundary
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(fem_mesh: &TetrahedralMesh, bem_boundary: &[usize]) -> KwaversResult<Self> {
        // Find FEM nodes that lie on the BEM boundary
        let mut fem_interface_nodes = Vec::new();
        let mut node_element_mapping = HashMap::new();

        // For each FEM node, check if it's on the BEM boundary
        for (node_idx, node) in fem_mesh.nodes.iter().enumerate() {
            // Check if this node is on the BEM interface
            // This would typically involve geometric checks or pre-computed mappings
            let is_interface = Self::is_node_on_interface(node, bem_boundary, fem_mesh);

            if is_interface {
                fem_interface_nodes.push(node_idx);

                // Find corresponding BEM element (simplified)
                // In practice, this would involve more sophisticated geometric queries
                let bem_element =
                    Self::find_corresponding_bem_element(node, bem_boundary, fem_mesh)?;
                node_element_mapping.insert(node_idx, bem_element);
            }
        }

        // Generate interface geometry
        let (quadrature_points, quadrature_weights) =
            Self::generate_interface_quadrature(&fem_interface_nodes, fem_mesh);
        let interface_normals = Self::compute_interface_normals(&fem_interface_nodes, fem_mesh);

        Ok(Self {
            fem_interface_nodes,
            bem_interface_elements: bem_boundary.to_vec(),
            node_element_mapping,
            quadrature_points,
            quadrature_weights,
            interface_normals,
        })
    }

    /// Check if a FEM node lies on the BEM interface
    fn is_node_on_interface(
        node: &crate::domain::mesh::tetrahedral::MeshNode,
        bem_boundary: &[usize],
        fem_mesh: &TetrahedralMesh,
    ) -> bool {
        // Fast path: if the node index is explicitly in the boundary list
        if bem_boundary.contains(&node.index) {
            return true;
        }

        // Geometric check: check if node coordinates lie on the boundary surface defined by BEM elements
        // Since bem_boundary contains node indices, we check proximity to any of these nodes.
        let tolerance_sq = 1e-12; // Corresponding to 1e-6 distance

        for &bem_node_idx in bem_boundary {
            if let Some(bem_node) = fem_mesh.nodes.get(bem_node_idx) {
                let dx = node.coordinates[0] - bem_node.coordinates[0];
                let dy = node.coordinates[1] - bem_node.coordinates[1];
                let dz = node.coordinates[2] - bem_node.coordinates[2];

                if dz.mul_add(dz, dx.mul_add(dx, dy * dy)) < tolerance_sq {
                    return true;
                }
            }
        }

        false
    }

    /// Find corresponding BEM element for a FEM node
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn find_corresponding_bem_element(
        node: &crate::domain::mesh::tetrahedral::MeshNode,
        bem_boundary: &[usize],
        fem_mesh: &TetrahedralMesh,
    ) -> KwaversResult<usize> {
        bem_boundary
            .iter()
            .copied()
            .filter_map(|idx| {
                fem_mesh.nodes.get(idx).map(|bem_node| {
                    let dx = node.coordinates[0] - bem_node.coordinates[0];
                    let dy = node.coordinates[1] - bem_node.coordinates[1];
                    let dz = node.coordinates[2] - bem_node.coordinates[2];
                    let dist_sq = dz.mul_add(dz, dx.mul_add(dx, dy * dy));
                    (dist_sq, idx)
                })
            })
            .min_by(|(d1, _), (d2, _)| d1.total_cmp(d2))
            .map(|(_, idx)| idx)
            .ok_or_else(|| {
                crate::core::error::KwaversError::InvalidInput(
                    "Could not find corresponding BEM element: boundary list empty or invalid indices".to_owned(),
                )
            })
    }

    /// Generate quadrature points and weights for interface integration
    fn generate_interface_quadrature(
        fem_nodes: &[usize],
        fem_mesh: &TetrahedralMesh,
    ) -> (Vec<(f64, f64, f64)>, Vec<f64>) {
        let mut points = Vec::new();
        let mut weights = Vec::new();

        for &node_idx in fem_nodes {
            if let Some(node) = fem_mesh.nodes.get(node_idx) {
                points.push((
                    node.coordinates[0],
                    node.coordinates[1],
                    node.coordinates[2],
                ));
                weights.push(1.0 / fem_nodes.len() as f64); // Equal weights
            }
        }

        (points, weights)
    }

    /// Compute interface normals
    pub(super) fn compute_interface_normals(
        fem_nodes: &[usize],
        fem_mesh: &TetrahedralMesh,
    ) -> Vec<(f64, f64, f64)> {
        let mut normals = Vec::new();
        let mut accumulated_normals: HashMap<usize, Vector3<f64>> = HashMap::new();

        // Accumulate weighted normals from boundary faces
        for (face_nodes, &(elem_idx, _)) in &fem_mesh.boundary_faces {
            // Get element to determine orientation
            if let Some(element) = fem_mesh.elements.get(elem_idx) {
                // Get face vertices
                // Note: face_nodes are sorted, so we can't trust winding order from them.
                // We use local_face_idx to get proper winding if needed, or check against 4th node.
                // Let's get the 3 nodes of the face.
                let n1_idx = face_nodes[0];
                let n2_idx = face_nodes[1];
                let n3_idx = face_nodes[2];

                if let (Some(n1), Some(n2), Some(n3)) = (
                    fem_mesh.nodes.get(n1_idx),
                    fem_mesh.nodes.get(n2_idx),
                    fem_mesh.nodes.get(n3_idx),
                ) {
                    let v1 = Vector3::from(n1.coordinates);
                    let v2 = Vector3::from(n2.coordinates);
                    let v3 = Vector3::from(n3.coordinates);

                    // Compute face normal (unnormalized to weight by area)
                    let edge1 = v2 - v1;
                    let edge2 = v3 - v1;
                    let mut face_normal = edge1.cross(&edge2);

                    // Identify the 4th node (opposite node)
                    // element.nodes has 4 indices. face_nodes has 3.
                    // The one missing in face_nodes is the opposite node.
                    let opp_node_idx = element
                        .nodes
                        .iter()
                        .find(|&&idx| !face_nodes.contains(&idx))
                        .copied();

                    if let Some(opp_idx) = opp_node_idx {
                        if let Some(opp_node) = fem_mesh.nodes.get(opp_idx) {
                            let v_opp = Vector3::from(opp_node.coordinates);
                            // Vector from a face vertex to opposite node
                            let to_interior = v_opp - v1;

                            // If normal points towards interior (dot > 0), flip it
                            if face_normal.dot(&to_interior) > 0.0 {
                                face_normal = -face_normal;
                            }
                        }
                    }

                    // Accumulate to vertices
                    for &idx in face_nodes {
                        accumulated_normals
                            .entry(idx)
                            .and_modify(|n| *n += face_normal)
                            .or_insert(face_normal);
                    }
                }
            }
        }

        for &node_idx in fem_nodes {
            if let Some(normal) = accumulated_normals.get(&node_idx) {
                let norm = normal.norm();
                if norm > 1e-12 {
                    let n = normal / norm;
                    normals.push((n.x, n.y, n.z));
                } else {
                    normals.push((0.0, 0.0, 1.0)); // Fallback for degenerate geometry
                }
            } else {
                // Node might not be on the boundary mesh stored in boundary_faces
                // or mesh connectivity issue. Return default.
                normals.push((0.0, 0.0, 1.0));
            }
        }

        normals
    }
}
