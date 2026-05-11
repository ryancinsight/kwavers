use super::{BemConfig, BemSolver};
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::boundary::BemBoundaryManager;
use crate::domain::mesh::tetrahedral::TetrahedralMesh;
use std::collections::HashMap;

impl BemSolver {
    /// Create BEM solver directly from pre-extracted boundary vertices and triangles.
    ///
    /// Use this constructor when the boundary surface has already been extracted.
    /// Triangles must follow CCW outward-normal winding convention.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn new(
        config: BemConfig,
        vertices: Vec<[f64; 3]>,
        triangles: Vec<[usize; 3]>,
    ) -> KwaversResult<Self> {
        let n = vertices.len();
        for tri in &triangles {
            for &idx in tri {
                if idx >= n {
                    return Err(KwaversError::InvalidInput(format!(
                        "BEM triangle index {} out of bounds (vertices: {})",
                        idx, n
                    )));
                }
            }
        }
        Ok(Self {
            config,
            vertices,
            triangles,
            global_to_local_node: HashMap::new(),
            boundary_manager: BemBoundaryManager::new(),
            h_matrix: None,
            g_matrix: None,
        })
    }

    /// Create BEM solver by extracting the boundary surface from a tetrahedral mesh.
    ///
    /// Only faces shared by exactly one element (boundary faces) are included.
    /// Outward normal orientation is determined by the position of the interior node.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn from_mesh(config: BemConfig, mesh: &TetrahedralMesh) -> KwaversResult<Self> {
        let mut nodes: Vec<[f64; 3]> = Vec::new();
        let mut triangles_local: Vec<[usize; 3]> = Vec::new();
        let mut global_to_local_node = HashMap::new();

        for (sorted_nodes, &(elem_idx, _face_idx)) in &mesh.boundary_faces {
            let element = &mesh.elements[elem_idx];
            let elem_nodes = element.nodes;

            let mut face_nodes = Vec::new();

            let interior_node: usize = elem_nodes
                .iter()
                .find(|&&n| !sorted_nodes.contains(&n))
                .copied()
                .ok_or_else(|| {
                    KwaversError::InvalidInput(format!(
                        "BEM mesh element {:?} has no interior node — degenerate boundary face \
                         (all 4 nodes are on the boundary surface)",
                        elem_nodes
                    ))
                })?;

            for &n_idx in &elem_nodes {
                if sorted_nodes.contains(&n_idx) {
                    face_nodes.push(n_idx);
                }
            }

            if face_nodes.len() != 3 {
                continue;
            }

            let p0 = mesh.nodes[face_nodes[0]].coordinates;
            let p1 = mesh.nodes[face_nodes[1]].coordinates;
            let p2 = mesh.nodes[face_nodes[2]].coordinates;
            let p_in = mesh.nodes[interior_node].coordinates;

            let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

            let nx = v1[1].mul_add(v2[2], -(v1[2] * v2[1]));
            let ny = v1[2].mul_add(v2[0], -(v1[0] * v2[2]));
            let nz = v1[0].mul_add(v2[1], -(v1[1] * v2[0]));

            let v_in = [p_in[0] - p0[0], p_in[1] - p0[1], p_in[2] - p0[2]];
            let dot = nz.mul_add(v_in[2], nx.mul_add(v_in[0], ny * v_in[1]));

            let final_face_nodes = if dot > 0.0 {
                [face_nodes[0], face_nodes[2], face_nodes[1]]
            } else {
                [face_nodes[0], face_nodes[1], face_nodes[2]]
            };

            let mut local_face = [0; 3];
            for (i, &global_idx) in final_face_nodes.iter().enumerate() {
                if let Some(&local_idx) = global_to_local_node.get(&global_idx) {
                    local_face[i] = local_idx;
                } else {
                    let new_idx = nodes.len();
                    nodes.push(mesh.nodes[global_idx].coordinates);
                    global_to_local_node.insert(global_idx, new_idx);
                    local_face[i] = new_idx;
                }
            }
            triangles_local.push(local_face);
        }

        Ok(Self {
            config,
            vertices: nodes,
            triangles: triangles_local,
            global_to_local_node,
            boundary_manager: BemBoundaryManager::new(),
            h_matrix: None,
            g_matrix: None,
        })
    }
}
