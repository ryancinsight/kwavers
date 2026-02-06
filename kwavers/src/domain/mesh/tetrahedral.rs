//! Tetrahedral Mesh for 3D Finite Element Methods
//!
//! Provides data structures and algorithms for tetrahedral meshes used in:
//! - Helmholtz equation FEM solvers
//! - Elastic wave propagation
//! - Thermal diffusion problems
//!
//! ## Data Structures
//!
//! - `TetrahedralMesh`: Main mesh structure with nodes, elements, and topology
//! - `Tetrahedron`: Individual element with 4 nodes and material properties
//! - `MeshNode`: 3D node with coordinates and boundary conditions
//!
//! ## Features
//!
//! - Efficient adjacency queries
//! - Boundary face detection
//! - Volume and quality calculations
//! - Parallel mesh operations
//!
//! ## References
//!
//! - Shewchuk (1996): "Triangle: Engineering a 2D quality mesh generator"
//! - Si (2015): "TetGen: A quality tetrahedral mesh generator"

use crate::core::error::{KwaversError, KwaversResult};
use nalgebra::{Matrix3, Vector3};
use std::collections::HashMap;

/// 3D node in tetrahedral mesh
#[derive(Debug, Clone, Copy)]
pub struct MeshNode {
    /// Node coordinates (x, y, z)
    pub coordinates: [f64; 3],
    /// Boundary condition type
    pub boundary_type: BoundaryType,
    /// Node index in global mesh
    pub index: usize,
}

/// Tetrahedron element with 4 nodes
#[derive(Debug, Clone)]
pub struct Tetrahedron {
    /// Node indices (4 vertices)
    pub nodes: [usize; 4],
    /// Material ID for heterogeneous media
    pub material_id: usize,
    /// Element volume
    pub volume: f64,
    /// Element quality metric (0-1, higher is better)
    pub quality: f64,
}

/// Boundary condition types for mesh nodes/faces
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    /// Interior node/face
    Interior,
    /// Dirichlet boundary (prescribed field)
    Dirichlet,
    /// Neumann boundary (prescribed normal derivative)
    Neumann,
    /// Robin boundary (mixed condition)
    Robin,
    /// Radiation/absorbing boundary
    Radiation,
}

/// Tetrahedral mesh for 3D FEM
#[derive(Debug, Clone)]
pub struct TetrahedralMesh {
    /// Mesh nodes
    pub nodes: Vec<MeshNode>,
    /// Tetrahedron elements
    pub elements: Vec<Tetrahedron>,
    /// Element adjacency list (element -> neighboring elements)
    pub adjacency: Vec<Vec<usize>>,
    /// Boundary faces (sorted face nodes -> (element index, local face index))
    pub boundary_faces: HashMap<[usize; 3], (usize, usize)>,
    /// Face-to-element connectivity
    pub face_elements: HashMap<[usize; 3], Vec<usize>>,
    /// Mesh bounding box
    pub bounding_box: BoundingBox,
}

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl TetrahedralMesh {
    /// Create empty tetrahedral mesh
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            elements: Vec::new(),
            adjacency: Vec::new(),
            boundary_faces: HashMap::new(),
            face_elements: HashMap::new(),
            bounding_box: BoundingBox {
                min: [f64::INFINITY; 3],
                max: [f64::NEG_INFINITY; 3],
            },
        }
    }

    /// Add node to mesh
    pub fn add_node(&mut self, coordinates: [f64; 3], boundary_type: BoundaryType) -> usize {
        let index = self.nodes.len();
        self.nodes.push(MeshNode {
            coordinates,
            boundary_type,
            index,
        });
        self.update_bounding_box(coordinates);
        index
    }

    /// Add tetrahedron element
    pub fn add_element(&mut self, nodes: [usize; 4], material_id: usize) -> KwaversResult<usize> {
        // Validate node indices
        for &node_idx in &nodes {
            if node_idx >= self.nodes.len() {
                return Err(KwaversError::InvalidInput(format!(
                    "Node index {} out of bounds",
                    node_idx
                )));
            }
        }

        for (face_idx, face) in self.get_element_faces(nodes).iter().enumerate() {
            let mut sorted_face = *face;
            sorted_face.sort();

            if let Some(existing) = self.face_elements.get(&sorted_face) {
                if existing.len() >= 2 {
                    return Err(KwaversError::InvalidInput(format!(
                        "Non-manifold face encountered at face_idx {} with nodes {:?}",
                        face_idx, sorted_face
                    )));
                }
            }
        }

        // Calculate element properties
        let volume = self.calculate_tetrahedron_volume(nodes)?;
        let quality = self.calculate_element_quality(nodes)?;

        let element = Tetrahedron {
            nodes,
            material_id,
            volume,
            quality,
        };

        let element_idx = self.elements.len();
        self.elements.push(element);

        // Update connectivity
        self.update_connectivity(element_idx)?;

        Ok(element_idx)
    }

    /// Calculate tetrahedron volume using scalar triple product
    fn calculate_tetrahedron_volume(&self, nodes: [usize; 4]) -> KwaversResult<f64> {
        let p0 = self.nodes[nodes[0]].coordinates;
        let p1 = self.nodes[nodes[1]].coordinates;
        let p2 = self.nodes[nodes[2]].coordinates;
        let p3 = self.nodes[nodes[3]].coordinates;

        // Vectors from p0 to other points
        let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let v3 = [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]];

        // Scalar triple product: (v1 × v2) · v3
        let cross_x = v1[1] * v2[2] - v1[2] * v2[1];
        let cross_y = v1[2] * v2[0] - v1[0] * v2[2];
        let cross_z = v1[0] * v2[1] - v1[1] * v2[0];

        let volume = (cross_x * v3[0] + cross_y * v3[1] + cross_z * v3[2]).abs() / 6.0;

        if volume < 1e-12 {
            return Err(KwaversError::InvalidInput(
                "Degenerate tetrahedron with zero volume".to_string(),
            ));
        }

        Ok(volume)
    }

    /// Calculate element quality metric (0-1, higher is better)
    fn calculate_element_quality(&self, nodes: [usize; 4]) -> KwaversResult<f64> {
        // Volume-based quality metric
        // Q = 6√2 * V / (∑_{edges} L_e^2)^{3/2}
        // where V is volume, L_e are edge lengths

        let volume = self.calculate_tetrahedron_volume(nodes)?;

        // Calculate edge lengths
        let edges = [
            (nodes[0], nodes[1]),
            (nodes[0], nodes[2]),
            (nodes[0], nodes[3]),
            (nodes[1], nodes[2]),
            (nodes[1], nodes[3]),
            (nodes[2], nodes[3]),
        ];

        let mut edge_sum_sq = 0.0;
        for (i, j) in edges {
            let p1 = self.nodes[i].coordinates;
            let p2 = self.nodes[j].coordinates;
            let dx = p1[0] - p2[0];
            let dy = p1[1] - p2[1];
            let dz = p1[2] - p2[2];
            edge_sum_sq += dx * dx + dy * dy + dz * dz;
        }

        if edge_sum_sq < 1e-12 {
            return Ok(0.0); // Degenerate element
        }

        let quality = 6.0 * 2.0_f64.sqrt() * volume / edge_sum_sq.powf(1.5);
        Ok(quality.clamp(0.0, 1.0))
    }

    /// Update mesh connectivity information
    fn update_connectivity(&mut self, element_idx: usize) -> KwaversResult<()> {
        let nodes = self.elements[element_idx].nodes;
        let faces = self.get_element_faces(nodes);

        while self.adjacency.len() <= element_idx {
            self.adjacency.push(Vec::new());
        }

        for (face_idx, face) in faces.into_iter().enumerate() {
            // Sort face nodes for consistent hashing
            let mut sorted_face = face;
            sorted_face.sort();

            // Update face-to-element mapping
            let entry = self.face_elements.entry(sorted_face).or_default();
            entry.push(element_idx);
            if entry.len() > 2 {
                return Err(KwaversError::InvalidInput(format!(
                    "Non-manifold face encountered with nodes {:?}",
                    sorted_face
                )));
            }

            match entry.len() {
                1 => {
                    self.boundary_faces
                        .insert(sorted_face, (element_idx, face_idx));
                }
                2 => {
                    self.boundary_faces.remove(&sorted_face);

                    let other_idx = entry[0];
                    if other_idx != element_idx && !self.adjacency[element_idx].contains(&other_idx)
                    {
                        self.adjacency[element_idx].push(other_idx);
                    }
                    if other_idx != element_idx {
                        while self.adjacency.len() <= other_idx {
                            self.adjacency.push(Vec::new());
                        }
                        if !self.adjacency[other_idx].contains(&element_idx) {
                            self.adjacency[other_idx].push(element_idx);
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Get the 4 triangular faces of a tetrahedron
    fn get_element_faces(&self, nodes: [usize; 4]) -> [[usize; 3]; 4] {
        [
            [nodes[0], nodes[1], nodes[2]], // Face 0: opposite node 3
            [nodes[0], nodes[1], nodes[3]], // Face 1: opposite node 2
            [nodes[0], nodes[2], nodes[3]], // Face 2: opposite node 1
            [nodes[1], nodes[2], nodes[3]], // Face 3: opposite node 0
        ]
    }

    /// Update bounding box with new point
    fn update_bounding_box(&mut self, point: [f64; 3]) {
        for (i, coord) in point.iter().enumerate() {
            self.bounding_box.min[i] = self.bounding_box.min[i].min(*coord);
            self.bounding_box.max[i] = self.bounding_box.max[i].max(*coord);
        }
    }

    /// Get mesh statistics
    #[must_use]
    pub fn statistics(&self) -> MeshStatistics {
        let total_volume: f64 = self.elements.iter().map(|e| e.volume).sum();
        let (average_quality, minimum_quality) = if self.elements.is_empty() {
            (0.0, 0.0)
        } else {
            let average_quality =
                self.elements.iter().map(|e| e.quality).sum::<f64>() / self.elements.len() as f64;
            let minimum_quality = self
                .elements
                .iter()
                .map(|e| e.quality)
                .fold(f64::INFINITY, f64::min);
            (average_quality, minimum_quality)
        };

        MeshStatistics {
            num_nodes: self.nodes.len(),
            num_elements: self.elements.len(),
            num_boundary_faces: self.boundary_faces.len(),
            total_volume,
            average_quality,
            minimum_quality,
        }
    }

    /// Find elements containing a point
    pub fn locate_point(&self, point: [f64; 3]) -> Vec<usize> {
        let mut containing_elements = Vec::new();

        if !self.point_in_bounding_box(point) {
            return containing_elements;
        }

        for (idx, element) in self.elements.iter().enumerate() {
            if self.point_in_element(point, element) {
                containing_elements.push(idx);
            }
        }

        containing_elements
    }

    /// Check if point is inside tetrahedron (barycentric coordinates)
    fn point_in_element(&self, point: [f64; 3], element: &Tetrahedron) -> bool {
        let a = self.nodes[element.nodes[0]].coordinates;
        let b = self.nodes[element.nodes[1]].coordinates;
        let c = self.nodes[element.nodes[2]].coordinates;
        let d = self.nodes[element.nodes[3]].coordinates;

        let a = Vector3::new(a[0], a[1], a[2]);
        let b = Vector3::new(b[0], b[1], b[2]);
        let c = Vector3::new(c[0], c[1], c[2]);
        let d = Vector3::new(d[0], d[1], d[2]);
        let p = Vector3::new(point[0], point[1], point[2]);

        let m = Matrix3::from_columns(&[b - a, c - a, d - a]);
        let Some(inv) = m.try_inverse() else {
            return false;
        };

        let uvw = inv * (p - a);
        let u = uvw[0];
        let v = uvw[1];
        let w = uvw[2];
        let t = 1.0 - u - v - w;

        let eps = 1e-12;
        u >= -eps && v >= -eps && w >= -eps && t >= -eps
    }

    fn point_in_bounding_box(&self, point: [f64; 3]) -> bool {
        let eps = 1e-12;
        point[0] >= self.bounding_box.min[0] - eps
            && point[0] <= self.bounding_box.max[0] + eps
            && point[1] >= self.bounding_box.min[1] - eps
            && point[1] <= self.bounding_box.max[1] + eps
            && point[2] >= self.bounding_box.min[2] - eps
            && point[2] <= self.bounding_box.max[2] + eps
    }
}

/// Mesh statistics for quality assessment
#[derive(Debug, Clone)]
pub struct MeshStatistics {
    pub num_nodes: usize,
    pub num_elements: usize,
    pub num_boundary_faces: usize,
    pub total_volume: f64,
    pub average_quality: f64,
    pub minimum_quality: f64,
}

impl Default for TetrahedralMesh {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{BoundaryType, TetrahedralMesh};

    #[test]
    fn tetrahedron_volume_unit_is_one_sixth() {
        let mut mesh = TetrahedralMesh::new();
        let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
        let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
        let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
        let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);
        let e0 = mesh.add_element([n0, n1, n2, n3], 0).unwrap();
        let vol = mesh.elements[e0].volume;
        assert!((vol - 1.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn point_location_barycentric_is_correct() {
        let mut mesh = TetrahedralMesh::new();
        let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
        let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
        let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
        let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);
        mesh.add_element([n0, n1, n2, n3], 0).unwrap();

        let inside = mesh.locate_point([0.1, 0.1, 0.1]);
        assert_eq!(inside, vec![0]);

        let on_face = mesh.locate_point([0.3, 0.3, 0.0]);
        assert_eq!(on_face, vec![0]);

        let outside = mesh.locate_point([1.0, 1.0, 1.0]);
        assert!(outside.is_empty());
    }

    #[test]
    fn adjacency_is_symmetric_for_shared_face() {
        let mut mesh = TetrahedralMesh::new();
        let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
        let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
        let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
        let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);
        let n4 = mesh.add_node([0.0, 0.0, -1.0], BoundaryType::Interior);
        let e0 = mesh.add_element([n0, n1, n2, n3], 0).unwrap();
        let e1 = mesh.add_element([n0, n1, n2, n4], 0).unwrap();

        assert!(mesh.adjacency[e0].contains(&e1));
        assert!(mesh.adjacency[e1].contains(&e0));
    }

    #[test]
    fn non_manifold_face_is_rejected() {
        let mut mesh = TetrahedralMesh::new();
        let n0 = mesh.add_node([0.0, 0.0, 0.0], BoundaryType::Interior);
        let n1 = mesh.add_node([1.0, 0.0, 0.0], BoundaryType::Interior);
        let n2 = mesh.add_node([0.0, 1.0, 0.0], BoundaryType::Interior);
        let n3 = mesh.add_node([0.0, 0.0, 1.0], BoundaryType::Interior);
        let n4 = mesh.add_node([0.0, 0.0, -1.0], BoundaryType::Interior);
        let n5 = mesh.add_node([0.0, 0.0, 2.0], BoundaryType::Interior);

        mesh.add_element([n0, n1, n2, n3], 0).unwrap();
        mesh.add_element([n0, n1, n2, n4], 0).unwrap();
        let err = mesh.add_element([n0, n1, n2, n5], 0).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Non-manifold face encountered"));
    }
}
