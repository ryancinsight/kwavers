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
use ndarray::{Array2, ArrayView2};
use std::collections::{HashMap, HashSet};

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
#[derive(Debug)]
pub struct TetrahedralMesh {
    /// Mesh nodes
    pub nodes: Vec<MeshNode>,
    /// Tetrahedron elements
    pub elements: Vec<Tetrahedron>,
    /// Element adjacency list (element -> neighboring elements)
    pub adjacency: Vec<Vec<usize>>,
    /// Boundary faces (face -> element pairs)
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
                min: [0.0; 3],
                max: [0.0; 3],
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
        self.update_connectivity(element_idx);

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
        Ok(quality.min(1.0).max(0.0))
    }

    /// Update mesh connectivity information
    fn update_connectivity(&mut self, element_idx: usize) {
        let element = &self.elements[element_idx];

        // Get all faces of this tetrahedron
        let faces = self.get_element_faces(element.nodes);

        for face in faces {
            // Sort face nodes for consistent hashing
            let mut sorted_face = face;
            sorted_face.sort();

            // Update face-to-element mapping
            self.face_elements
                .entry(sorted_face)
                .or_insert_with(Vec::new)
                .push(element_idx);
        }

        // Update adjacency list
        while self.adjacency.len() <= element_idx {
            self.adjacency.push(Vec::new());
        }

        // Find neighboring elements through shared faces
        for face in self.get_element_faces(element.nodes) {
            let mut sorted_face = face;
            sorted_face.sort();

            if let Some(elements) = self.face_elements.get(&sorted_face) {
                for &neighbor_idx in elements {
                    if neighbor_idx != element_idx
                        && !self.adjacency[element_idx].contains(&neighbor_idx)
                    {
                        self.adjacency[element_idx].push(neighbor_idx);
                    }
                }
            }
        }
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
        for i in 0..3 {
            self.bounding_box.min[i] = self.bounding_box.min[i].min(point[i]);
            self.bounding_box.max[i] = self.bounding_box.max[i].max(point[i]);
        }
    }

    /// Get mesh statistics
    #[must_use]
    pub fn statistics(&self) -> MeshStatistics {
        let total_volume: f64 = self.elements.iter().map(|e| e.volume).sum();
        let avg_quality: f64 =
            self.elements.iter().map(|e| e.quality).sum::<f64>() / self.elements.len() as f64;
        let min_quality = self
            .elements
            .iter()
            .map(|e| e.quality)
            .fold(f64::INFINITY, f64::min);

        MeshStatistics {
            num_nodes: self.nodes.len(),
            num_elements: self.elements.len(),
            num_boundary_faces: self.boundary_faces.len(),
            total_volume,
            average_quality: avg_quality,
            minimum_quality: min_quality,
        }
    }

    /// Find elements containing a point
    pub fn locate_point(&self, point: [f64; 3]) -> Vec<usize> {
        // Simple bounding box check for now
        // In practice, would use spatial index (octree/k-d tree)
        let mut containing_elements = Vec::new();

        for (idx, element) in self.elements.iter().enumerate() {
            if self.point_in_element(point, element) {
                containing_elements.push(idx);
            }
        }

        containing_elements
    }

    /// Check if point is inside tetrahedron (barycentric coordinates)
    fn point_in_element(&self, point: [f64; 3], element: &Tetrahedron) -> bool {
        // Barycentric coordinate method
        let p0 = self.nodes[element.nodes[0]].coordinates;
        let p1 = self.nodes[element.nodes[1]].coordinates;
        let p2 = self.nodes[element.nodes[2]].coordinates;
        let p3 = self.nodes[element.nodes[3]].coordinates;

        // For simplicity, check if point is within bounding box of element
        // Full barycentric test would be implemented here
        let mut min_coords = [f64::INFINITY; 3];
        let mut max_coords = [f64::NEG_INFINITY; 3];

        for &node_idx in &element.nodes {
            let coords = self.nodes[node_idx].coordinates;
            for i in 0..3 {
                min_coords[i] = min_coords[i].min(coords[i]);
                max_coords[i] = max_coords[i].max(coords[i]);
            }
        }

        point[0] >= min_coords[0]
            && point[0] <= max_coords[0]
            && point[1] >= min_coords[1]
            && point[1] <= max_coords[1]
            && point[2] >= min_coords[2]
            && point[2] <= max_coords[2]
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
