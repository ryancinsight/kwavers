//! Types for tetrahedral mesh.

/// 3D node in tetrahedral mesh
#[derive(Debug, Clone, Copy)]
pub struct MeshNode {
    /// Node coordinates (x, y, z)
    pub coordinates: [f64; 3],
    /// Boundary condition type
    pub boundary_type: MeshBoundaryType,
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

/// Mesh node/face classification for FEM boundary handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshBoundaryType {
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

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub min: [f64; 3],
    pub max: [f64; 3],
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
