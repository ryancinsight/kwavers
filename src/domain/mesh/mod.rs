//! Mesh Infrastructure for Finite Element Methods
//!
//! Provides geometric data structures and operations for:
//! - Tetrahedral meshes for 3D FEM
//! - Mesh quality metrics and optimization
//! - Geometry queries and interpolation

pub mod tetrahedral;

pub use tetrahedral::{
    BoundaryType, BoundingBox, MeshNode, MeshStatistics, TetrahedralMesh, Tetrahedron,
};
