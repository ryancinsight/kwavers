//! Mesh Infrastructure for Finite Element Methods
//!
//! Provides geometric data structures and operations for:
//! - Tetrahedral meshes for 3D FEM
//! - Mesh quality metrics and optimization
//! - Geometry queries and interpolation
//!
//! gaia is the authoritative external mesh/STL boundary. This crate keeps
//! kwavers' solver-facing [`TetrahedralMesh`] representation and converts from
//! `gaia::IndexedMesh` through [`TetrahedralMesh::from_gaia_indexed_mesh`]; it does
//! not maintain an independent STL writer.

pub mod tetrahedral;

pub use tetrahedral::{
    BoundingBox, MeshBoundaryType, MeshNode, MeshStatistics, TetrahedralMesh, Tetrahedron,
};
