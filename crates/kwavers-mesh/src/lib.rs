#![doc = include_str!("../README.md")]

pub mod tetrahedral;

pub use tetrahedral::{
    BoundingBox, MeshBoundaryType, MeshNode, MeshStatistics, TetrahedralMesh, Tetrahedron,
};
