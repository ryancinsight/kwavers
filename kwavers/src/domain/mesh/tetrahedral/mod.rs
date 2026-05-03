//! Tetrahedral Mesh for 3D Finite Element Methods
//!
//! Provides data structures and algorithms for tetrahedral meshes used in:
//! - Helmholtz equation FEM solvers
//! - Elastic wave propagation
//! - Thermal diffusion problems
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

pub mod mesh;
#[cfg(test)]
mod tests;
pub mod types;

pub use mesh::TetrahedralMesh;
pub use types::{BoundaryType, BoundingBox, MeshNode, MeshStatistics, Tetrahedron};
