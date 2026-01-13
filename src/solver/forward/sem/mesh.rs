//! SEM Mesh Management
//!
//! Handles the creation and management of spectral element meshes.
//! Provides utilities for mesh validation, quality assessment, and
//! conversion from standard mesh formats.

use crate::core::error::KwaversResult;
use ndarray::Array2;
use std::sync::Arc;

use super::basis::SemBasis;
use super::elements::SemMesh;

/// Mesh quality metrics for spectral elements
#[derive(Debug, Clone)]
pub struct MeshQuality {
    /// Minimum scaled Jacobian (should be > 0.2)
    pub min_scaled_jacobian: f64,
    /// Maximum scaled Jacobian
    pub max_scaled_jacobian: f64,
    /// Average scaled Jacobian
    pub avg_scaled_jacobian: f64,
    /// Number of elements with negative Jacobian
    pub negative_jacobians: usize,
    /// Element aspect ratios
    pub aspect_ratios: Vec<f64>,
}

impl MeshQuality {
    /// Assess mesh quality
    #[must_use]
    pub fn assess(mesh: &SemMesh) -> Self {
        let mut min_jac = f64::INFINITY;
        let mut max_jac = f64::NEG_INFINITY;
        let mut sum_jac = 0.0;
        let mut negative_count = 0;
        let mut aspect_ratios = Vec::new();

        let n_gll = mesh.basis.n_points();

        for element in &mesh.elements {
            for i in 0..n_gll {
                for j in 0..n_gll {
                    for k in 0..n_gll {
                        let det = element.jacobian_det[[i, j, k]];
                        let mut col_norms = [0.0; 3];
                        for col in 0..3 {
                            let mut s = 0.0;
                            for row in 0..3 {
                                let v = element.jacobian[[i, j, k, row, col]];
                                s += v * v;
                            }
                            col_norms[col] = s.sqrt();
                        }
                        let denom = col_norms[0] * col_norms[1] * col_norms[2];
                        let scaled = if denom > 0.0 { det / denom } else { 0.0 };

                        min_jac = min_jac.min(scaled);
                        max_jac = max_jac.max(scaled);
                        sum_jac += scaled;

                        if scaled < 0.0 {
                            negative_count += 1;
                        }
                    }
                }
            }

            // Calculate element aspect ratio (simplified)
            let dims = element.nodes.outer_iter().fold(
                [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ],
                |[min_x, min_y, min_z, max_x, max_y, max_z], node| {
                    [
                        min_x.min(node[0]),
                        min_y.min(node[1]),
                        min_z.min(node[2]),
                        max_x.max(node[0]),
                        max_y.max(node[1]),
                        max_z.max(node[2]),
                    ]
                },
            );

            let length_x = dims[3] - dims[0];
            let length_y = dims[4] - dims[1];
            let length_z = dims[5] - dims[2];

            let max_len = length_x.max(length_y).max(length_z);
            let min_len = length_x.min(length_y).min(length_z);
            let aspect_ratio = if min_len > 0.0 {
                max_len / min_len
            } else {
                f64::INFINITY
            };
            aspect_ratios.push(aspect_ratio);
        }

        let total_points = mesh.elements.len() * n_gll.pow(3);
        let avg_jac = sum_jac / total_points as f64;

        Self {
            min_scaled_jacobian: min_jac,
            max_scaled_jacobian: max_jac,
            avg_scaled_jacobian: avg_jac,
            negative_jacobians: negative_count,
            aspect_ratios,
        }
    }

    /// Check if mesh quality is acceptable
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        self.min_scaled_jacobian > 0.1
            && self.negative_jacobians == 0
            && self.aspect_ratios.iter().all(|&ar| ar < 10.0)
    }
}

/// Utilities for creating SEM meshes from standard formats
#[derive(Debug)]
pub struct MeshBuilder;

impl MeshBuilder {
    /// Create SEM mesh from hexahedral connectivity
    ///
    /// # Arguments
    /// * `connectivity` - Vector of element connectivity (each element has 8 node indices)
    /// * `node_coords` - Node coordinates array (n_nodes × 3)
    /// * `polynomial_degree` - SEM polynomial degree
    ///
    /// # Returns
    /// SEM mesh ready for simulation
    pub fn from_hexahedral_mesh(
        connectivity: Vec<Vec<usize>>,
        node_coords: Array2<f64>,
        polynomial_degree: usize,
    ) -> KwaversResult<SemMesh> {
        let basis = Arc::new(SemBasis::new(polynomial_degree));
        SemMesh::new(&connectivity, &node_coords, basis)
    }

    /// Create simple rectangular mesh for testing
    ///
    /// Creates a single hexahedral element spanning [0,Lx] × [0,Ly] × [0,Lz]
    #[must_use]
    pub fn create_rectangular_mesh(lx: f64, ly: f64, lz: f64, polynomial_degree: usize) -> SemMesh {
        // Define 8 corner nodes of a cube
        let mut node_coords = Array2::<f64>::zeros((8, 3));
        node_coords[[0, 0]] = 0.0;
        node_coords[[0, 1]] = 0.0;
        node_coords[[0, 2]] = 0.0; // (0,0,0)
        node_coords[[1, 0]] = lx;
        node_coords[[1, 1]] = 0.0;
        node_coords[[1, 2]] = 0.0; // (Lx,0,0)
        node_coords[[2, 0]] = lx;
        node_coords[[2, 1]] = ly;
        node_coords[[2, 2]] = 0.0; // (Lx,Ly,0)
        node_coords[[3, 0]] = 0.0;
        node_coords[[3, 1]] = ly;
        node_coords[[3, 2]] = 0.0; // (0,Ly,0)
        node_coords[[4, 0]] = 0.0;
        node_coords[[4, 1]] = 0.0;
        node_coords[[4, 2]] = lz; // (0,0,Lz)
        node_coords[[5, 0]] = lx;
        node_coords[[5, 1]] = 0.0;
        node_coords[[5, 2]] = lz; // (Lx,0,Lz)
        node_coords[[6, 0]] = lx;
        node_coords[[6, 1]] = ly;
        node_coords[[6, 2]] = lz; // (Lx,Ly,Lz)
        node_coords[[7, 0]] = 0.0;
        node_coords[[7, 1]] = ly;
        node_coords[[7, 2]] = lz; // (0,Ly,Lz)

        // Single element connectivity (0-based node indices)
        let connectivity = vec![vec![0, 1, 2, 3, 4, 5, 6, 7]];

        let basis = Arc::new(SemBasis::new(polynomial_degree));
        SemMesh::new(&connectivity, &node_coords, basis).unwrap()
    }

    /// Validate mesh for SEM compatibility
    pub fn validate_mesh(mesh: &SemMesh) -> KwaversResult<MeshQuality> {
        let quality = MeshQuality::assess(mesh);

        if !quality.is_acceptable() {
            return Err(crate::core::error::KwaversError::InvalidInput(format!(
                "Mesh quality unacceptable: min Jacobian = {:.3}, {} negative Jacobians",
                quality.min_scaled_jacobian, quality.negative_jacobians
            )));
        }

        Ok(quality)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rectangular_mesh_creation() {
        let mesh = MeshBuilder::create_rectangular_mesh(1.0, 2.0, 3.0, 3);

        assert_eq!(mesh.elements.len(), 1);
        assert_eq!(mesh.basis.degree, 3);
        assert_eq!(mesh.n_dofs, 64); // (3+1)³ = 64 DOFs per element

        // Check volume calculation
        assert_relative_eq!(mesh.volume(), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mesh_quality_assessment() {
        let mesh = MeshBuilder::create_rectangular_mesh(1.0, 1.0, 1.0, 2);
        let quality = MeshQuality::assess(&mesh);

        // Unit cube should have good quality
        assert!(quality.min_scaled_jacobian > 0.5);
        assert_eq!(quality.negative_jacobians, 0);
        assert!(quality.is_acceptable());
    }

    #[test]
    fn test_mesh_validation() {
        let mesh = MeshBuilder::create_rectangular_mesh(1.0, 1.0, 1.0, 2);
        let quality = MeshBuilder::validate_mesh(&mesh).unwrap();

        assert!(quality.avg_scaled_jacobian > 0.0);
    }
}
