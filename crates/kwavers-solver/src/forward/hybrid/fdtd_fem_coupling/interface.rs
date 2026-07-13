use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_mesh::tetrahedral::TetrahedralMesh;

/// Paired interface nodes: FDTD grid indices and the matching FEM node indices.
type InterfaceNodePair = (Vec<(usize, usize, usize)>, Vec<usize>);
/// Per-interface-point geometry: outward `(x, y, z)` normals and surface areas.
type InterfaceGeometryData = (Vec<(f64, f64, f64)>, Vec<f64>);

/// Interface definition between FDTD and FEM domains
#[derive(Debug, Clone)]
pub struct FdtdFemInterface {
    /// FDTD grid indices at interface
    pub fdtd_indices: Vec<(usize, usize, usize)>,
    /// FEM node indices at interface
    pub fem_indices: Vec<usize>,
    /// Interpolation weights for conservative transfer
    pub interpolation_weights: Vec<f64>,
    /// Interface normal vectors (outward from FDTD domain)
    pub normals: Vec<(f64, f64, f64)>,
    /// Interface area elements
    pub areas: Vec<f64>,
}

impl FdtdFemInterface {
    /// Create coupling interface from FDTD grid and FEM mesh
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn new(fdtd_grid: &Grid, fem_mesh: &TetrahedralMesh) -> KwaversResult<Self> {
        let (fdtd_indices, fem_indices) = Self::find_interface_nodes(fdtd_grid, fem_mesh)?;
        let interpolation_weights =
            Self::compute_interpolation_weights(&fdtd_indices, &fem_indices, fdtd_grid, fem_mesh)?;
        let (normals, areas) = Self::compute_interface_geometry(&fdtd_indices, fdtd_grid)?;

        Ok(Self {
            fdtd_indices,
            fem_indices,
            interpolation_weights,
            normals,
            areas,
        })
    }

    /// Find nodes at FDTD-FEM interface
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    fn find_interface_nodes(
        fdtd_grid: &Grid,
        fem_mesh: &TetrahedralMesh,
    ) -> KwaversResult<InterfaceNodePair> {
        let mut fdtd_indices = Vec::new();
        let mut fem_indices = Vec::new();

        for i in 0..fdtd_grid.nx {
            for j in 0..fdtd_grid.ny {
                for k in 0..fdtd_grid.nz {
                    let (x, y, z) = fdtd_grid.indices_to_coordinates(i, j, k);

                    for (node_idx, node) in fem_mesh.nodes.iter().enumerate() {
                        let dx = x - node.coordinates[0];
                        let dy = y - node.coordinates[1];
                        let dz = z - node.coordinates[2];
                        let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

                        if distance <= fdtd_grid.dx {
                            fdtd_indices.push((i, j, k));
                            fem_indices.push(node_idx);
                            break;
                        }
                    }
                }
            }
        }

        if fdtd_indices.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No interface nodes found between FDTD and FEM domains".to_owned(),
            ));
        }

        Ok((fdtd_indices, fem_indices))
    }

    /// Compute interpolation weights for conservative transfer
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_interpolation_weights(
        fdtd_indices: &[(usize, usize, usize)],
        fem_indices: &[usize],
        fdtd_grid: &Grid,
        fem_mesh: &TetrahedralMesh,
    ) -> KwaversResult<Vec<f64>> {
        let mut weights = Vec::with_capacity(fdtd_indices.len());

        for (&fdtd_idx, &fem_idx) in fdtd_indices.iter().zip(fem_indices.iter()) {
            let (fdtd_x, fdtd_y, fdtd_z) =
                fdtd_grid.indices_to_coordinates(fdtd_idx.0, fdtd_idx.1, fdtd_idx.2);
            let fem_node = &fem_mesh.nodes[fem_idx];

            let dx = fdtd_x - fem_node.coordinates[0];
            let dy = fdtd_y - fem_node.coordinates[1];
            let dz = fdtd_z - fem_node.coordinates[2];
            let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

            let sigma = fdtd_grid.dx;
            let weight = (-distance * distance / (2.0 * sigma * sigma)).exp();

            weights.push(weight);
        }

        Ok(weights)
    }

    /// Compute interface geometry (normals and areas)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_interface_geometry(
        fdtd_indices: &[(usize, usize, usize)],
        fdtd_grid: &Grid,
    ) -> KwaversResult<InterfaceGeometryData> {
        let mut normals = Vec::new();
        let mut areas = Vec::new();

        for _ in fdtd_indices {
            normals.push((1.0, 0.0, 0.0));
            let area = fdtd_grid.dy * fdtd_grid.dz;
            areas.push(area);
        }

        Ok((normals, areas))
    }
}
