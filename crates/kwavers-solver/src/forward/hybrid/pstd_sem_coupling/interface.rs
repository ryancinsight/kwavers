use super::{PstdSemCouplingConfig, SpectralCouplingInterface};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_mesh::tetrahedral::TetrahedralMesh;
use ndarray::Array2;

/// Paired interface region: PSTD grid points and the matching SEM node indices.
type PstdSemRegion = (Vec<(usize, usize, usize)>, Vec<usize>);

impl SpectralCouplingInterface {
    /// Create spectral coupling interface
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        pstd_grid: &Grid,
        sem_mesh: &TetrahedralMesh,
        config: &PstdSemCouplingConfig,
    ) -> KwaversResult<Self> {
        let (pstd_points, sem_nodes) =
            Self::find_interface_region(pstd_grid, sem_mesh, config.overlap_thickness)?;

        let modal_transform =
            Self::compute_modal_transform(&pstd_points, &sem_nodes, pstd_grid, sem_mesh, config)?;
        let projection_matrix =
            Self::compute_projection_matrix(&pstd_points, &sem_nodes, pstd_grid, sem_mesh, config)?;

        Ok(Self {
            pstd_interface_points: pstd_points,
            sem_interface_nodes: sem_nodes,
            modal_transform,
            projection_matrix,
        })
    }

    fn find_interface_region(
        pstd_grid: &Grid,
        sem_mesh: &TetrahedralMesh,
        overlap_thickness: usize,
    ) -> KwaversResult<PstdSemRegion> {
        let mut pstd_points = Vec::new();
        let mut sem_nodes = Vec::new();

        for i in 0..pstd_grid.nx {
            for j in 0..pstd_grid.ny {
                for k in 0..pstd_grid.nz {
                    let (x, y, z) = pstd_grid.indices_to_coordinates(i, j, k);

                    let mut is_interface = false;
                    let mut nearest_sem_node = None;
                    let mut min_distance = f64::INFINITY;

                    for (node_idx, node) in sem_mesh.nodes.iter().enumerate() {
                        let dx = x - node.coordinates[0];
                        let dy = y - node.coordinates[1];
                        let dz = z - node.coordinates[2];
                        let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

                        if distance < min_distance {
                            min_distance = distance;
                            nearest_sem_node = Some(node_idx);
                        }

                        if distance <= overlap_thickness as f64 * pstd_grid.dx {
                            is_interface = true;
                        }
                    }

                    if is_interface {
                        pstd_points.push((i, j, k));
                        if let Some(node_idx) = nearest_sem_node {
                            sem_nodes.push(node_idx);
                        }
                    }
                }
            }
        }

        if pstd_points.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No interface points found between PSTD and SEM domains".to_owned(),
            ));
        }

        Ok((pstd_points, sem_nodes))
    }

    fn compute_modal_transform(
        pstd_points: &[(usize, usize, usize)],
        sem_nodes: &[usize],
        pstd_grid: &Grid,
        sem_mesh: &TetrahedralMesh,
        config: &PstdSemCouplingConfig,
    ) -> KwaversResult<Array2<f64>> {
        let n_pstd = pstd_points.len();
        let n_sem = sem_nodes.len();
        let mut transform = Array2::<f64>::zeros((n_pstd, n_sem));

        for (i, &(pi, pj, pk)) in pstd_points.iter().enumerate() {
            let (x, y, z) = pstd_grid.indices_to_coordinates(pi, pj, pk);
            let weights = Self::compute_sem_weights(x, y, z, sem_nodes, sem_mesh, config)?;
            for (j, &weight) in weights.iter().enumerate() {
                if j < n_sem {
                    transform[[i, j]] = weight;
                }
            }
        }

        Ok(transform)
    }

    fn compute_projection_matrix(
        pstd_points: &[(usize, usize, usize)],
        sem_nodes: &[usize],
        pstd_grid: &Grid,
        sem_mesh: &TetrahedralMesh,
        config: &PstdSemCouplingConfig,
    ) -> KwaversResult<Array2<f64>> {
        let n_pstd = pstd_points.len();
        let n_sem = sem_nodes.len();
        let mut projection = Array2::<f64>::zeros((n_sem, n_pstd));

        for (i, &sem_node_idx) in sem_nodes.iter().enumerate() {
            let sem_node = &sem_mesh.nodes[sem_node_idx];
            let x = sem_node.coordinates[0];
            let y = sem_node.coordinates[1];
            let z = sem_node.coordinates[2];

            let weights = Self::compute_pstd_weights(x, y, z, pstd_points, pstd_grid, config)?;
            for (j, &weight) in weights.iter().enumerate() {
                if j < n_pstd {
                    projection[[i, j]] = weight;
                }
            }
        }

        Ok(projection)
    }

    fn compute_sem_weights(
        x: f64,
        y: f64,
        z: f64,
        sem_nodes: &[usize],
        sem_mesh: &TetrahedralMesh,
        config: &PstdSemCouplingConfig,
    ) -> KwaversResult<Vec<f64>> {
        let mut weights = Vec::new();
        let mut total_weight = 0.0;

        for &node_idx in sem_nodes {
            if node_idx >= sem_mesh.nodes.len() {
                continue;
            }
            let node = &sem_mesh.nodes[node_idx];
            let dx = x - node.coordinates[0];
            let dy = y - node.coordinates[1];
            let dz = z - node.coordinates[2];
            let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

            let sigma = config.coupling_order as f64 * 0.1;
            let weight = if distance < 1e-12 {
                1.0
            } else {
                (-distance * distance / (2.0 * sigma * sigma)).exp()
            };

            weights.push(weight);
            total_weight += weight;
        }

        if total_weight > 1e-12 {
            for weight in &mut weights {
                *weight /= total_weight;
            }
        }

        Ok(weights)
    }

    fn compute_pstd_weights(
        x: f64,
        y: f64,
        z: f64,
        pstd_points: &[(usize, usize, usize)],
        pstd_grid: &Grid,
        config: &PstdSemCouplingConfig,
    ) -> KwaversResult<Vec<f64>> {
        let mut weights = Vec::new();
        let mut total_weight = 0.0;

        for &(pi, pj, pk) in pstd_points {
            let (px, py, pz) = pstd_grid.indices_to_coordinates(pi, pj, pk);
            let dx = x - px;
            let dy = y - py;
            let dz = z - pz;
            let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

            let sigma = config.coupling_order as f64 * 0.1;
            let weight = if distance < 1e-12 {
                1.0
            } else {
                (-distance * distance / (2.0 * sigma * sigma)).exp()
            };

            weights.push(weight);
            total_weight += weight;
        }

        if total_weight > 1e-12 {
            for weight in &mut weights {
                *weight /= total_weight;
            }
        }

        Ok(weights)
    }
}
