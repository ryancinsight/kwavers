use super::{PstdSemCouplingConfig, SpectralCouplingInterface};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_mesh::tetrahedral::TetrahedralMesh;
use leto::Array3;

/// PSTD-SEM Spectral Coupler
#[derive(Debug)]
pub struct PstdSemCoupler {
    pub(super) config: PstdSemCouplingConfig,
    pub(super) interface: SpectralCouplingInterface,
    convergence_history: Vec<f64>,
    time_step: usize,
}

impl PstdSemCoupler {
    /// Create new PSTD-SEM spectral coupler
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        config: PstdSemCouplingConfig,
        pstd_grid: &Grid,
        sem_mesh: &TetrahedralMesh,
    ) -> KwaversResult<Self> {
        let interface = SpectralCouplingInterface::new(pstd_grid, sem_mesh, &config)?;

        Ok(Self {
            config,
            interface,
            convergence_history: Vec::new(),
            time_step: 0,
        })
    }

    /// Perform spectral coupling between PSTD and SEM fields
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn couple_fields(
        &mut self,
        pstd_field: &mut Array3<f64>,
        sem_field: &mut Vec<f64>,
        pstd_grid: &Grid,
        sem_mesh: &TetrahedralMesh,
    ) -> KwaversResult<f64> {
        let pstd_interface = self.extract_pstd_interface(pstd_field)?;
        let sem_interface = self.extract_sem_interface(sem_field.as_slice())?;

        let transformed_field = self.apply_modal_transform(&pstd_interface)?;
        let residual = self.enforce_continuity(&transformed_field, &sem_interface)?;

        self.apply_conservative_projection(
            pstd_field,
            sem_field.as_mut_slice(),
            pstd_grid,
            sem_mesh,
        )?;

        if self.config.stabilization_alpha > 0.0 {
            self.apply_stabilization(pstd_field, pstd_grid)?;
        }

        self.convergence_history.push(residual);
        self.time_step += 1;

        Ok(residual)
    }

    fn extract_pstd_interface(&self, pstd_field: &Array3<f64>) -> KwaversResult<Vec<f64>> {
        let mut interface_values = Vec::with_capacity(self.interface.pstd_interface_points.len());
        for &(i, j, k) in &self.interface.pstd_interface_points {
            interface_values.push(pstd_field[[i, j, k]]);
        }
        Ok(interface_values)
    }

    fn extract_sem_interface(&self, sem_field: &[f64]) -> KwaversResult<Vec<f64>> {
        let mut interface_values = Vec::with_capacity(self.interface.sem_interface_nodes.len());
        for &node_idx in &self.interface.sem_interface_nodes {
            let value = sem_field.get(node_idx).ok_or_else(|| {
                KwaversError::InvalidInput(format!(
                    "SEM interface node index {} is out of bounds (sem_field len {})",
                    node_idx,
                    sem_field.len()
                ))
            })?;
            interface_values.push(*value);
        }
        Ok(interface_values)
    }

    fn apply_modal_transform(&self, pstd_values: &[f64]) -> KwaversResult<Vec<f64>> {
        let mut transformed = vec![0.0; self.interface.modal_transform.ncols()];
        for i in 0..self.interface.modal_transform.nrows() {
            let pstd_value = match pstd_values.get(i) {
                Some(v) => *v,
                None => continue,
            };
            for (j, transformed_j) in transformed.iter_mut().enumerate() {
                *transformed_j += self.interface.modal_transform[[i, j]] * pstd_value;
            }
        }
        Ok(transformed)
    }

    fn enforce_continuity(&self, transformed: &[f64], sem_interface: &[f64]) -> KwaversResult<f64> {
        let mut max_residual = 0.0;
        for (&trans, &sem) in transformed.iter().zip(sem_interface.iter()) {
            let residual = (trans - sem).abs();
            if residual > max_residual {
                max_residual = residual;
            }
        }
        Ok(max_residual)
    }

    fn apply_conservative_projection(
        &mut self,
        pstd_field: &mut Array3<f64>,
        sem_field: &mut [f64],
        _pstd_grid: &Grid,
        _sem_mesh: &TetrahedralMesh,
    ) -> KwaversResult<()> {
        for (i, &sem_node) in self.interface.sem_interface_nodes.iter().enumerate() {
            let sem_value = *sem_field.get(sem_node).ok_or_else(|| {
                KwaversError::InvalidInput(format!(
                    "SEM interface node index {} is out of bounds (sem_field len {})",
                    sem_node,
                    sem_field.len()
                ))
            })?;

            for (j, &(pi, pj, pk)) in self.interface.pstd_interface_points.iter().enumerate() {
                if i < self.interface.projection_matrix.nrows()
                    && j < self.interface.projection_matrix.ncols()
                {
                    let weight = self.interface.projection_matrix[[i, j]];
                    pstd_field[[pi, pj, pk]] += weight * sem_value;
                }
            }
        }
        Ok(())
    }

    fn apply_stabilization(&self, field: &mut Array3<f64>, grid: &Grid) -> KwaversResult<()> {
        for &(i, j, k) in &self.interface.pstd_interface_points {
            if i > 0 && i < grid.nx - 1 && j > 0 && j < grid.ny - 1 && k > 0 && k < grid.nz - 1 {
                let laplacian = 6.0f64.mul_add(
                    -field[[i, j, k]],
                    field[[i - 1, j, k]]
                        + field[[i + 1, j, k]]
                        + field[[i, j - 1, k]]
                        + field[[i, j + 1, k]]
                        + field[[i, j, k - 1]]
                        + field[[i, j, k + 1]],
                );
                field[[i, j, k]] += self.config.stabilization_alpha * laplacian;
            }
        }
        Ok(())
    }

    /// Get convergence history
    #[must_use]
    pub fn convergence_history(&self) -> &[f64] {
        &self.convergence_history
    }

    /// Check if coupling has converged
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[must_use]
    pub fn has_converged(&self, tolerance: f64) -> bool {
        if self.convergence_history.len() < 2 {
            return false;
        }
        let last_residual = *self.convergence_history.last().unwrap();
        last_residual < tolerance
    }

    /// Reset convergence tracking
    pub fn reset_convergence(&mut self) {
        self.convergence_history.clear();
    }
}
