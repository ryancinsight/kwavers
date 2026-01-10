//! DG projection and basis operations
//!
//! This module handles projection of fields onto the DG basis and
//! related operations for discontinuous Galerkin methods.

use super::core::DGSolver;
use crate::domain::core::error::KwaversError;
use crate::domain::core::error::KwaversResult;
use ndarray::Array3;

impl DGSolver {
    /// Project a field onto the DG basis
    pub fn project_to_dg(&mut self, field: &Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);

        // Determine number of elements for structured Cartesian grid
        // Each element contains n_nodes in each direction
        // Unstructured grids would require mesh data structure
        let n_elements_x = nx / self.n_nodes;
        let n_elements_y = ny / self.n_nodes;
        let n_elements_z = nz / self.n_nodes;

        let mut coeffs = Array3::zeros((
            n_elements_x * n_elements_y * n_elements_z,
            self.n_nodes,
            1, // Scalar field (pressure); multi-component systems require tensor extension
        ));

        // Project field onto modal basis
        for elem_idx in 0..coeffs.shape()[0] {
            // Map element index to grid location
            let elem_z = elem_idx / (n_elements_x * n_elements_y);
            let elem_y = (elem_idx % (n_elements_x * n_elements_y)) / n_elements_x;
            let elem_x = elem_idx % n_elements_x;

            // Extract element data
            for node in 0..self.n_nodes {
                let grid_x = elem_x * self.n_nodes + node;
                let grid_y = elem_y * self.n_nodes;
                let grid_z = elem_z * self.n_nodes;

                if grid_x < nx && grid_y < ny && grid_z < nz {
                    coeffs[(elem_idx, node, 0)] = field[(grid_x, grid_y, grid_z)];
                }
            }
        }

        self.modal_coefficients = Some(coeffs);
        Ok(())
    }

    /// Project modal coefficients back to grid
    pub fn project_to_grid(&self, field: &mut Array3<f64>) -> KwaversResult<()> {
        let coeffs = self.modal_coefficients.as_ref().ok_or_else(|| {
            KwaversError::InvalidInput("Modal coefficients not initialized".to_string())
        })?;

        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let n_elements_x = nx / self.n_nodes;
        let n_elements_y = ny / self.n_nodes;
        let _n_elements_z = nz / self.n_nodes;

        // Project coefficients back to grid
        for elem_idx in 0..coeffs.shape()[0] {
            let elem_z = elem_idx / (n_elements_x * n_elements_y);
            let elem_y = (elem_idx % (n_elements_x * n_elements_y)) / n_elements_x;
            let elem_x = elem_idx % n_elements_x;

            for node in 0..self.n_nodes {
                let grid_x = elem_x * self.n_nodes + node;
                let grid_y = elem_y * self.n_nodes;
                let grid_z = elem_z * self.n_nodes;

                if grid_x < nx && grid_y < ny && grid_z < nz {
                    field[(grid_x, grid_y, grid_z)] = coeffs[(elem_idx, node, 0)];
                }
            }
        }

        Ok(())
    }

    /// Get modal coefficients reference
    pub fn modal_coefficients(&self) -> Option<&Array3<f64>> {
        self.modal_coefficients.as_ref()
    }

    /// Get mutable modal coefficients reference
    pub fn modal_coefficients_mut(&mut self) -> Option<&mut Array3<f64>> {
        self.modal_coefficients.as_mut()
    }

    /// Initialize modal coefficients with given dimensions
    pub fn initialize_modal_coefficients(&mut self, n_elements: usize, n_vars: usize) {
        self.modal_coefficients = Some(Array3::zeros((n_elements, self.n_nodes, n_vars)));
    }
}
