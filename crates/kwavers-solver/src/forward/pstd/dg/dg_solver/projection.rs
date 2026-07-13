//! DG projection and basis operations
//!
//! This module handles projection of fields onto the DG basis and
//! related operations for discontinuous Galerkin methods.

use super::core::DGSolver;
use super::topology::{CoefficientLayout, DgTopology};
use kwavers_core::error::KwaversError;
use kwavers_core::error::KwaversResult;
use leto::Array3;

impl DGSolver {
    pub(super) fn ensure_rk_workspace(&mut self, dim: (usize, usize, usize)) {
        if self.rk_original.shape() != [dim.0, dim.1, dim.2] {
            self.rk_original = Array3::zeros(dim);
            self.rk_stage = Array3::zeros(dim);
            self.rk_rhs = Array3::zeros(dim);
        }
    }

    /// Project a field onto the DG basis
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn project_to_dg(&mut self, field: &Array3<f64>) -> KwaversResult<()> {
        if field.shape() != [self.grid.nx, self.grid.ny, self.grid.nz] {
            return Err(KwaversError::InvalidInput(format!(
                "project_to_dg field shape {:?} does not match grid ({}, {}, {})",
                field.shape(),
                self.grid.nx,
                self.grid.ny,
                self.grid.nz
            )));
        }

        let topology = DgTopology::from_grid(&self.grid, self.n_nodes)?;
        let dim = (topology.n_elements, topology.nodes_per_element, 1);
        if self
            .modal_coefficients
            .as_ref()
            .is_none_or(|coeffs| coeffs.shape() != [dim.0, dim.1, dim.2])
        {
            self.modal_coefficients = Some(Array3::zeros(dim));
        }

        let coeffs = self.modal_coefficients.as_mut().ok_or_else(|| {
            KwaversError::InternalError("DG coefficient allocation failed".to_owned())
        })?;

        for elem in 0..topology.n_elements {
            for node in 0..topology.nodes_per_element {
                let [i, j, k] = topology.grid_index(elem, node);
                coeffs[[elem, node, 0]] = field[[i, j, k]];
            }
        }

        self.ensure_rk_workspace(dim);
        self.coefficient_layout = CoefficientLayout::TensorProduct(topology);
        Ok(())
    }

    /// Project modal coefficients back to grid
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn project_to_grid(&self, field: &mut Array3<f64>) -> KwaversResult<()> {
        let coeffs = self.modal_coefficients.as_ref().ok_or_else(|| {
            KwaversError::InvalidInput("Modal coefficients not initialized".to_owned())
        })?;

        match self.coefficient_layout {
            CoefficientLayout::TensorProduct(topology) => {
                if coeffs.shape() != [topology.n_elements, topology.nodes_per_element, 1] {
                    return Err(KwaversError::InvalidInput(format!(
                        "project_to_grid tensor coefficient shape {:?} does not match topology ({}, {}, 1)",
                        coeffs.shape(),
                        topology.n_elements,
                        topology.nodes_per_element
                    )));
                }
                if field.shape() != [self.grid.nx, self.grid.ny, self.grid.nz] {
                    return Err(KwaversError::InvalidInput(format!(
                        "project_to_grid field shape {:?} does not match grid ({}, {}, {})",
                        field.shape(),
                        self.grid.nx,
                        self.grid.ny,
                        self.grid.nz
                    )));
                }

                for elem in 0..topology.n_elements {
                    for node in 0..topology.nodes_per_element {
                        let [i, j, k] = topology.grid_index(elem, node);
                        field[[i, j, k]] = coeffs[[elem, node, 0]];
                    }
                }
            }
            CoefficientLayout::Line1D => {
                let n_elements = coeffs.shape()[0].min(self.grid.nx / self.n_nodes);
                for elem in 0..n_elements {
                    for node in 0..self.n_nodes {
                        field[[elem * self.n_nodes + node, 0, 0]] = coeffs[[elem, node, 0]];
                    }
                }
            }
        }

        Ok(())
    }

    /// Get modal coefficients reference
    #[must_use]
    pub fn modal_coefficients(&self) -> Option<&Array3<f64>> {
        self.modal_coefficients.as_ref()
    }

    /// Get mutable modal coefficients reference
    pub fn modal_coefficients_mut(&mut self) -> Option<&mut Array3<f64>> {
        self.modal_coefficients.as_mut()
    }

    /// Initialize modal coefficients with given dimensions
    pub fn initialize_modal_coefficients(&mut self, n_elements: usize, n_vars: usize) {
        let dim = (n_elements, self.n_nodes, n_vars);
        self.modal_coefficients = Some(Array3::zeros(dim));
        self.coefficient_layout = CoefficientLayout::Line1D;
        self.ensure_rk_workspace(dim);
    }
}
