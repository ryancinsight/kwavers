//! Subgrid refinement for FDTD solver
//!
//! This module implements local mesh refinement following the literature:
//! - Berenger, J. P. (2002). "Application of the CFS PML to the absorption of
//!   evanescent waves in waveguides." IEEE Microwave and Wireless Components
//!   Letters, 12(6), 218-220.

use crate::error::{ConfigError, KwaversError, KwaversResult};
use ndarray::Array3;

/// Subgrid region for local refinement
#[derive(Debug, Clone)]
pub struct SubgridRegion {
    /// Start indices in coarse grid
    pub start: (usize, usize, usize),
    /// End indices in coarse grid
    pub end: (usize, usize, usize),
    /// Refinement factor
    pub refinement_factor: usize,
    /// Fine grid data
    pub fine_pressure: Array3<f64>,
    pub fine_vx: Array3<f64>,
    pub fine_vy: Array3<f64>,
    pub fine_vz: Array3<f64>,
}

impl SubgridRegion {
    /// Create a new subgrid region
    pub fn new(
        start: (usize, usize, usize),
        end: (usize, usize, usize),
        refinement_factor: usize,
    ) -> KwaversResult<Self> {
        // Validate bounds
        if start.0 >= end.0 || start.1 >= end.1 || start.2 >= end.2 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "subgrid_bounds".to_string(),
                value: format!("start: {:?}, end: {:?}", start, end),
                constraint: "end indices must be greater than start indices".to_string(),
            }));
        }

        if refinement_factor < 2 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "refinement_factor".to_string(),
                value: refinement_factor.to_string(),
                constraint: "must be at least 2".to_string(),
            }));
        }

        // Calculate fine grid dimensions
        let nx_fine = (end.0 - start.0) * refinement_factor;
        let ny_fine = (end.1 - start.1) * refinement_factor;
        let nz_fine = (end.2 - start.2) * refinement_factor;

        Ok(Self {
            start,
            end,
            refinement_factor,
            fine_pressure: Array3::zeros((nx_fine, ny_fine, nz_fine)),
            fine_vx: Array3::zeros((nx_fine, ny_fine, nz_fine)),
            fine_vy: Array3::zeros((nx_fine, ny_fine, nz_fine)),
            fine_vz: Array3::zeros((nx_fine, ny_fine, nz_fine)),
        })
    }

    /// Get the dimensions of the fine grid
    pub fn fine_dimensions(&self) -> (usize, usize, usize) {
        self.fine_pressure.dim()
    }

    /// Check if a point is within this subgrid region
    pub fn contains(&self, i: usize, j: usize, k: usize) -> bool {
        i >= self.start.0
            && i < self.end.0
            && j >= self.start.1
            && j < self.end.1
            && k >= self.start.2
            && k < self.end.2
    }

    /// Convert coarse grid indices to fine grid indices
    pub fn coarse_to_fine(&self, i: usize, j: usize, k: usize) -> Option<(usize, usize, usize)> {
        if !self.contains(i, j, k) {
            return None;
        }

        let i_fine = (i - self.start.0) * self.refinement_factor;
        let j_fine = (j - self.start.1) * self.refinement_factor;
        let k_fine = (k - self.start.2) * self.refinement_factor;

        Some((i_fine, j_fine, k_fine))
    }

    /// Convert fine grid indices to coarse grid indices
    pub fn fine_to_coarse(&self, i_fine: usize, j_fine: usize, k_fine: usize) -> (usize, usize, usize) {
        let i_coarse = self.start.0 + i_fine / self.refinement_factor;
        let j_coarse = self.start.1 + j_fine / self.refinement_factor;
        let k_coarse = self.start.2 + k_fine / self.refinement_factor;

        (i_coarse, j_coarse, k_coarse)
    }
}

/// Deprecated subgridding functionality warning
///
/// This feature is not fully implemented and should not be used.
/// The interface is retained for API compatibility but returns an error.
#[deprecated(
    since = "0.4.0",
    note = "Subgridding feature is not fully implemented and is not ready for use."
)]
pub fn deprecated_subgridding() -> KwaversResult<()> {
    Err(KwaversError::Config(ConfigError::InvalidValue {
        parameter: "subgridding".to_string(),
        value: "enabled".to_string(),
        constraint: "Subgridding is not fully implemented. The feature requires stable interface schemes between coarse and fine grids which are not yet available.".to_string(),
    }))
}