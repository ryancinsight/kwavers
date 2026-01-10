//! Aberration correction for transcranial ultrasound
//!
//! Reference: Aubry et al. (2003) "Experimental demonstration of noninvasive
//! transskull adaptive focusing"

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::physics::skull::HeterogeneousSkull;
use ndarray::Array3;
use std::f64::consts::PI;

/// Aberration correction using time-reversal methods
#[derive(Debug)]
pub struct AberrationCorrection<'a> {
    grid: &'a Grid,
    skull: &'a HeterogeneousSkull,
}

impl<'a> AberrationCorrection<'a> {
    /// Create new aberration correction calculator
    pub fn new(grid: &'a Grid, skull: &'a HeterogeneousSkull) -> Self {
        Self { grid, skull }
    }

    /// Compute time-reversal phase corrections
    ///
    /// Returns phase corrections in radians for each grid point
    pub fn compute_time_reversal_phases(&self, frequency: f64) -> KwaversResult<Array3<f64>> {
        let k = 2.0 * PI * frequency / 1500.0; // Water wavenumber

        let mut phases = Array3::zeros((self.grid.nx, self.grid.ny, self.grid.nz));

        // Simplified phase aberration model
        // In practice, this would use ray tracing or full wave simulation
        let center = (self.grid.nx / 2, self.grid.ny / 2, self.grid.nz / 2);

        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k_idx in 0..self.grid.nz {
                    let dx = (i as f64 - center.0 as f64) * self.grid.dx;
                    let dy = (j as f64 - center.1 as f64) * self.grid.dy;
                    let dz = (k_idx as f64 - center.2 as f64) * self.grid.dz;

                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    let c_local = self.skull.sound_speed[[i, j, k_idx]];

                    // Phase delay due to sound speed variation
                    let k_local = 2.0 * PI * frequency / c_local;
                    phases[[i, j, k_idx]] = (k_local - k) * distance;
                }
            }
        }

        Ok(phases)
    }
}
