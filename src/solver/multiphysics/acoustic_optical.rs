//! Acoustic-optical coupling module
//!
//! This module provides specialized coupling between acoustic and optical fields.

use crate::domain::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;

/// Acoustic-optical solver for coupled simulations
#[derive(Debug)]
pub struct AcousticOpticalSolver {
    /// Photoelastic coefficient
    photoelastic_coefficient: f64,
    /// Grid reference
    _grid: Grid,
}

impl AcousticOpticalSolver {
    /// Create a new acoustic-optical solver
    pub fn new(grid: Grid, photoelastic_coefficient: f64) -> Self {
        Self {
            photoelastic_coefficient,
            _grid: grid,
        }
    }

    /// Couple acoustic pressure to optical intensity
    pub fn couple_fields(
        &self,
        pressure: &Array3<f64>,
        intensity: &mut Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        // Photoelastic effect: pressure changes refractive index
        // which affects optical intensity
        for ((i, j, k), &p) in pressure.indexed_iter() {
            let delta_n = self.photoelastic_coefficient * p;
            let modulation = 1.0 + delta_n * dt;
            intensity[[i, j, k]] *= modulation;
        }

        Ok(())
    }
}
