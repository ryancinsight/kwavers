//! Legacy linear polarization model (deprecated — use JonesPolarizationModel)

use super::PolarizationModel;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use log::debug;
use ndarray::{Array3, Array4, Zip};
use num_complex::Complex64;

/// Legacy linear polarization model (deprecated — use JonesPolarizationModel)
#[derive(Debug)]
pub struct LinearPolarization {
    polarization_factor: f64,
}

impl LinearPolarization {
    #[must_use]
    pub fn new(polarization_factor: f64) -> Self {
        Self {
            polarization_factor: polarization_factor.clamp(0.0, 1.0),
        }
    }
}

impl PolarizationModel for LinearPolarization {
    fn apply_polarization(
        &mut self,
        fluence: &mut Array3<f64>,
        _polarization_state: &mut Array4<Complex64>,
        _grid: &Grid,
        _medium: &dyn Medium,
    ) {
        debug!("WARNING: Using deprecated LinearPolarization model. Consider using JonesPolarizationModel for mathematical accuracy.");
        Zip::from(fluence).for_each(|f| {
            *f *= 1.0 + self.polarization_factor * f.abs();
        });
    }
}
