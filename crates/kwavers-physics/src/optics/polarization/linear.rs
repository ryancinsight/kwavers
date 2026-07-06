//! Legacy linear polarization model (deprecated — use JonesPolarizationModel)

use super::PolarizationModel;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use log::debug;
use moirai_parallel::{for_each_mut_with, Adaptive};
use ndarray::{Array3, Array4};
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
        let values = fluence
            .as_slice_memory_order_mut()
            .expect("invariant: linear-polarization fluence field must be contiguous");
        for_each_mut_with::<Adaptive, _, _>(values, |f| {
            *f *= self.polarization_factor.mul_add(f.abs(), 1.0);
        });
    }
}
