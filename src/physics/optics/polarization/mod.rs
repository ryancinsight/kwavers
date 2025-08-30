// physics/optics/polarization/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};

use std::fmt::Debug;

pub trait PolarizationModel: Debug + Send + Sync {
    fn apply_polarization(
        &mut self,
        fluence: &mut Array3<f64>,
        emission_spectrum: &Array3<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
    );
}

#[derive(Debug, Debug))]
pub struct LinearPolarization {
    polarization_factor: f64, // Polarization strength (0 to 1)
}

impl Default for LinearPolarization {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl LinearPolarization {
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
        emission_spectrum: &Array3<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
    ) {
        debug!("Applying polarization effects to light fluence");
        Zip::from(fluence).and(emission_spectrum).for_each(|f, &e| {
            *f *= 1.0 + self.polarization_factor * e.abs();
        });
    }
}
