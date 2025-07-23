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

#[derive(Debug)]
pub struct SimplePolarizationModel {
    polarization_factor: f64, // Simplified polarization strength (0 to 1)
}

impl Default for SimplePolarizationModel {
    fn default() -> Self {
        Self::new()
    }
}

impl SimplePolarizationModel {
    pub fn new() -> Self {
        debug!("Initializing SimplePolarizationModel");
        Self {
            polarization_factor: 0.5, // Default: moderate polarization
        }
    }
}

impl PolarizationModel for SimplePolarizationModel {
    fn apply_polarization(
        &mut self,
        fluence: &mut Array3<f64>,
        emission_spectrum: &Array3<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
    ) {
        debug!("Applying polarization effects to light fluence");
        Zip::from(fluence)
            .and(emission_spectrum)
            .for_each(|f, &spec| {
                let x = (spec / 1e-9).cos(); // Simple polarization modulation based on wavelength
                *f *= 1.0 + self.polarization_factor * x; // Adjust fluence by polarization
                *f = f.max(0.0);
            });
    }
}