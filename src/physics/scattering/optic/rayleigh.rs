// physics/scattering/optic/rayleigh.rs
use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};

#[derive(Debug, Clone)]
pub struct RayleighOpticalScatteringModel {
    scattering_coefficient: f64, // Base scattering coefficient (m⁻¹)
}

impl RayleighOpticalScatteringModel {
    pub fn new() -> Self {
        debug!("Initializing RayleighOpticalScatteringModel");
        Self {
            scattering_coefficient: 0.1, // Placeholder value (m⁻¹)
        }
    }
}

impl super::OpticalScatteringModel for RayleighOpticalScatteringModel {
    fn apply_scattering(&mut self, fluence: &mut Array3<f64>, grid: &Grid, medium: &dyn Medium) {
        debug!("Applying Rayleigh optical scattering");
        Zip::indexed(fluence).par_for_each(|(i, j, k), f| {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;
            let mu_s_prime = medium.reduced_scattering_coefficient_light(x, y, z, grid);
            let scatter_factor = self.scattering_coefficient * mu_s_prime * grid.dx;
            *f *= (1.0 - scatter_factor).max(0.0);
            *f = f.max(0.0);
        });
    }
}