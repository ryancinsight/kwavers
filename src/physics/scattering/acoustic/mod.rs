// physics/scattering/acoustic/mod.rs
pub mod bubble_interactions;
pub mod mie;
pub mod rayleigh;

pub use bubble_interactions::compute_bubble_interactions;
pub use mie::compute_mie_scattering;
pub use rayleigh::compute_rayleigh_scattering;

use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct AcousticScatteringModel {
    pub scattered_field: Array3<f64>,
}

impl AcousticScatteringModel {
    pub fn new(grid: &Grid) -> Self {
        debug!("Initializing AcousticScatteringModel");
        Self {
            scattered_field: Array3::zeros((grid.nx, grid.ny, grid.nz)),
        }
    }

    pub fn compute_scattering(
        &mut self,
        incident_field: &Array3<f64>,
        bubble_radius: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        frequency: f64,
    ) {
        debug!("Computing combined acoustic scattering");
        let mut rayleigh_scatter = Array3::zeros(incident_field.dim());
        let mut mie_scatter = Array3::zeros(incident_field.dim());
        let mut interaction_scatter = Array3::zeros(incident_field.dim());

        compute_rayleigh_scattering(&mut rayleigh_scatter, bubble_radius, incident_field, grid, medium, frequency);
        compute_mie_scattering(&mut mie_scatter, bubble_radius, incident_field, grid, medium, frequency);
        compute_bubble_interactions(&mut interaction_scatter, bubble_radius, bubble_radius, incident_field, grid, medium, frequency);

        Zip::from(&mut self.scattered_field)
            .and(&rayleigh_scatter)
            .and(&mie_scatter)
            .and(&interaction_scatter)
            .par_for_each(|s, &ray, &mie, &inter| {
                *s = ray + mie + inter;
                if s.is_nan() || s.is_infinite() {
                    *s = 0.0;
                }
            });
    }

    pub fn scattered_field(&self) -> &Array3<f64> {
        &self.scattered_field
    }
}