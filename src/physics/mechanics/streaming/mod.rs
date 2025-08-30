// physics/mechanics/streaming/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};

#[derive(Debug, Debug))]
pub struct StreamingModel {
    pub velocity: Array3<f64>,
}

impl StreamingModel {
    pub fn new(grid: &Grid) -> Self {
        debug!("Initializing StreamingModel");
        Self {
            velocity: Array3::zeros((grid.nx, grid.ny, grid.nz)),
        }
    }

    pub fn update_velocity(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        debug!("Updating streaming velocity with viscosity");
        let mut force = Array3::zeros(pressure.dim());
        Zip::indexed(&mut force)
            .and(pressure)
            .for_each(|(i, j, k), f, &p| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let rho = medium.density(x, y, z, grid);
                *f = -p / rho; // Acoustic force
            });

        Zip::indexed(&mut self.velocity)
            .and(&force)
            .for_each(|(i, j, k), v, &f| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                let mu = medium.viscosity(x, y, z, grid);
                *v += (dt / mu) * f; // Viscosity modulates velocity update
                if v.is_nan() || v.is_infinite() {
                    *v = 0.0;
                }
            });
    }

    pub fn velocity(&self) -> &Array3<f64> {
        &self.velocity
    }
}

// ADDED:
use crate::physics::traits::StreamingModelTrait;
// Note: The inherent update_velocity uses Zip...par_for_each, so rayon::prelude might be needed here
// if it's not already in scope. However, the trait implementation just calls the inherent method,
// so the inherent method's own imports should cover it. Let's not add `use rayon::prelude::*;` here
// unless the compiler complains about it specifically for the trait impl.

impl StreamingModelTrait for StreamingModel {
    fn update_velocity(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) {
        self.update_velocity(pressure, grid, medium, dt);
    }

    fn velocity(&self) -> &Array3<f64> {
        self.velocity()
    }
}
