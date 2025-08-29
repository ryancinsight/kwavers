// physics/chemistry/reaction_kinetics/mod.rs
use crate::grid::Grid;
use crate::medium::Medium;
use log::debug;
use ndarray::{Array3, Zip};

#[derive(Debug, Clone)]
pub struct ReactionKinetics {
    pub hydroxyl_concentration: Array3<f64>,
    pub hydrogen_peroxide: Array3<f64>,
}

impl ReactionKinetics {
    pub fn new(grid: &Grid) -> Self {
        debug!("Initializing ReactionKinetics");
        Self {
            hydroxyl_concentration: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            hydrogen_peroxide: Array3::zeros((grid.nx, grid.ny, grid.nz)),
        }
    }

    pub fn update_reactions(
        &mut self,
        radical_init: &Array3<f64>,
        temperature: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        _medium: &dyn Medium,
    ) {
        debug!("Updating reaction kinetics");

        Zip::indexed(&mut self.hydroxyl_concentration)
            .and(&mut self.hydrogen_peroxide)
            .and(radical_init)
            .and(temperature)
            .for_each(|(i, j, k), oh, h2o2, &r_init, &t| {
                let _x = i as f64 * grid.dx;
                let _y = j as f64 * grid.dy;
                let _z = k as f64 * grid.dz;

                use crate::constants::thermodynamics::{
                    REACTION_REFERENCE_TEMPERATURE, SECONDARY_REACTION_RATE,
                    SONOCHEMISTRY_BASE_RATE,
                };
                let k1 = SONOCHEMISTRY_BASE_RATE * (t / REACTION_REFERENCE_TEMPERATURE).exp();
                let k2 = SECONDARY_REACTION_RATE * (t / REACTION_REFERENCE_TEMPERATURE);

                *oh += k1 * r_init * dt;
                *oh = oh.max(0.0);

                let h2o2_prod = k2 * *oh * *oh * dt; // Dereference oh
                *h2o2 += h2o2_prod;
                *oh -= 2.0 * h2o2_prod;
                *h2o2 = h2o2.max(0.0);
                *oh = oh.max(0.0);
            });
    }

    pub fn hydroxyl_concentration(&self) -> &Array3<f64> {
        &self.hydroxyl_concentration
    }

    pub fn hydrogen_peroxide(&self) -> &Array3<f64> {
        &self.hydrogen_peroxide
    }
}
