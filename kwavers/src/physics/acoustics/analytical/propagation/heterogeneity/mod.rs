// physics/heterogeneity/mod.rs
use crate::domain::grid::Grid;
use ndarray::Array3;
use rand::Rng;

#[derive(Debug)]
pub struct HeterogeneityModel {
    pub sound_speed_var: Array3<f64>,
    base_speed: f64,
    variance: f64,
}

impl HeterogeneityModel {
    pub fn new(grid: &Grid, base_speed: f64, variance: f64) -> Self {
        let mut rng = rand::rngs::ThreadRng::default();
        let sound_speed_var = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |_| {
            base_speed * (1.0 + rng.gen_range(-variance..=variance))
        });
        Self {
            sound_speed_var,
            base_speed,
            variance,
        }
    }

    pub fn adjust_sound_speed(&self, _grid: &Grid) -> Array3<f64> {
        self.sound_speed_var.clone()
    }

    /// Regenerate heterogeneity with new parameters (following Open/Closed Principle)
    pub fn regenerate(
        &mut self,
        grid: &Grid,
        new_base_speed: Option<f64>,
        new_variance: Option<f64>,
    ) {
        if let Some(speed) = new_base_speed {
            self.base_speed = speed;
        }
        if let Some(var) = new_variance {
            self.variance = var;
        }

        let mut rng = rand::rngs::ThreadRng::default();
        self.sound_speed_var = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |_| {
            self.base_speed * (1.0 + rng.gen_range(-self.variance..=self.variance))
        });
    }
}

// ADDED:
use crate::physics::traits::HeterogeneityModelTrait;

impl HeterogeneityModelTrait for HeterogeneityModel {
    fn adjust_sound_speed(&self, grid: &Grid) -> Array3<f64> {
        self.adjust_sound_speed(grid)
    }
}
