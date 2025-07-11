// physics/heterogeneity/mod.rs
use crate::grid::Grid;
use ndarray::Array3;
use rand::Rng;

#[derive(Debug)]
pub struct HeterogeneityModel {
    pub sound_speed_var: Array3<f64>,
}

impl HeterogeneityModel {
    pub fn new(grid: &Grid, base_speed: f64, variance: f64) -> Self {
        let mut rng = rand::rngs::ThreadRng::default(); // Correct way to get a ThreadRng
        let sound_speed_var = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |_| {
            base_speed * (1.0 + rng.gen_range(-variance..=variance)) // Inclusive upper bound
        });
        Self { sound_speed_var }
    }

    pub fn adjust_sound_speed(&self, _grid: &Grid) -> Array3<f64> {
        self.sound_speed_var.clone()
    }
}

// ADDED:
use crate::physics::traits::HeterogeneityModelTrait;

impl HeterogeneityModelTrait for HeterogeneityModel {
    fn adjust_sound_speed(&self, grid: &Grid) -> Array3<f64> {
        // Call the inherent method, which has the same name and signature
        // (though the inherent method currently ignores the grid argument)
        self.adjust_sound_speed(grid)
    }
}