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

#[cfg(test)]
mod tests {
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use super::*;

    fn small_grid() -> Grid {
        Grid::new(4, 4, 4, 1e-3, 1e-3, 1e-3).expect("grid creation must succeed")
    }

    /// new() constructs a field with the correct shape.
    #[test]
    fn new_produces_correct_shape() {
        let grid = small_grid();
        let model = HeterogeneityModel::new(&grid, SOUND_SPEED_WATER_SIM, 0.05);
        assert_eq!(model.sound_speed_var.dim(), (4, 4, 4));
    }

    /// All values are within base_speed ± variance * base_speed.
    #[test]
    fn values_within_variance_bounds() {
        let base = SOUND_SPEED_WATER_SIM;
        let var = 0.05_f64;
        let grid = small_grid();
        let model = HeterogeneityModel::new(&grid, base, var);
        for &v in model.sound_speed_var.iter() {
            assert!(
                v >= base * (1.0 - var) && v <= base * (1.0 + var),
                "value {v} outside [{}, {}]",
                base * (1.0 - var),
                base * (1.0 + var)
            );
        }
    }

    /// adjust_sound_speed returns the stored field (same values).
    #[test]
    fn adjust_sound_speed_returns_stored_field() {
        let grid = small_grid();
        let model = HeterogeneityModel::new(&grid, SOUND_SPEED_WATER_SIM, 0.0); // zero variance → uniform
        let returned = model.adjust_sound_speed(&grid);
        for (&orig, &ret) in model.sound_speed_var.iter().zip(returned.iter()) {
            assert!((orig - ret).abs() < 1e-14);
        }
    }

    /// Zero variance produces a uniform field equal to base_speed.
    #[test]
    fn zero_variance_produces_uniform_field() {
        let base = 1480.0_f64;
        let grid = small_grid();
        let model = HeterogeneityModel::new(&grid, base, 0.0);
        for &v in model.sound_speed_var.iter() {
            assert!(
                (v - base).abs() < 1e-10,
                "expected {base} with zero variance, got {v}"
            );
        }
    }
}
