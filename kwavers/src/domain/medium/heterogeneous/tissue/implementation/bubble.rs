//! `BubbleProperties` and `BubbleState` impls for `HeterogeneousTissueMedium`

use crate::core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use super::HeterogeneousTissueMedium;
use crate::domain::grid::Grid;
use crate::domain::medium::bubble::{BubbleProperties, BubbleState};
use ndarray::Array3;

impl BubbleProperties for HeterogeneousTissueMedium {
    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).surface_tension
    }

    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        ATMOSPHERIC_PRESSURE
    }

    fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        2330.0 // Water vapor pressure at 20C approx
    }

    fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        1.4 // Air
    }

    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k)
            .gas_diffusion_coefficient
    }
}

impl BubbleState for HeterogeneousTissueMedium {
    fn bubble_radius(&self) -> &Array3<f64> {
        &self.bubble_radius
    }

    fn bubble_velocity(&self) -> &Array3<f64> {
        &self.bubble_velocity
    }

    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>) {
        self.bubble_radius = radius.clone();
        self.bubble_velocity = velocity.clone();
    }
}
