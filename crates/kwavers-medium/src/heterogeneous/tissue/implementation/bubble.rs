//! `BubbleProperties` and `BubbleState` impls for `HeterogeneousTissueMedium`

use super::HeterogeneousTissueMedium;
use crate::bubble::{BubbleProperties, BubbleState};
use kwavers_core::constants::acoustic_parameters::AIR_POLYTROPIC_INDEX;
use kwavers_core::constants::cavitation::VAPOR_PRESSURE_WATER;
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use kwavers_grid::Grid;
use leto::Array3;

impl BubbleProperties for HeterogeneousTissueMedium {
    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).surface_tension
    }

    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        ATMOSPHERIC_PRESSURE
    }

    fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        VAPOR_PRESSURE_WATER
    }

    fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        AIR_POLYTROPIC_INDEX
    }

    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
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
