//! Bubble dynamics properties implementation for heterogeneous media

use crate::{
    bubble::{BubbleProperties, BubbleState},
    heterogeneous::core::HeterogeneousMedium,
};
use aequitas::systems::si::quantities::{AreaPerTime, Dimensionless, Pressure, SurfaceTension};
use kwavers_grid::Grid;
use leto::Array3;

impl BubbleProperties for HeterogeneousMedium {
    #[inline]
    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> SurfaceTension<f64> {
        SurfaceTension::from_base(self.lookup_field(&self.surface_tension, x, y, z, grid))
    }

    #[inline]
    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> Pressure<f64> {
        Pressure::from_base(self.ambient_pressure)
    }

    #[inline]
    fn vapor_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> Pressure<f64> {
        Pressure::from_base(self.lookup_field(&self.vapor_pressure, x, y, z, grid))
    }

    #[inline]
    fn polytropic_index(&self, x: f64, y: f64, z: f64, grid: &Grid) -> Dimensionless<f64> {
        Dimensionless::from_base(self.lookup_field(&self.polytropic_index, x, y, z, grid))
    }

    #[inline]
    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> AreaPerTime<f64> {
        AreaPerTime::from_base(self.lookup_field(&self.gas_diffusion_coeff, x, y, z, grid))
    }
}

impl BubbleState for HeterogeneousMedium {
    #[inline]
    fn bubble_radius(&self) -> &Array3<f64> {
        &self.bubble_radius
    }

    #[inline]
    fn bubble_velocity(&self) -> &Array3<f64> {
        &self.bubble_velocity
    }

    #[inline]
    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>) {
        self.bubble_radius = radius.clone();
        self.bubble_velocity = velocity.clone();
    }
}
