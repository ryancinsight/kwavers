//! Bubble dynamics properties implementation for heterogeneous media

use crate::domain::grid::Grid;
use crate::domain::medium::{
    bubble::{BubbleProperties, BubbleState},
    heterogeneous::{core::HeterogeneousMedium, interpolation::TrilinearInterpolator},
};
use ndarray::Array3;

impl BubbleProperties for HeterogeneousMedium {
    #[inline]
    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        TrilinearInterpolator::get_field_value(
            &self.surface_tension,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }

    #[inline]
    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.ambient_pressure
    }

    #[inline]
    fn vapor_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        TrilinearInterpolator::get_field_value(
            &self.vapor_pressure,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }

    #[inline]
    fn polytropic_index(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        TrilinearInterpolator::get_field_value(
            &self.polytropic_index,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
    }

    #[inline]
    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        TrilinearInterpolator::get_field_value(
            &self.gas_diffusion_coeff,
            x,
            y,
            z,
            grid,
            self.use_trilinear_interpolation,
        )
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
