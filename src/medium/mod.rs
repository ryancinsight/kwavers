// src/medium/mod.rs
use crate::grid::Grid;
use ndarray::Array3;
use std::fmt::Debug;

pub mod absorption;
pub mod heterogeneous;
pub mod homogeneous;

pub use absorption::power_law_absorption;
pub use absorption::tissue_specific;

pub trait Medium: Debug + Sync + Send {
    fn density(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn sound_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn is_homogeneous(&self) -> bool { false }
    fn viscosity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn surface_tension(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn ambient_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn vapor_pressure(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn polytropic_index(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64;
    fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn gas_diffusion_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn nonlinearity_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn absorption_coefficient_light(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn reduced_scattering_coefficient_light(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn reference_frequency(&self) -> f64; // Added for absorption calculations
    /// Get the tissue type at a specific position (if medium supports tissue types)
    fn tissue_type(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> Option<tissue_specific::TissueType> { None }

    fn update_temperature(&mut self, temperature: &Array3<f64>);
    fn temperature(&self) -> &Array3<f64>;
    fn bubble_radius(&self) -> &Array3<f64>;
    fn bubble_velocity(&self) -> &Array3<f64>;
    fn update_bubble_state(&mut self, radius: &Array3<f64>, velocity: &Array3<f64>);
    fn density_array(&self) -> Array3<f64>;
    fn sound_speed_array(&self) -> Array3<f64>;
}