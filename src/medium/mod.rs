// src/medium/mod.rs
use crate::grid::Grid;
use ndarray::Array3;
use num_complex::Complex;
use std::fmt::Debug;

pub mod absorption;
pub mod dispersion;
pub mod heterogeneous;
pub mod homogeneous;

pub use absorption::power_law_absorption;
pub use absorption::tissue_specific;
pub use dispersion::DispersiveMedium;

use std::any::Any;

pub trait Medium: Debug + Sync + Send + Any {
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
    
    // Elastic properties
    fn shear_modulus(&self, x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        3.0e3  // Default: 3 kPa (typical soft tissue)
    }
    
    fn youngs_modulus(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let mu = self.shear_modulus(x, y, z, grid);
        let lambda = self.lame_first_parameter(x, y, z, grid);
        mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu)
    }
    
    fn lame_first_parameter(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let c = self.sound_speed(x, y, z, grid);
        let rho = self.density(x, y, z, grid);
        let mu = self.shear_modulus(x, y, z, grid);
        rho * c * c - 2.0 * mu  // λ = ρc² - 2μ
    }
    
    fn poissons_ratio(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let mu = self.shear_modulus(x, y, z, grid);
        let lambda = self.lame_first_parameter(x, y, z, grid);
        lambda / (2.0 * (lambda + mu))
    }
    
    fn bulk_modulus(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let lambda = self.lame_first_parameter(x, y, z, grid);
        let mu = self.shear_modulus(x, y, z, grid);
        lambda + 2.0 * mu / 3.0
    }
    
    // Dispersion-related methods with default implementations
    fn phase_velocity(&self, x: f64, y: f64, z: f64, grid: &Grid, _frequency: f64) -> f64 {
        self.sound_speed(x, y, z, grid)  // Default: frequency-independent speed
    }
    
    fn group_velocity(&self, x: f64, y: f64, z: f64, grid: &Grid, _frequency: f64) -> f64 {
        self.sound_speed(x, y, z, grid)  // Default: same as phase velocity
    }
    
    fn complex_wave_number(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> Complex<f64> {
        let omega = 2.0 * std::f64::consts::PI * frequency;
        let c = self.phase_velocity(x, y, z, grid, frequency);
        let alpha = self.absorption_coefficient(x, y, z, grid, frequency);
        Complex::new(omega / c, alpha)
    }
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
    
    /// Optional method to compute shear modulus array for the entire grid
    fn compute_shear_modulus_array(&self, grid: &Grid) -> Array3<f64> {
        let mut shear_modulus = Array3::zeros((grid.nx(), grid.ny(), grid.nz()));
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let (x, y, z) = grid.idx_to_pos(i, j, k);
                    shear_modulus[[i, j, k]] = self.shear_modulus(x, y, z, grid);
                }
            }
        }
        shear_modulus
    }
}