//! Homogeneous medium implementation with uniform properties

use crate::grid::Grid;
use crate::medium::{Medium, TissueType};
use ndarray::{Array3, Zip};
use std::fmt::Debug;

/// Medium with uniform properties throughout the spatial domain
#[derive(Debug, Clone)]
pub struct HomogeneousMedium {
    density: f64,
    sound_speed: f64,
    viscosity: f64,
    surface_tension: f64,
    ambient_pressure: f64,
    vapor_pressure: f64,
    polytropic_index: f64,
    specific_heat: f64,
    thermal_conductivity: f64,
    shear_viscosity: f64,
    bulk_viscosity: f64,
    absorption_alpha: f64,
    absorption_power: f64,
    thermal_expansion: f64,
    gas_diffusion: f64,
    nonlinearity: f64,
    optical_absorption: f64,
    optical_scattering: f64,
    reference_frequency: f64,
    temperature: Array3<f64>,
    bubble_radius: Array3<f64>,
    bubble_velocity: Array3<f64>,
    density_cache: Array3<f64>,
    sound_speed_cache: Array3<f64>,
    lame_lambda: f64,
    lame_mu: f64,
    grid_shape: (usize, usize, usize),
}

impl HomogeneousMedium {
    /// Create a new homogeneous medium with specified properties
    pub fn new(
        density: f64,
        sound_speed: f64,
        mu_a: f64,
        mu_s_prime: f64,
        grid: &Grid,
    ) -> Self {
        let viscosity = 1.0e-3; // Default water viscosity
        let shape = (grid.nx, grid.ny, grid.nz);
        Self {
            density,
            sound_speed,
            viscosity,
            surface_tension: 0.0728, // Water at 20°C [N/m]
            ambient_pressure: 101325.0, // 1 atm [Pa]
            vapor_pressure: 2339.0, // Water at 20°C [Pa]
            polytropic_index: 1.4, // Diatomic gas approximation
            specific_heat: 4182.0, // Water [J/(kg·K)]
            thermal_conductivity: 0.598, // Water at 20°C [W/(m·K)]
            shear_viscosity: viscosity,
            bulk_viscosity: 2.5 * viscosity, // Stokes' hypothesis
            absorption_alpha: 0.0022, // Water absorption coefficient
            absorption_power: 1.05, // Power law exponent
            thermal_expansion: 2.07e-4, // Water at 20°C [1/K]
            gas_diffusion: 2.0e-9, // O2 in water [m²/s]
            nonlinearity: 5.0, // B/A parameter for water
            optical_absorption: mu_a, // [1/m]
            optical_scattering: mu_s_prime, // [1/m]
            reference_frequency: 1e6, // 1 MHz
            temperature: Array3::from_elem(shape, 293.15),
            bubble_radius: Array3::zeros(shape),
            bubble_velocity: Array3::zeros(shape),
            density_cache: Array3::from_elem(shape, density),
            sound_speed_cache: Array3::from_elem(shape, sound_speed),
            // For fluids, lambda is the bulk modulus, mu is 0
            lame_lambda: density * sound_speed * sound_speed,
            lame_mu: 0.0, // Fluid has no shear modulus
            grid_shape: shape,
        }
    }

    /// Create a water medium with standard properties at 20°C
    pub fn water(grid: &Grid) -> Self {
        let mut medium = Self::new(
            998.0,  // Density [kg/m³]
            1482.0, // Sound speed [m/s]
            0.01,   // Optical absorption [1/m]
            0.1,    // Optical scattering [1/m]
            grid,
        );
        let shape = (grid.nx, grid.ny, grid.nz);
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, 293.15); // 20°C
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, medium.density);
        medium.sound_speed_cache = Array3::from_elem(shape, medium.sound_speed);
        medium
    }

    /// Create a blood medium with standard properties at 37°C
    pub fn blood(grid: &Grid) -> Self {
        let mut medium = Self::new(
            1060.0, // Density [kg/m³]
            1570.0, // Sound speed [m/s]
            0.15,   // Optical absorption [1/m] - higher for blood
            0.5,    // Optical scattering [1/m] - higher for blood
            grid,
        );
        let shape = (grid.nx, grid.ny, grid.nz);
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, 310.15); // 37°C
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, medium.density);
        medium.sound_speed_cache = Array3::from_elem(shape, medium.sound_speed);
        medium
    }

    /// Create an air medium with standard properties at 20°C
    pub fn air(grid: &Grid) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        Self {
            density: 1.204,
            sound_speed: 343.0,
            viscosity: 1.81e-5,
            surface_tension: 0.0, // No surface tension for gas
            ambient_pressure: 101325.0,
            vapor_pressure: 0.0, // Not applicable for air
            polytropic_index: 1.4,
            specific_heat: 1005.0, // Air [J/(kg·K)]
            thermal_conductivity: 0.0257, // Air at 20°C [W/(m·K)]
            shear_viscosity: 1.81e-5,
            bulk_viscosity: 0.0, // Negligible for ideal gas
            absorption_alpha: 1.84e-11, // Air absorption
            absorption_power: 2.0,
            thermal_expansion: 3.43e-3, // Air [1/K]
            gas_diffusion: 2.0e-5, // Self-diffusion in air [m²/s]
            nonlinearity: 0.4, // B/A for air
            optical_absorption: 0.0,
            optical_scattering: 0.0,
            reference_frequency: 1e6,
            temperature: Array3::from_elem(shape, 293.15),
            bubble_radius: Array3::zeros(shape),
            bubble_velocity: Array3::zeros(shape),
            density_cache: Array3::from_elem(shape, 1.204),
            sound_speed_cache: Array3::from_elem(shape, 343.0),
            lame_lambda: 1.204 * 343.0 * 343.0, // Bulk modulus
            lame_mu: 0.0, // Gas has no shear modulus
            grid_shape: shape,
        }
    }

    /// Create from minimal parameters (for compatibility)
    pub fn from_minimal(density: f64, sound_speed: f64, grid: &Grid) -> Self {
        let mut medium = Self::new(density, sound_speed, 0.01, 0.1, grid);
        let shape = (grid.nx, grid.ny, grid.nz);
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, 293.15);
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, density);
        medium.sound_speed_cache = Array3::from_elem(shape, sound_speed);
        medium
    }
}

// Implementation of Medium trait continues...