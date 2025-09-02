//! Homogeneous medium implementation with uniform properties

use crate::grid::Grid;
use crate::medium::{
    acoustic::AcousticProperties,
    bubble::{BubbleProperties, BubbleState},
    core::{ArrayAccess, CoreMedium},
    elastic::{ElasticArrayAccess, ElasticProperties},
    optical::OpticalProperties,
    thermal::{ThermalField, ThermalProperties},
    viscous::ViscousProperties,
};
use crate::physics::constants::*;
use ndarray::{Array3, ArrayView3};
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
    pub(crate) nonlinearity: f64,
    optical_absorption: f64,
    optical_scattering: f64,
    reference_frequency: f64,
    temperature: Array3<f64>,
    bubble_radius: Array3<f64>,
    bubble_velocity: Array3<f64>,
    density_cache: Array3<f64>,
    sound_speed_cache: Array3<f64>,
    absorption_cache: Array3<f64>,
    nonlinearity_cache: Array3<f64>,
    lame_lambda: f64,
    lame_mu: f64,
    grid_shape: (usize, usize, usize),
}

impl HomogeneousMedium {
    /// Create a new homogeneous medium with specified properties
    pub fn new(density: f64, sound_speed: f64, mu_a: f64, mu_s_prime: f64, grid: &Grid) -> Self {
        let viscosity = 1.0e-3; // Default water viscosity
        Self {
            density,
            sound_speed,
            viscosity,
            surface_tension: WATER_SURFACE_TENSION_20C, // Water at 20°C [N/m]
            ambient_pressure: ATMOSPHERIC_PRESSURE,     // 1 atm [Pa]
            vapor_pressure: WATER_VAPOR_PRESSURE_20C,   // Water at 20°C [Pa]
            polytropic_index: AIR_POLYTROPIC_INDEX,     // Diatomic gas approximation
            specific_heat: WATER_SPECIFIC_HEAT,         // Water [J/(kg·K)]
            thermal_conductivity: WATER_THERMAL_CONDUCTIVITY, // Water at 20°C [W/(m·K)]
            shear_viscosity: viscosity,
            bulk_viscosity: 2.5 * viscosity, // Stokes' hypothesis
            absorption_alpha: WATER_ABSORPTION_ALPHA_0, // Water absorption coefficient
            absorption_power: WATER_ABSORPTION_POWER, // Power law exponent
            thermal_expansion: 2.07e-4,      // Water at 20°C [1/K]
            gas_diffusion: 2.0e-9,           // O2 in water [m²/s]
            nonlinearity: 5.0,               // B/A parameter for water
            optical_absorption: mu_a,        // [1/m]
            optical_scattering: mu_s_prime,  // [1/m]
            reference_frequency: REFERENCE_FREQUENCY_MHZ, // 1 MHz
            temperature: Array3::zeros((1, 1, 1)),
            bubble_radius: Array3::zeros((1, 1, 1)),
            bubble_velocity: Array3::zeros((1, 1, 1)),
            density_cache: Array3::from_elem((grid.nx, grid.ny, grid.nz), density),
            sound_speed_cache: Array3::from_elem((grid.nx, grid.ny, grid.nz), sound_speed),
            absorption_cache: Array3::from_elem(
                (grid.nx, grid.ny, grid.nz),
                0.0022 * (1e6_f64 / 1e6).powf(1.05),
            ), // α at 1 MHz
            nonlinearity_cache: Array3::from_elem((grid.nx, grid.ny, grid.nz), 5.0), // B/A for water
            // For fluids, lambda is the bulk modulus, mu is 0
            lame_lambda: density * sound_speed * sound_speed,
            lame_mu: 0.0, // Fluid has no shear modulus
            grid_shape: (grid.nx, grid.ny, grid.nz),
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

        // Compute absorption at reference frequency
        let alpha = medium.absorption_alpha
            * (medium.reference_frequency / 1e6).powf(medium.absorption_power);
        medium.absorption_cache = Array3::from_elem(shape, alpha);
        medium.nonlinearity_cache = Array3::from_elem(shape, medium.nonlinearity);

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
        // Blood has higher viscosity than water
        medium.viscosity = BLOOD_VISCOSITY_37C;
        medium.shear_viscosity = BLOOD_VISCOSITY_37C;
        medium.bulk_viscosity = 2.5 * BLOOD_VISCOSITY_37C;

        // Update caches
        let alpha = medium.absorption_alpha
            * (medium.reference_frequency / 1e6).powf(medium.absorption_power);
        medium.absorption_cache = Array3::from_elem(shape, alpha);
        medium.nonlinearity_cache = Array3::from_elem(shape, medium.nonlinearity);

        medium
    }

    /// Create an air medium with standard properties at 20°C
    pub fn air(grid: &Grid) -> Self {
        Self {
            density: 1.204,
            sound_speed: 343.0,
            viscosity: 1.81e-5,
            surface_tension: 0.0, // No surface tension for gas
            ambient_pressure: 101325.0,
            vapor_pressure: 0.0, // Not applicable for air
            polytropic_index: 1.4,
            specific_heat: 1005.0,        // Air [J/(kg·K)]
            thermal_conductivity: 0.0257, // Air at 20°C [W/(m·K)]
            shear_viscosity: 1.81e-5,
            bulk_viscosity: 0.0,        // Negligible for ideal gas
            absorption_alpha: 1.84e-11, // Air absorption
            absorption_power: 2.0,
            thermal_expansion: 3.43e-3, // Air [1/K]
            gas_diffusion: 2.0e-5,      // Self-diffusion in air [m²/s]
            nonlinearity: 0.4,          // B/A for air
            optical_absorption: 0.0,
            optical_scattering: 0.0,
            reference_frequency: 1e6,
            temperature: Array3::from_elem((grid.nx, grid.ny, grid.nz), 293.15),
            bubble_radius: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            bubble_velocity: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            density_cache: Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.204),
            sound_speed_cache: Array3::from_elem((grid.nx, grid.ny, grid.nz), 343.0),
            absorption_cache: Array3::from_elem(
                (grid.nx, grid.ny, grid.nz),
                1.84e-11 * (1e6_f64 / 1e6).powf(2.0),
            ),
            nonlinearity_cache: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.4),
            lame_lambda: 1.204 * 343.0 * 343.0, // Bulk modulus
            lame_mu: 0.0,                       // Gas has no shear modulus
            grid_shape: (grid.nx, grid.ny, grid.nz),
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

        // Update absorption and nonlinearity caches
        let alpha = medium.absorption_alpha
            * (medium.reference_frequency / 1e6).powf(medium.absorption_power);
        medium.absorption_cache = Array3::from_elem(shape, alpha);
        medium.nonlinearity_cache = Array3::from_elem(shape, medium.nonlinearity);

        medium
    }
}

// Core medium properties
impl CoreMedium for HomogeneousMedium {
    fn density(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.density
    }

    fn sound_speed(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.sound_speed
    }

    fn reference_frequency(&self) -> f64 {
        self.reference_frequency
    }
}

// Array-based access
impl ArrayAccess for HomogeneousMedium {
    fn density_array(&self) -> ArrayView3<f64> {
        self.density_cache.view()
    }

    fn sound_speed_array(&self) -> ArrayView3<f64> {
        self.sound_speed_cache.view()
    }

    fn density_array_mut(&mut self) -> Option<&mut Array3<f64>> {
        Some(&mut self.density_cache)
    }

    fn sound_speed_array_mut(&mut self) -> Option<&mut Array3<f64>> {
        Some(&mut self.sound_speed_cache)
    }

    fn absorption_array(&self) -> ArrayView3<f64> {
        self.absorption_cache.view()
    }

    fn nonlinearity_array(&self) -> ArrayView3<f64> {
        self.nonlinearity_cache.view()
    }
}

// Acoustic properties
impl AcousticProperties for HomogeneousMedium {
    fn absorption_coefficient(
        &self,
        _x: f64,
        _y: f64,
        _z: f64,
        _grid: &Grid,
        frequency: f64,
    ) -> f64 {
        // Power law absorption: α = α₀ * (f/f₀)^y
        self.absorption_alpha * (frequency / self.reference_frequency).powf(self.absorption_power)
    }

    fn nonlinearity_parameter(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.nonlinearity
    }

    fn acoustic_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        // α = k / (ρ * Cp * c)
        self.thermal_conductivity / (self.density * self.specific_heat * self.sound_speed)
    }
}

// Elastic properties
impl ElasticProperties for HomogeneousMedium {
    fn lame_lambda(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.lame_lambda
    }

    fn lame_mu(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.lame_mu
    }
}

// Elastic array access
impl ElasticArrayAccess for HomogeneousMedium {
    fn lame_lambda_array(&self) -> Array3<f64> {
        Array3::from_elem(self.grid_shape, self.lame_lambda)
    }

    fn lame_mu_array(&self) -> Array3<f64> {
        Array3::from_elem(self.grid_shape, self.lame_mu)
    }
}

// Thermal properties
impl ThermalProperties for HomogeneousMedium {
    fn specific_heat(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.specific_heat
    }

    fn thermal_conductivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.thermal_conductivity
    }

    fn thermal_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        // α = k / (ρ * Cp)
        self.thermal_conductivity / (self.density * self.specific_heat)
    }

    fn thermal_expansion(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.thermal_expansion
    }
}

// Thermal field management
impl ThermalField for HomogeneousMedium {
    fn update_thermal_field(&mut self, temperature: &Array3<f64>) {
        self.temperature = temperature.clone();
    }

    fn thermal_field(&self) -> &Array3<f64> {
        &self.temperature
    }
}

// Optical properties
impl OpticalProperties for HomogeneousMedium {
    fn optical_absorption_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.optical_absorption
    }

    fn optical_scattering_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.optical_scattering
    }
}

// Viscous properties
impl ViscousProperties for HomogeneousMedium {
    fn viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.viscosity
    }

    fn shear_viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.shear_viscosity
    }

    fn bulk_viscosity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.bulk_viscosity
    }
}

// Bubble properties
impl BubbleProperties for HomogeneousMedium {
    fn surface_tension(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.surface_tension
    }

    fn ambient_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.ambient_pressure
    }

    fn vapor_pressure(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.vapor_pressure
    }

    fn polytropic_index(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.polytropic_index
    }

    fn gas_diffusion_coefficient(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        self.gas_diffusion
    }
}

// Bubble state management
impl BubbleState for HomogeneousMedium {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    #[test]
    fn test_water_properties() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
        let water = HomogeneousMedium::water(&grid);

        assert_eq!(water.density(0.0, 0.0, 0.0, &grid), 998.0);
        assert_eq!(water.sound_speed(0.0, 0.0, 0.0, &grid), 1482.0);
        assert_eq!(water.viscosity(0.0, 0.0, 0.0, &grid), 1.0e-3);
    }

    #[test]
    fn test_blood_properties() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
        let blood = HomogeneousMedium::blood(&grid);

        assert_eq!(blood.density(0.0, 0.0, 0.0, &grid), 1060.0);
        assert_eq!(blood.sound_speed(0.0, 0.0, 0.0, &grid), 1570.0);
        assert_eq!(blood.viscosity(0.0, 0.0, 0.0, &grid), 3.5e-3);
    }

    #[test]
    fn test_air_properties() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
        let air = HomogeneousMedium::air(&grid);

        assert_eq!(air.density(0.0, 0.0, 0.0, &grid), 1.204);
        assert_eq!(air.sound_speed(0.0, 0.0, 0.0, &grid), 343.0);
    }
}
