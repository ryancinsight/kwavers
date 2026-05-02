use crate::core::constants::BLOOD_VISCOSITY_37C;
use crate::domain::grid::Grid;
use ndarray::Array3;

use super::HomogeneousMedium;

impl HomogeneousMedium {
    /// Create a tissue medium with standard properties
    pub fn tissue(grid: &Grid) -> Self {
        use crate::core::constants::{DENSITY_TISSUE, SOUND_SPEED_TISSUE};
        let mut medium = Self::new(DENSITY_TISSUE, SOUND_SPEED_TISSUE, 0.75, 15.0, grid);
        let shape = (grid.nx, grid.ny, grid.nz);
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, 310.15);
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, medium.density);
        medium.sound_speed_cache = Array3::from_elem(shape, medium.sound_speed);
        let alpha = medium.absorption_alpha
            * (medium.reference_frequency / 1e6).powf(medium.absorption_power);
        medium.absorption_cache = Array3::from_elem(shape, alpha);
        medium.nonlinearity_cache = Array3::from_elem(shape, medium.nonlinearity);
        medium
    }

    /// Create a water medium with standard properties at 20°C
    pub fn water(grid: &Grid) -> Self {
        let mut medium = Self::new(998.0, 1482.0, 0.01, 0.1, grid);
        let shape = (grid.nx, grid.ny, grid.nz);
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, 293.15);
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, medium.density);
        medium.sound_speed_cache = Array3::from_elem(shape, medium.sound_speed);
        let alpha = medium.absorption_alpha
            * (medium.reference_frequency / 1e6).powf(medium.absorption_power);
        medium.absorption_cache = Array3::from_elem(shape, alpha);
        medium.nonlinearity_cache = Array3::from_elem(shape, medium.nonlinearity);
        medium
    }

    /// Create a blood medium with standard properties at 37°C
    pub fn blood(grid: &Grid) -> Self {
        let mut medium = Self::new(1060.0, 1570.0, 0.15, 0.5, grid);
        let shape = (grid.nx, grid.ny, grid.nz);
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, 310.15);
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, medium.density);
        medium.sound_speed_cache = Array3::from_elem(shape, medium.sound_speed);
        medium.viscosity = BLOOD_VISCOSITY_37C;
        medium.shear_viscosity = BLOOD_VISCOSITY_37C;
        medium.bulk_viscosity = 2.5 * BLOOD_VISCOSITY_37C;
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
            surface_tension: 0.0,
            ambient_pressure: 101325.0,
            vapor_pressure: 0.0,
            polytropic_index: 1.4,
            specific_heat: 1005.0,
            thermal_conductivity: 0.0257,
            shear_viscosity: 1.81e-5,
            bulk_viscosity: 0.0,
            absorption_alpha: 1.84e-11,
            absorption_power: 2.0,
            thermal_expansion: 3.43e-3,
            gas_diffusion: 2.0e-5,
            nonlinearity: 0.4,
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
                1.84e-11 * 1.0_f64.powf(2.0),
            ),
            nonlinearity_cache: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.4),
            lame_lambda: 1.204 * 343.0 * 343.0,
            lame_mu: 0.0,
            grid_shape: (grid.nx, grid.ny, grid.nz),
        }
    }

    /// Create from minimal parameters
    pub fn from_minimal(density: f64, sound_speed: f64, grid: &Grid) -> Self {
        let mut medium = Self::new(density, sound_speed, 0.01, 0.1, grid);
        let shape = (grid.nx, grid.ny, grid.nz);
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, 293.15);
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, density);
        medium.sound_speed_cache = Array3::from_elem(shape, sound_speed);
        let alpha = medium.absorption_alpha
            * (medium.reference_frequency / 1e6).powf(medium.absorption_power);
        medium.absorption_cache = Array3::from_elem(shape, alpha);
        medium.nonlinearity_cache = Array3::from_elem(shape, medium.nonlinearity);
        medium
    }

    /// Create a soft tissue medium for elastography simulations
    ///
    /// λ = Eν/((1+ν)(1-2ν)), μ = E/(2(1+ν))
    pub fn soft_tissue(youngs_modulus: f64, poisson_ratio: f64, grid: &Grid) -> Self {
        let density = 1060.0;
        let sound_speed = 1580.0;

        let mut medium = Self::new(density, sound_speed, 0.01, 0.1, grid);
        let shape = (grid.nx, grid.ny, grid.nz);
        medium.grid_shape = shape;
        medium.temperature = Array3::from_elem(shape, 310.15);
        medium.bubble_radius = Array3::zeros(shape);
        medium.bubble_velocity = Array3::zeros(shape);
        medium.density_cache = Array3::from_elem(shape, density);
        medium.sound_speed_cache = Array3::from_elem(shape, sound_speed);

        let nu = poisson_ratio;
        medium.lame_lambda = youngs_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        medium.lame_mu = youngs_modulus / (2.0 * (1.0 + nu));

        medium.viscosity = 0.001;
        medium.shear_viscosity = 0.001;
        medium.bulk_viscosity = 2.5 * 0.001;

        let alpha = 0.5 * (medium.reference_frequency / 1e6).powf(1.1);
        medium.absorption_cache = Array3::from_elem(shape, alpha);
        medium.nonlinearity_cache = Array3::from_elem(shape, 7.0);

        medium
    }

    /// Create liver tissue medium for SWE simulations
    pub fn liver_tissue(fibrosis_stage: u8, grid: &Grid) -> Self {
        let youngs_modulus_kpa = match fibrosis_stage {
            0 => 5.0,
            1 => 6.5,
            2 => 8.0,
            3 => 11.0,
            4 => 18.0,
            _ => 8.0,
        };
        Self::soft_tissue(youngs_modulus_kpa * 1000.0, 0.49, grid)
    }
}
