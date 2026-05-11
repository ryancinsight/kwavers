//! Homogeneous medium implementation with uniform properties

use crate::core::constants::{
    AIR_POLYTROPIC_INDEX, ATMOSPHERIC_PRESSURE, REFERENCE_FREQUENCY_MHZ, WATER_ABSORPTION_ALPHA_0,
    WATER_ABSORPTION_POWER, WATER_SPECIFIC_HEAT, WATER_SURFACE_TENSION_20C,
    WATER_THERMAL_CONDUCTIVITY, WATER_VAPOR_PRESSURE_20C,
};
use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::domain::grid::Grid;
use ndarray::Array3;

/// Medium with uniform properties throughout the spatial domain
#[derive(Debug, Clone)]
pub struct HomogeneousMedium {
    pub(super) density: f64,
    pub(super) sound_speed: f64,
    pub(super) viscosity: f64,
    pub(super) surface_tension: f64,
    pub(super) ambient_pressure: f64,
    pub(super) vapor_pressure: f64,
    pub(super) polytropic_index: f64,
    pub(super) specific_heat: f64,
    pub(super) thermal_conductivity: f64,
    pub(super) shear_viscosity: f64,
    pub(super) bulk_viscosity: f64,
    pub(super) absorption_alpha: f64,
    pub(super) absorption_power: f64,
    pub(super) thermal_expansion: f64,
    pub(super) gas_diffusion: f64,
    pub(crate) nonlinearity: f64,
    pub(super) optical_absorption: f64,
    pub(super) optical_scattering: f64,
    pub(super) reference_frequency: f64,
    pub(super) temperature: Array3<f64>,
    pub(super) bubble_radius: Array3<f64>,
    pub(super) bubble_velocity: Array3<f64>,
    pub(super) density_cache: Array3<f64>,
    pub(super) sound_speed_cache: Array3<f64>,
    pub(super) absorption_cache: Array3<f64>,
    pub(super) nonlinearity_cache: Array3<f64>,
    pub(super) lame_lambda: f64,
    pub(super) lame_mu: f64,
    pub(super) grid_shape: (usize, usize, usize),
}

impl HomogeneousMedium {
    /// Create a new homogeneous medium with specified properties
    pub fn new(density: f64, sound_speed: f64, mu_a: f64, mu_s_prime: f64, grid: &Grid) -> Self {
        let viscosity = 1.0e-3;
        Self {
            density,
            sound_speed,
            viscosity,
            surface_tension: WATER_SURFACE_TENSION_20C,
            ambient_pressure: ATMOSPHERIC_PRESSURE,
            vapor_pressure: WATER_VAPOR_PRESSURE_20C,
            polytropic_index: AIR_POLYTROPIC_INDEX,
            specific_heat: WATER_SPECIFIC_HEAT,
            thermal_conductivity: WATER_THERMAL_CONDUCTIVITY,
            shear_viscosity: viscosity,
            bulk_viscosity: 2.5 * viscosity,
            absorption_alpha: WATER_ABSORPTION_ALPHA_0,
            absorption_power: WATER_ABSORPTION_POWER,
            thermal_expansion: 2.07e-4,
            gas_diffusion: 2.0e-9,
            nonlinearity: 5.0,
            optical_absorption: mu_a,
            optical_scattering: mu_s_prime,
            reference_frequency: REFERENCE_FREQUENCY_MHZ,
            temperature: Array3::zeros((1, 1, 1)),
            bubble_radius: Array3::zeros((1, 1, 1)),
            bubble_velocity: Array3::zeros((1, 1, 1)),
            density_cache: Array3::from_elem((grid.nx, grid.ny, grid.nz), density),
            sound_speed_cache: Array3::from_elem((grid.nx, grid.ny, grid.nz), sound_speed),
            absorption_cache: Array3::from_elem(
                (grid.nx, grid.ny, grid.nz),
                0.0022 * 1.0_f64.powf(1.05),
            ),
            nonlinearity_cache: Array3::from_elem((grid.nx, grid.ny, grid.nz), 5.0),
            lame_lambda: density * sound_speed * sound_speed,
            lame_mu: 0.0,
            grid_shape: (grid.nx, grid.ny, grid.nz),
        }
    }
    /// Set acoustic properties.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn set_acoustic_properties(
        &mut self,
        absorption_alpha: f64,
        absorption_power: f64,
        nonlinearity: f64,
    ) -> KwaversResult<()> {
        if !absorption_alpha.is_finite() || absorption_alpha < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "absorption_alpha".to_owned(),
                value: absorption_alpha,
                reason: "Absorption coefficient must be finite and non-negative".to_owned(),
            }));
        }

        if !absorption_power.is_finite() || absorption_power < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "absorption_power".to_owned(),
                value: absorption_power,
                reason: "Absorption power must be finite and non-negative".to_owned(),
            }));
        }

        if !nonlinearity.is_finite() || nonlinearity < 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "nonlinearity".to_owned(),
                value: nonlinearity,
                reason: "Nonlinearity must be finite and non-negative".to_owned(),
            }));
        }

        self.absorption_alpha = absorption_alpha;
        self.absorption_power = absorption_power;
        self.nonlinearity = nonlinearity;

        let alpha_at_ref =
            self.absorption_alpha * (self.reference_frequency / 1e6).powf(self.absorption_power);
        self.absorption_cache = Array3::from_elem(self.grid_shape, alpha_at_ref);
        self.nonlinearity_cache = Array3::from_elem(self.grid_shape, self.nonlinearity);

        Ok(())
    }

    /// Set thermal properties on an existing homogeneous medium.
    ///
    /// `thermal_conductivity` [W/(m·K)] and `specific_heat` [J/(kg·K)] must be
    /// finite and strictly positive. `thermal_diffusivity = k / (ρ·cp)` is computed
    /// implicitly through the `ThermalProperties` trait implementation.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn set_thermal_properties(
        &mut self,
        thermal_conductivity: f64,
        specific_heat: f64,
    ) -> KwaversResult<()> {
        if !thermal_conductivity.is_finite() || thermal_conductivity <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "thermal_conductivity".to_owned(),
                value: thermal_conductivity,
                reason: "must be finite and positive".to_owned(),
            }));
        }
        if !specific_heat.is_finite() || specific_heat <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "specific_heat".to_owned(),
                value: specific_heat,
                reason: "must be finite and positive".to_owned(),
            }));
        }
        self.thermal_conductivity = thermal_conductivity;
        self.specific_heat = specific_heat;
        Ok(())
    }
}

mod constructors;
#[cfg(test)]
mod tests;
mod traits_core;
mod traits_physical;
