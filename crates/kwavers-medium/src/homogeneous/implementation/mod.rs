//! Homogeneous medium implementation with uniform properties

use kwavers_core::constants::cavitation::SURFACE_TENSION_WATER;
use kwavers_core::constants::thermodynamic::THERMAL_EXPANSION_WATER_20C;
use kwavers_core::constants::thermodynamic::{SPECIFIC_HEAT_WATER, THERMAL_CONDUCTIVITY_WATER};
use kwavers_core::constants::tissue_acoustics::B_OVER_A_WATER;
use kwavers_core::constants::{
    AIR_POLYTROPIC_INDEX, ATMOSPHERIC_PRESSURE, REFERENCE_FREQUENCY_HZ, VISCOSITY_WATER,
    WATER_ABSORPTION_ALPHA_0, WATER_ABSORPTION_POWER, WATER_VAPOR_PRESSURE_20C,
};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_grid::Grid;
use leto::{Array3, ArrayView3};
use std::sync::OnceLock;

/// Lazily-allocated 3D uniform array used to defer memory allocation until
/// first access.
///
/// `HomogeneousMedium` uses this wrapper for its cache arrays so that the
/// constructor does not pay the cost of allocating full-grid arrays unless a
/// caller actually needs them (e.g. a solver that operates on array views).
#[derive(Debug)]
pub(super) struct LazyUniformArray(OnceLock<Array3<f64>>);

impl LazyUniformArray {
    fn new() -> Self {
        Self(OnceLock::new())
    }

    fn get_or_init(&self, shape: [usize; 3], value: f64) -> ArrayView3<'_, f64> {
        self.0
            .get_or_init(|| Array3::from_elem(shape, value))
            .view()
    }

    /// Initialize with a closure so that expensive scalar computations run only
    /// on first allocation, not on every cache hit.
    fn get_or_init_with<F: FnOnce() -> Array3<f64>>(&self, f: F) -> ArrayView3<'_, f64> {
        self.0.get_or_init(f).view()
    }
}

impl Clone for LazyUniformArray {
    fn clone(&self) -> Self {
        let new = LazyUniformArray::new();
        if let Some(arr) = self.0.get() {
            // Best-effort: if already initialized, clone the cached array.
            let _ = new.0.set(arr.clone());
        }
        new
    }
}

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
    pub(super) density_cache: LazyUniformArray,
    pub(super) sound_speed_cache: LazyUniformArray,
    pub(super) absorption_cache: LazyUniformArray,
    pub(super) nonlinearity_cache: LazyUniformArray,
    pub(super) lame_lambda: f64,
    pub(super) lame_mu: f64,
    pub(super) grid_shape: [usize; 3],
}

impl HomogeneousMedium {
    /// Create a new homogeneous medium with specified properties
    pub fn new(density: f64, sound_speed: f64, mu_a: f64, mu_s_prime: f64, grid: &Grid) -> Self {
        let viscosity = VISCOSITY_WATER;
        Self {
            density,
            sound_speed,
            viscosity,
            surface_tension: SURFACE_TENSION_WATER,
            ambient_pressure: ATMOSPHERIC_PRESSURE,
            vapor_pressure: WATER_VAPOR_PRESSURE_20C,
            polytropic_index: AIR_POLYTROPIC_INDEX,
            specific_heat: SPECIFIC_HEAT_WATER,
            thermal_conductivity: THERMAL_CONDUCTIVITY_WATER,
            shear_viscosity: viscosity,
            bulk_viscosity: 2.5 * viscosity,
            absorption_alpha: WATER_ABSORPTION_ALPHA_0,
            absorption_power: WATER_ABSORPTION_POWER,
            thermal_expansion: THERMAL_EXPANSION_WATER_20C,
            gas_diffusion: 2.0e-9,
            nonlinearity: B_OVER_A_WATER, // 5.2 at 20°C (Duck 1990 Table 4.16)
            optical_absorption: mu_a,
            optical_scattering: mu_s_prime,
            reference_frequency: REFERENCE_FREQUENCY_HZ,
            temperature: Array3::zeros([1, 1, 1]),
            bubble_radius: Array3::zeros([1, 1, 1]),
            bubble_velocity: Array3::zeros([1, 1, 1]),
            density_cache: LazyUniformArray::new(),
            sound_speed_cache: LazyUniformArray::new(),
            absorption_cache: LazyUniformArray::new(),
            nonlinearity_cache: LazyUniformArray::new(),
            lame_lambda: density * sound_speed * sound_speed,
            lame_mu: 0.0,
            grid_shape: [grid.nx, grid.ny, grid.nz],
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

        // Invalidate lazy caches so the next access rebuilds them with the new
        // scalar values.  This avoids re-allocating full-grid arrays here.
        self.absorption_cache = LazyUniformArray::new();
        self.nonlinearity_cache = LazyUniformArray::new();

        Ok(())
    }

    /// Scalar acoustic nonlinearity coefficient `B/A`.
    #[must_use]
    pub fn nonlinearity_coefficient(&self) -> f64 {
        self.nonlinearity
    }

    /// Set the scalar acoustic nonlinearity coefficient `B/A`, refreshing the
    /// per-voxel cache so trait accessors stay consistent. (Public accessor for
    /// cross-crate callers; the field itself is crate-private.)
    pub fn set_nonlinearity(&mut self, b_over_a: f64) {
        self.nonlinearity = b_over_a;
        self.nonlinearity_cache = LazyUniformArray::new();
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
