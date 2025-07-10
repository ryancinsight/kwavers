// src/physics/traits.rs
use crate::grid::Grid;
use crate::medium::Medium;
use crate::source::Source;
use ndarray::{Array3, Array4};
use std::fmt::Debug;

/// Trait for acoustic wave propagation models.
///
/// Implementors of this trait are responsible for updating the acoustic wave field
/// over a time step `dt`, considering nonlinear effects, source terms, and medium properties.
pub trait AcousticWaveModel: Debug + Send + Sync {
    /// Advances the acoustic wave simulation by a single time step `dt`.
    ///
    /// # Arguments
    ///
    /// * `fields` - A mutable reference to a 4D array representing the simulation fields.
    ///   The pressure field, typically at `fields.index_axis(Axis(0), PRESSURE_IDX)`, is updated in place.
    /// * `prev_pressure` - A reference to the 3D pressure field from the previous time step.
    /// * `source` - A trait object implementing `Source`, defining the acoustic source.
    /// * `grid` - A reference to the `Grid` structure.
    /// * `medium` - A trait object implementing `Medium`, providing material properties.
    /// * `dt` - The time step size for this update.
    /// * `t` - The current simulation time.
    fn update_wave(
        &mut self,
        fields: &mut Array4<f64>,
        prev_pressure: &Array3<f64>,
        source: &dyn Source,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    );

    /// Reports performance metrics of the wave model.
    ///
    /// This method should log or print information about the computational performance,
    /// such as time spent in different parts of the simulation or number of calls.
    fn report_performance(&self);

    /// Sets the scaling factor for the nonlinearity term.
    fn set_nonlinearity_scaling(&mut self, scaling: f64);

    /// Sets the order of k-space correction for dispersion.
    fn set_k_space_correction_order(&mut self, order: usize);

    // Consider adding methods for configuration if common settings are identifiable
    // e.g., set_adaptive_timestep, etc.
    // However, these might be too implementation-specific for a general trait.
    // For now, they are omitted. Implementations can provide their own config methods.
}

/// Trait for cavitation behavior models.
///
/// Implementors of this trait simulate the dynamics of cavitation bubbles and their
/// effects on the acoustic field and other physical phenomena (e.g., sonoluminescence).
pub trait CavitationModelBehavior: Debug + Send + Sync {
    /// Advances the cavitation simulation by a single time step `dt`.
    ///
    /// # Arguments
    ///
    /// * `p_update` - A mutable reference to a 3D array representing the acoustic pressure field.
    ///   This field will be modified to include pressure changes due to bubble activity.
    /// * `p` - A reference to the current 3D acoustic pressure field driving bubble dynamics.
    /// * `grid` - A reference to the `Grid` structure.
    /// * `dt` - The time step size for this update.
    /// * `medium` - A trait object implementing `Medium`.
    /// * `frequency` - The driving acoustic frequency.
    ///
    /// # Returns
    ///
    /// An `Array3<f64>` representing the light emission power density (W/m^3) from sonoluminescence.
    fn update_cavitation(
        &mut self,
        p_update: &mut Array3<f64>,
        p: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        medium: &dyn Medium,
        frequency: f64,
    ) -> Array3<f64>;

    /// Returns a reference to the 3D array of bubble radii (meters).
    fn radius(&self) -> &Array3<f64>;

    /// Returns a reference to the 3D array of bubble wall velocities (m/s).
    fn velocity(&self) -> &Array3<f64>;

    /// Returns a reference to the 3D array of bubble temperatures (Kelvin).
    fn temperature(&self) -> &Array3<f64>;

    /// Sets the 3D array of bubble radii.
    /// Required for numerical solvers that manage state externally.
    fn set_radius(&mut self, new_radius: &Array3<f64>);
}

// Placeholder for other physics model traits that will be added later:
// pub trait LightDiffusionModelTrait: Debug + Send + Sync { /* ... */ }
// pub trait ThermalModelTrait: Debug + Send + Sync { /* ... */ }
// pub trait ChemicalModelTrait: Debug + Send + Sync { /* ... */ }
// pub trait StreamingModelTrait: Debug + Send + Sync { /* ... */ }

/// Trait for light diffusion models.
///
/// Implementors of this trait simulate the diffusion of light through a medium,
/// considering sources, absorption, scattering, and potentially other optical effects.
pub trait LightDiffusionModelTrait: Debug + Send + Sync {
    /// Advances the light diffusion simulation by a single time step `dt`.
    ///
    /// # Arguments
    ///
    /// * `fields` - A mutable reference to a 4D array representing the simulation fields.
    ///   The light field (e.g., fluence rate) is updated in place.
    /// * `light_source` - A 3D array representing the light source term (e.g., from sonoluminescence).
    /// * `grid` - A reference to the `Grid` structure.
    /// * `medium` - A trait object implementing `Medium`, providing optical properties.
    /// * `dt` - The time step size for this update.
    fn update_light(
        &mut self,
        fields: &mut Array4<f64>,
        light_source: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    );

    /// Returns a reference to the 3D array of the emission spectrum.
    fn emission_spectrum(&self) -> &Array3<f64>;

    /// Returns a reference to the 4D array of the fluence rate.
    /// The first dimension typically represents different wavelengths or components if applicable,
    /// or is singular if monochromatic.
    fn fluence_rate(&self) -> &Array4<f64>;

    /// Reports performance metrics of the light diffusion model.
    fn report_performance(&self);
}

/// Trait for thermal models.
///
/// Implementors of this trait simulate heat transfer and temperature changes within the medium,
/// considering heat sources (e.g., acoustic, optical) and thermal diffusion.
pub trait ThermalModelTrait: Debug + Send + Sync {
    /// Advances the thermal simulation by a single time step `dt`.
    ///
    /// # Arguments
    ///
    /// * `fields` - A mutable reference to a 4D array representing the simulation fields.
    ///   The temperature field is updated in place.
    /// * `grid` - A reference to the `Grid` structure.
    /// * `medium` - A trait object implementing `Medium`, providing thermal properties.
    /// * `dt` - The time step size for this update.
    /// * `frequency` - The acoustic frequency, relevant for some heat source calculations.
    fn update_thermal(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        frequency: f64,
    );

    /// Returns a reference to the 3D array of the current temperature field (Kelvin).
    fn temperature(&self) -> &Array3<f64>;

    /// Sets the 3D temperature field.
    /// Required for numerical solvers that manage state externally or for initialization.
    fn set_temperature(&mut self, new_temp: &Array3<f64>);

    /// Reports performance metrics of the thermal model.
    fn report_performance(&self);
}

/// Trait for chemical reaction models.
///
/// Implementors of this trait simulate chemical reactions, radical formation,
/// and other chemical processes occurring within the medium.
pub trait ChemicalModelTrait: Debug + Send + Sync {
    /// Advances the chemical simulation by a single time step `dt`.
    ///
    /// # Arguments
    ///
    /// * `p` - Acoustic pressure field.
    /// * `light` - Light intensity field.
    /// * `emission_spectrum` - Emission spectrum from other processes (e.g., sonoluminescence).
    /// * `bubble_radius` - Field of bubble radii.
    /// * `temperature` - Temperature field.
    /// * `grid` - Simulation grid.
    /// * `dt` - Time step.
    /// * `medium` - Medium properties.
    /// * `frequency` - Acoustic frequency.
    fn update_chemical(
        &mut self,
        p: &Array3<f64>,
        light: &Array3<f64>,
        emission_spectrum: &Array3<f64>,
        bubble_radius: &Array3<f64>,
        temperature: &Array3<f64>,
        grid: &Grid,
        dt: f64,
        medium: &dyn Medium,
        frequency: f64,
    );

    /// Returns a reference to the 3D array of the primary radical concentration.
    fn radical_concentration(&self) -> &Array3<f64>;

    // Other accessors like hydroxyl_concentration, etc., could be added if needed by consumers
    // using the trait object. For now, only the one used by solver::numerics is included.
}

/// Trait for acoustic streaming models.
///
/// Implementors of this trait simulate the fluid motion induced by acoustic waves.
pub trait StreamingModelTrait: Debug + Send + Sync {
    /// Advances the streaming velocity field by a single time step `dt`.
    ///
    /// # Arguments
    ///
    /// * `pressure` - Current acoustic pressure field.
    /// * `grid` - Simulation grid.
    /// * `medium` - Medium properties.
    /// * `dt` - Time step.
    fn update_velocity(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    );

    /// Returns a reference to the 3D array of the streaming velocity field (m/s).
    /// Note: This might represent a single component or magnitude depending on implementation.
    /// For simplicity, matching current `StreamingModel` which has one velocity field.
    fn velocity(&self) -> &Array3<f64>;
}

/// Trait for acoustic scattering models.
///
/// Implementors of this trait simulate the scattering of acoustic waves from
/// particles, bubbles, or other inhomogeneities in the medium.
pub trait AcousticScatteringModelTrait: Debug + Send + Sync {
    /// Computes the acoustic scattering effects.
    ///
    /// # Arguments
    ///
    /// * `incident_field` - The incident acoustic pressure field.
    /// * `bubble_radius` - Radius of bubbles/particles.
    /// * `bubble_velocity` - Velocity of bubbles/particles.
    /// * `grid` - Simulation grid.
    /// * `medium` - Medium properties.
    /// * `frequency` - Acoustic frequency.
    fn compute_scattering(
        &mut self,
        incident_field: &Array3<f64>,
        bubble_radius: &Array3<f64>,
        bubble_velocity: &Array3<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        frequency: f64,
    );

    /// Returns a reference to the 3D array of the computed scattered field.
    fn scattered_field(&self) -> &Array3<f64>;
}

/// Trait for models dealing with medium heterogeneity.
///
/// Implementors of this trait provide ways to represent and apply spatial variations
/// in medium properties, such as sound speed.
pub trait HeterogeneityModelTrait: Debug + Send + Sync {
    /// Calculates or returns the adjusted sound speed field considering heterogeneity.
    ///
    /// # Arguments
    ///
    /// * `grid` - Simulation grid.
    ///
    /// # Returns
    ///
    /// A 3D array representing the sound speed at each grid point.
    fn adjust_sound_speed(&self, grid: &Grid) -> Array3<f64>;

    // Potentially add an accessor for the raw variance field if needed:
    // fn sound_speed_variance_field(&self) -> &Array3<f64>;
}
