//! Convolutional Perfectly Matched Layer (C-PML) implementation
//!
//! This module provides a complete C-PML boundary condition implementation for absorbing
//! outgoing waves at domain boundaries. Based on the formulation by Roden & Gedney (2000)
//! and Komatitsch & Martin (2007).
//!
//! ## Implementation Features
//! - Full recursive convolution with memory variables
//! - Support for acoustic, elastic, and dispersive media
//! - Configured for grazing angle absorption
//! - Polynomial grading profiles with κ stretching and α frequency shifting
//!
//! ## References
//! - Roden & Gedney (2000) "Convolutional PML (CPML): An efficient FDTD implementation"
//! - Komatitsch & Martin (2007) "An unsplit convolutional perfectly matched layer"

mod boundary_condition_impl;
mod boundary_impl;
mod config;
mod dispersive;
mod memory;
mod profiles;
mod update;

pub use config::{CPMLConfig, PerDimensionAlpha, PerDimensionPML};
pub use dispersive::DispersiveParameters;
pub use memory::CPMLMemory;
pub use profiles::CPMLProfiles;
pub use update::CPMLUpdater;

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;

#[doc(hidden)]
pub trait CpmlGradientField {
    fn with_leto_mut<R, F>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut leto::Array3<f64>) -> R;
}

impl CpmlGradientField for leto::Array3<f64> {
    fn with_leto_mut<R, F>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut leto::Array3<f64>) -> R,
    {
        f(self)
    }
}

/// Main CPML boundary struct that coordinates all components
#[derive(Debug)]
pub struct CPMLBoundary {
    config: CPMLConfig,
    profiles: CPMLProfiles,
    memory: CPMLMemory,
    updater: CPMLUpdater,
    reference_sound_speed: f64,
    reference_time_step: Option<f64>,
}

impl CPMLBoundary {
    /// Create a new CPML boundary
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: CPMLConfig, grid: &Grid, sound_speed: f64) -> KwaversResult<Self> {
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let dt = 0.3 * dx_min / sound_speed;
        Self::new_with_time_step(config, grid, sound_speed, Some(dt))
    }

    /// Create a new CPML boundary with optional explicit solver time step.
    /// # Errors
    /// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
    ///
    pub fn new_with_time_step(
        config: CPMLConfig,
        grid: &Grid,
        sound_speed: f64,
        dt_opt: Option<f64>,
    ) -> KwaversResult<Self> {
        config.validate()?;
        let dt = dt_opt.unwrap_or_else(|| {
            let dx_min = grid.dx.min(grid.dy).min(grid.dz);
            0.3 * dx_min / sound_speed
        });
        let profiles = CPMLProfiles::new(&config, grid, sound_speed, dt)?;
        let memory = CPMLMemory::new(&config, grid);
        let updater = CPMLUpdater::new();

        Ok(Self {
            config,
            profiles,
            memory,
            updater,
            reference_sound_speed: sound_speed,
            reference_time_step: Some(dt),
        })
    }

    /// Get the CPML thickness
    #[must_use]
    pub fn thickness(&self) -> usize {
        self.config.thickness
    }

    /// Reset memory variables
    pub fn reset(&mut self) {
        self.memory.reset();
    }

    /// Updates CPML memory and applies the correction to a pressure gradient field (for velocity update).
    pub fn update_and_apply_p_gradient_correction<G>(&mut self, gradient: &mut G, component: usize)
    where
        G: CpmlGradientField,
    {
        gradient.with_leto_mut(|gradient| {
            self.updater
                .update_p_memory(&mut self.memory, gradient, component, &self.profiles);
            self.updater
                .apply_p_correction(gradient, &self.memory, component, &self.profiles);
        });
    }

    /// Updates CPML memory and applies the correction to a velocity gradient field (for pressure update).
    pub fn update_and_apply_v_gradient_correction<G>(
        &mut self,
        v_gradient: &mut G,
        component: usize,
    ) where
        G: CpmlGradientField,
    {
        v_gradient.with_leto_mut(|v_gradient| {
            self.updater
                .update_v_memory(&mut self.memory, v_gradient, component, &self.profiles);
            self.updater
                .apply_v_correction(v_gradient, &self.memory, component, &self.profiles);
        });
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &CPMLConfig {
        &self.config
    }

    // Dispersive support is configured via CPMLConfig and is handled
    // automatically during initialization. See CPMLConfig documentation.

    /// Estimate reflection coefficient
    #[must_use]
    pub fn estimate_reflection(&self, angle: f64, dx: f64, sound_speed: f64) -> f64 {
        self.config
            .theoretical_reflection(angle.cos(), dx, sound_speed)
    }

    fn estimate_dt_from_grid(&self, grid: &Grid) -> f64 {
        if let Some(dt) = self.reference_time_step {
            return dt;
        }
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let c_ref = self.reference_sound_speed.max(f64::EPSILON);
        0.3 * dx_min / c_ref
    }

    fn estimate_dt_from_spacing(&self, spacing: &[f64]) -> f64 {
        if let Some(dt) = self.reference_time_step {
            return dt;
        }
        let dx_min = spacing.iter().copied().fold(f64::INFINITY, f64::min);
        let c_ref = self.reference_sound_speed.max(f64::EPSILON);
        0.3 * dx_min / c_ref
    }
}

// Note: Efficient Clone implementation provided to share profile data
// and provide a fresh memory state. The profiles are reused as they
// only depend on the grid and configuration, not the simulation state.

impl Clone for CPMLBoundary {
    /// Creates a new `CPMLBoundary` with the same configuration and profiles,
    /// but with a fresh, zeroed memory state. This is an efficient way to
    /// get a new boundary instance for a new simulation run without
    /// re-calculating the expensive static profiles.
    fn clone(&self) -> Self {
        // Profiles can be cloned cheaply as they're static for a given configuration
        let profiles = self.profiles.clone();

        // Create fresh memory state by cloning and resetting
        let mut memory = self.memory.clone();
        memory.reset();

        // Clone the other components
        let updater = CPMLUpdater::new();

        Self {
            config: self.config.clone(),
            profiles,
            memory,
            updater,
            reference_sound_speed: self.reference_sound_speed,
            reference_time_step: self.reference_time_step,
        }
    }
}
