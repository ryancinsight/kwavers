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

mod config;
mod dispersive;
mod memory;
mod profiles;
mod update;

pub use config::CPMLConfig;
pub use dispersive::DispersiveParameters;
pub use memory::CPMLMemory;
pub use profiles::CPMLProfiles;
pub use update::CPMLUpdater;

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::Array4;

/// Main CPML boundary struct that coordinates all components
#[derive(Debug)]
pub struct CPMLBoundary {
    config: CPMLConfig,
    profiles: CPMLProfiles,
    memory: CPMLMemory,
    updater: CPMLUpdater,
}

impl CPMLBoundary {
    /// Create a new CPML boundary
    pub fn new(config: CPMLConfig, grid: &Grid, sound_speed: f64) -> KwaversResult<Self> {
        config.validate()?;
        let profiles = CPMLProfiles::new(&config, grid, sound_speed)?;
        let memory = CPMLMemory::new(&config, grid);
        let updater = CPMLUpdater::new(&config);

        Ok(Self {
            config,
            profiles,
            memory,
            updater,
        })
    }

    /// # Deprecated
    /// The CFL factor is now handled by the solver. This constructor is provided
    /// for backward compatibility, and the `_cfl` parameter is ignored.
    /// Use `CPMLBoundary::new` instead.
    #[deprecated(
        since = "3.0.0",
        note = "CFL factor is now handled by the solver. Use `CPMLBoundary::new` instead."
    )]
    pub fn with_cfl(
        config: CPMLConfig,
        grid: &Grid,
        _cfl: f64,
        sound_speed: f64,
    ) -> KwaversResult<Self> {
        Self::new(config, grid, sound_speed)
    }

    /// Apply CPML update to fields
    pub fn apply(&mut self, fields: &mut Array4<f64>, dt: f64, grid: &Grid) -> KwaversResult<()> {
        self.updater.update_fields(
            fields,
            &mut self.memory,
            &self.profiles,
            dt,
            grid,
            &self.config,
        )
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

    /// Updates CPML memory and applies the correction to a gradient field.
    /// This is the primary method for applying the CPML correction.
    pub fn update_and_apply_gradient_correction(
        &mut self,
        gradient: &mut ndarray::Array3<f64>,
        component: usize,
    ) {
        // Step 1: Update memory from the original gradient
        self.updater
            .update_memory_component(&mut self.memory, gradient, component, &self.profiles);

        // Step 2: Apply the correction to the gradient
        self.updater
            .apply_gradient_correction(gradient, &self.memory, component, &self.profiles);
    }

    /// Update acoustic memory for gradient component
    /// # Deprecated - Use update_and_apply_gradient_correction instead
    #[deprecated(
        since = "3.0.0",
        note = "Use `update_and_apply_gradient_correction` for the complete CPML update"
    )]
    pub fn update_acoustic_memory(&mut self, gradient: &ndarray::Array3<f64>, component: usize) {
        self.updater
            .update_memory_component(&mut self.memory, gradient, component, &self.profiles);
    }

    /// Apply CPML gradient correction
    /// # Deprecated - Use update_and_apply_gradient_correction instead
    #[deprecated(
        since = "3.0.0",
        note = "Use `update_and_apply_gradient_correction` for the complete CPML update"
    )]
    pub fn apply_cpml_gradient(&mut self, gradient: &mut ndarray::Array3<f64>, component: usize) {
        self.updater
            .apply_gradient_correction(gradient, &self.memory, component, &self.profiles);
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
    pub fn estimate_reflection(&self, angle: f64) -> f64 {
        self.config.theoretical_reflection(angle.cos())
    }
}

// Note: Clone implementation removed to prevent accidental expensive copies.
// Use `recreate` method to create a new boundary with fresh state.

impl CPMLBoundary {
    /// Creates a new `CPMLBoundary` from the existing configuration,
    /// with a fresh (zeroed) state.
    pub fn recreate(&self, grid: &Grid, sound_speed: f64) -> KwaversResult<Self> {
        Self::new(self.config.clone(), grid, sound_speed)
    }
}
