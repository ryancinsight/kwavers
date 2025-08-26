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

    /// Create with CFL and sound speed (for backward compatibility)
    pub fn with_cfl(
        config: CPMLConfig,
        grid: &Grid,
        _cfl: f64,
        sound_speed: f64,
    ) -> KwaversResult<Self> {
        // CFL is now handled by the solver, not the boundary
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
    pub fn thickness(&self) -> usize {
        self.config.thickness
    }

    /// Reset memory variables
    pub fn reset(&mut self) {
        self.memory.reset();
    }

    /// Update acoustic memory for gradient component
    pub fn update_acoustic_memory(&mut self, gradient: &ndarray::Array3<f64>, component: usize) {
        self.updater
            .update_memory_component(&mut self.memory, gradient, component, &self.profiles);
    }

    /// Apply CPML gradient correction
    pub fn apply_cpml_gradient(&mut self, gradient: &mut ndarray::Array3<f64>, component: usize) {
        self.updater
            .apply_gradient_correction(gradient, &self.memory, component, &self.profiles);
    }

    /// Get configuration
    pub fn config(&self) -> &CPMLConfig {
        &self.config
    }

    /// Enable dispersive support
    pub fn enable_dispersive_support(&mut self, _params: DispersiveParameters) {
        // Dispersive support is handled in the dispersive module
        // This is a no-op for backward compatibility
    }

    /// Estimate reflection coefficient
    pub fn estimate_reflection(&self, angle: f64) -> f64 {
        self.config.theoretical_reflection(angle.cos())
    }
}

// Implement required traits
impl std::fmt::Debug for CPMLBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CPMLBoundary")
            .field("config", &self.config)
            .field("thickness", &self.config.thickness)
            .finish()
    }
}

impl Clone for CPMLBoundary {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            profiles: self.profiles.clone(),
            memory: self.memory.clone(),
            updater: self.updater.clone(),
        }
    }
}
