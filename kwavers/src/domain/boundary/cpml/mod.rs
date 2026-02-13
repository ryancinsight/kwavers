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

use crate::core::error::KwaversResult;
use crate::domain::boundary::traits::{AbsorbingBoundary, BoundaryCondition, BoundaryDirections};
use crate::domain::boundary::Boundary;
use crate::domain::grid::{Grid, GridTopology};
use ndarray::{Array3, ArrayViewMut3, Zip};
use rustfft::num_complex::Complex;

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
    pub fn new(config: CPMLConfig, grid: &Grid, sound_speed: f64) -> KwaversResult<Self> {
        Self::new_with_time_step(config, grid, sound_speed, None)
    }

    /// Create a new CPML boundary with optional explicit solver time step.
    pub fn new_with_time_step(
        config: CPMLConfig,
        grid: &Grid,
        sound_speed: f64,
        dt: Option<f64>,
    ) -> KwaversResult<Self> {
        config.validate()?;
        let profiles = CPMLProfiles::new(&config, grid, sound_speed)?;
        let memory = CPMLMemory::new(&config, grid);
        let updater = CPMLUpdater::new();

        Ok(Self {
            config,
            profiles,
            memory,
            updater,
            reference_sound_speed: sound_speed,
            reference_time_step: dt.filter(|value| value.is_finite() && *value > 0.0),
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

    /// Updates CPML memory and applies the correction to a gradient field.
    /// This is the primary method for applying the CPML correction in split-field FDTD solvers.
    ///
    /// # Arguments
    /// * `gradient`: The gradient field (e.g., ∂p/∂x) to be corrected. Modified in-place.
    /// * `component`: The vector component (0 for x, 1 for y, 2 for z) this gradient represents.
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

impl Boundary for CPMLBoundary {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn apply_acoustic(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &Grid,
        _time_step: usize,
    ) -> KwaversResult<()> {
        // For solvers that don't support full convolutional PML (like k-space),
        // we apply CPML as a damping layer using its sigma profiles.
        // This provides compatibility with the Boundary trait.
        let dt = self.estimate_dt_from_grid(grid);

        Zip::indexed(&mut field).for_each(|(i, j, k), val| {
            let s_x = self.profiles.sigma_x[i];
            let s_y = self.profiles.sigma_y[j];
            let s_z = self.profiles.sigma_z[k];
            let sigma_total = s_x + s_y + s_z;

            if sigma_total > 0.0 {
                *val *= (-sigma_total * dt).exp();
            }
        });

        Ok(())
    }

    fn apply_acoustic_freq(
        &mut self,
        field: &mut Array3<Complex<f64>>,
        grid: &Grid,
        _time_step: usize,
    ) -> KwaversResult<()> {
        let dt = self.estimate_dt_from_grid(grid);

        Zip::indexed(field).for_each(|(i, j, k), val| {
            let s_x = self.profiles.sigma_x[i];
            let s_y = self.profiles.sigma_y[j];
            let s_z = self.profiles.sigma_z[k];
            let sigma_total = s_x + s_y + s_z;

            if sigma_total > 0.0 {
                let decay = (-sigma_total * dt).exp();
                val.re *= decay;
                val.im *= decay;
            }
        });

        Ok(())
    }

    fn apply_light(&mut self, _field: ArrayViewMut3<f64>, _grid: &Grid, _time_step: usize) {
        // CPML for light is not implemented yet
    }
}

// Implement new BoundaryCondition trait system
impl BoundaryCondition for CPMLBoundary {
    fn name(&self) -> &str {
        "CPML (Convolutional PML)"
    }

    fn active_directions(&self) -> BoundaryDirections {
        BoundaryDirections::all()
    }

    fn apply_scalar_spatial(
        &mut self,
        field: ArrayViewMut3<f64>,
        grid: &dyn GridTopology,
        _time_step: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        let spacing = grid.spacing();
        let dt = if dt > 0.0 {
            dt
        } else {
            self.estimate_dt_from_spacing(&spacing)
        };

        // Apply damping using sigma profiles
        Zip::indexed(field).for_each(|(i, j, k), val| {
            let s_x = self.profiles.sigma_x[i];
            let s_y = self.profiles.sigma_y[j];
            let s_z = self.profiles.sigma_z[k];
            let sigma_total = s_x + s_y + s_z;

            if sigma_total > 0.0 {
                *val *= (-sigma_total * dt).exp();
            }
        });

        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        field: &mut Array3<Complex<f64>>,
        grid: &dyn GridTopology,
        _time_step: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        let spacing = grid.spacing();
        let dt = if dt > 0.0 {
            dt
        } else {
            self.estimate_dt_from_spacing(&spacing)
        };

        Zip::indexed(field).for_each(|(i, j, k), val| {
            let s_x = self.profiles.sigma_x[i];
            let s_y = self.profiles.sigma_y[j];
            let s_z = self.profiles.sigma_z[k];
            let sigma_total = s_x + s_y + s_z;

            if sigma_total > 0.0 {
                let decay = (-sigma_total * dt).exp();
                val.re *= decay;
                val.im *= decay;
            }
        });

        Ok(())
    }

    fn reflection_coefficient(&self, angle_degrees: f64, _frequency: f64, sound_speed: f64) -> f64 {
        // Use typical grid spacing for estimation
        let dx = 1e-4;
        self.estimate_reflection(angle_degrees, dx, sound_speed)
    }

    fn reset(&mut self) {
        self.memory.reset();
    }

    fn is_stateful(&self) -> bool {
        true
    }

    fn memory_usage(&self) -> usize {
        // Estimate memory usage from all components
        std::mem::size_of_val(self)
            + self.memory.psi_vx_x.len() * std::mem::size_of::<f64>()
            + self.memory.psi_vy_y.len() * std::mem::size_of::<f64>()
            + self.memory.psi_vz_z.len() * std::mem::size_of::<f64>()
            + self.memory.psi_p_x.len() * std::mem::size_of::<f64>()
            + self.memory.psi_p_y.len() * std::mem::size_of::<f64>()
            + self.memory.psi_p_z.len() * std::mem::size_of::<f64>()
    }
}

impl AbsorbingBoundary for CPMLBoundary {
    fn thickness(&self) -> usize {
        self.config.thickness
    }

    fn absorption_profile(&self, indices: [usize; 3], _grid: &dyn GridTopology) -> f64 {
        let s_x = self.profiles.sigma_x[indices[0]];
        let s_y = self.profiles.sigma_y[indices[1]];
        let s_z = self.profiles.sigma_z[indices[2]];
        s_x + s_y + s_z
    }

    fn target_reflection(&self) -> f64 {
        self.config.target_reflection
    }
}
