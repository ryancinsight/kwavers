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

pub use config::{CPMLConfig, PerDimensionAlpha, PerDimensionPML};
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
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let dt = 0.3 * dx_min / sound_speed;
        Self::new_with_time_step(config, grid, sound_speed, Some(dt))
    }

    /// Create a new CPML boundary with optional explicit solver time step.
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
    pub fn update_and_apply_p_gradient_correction(
        &mut self,
        gradient: &mut ndarray::Array3<f64>,
        component: usize,
    ) {
        self.updater
            .update_p_memory(&mut self.memory, gradient, component, &self.profiles);
        self.updater
            .apply_p_correction(gradient, &self.memory, component, &self.profiles);
    }

    /// Updates CPML memory and applies the correction to a velocity gradient field (for pressure update).
    pub fn update_and_apply_v_gradient_correction(
        &mut self,
        v_gradient: &mut ndarray::Array3<f64>,
        component: usize,
    ) {
        self.updater
            .update_v_memory(&mut self.memory, v_gradient, component, &self.profiles);
        self.updater
            .apply_v_correction(v_gradient, &self.memory, component, &self.profiles);
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
        // K-Wave applies PML as exp(-sigma * dt/2) to velocity AND density fields
        // (split-field PML, applied twice per step = net exp(-sigma*dt)).
        let dt = self.estimate_dt_from_grid(grid);

        Zip::indexed(&mut field).for_each(|(i, j, k), val| {
            let s_x = self.profiles.sigma_x[i];
            let s_y = self.profiles.sigma_y[j];
            let s_z = self.profiles.sigma_z[k];
            let sigma_total = s_x + s_y + s_z;

            if sigma_total > 0.0 {
                *val *= (-sigma_total * dt * 0.5).exp();
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
                let decay = (-sigma_total * dt * 0.5).exp();
                val.re *= decay;
                val.im *= decay;
            }
        });

        Ok(())
    }

    /// Apply directional (split-field) PML to a single field component.
    ///
    /// Applies `exp(-sigma_d * dt/2)` where `sigma_d` is the PML profile for
    /// dimension `axis` (0=x, 1=y, 2=z). This matches k-Wave's split-field PML:
    ///   rho_x *= pml_x,  rho_y *= pml_y,  rho_z *= pml_z
    ///   ux    *= pml_x,  uy    *= pml_y,  uz    *= pml_z
    ///
    /// Ref: Treeby & Cox (2010), J. Biomed. Opt. 15(2), Eq. (3)-(5)
    fn apply_acoustic_directional(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &Grid,
        _time_step: usize,
        axis: usize,
    ) -> KwaversResult<()> {
        let dt = self.estimate_dt_from_grid(grid);
        Zip::indexed(&mut field).for_each(|(i, j, k), val| {
            let sigma = match axis {
                0 => self.profiles.sigma_x[i],
                1 => self.profiles.sigma_y[j],
                _ => self.profiles.sigma_z[k],
            };
            if sigma > 0.0 {
                *val *= (-sigma * dt * 0.5).exp();
            }
        });
        Ok(())
    }

    /// Apply staggered-grid PML to a velocity component.
    ///
    /// Uses `sigma_x_sgx`, `sigma_y_sgy`, or `sigma_z_sgz` (half-cell-shifted sigma)
    /// matching k-Wave's `pml_x_sgx`, `pml_y_sgy`, `pml_z_sgz` arrays.
    ///
    /// At the deepest left PML cell (index 0), the staggered sigma is:
    ///   σ_sg = σ_max · ((pml_size − 0.5) / pml_size)⁴ ≈ 0.706 · σ_max
    /// vs the non-staggered σ_max, giving a less-absorbing PML for velocity.
    ///
    /// This matches k-Wave's behavior and corrects the ≈ 20% amplitude under-prediction
    /// that occurs when non-staggered sigma is applied to staggered velocity fields.
    fn apply_velocity_pml_directional(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        grid: &Grid,
        _time_step: usize,
        axis: usize,
    ) -> KwaversResult<()> {
        let dt = self.estimate_dt_from_grid(grid);
        Zip::indexed(&mut field).for_each(|(i, j, k), val| {
            let sigma = match axis {
                0 => self.profiles.sigma_x_sgx[i],
                1 => self.profiles.sigma_y_sgy[j],
                _ => self.profiles.sigma_z_sgz[k],
            };
            if sigma > 0.0 {
                *val *= (-sigma * dt * 0.5).exp();
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
                *val *= (-sigma_total * dt * 0.5).exp();
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
                let decay = (-sigma_total * dt * 0.5).exp();
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
            + self.memory.psi_v_x.len() * std::mem::size_of::<f64>()
            + self.memory.psi_v_y.len() * std::mem::size_of::<f64>()
            + self.memory.psi_v_z.len() * std::mem::size_of::<f64>()
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
