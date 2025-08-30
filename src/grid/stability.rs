//! Stability analysis and CFL conditions
//!
//! This module handles numerical stability calculations for time-stepping.

use crate::constants::numerical::CFL_SAFETY_FACTOR;
use crate::grid::structure::Grid;

/// Stability calculator for numerical schemes
#[derive(Debug))]
pub struct StabilityCalculator;

impl StabilityCalculator {
    /// Calculate CFL timestep for FDTD scheme
    pub fn cfl_timestep_fdtd(grid: &Grid, max_sound_speed: f64) -> f64 {
        let min_dx = grid.min_spacing();

        // For 3D FDTD, CFL condition is dt <= dx / (c * sqrt(3))
        // We use safety factor for stability
        CFL_SAFETY_FACTOR * min_dx / (max_sound_speed * 3.0_f64.sqrt())
    }

    /// Calculate CFL timestep for PSTD scheme
    pub fn cfl_timestep_pstd(grid: &Grid, max_sound_speed: f64) -> f64 {
        let min_dx = grid.min_spacing();

        // PSTD has less restrictive CFL condition
        // dt <= dx / (c * pi)
        CFL_SAFETY_FACTOR * min_dx / (max_sound_speed * std::f64::consts::PI)
    }

    /// Calculate CFL timestep for k-space method
    pub fn cfl_timestep_kspace(grid: &Grid, max_sound_speed: f64) -> f64 {
        // K-space method stability depends on k_max
        let k_max = std::f64::consts::PI / grid.min_spacing();

        // Stability condition: dt <= 2 / (c * k_max)
        CFL_SAFETY_FACTOR * 2.0 / (max_sound_speed * k_max)
    }

    /// Calculate Courant number for given timestep
    pub fn courant_number(grid: &Grid, dt: f64, sound_speed: f64) -> f64 {
        let min_dx = grid.min_spacing();
        sound_speed * dt / min_dx
    }

    /// Check if timestep is stable for FDTD
    pub fn is_stable_fdtd(grid: &Grid, dt: f64, max_sound_speed: f64) -> bool {
        dt <= Self::cfl_timestep_fdtd(grid, max_sound_speed)
    }

    /// Calculate diffusion stability for thermal problems
    pub fn diffusion_timestep(grid: &Grid, thermal_diffusivity: f64) -> f64 {
        let min_dx = grid.min_spacing();

        // For 3D diffusion: dt <= dx^2 / (6 * alpha)
        CFL_SAFETY_FACTOR * min_dx.powi(2) / (6.0 * thermal_diffusivity)
    }

    /// Calculate nonlinear stability timestep
    pub fn nonlinear_timestep(
        grid: &Grid,
        max_sound_speed: f64,
        nonlinearity_coefficient: f64,
    ) -> f64 {
        let linear_dt = Self::cfl_timestep_fdtd(grid, max_sound_speed);

        // Nonlinear effects require smaller timestep
        // Factor depends on B/A parameter
        let nonlinear_factor = 1.0 / (1.0 + nonlinearity_coefficient / 10.0);

        linear_dt * nonlinear_factor
    }

    /// Get recommended timestep for multi-physics simulation
    pub fn recommended_timestep(
        grid: &Grid,
        max_sound_speed: f64,
        thermal_diffusivity: Option<f64>,
        nonlinearity: Option<f64>,
    ) -> f64 {
        let mut dt = Self::cfl_timestep_fdtd(grid, max_sound_speed);

        if let Some(alpha) = thermal_diffusivity {
            dt = dt.min(Self::diffusion_timestep(grid, alpha));
        }

        if let Some(beta) = nonlinearity {
            dt = dt.min(Self::nonlinear_timestep(grid, max_sound_speed, beta));
        }

        dt
    }
}
