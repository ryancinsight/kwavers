//! Stability analysis and CFL conditions
//!
//! This module handles numerical stability calculations for time-stepping.

use crate::structure::Grid;
use aequitas::systems::si::quantities::{ThermalDiffusivity, Time, Velocity};
use kwavers_core::constants::numerical::CFL_SAFETY_FACTOR;

/// Stability calculator for numerical schemes
#[derive(Debug)]
pub struct StabilityCalculator;

impl StabilityCalculator {
    /// Calculate CFL timestep for FDTD scheme
    #[must_use]
    pub fn cfl_timestep_fdtd(grid: &Grid, max_sound_speed: Velocity) -> Time {
        let max_sound_speed = max_sound_speed.into_base();
        let min_dx = grid.min_spacing();
        let dim_factor = (grid.dimensionality as f64).sqrt();

        // For FDTD, CFL condition is dt <= dx / (c * sqrt(dim))
        // We use safety factor for stability
        Time::from_base(CFL_SAFETY_FACTOR * min_dx / (max_sound_speed * dim_factor))
    }

    /// Calculate CFL timestep for PSTD scheme
    #[must_use]
    pub fn cfl_timestep_pstd(grid: &Grid, max_sound_speed: Velocity) -> Time {
        let max_sound_speed = max_sound_speed.into_base();
        let min_dx = grid.min_spacing();

        // PSTD has less restrictive CFL condition
        // dt <= dx / (c * pi)
        Time::from_base(CFL_SAFETY_FACTOR * min_dx / (max_sound_speed * std::f64::consts::PI))
    }

    /// Calculate CFL timestep for k-space method
    #[must_use]
    pub fn cfl_timestep_kspace(grid: &Grid, max_sound_speed: Velocity) -> Time {
        let max_sound_speed = max_sound_speed.into_base();
        // K-space method stability depends on k_max
        let k_max = std::f64::consts::PI / grid.min_spacing();

        // Stability condition: dt <= 2 / (c * k_max)
        Time::from_base(CFL_SAFETY_FACTOR * 2.0 / (max_sound_speed * k_max))
    }

    /// Calculate Courant number for given timestep
    #[must_use]
    pub fn courant_number(grid: &Grid, dt: Time, sound_speed: Velocity) -> f64 {
        let min_dx = grid.min_spacing();
        sound_speed.into_base() * dt.into_base() / min_dx
    }

    /// Check if timestep is stable for FDTD
    pub fn is_stable_fdtd(grid: &Grid, dt: Time, max_sound_speed: Velocity) -> bool {
        dt <= Self::cfl_timestep_fdtd(grid, max_sound_speed)
    }

    /// Calculate diffusion stability for thermal problems
    #[must_use]
    pub fn diffusion_timestep(grid: &Grid, thermal_diffusivity: ThermalDiffusivity) -> Time {
        let min_dx = grid.min_spacing();

        // For 3D diffusion: dt <= dx^2 / (6 * alpha)
        Time::from_base(
            CFL_SAFETY_FACTOR * min_dx.powi(2) / (6.0 * thermal_diffusivity.into_base()),
        )
    }

    /// Calculate nonlinear stability timestep
    #[must_use]
    pub fn nonlinear_timestep(
        grid: &Grid,
        max_sound_speed: Velocity,
        nonlinearity_coefficient: f64,
    ) -> Time {
        let linear_dt = Self::cfl_timestep_fdtd(grid, max_sound_speed);

        // Nonlinear effects require smaller timestep
        // Factor depends on B/A parameter
        let nonlinear_factor = 1.0 / (1.0 + nonlinearity_coefficient / 10.0);

        Time::from_base(linear_dt.into_base() * nonlinear_factor)
    }

    /// Get recommended timestep for multi-physics simulation
    #[must_use]
    pub fn recommended_timestep(
        grid: &Grid,
        max_sound_speed: Velocity,
        thermal_diffusivity: Option<ThermalDiffusivity>,
        nonlinearity: Option<f64>,
    ) -> Time {
        let mut dt = Self::cfl_timestep_fdtd(grid, max_sound_speed);

        if let Some(alpha) = thermal_diffusivity {
            let diffusion_dt = Self::diffusion_timestep(grid, alpha);
            if diffusion_dt.into_base() < dt.into_base() {
                dt = diffusion_dt;
            }
        }

        if let Some(beta) = nonlinearity {
            let nonlinear_dt = Self::nonlinear_timestep(grid, max_sound_speed, beta);
            if nonlinear_dt.into_base() < dt.into_base() {
                dt = nonlinear_dt;
            }
        }

        dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::quantities::{ThermalDiffusivity, Time, Velocity};

    #[test]
    fn stability_contracts_return_typed_time() {
        let grid = Grid::new(16, 16, 16, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid grid");
        let sound_speed = Velocity::from_base(1.5e3);
        let dt = StabilityCalculator::cfl_timestep_fdtd(&grid, sound_speed);

        assert!(dt.into_base() > 0.0);
        assert!(StabilityCalculator::is_stable_fdtd(&grid, dt, sound_speed));
        assert!(StabilityCalculator::courant_number(&grid, dt, sound_speed) < 1.0);
    }

    #[test]
    fn recommended_timestep_respects_diffusion_bound() {
        let grid = Grid::new(16, 16, 16, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid grid");
        let sound_speed = Velocity::from_base(1.5e3);
        let diffusivity = ThermalDiffusivity::from_base(1.0e-4);
        let recommended =
            StabilityCalculator::recommended_timestep(&grid, sound_speed, Some(diffusivity), None);

        assert!(recommended <= StabilityCalculator::diffusion_timestep(&grid, diffusivity));
        assert!(recommended <= StabilityCalculator::cfl_timestep_fdtd(&grid, sound_speed));
        assert!(recommended > Time::from_base(0.0));
    }
}
