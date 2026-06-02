//! FDTD configuration: CFL validation and stability checks.

use super::InitialCondition;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::{KwaversError, KwaversResult};

/// FDTD solver configuration for 1D wave equation
#[derive(Debug, Clone)]
pub struct FDTDConfig {
    /// Wave speed (m/s)
    pub wave_speed: f64,
    /// Spatial step size (m)
    pub dx: f64,
    /// Temporal step size (s)
    pub dt: f64,
    /// Number of spatial points
    pub nx: usize,
    /// Number of time steps
    pub nt: usize,
    /// Initial condition type
    pub initial_condition: InitialCondition,
}

impl Default for FDTDConfig {
    fn default() -> Self {
        Self {
            wave_speed: SOUND_SPEED_WATER_SIM,
            dx: 0.01,
            dt: 0.000005,
            nx: 100,
            nt: 100,
            initial_condition: InitialCondition::GaussianPulse {
                width: 0.05,
                amplitude: 1.0,
            },
        }
    }
}

impl FDTDConfig {
    /// Validate configuration for numerical stability.
    ///
    /// Checks CFL condition: c×dt/dx ≤ 1
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if wave speed, steps, or grid size are invalid.
    /// - Returns [`KwaversError::InvalidInput`] if CFL > 1.0.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.wave_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Wave speed must be positive".to_owned(),
            ));
        }
        if self.dx <= 0.0 || self.dt <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Spatial and temporal steps must be positive".to_owned(),
            ));
        }
        if self.nx < 3 || self.nt < 3 {
            return Err(KwaversError::InvalidInput(
                "Grid must have at least 3 points in each dimension".to_owned(),
            ));
        }
        let cfl = self.wave_speed * self.dt / self.dx;
        if cfl > 1.0 {
            return Err(KwaversError::InvalidInput(format!(
                "CFL condition violated: c×dt/dx = {:.3} > 1.0. Reduce dt or increase dx.",
                cfl
            )));
        }
        Ok(())
    }

    /// CFL number c×dt/dx (must be ≤ 1 for stability).
    #[must_use]
    pub fn cfl_number(&self) -> f64 {
        self.wave_speed * self.dt / self.dx
    }
}
