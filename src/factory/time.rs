//! Time factory for creating time stepping configurations
//!
//! Follows Information Expert pattern for time step validation

use crate::physics::constants::numerical::CFL_SAFETY_FACTOR;
use crate::error::{ConfigError, KwaversResult};
use crate::grid::Grid;
use crate::time::Time;

/// Time configuration
#[derive(Debug, Clone)]
pub struct TimeConfig {
    pub dt: f64,
    pub num_steps: usize,
    pub cfl_factor: f64,
}

impl TimeConfig {
    /// Validate time configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.dt <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "dt".to_string(),
                value: self.dt.to_string(),
                constraint: "Time step must be positive".to_string(),
            }
            .into());
        }

        if self.num_steps == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "num_steps".to_string(),
                value: self.num_steps.to_string(),
                constraint: "Number of steps must be positive".to_string(),
            }
            .into());
        }

        if self.cfl_factor <= 0.0 || self.cfl_factor > 1.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "cfl_factor".to_string(),
                value: self.cfl_factor.to_string(),
                constraint: "CFL factor must be in (0, 1]".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for TimeConfig {
    fn default() -> Self {
        Self {
            dt: 1e-7, // Default time step
            num_steps: 1000,
            cfl_factor: CFL_SAFETY_FACTOR,
        }
    }
}

/// Factory for creating time configurations
#[derive(Debug)]
pub struct TimeFactory;

impl TimeFactory {
    /// Create time configuration from config
    pub fn create_time(config: &TimeConfig, _grid: &Grid) -> KwaversResult<Time> {
        config.validate()?;

        Ok(Time::new(config.dt, config.num_steps))
    }

    /// Create time configuration based on CFL condition
    pub fn create_from_cfl(
        grid: &Grid,
        sound_speed: f64,
        num_steps: usize,
        cfl_factor: f64,
    ) -> KwaversResult<Time> {
        let min_spacing = grid.dx.min(grid.dy).min(grid.dz);
        let dt = cfl_factor * min_spacing / sound_speed;

        let config = TimeConfig {
            dt,
            num_steps,
            cfl_factor,
        };

        Self::create_time(&config, grid)
    }
}
