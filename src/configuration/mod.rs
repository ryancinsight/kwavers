//! Modular configuration system implementing SSOT principles
//!
//! This module provides a hierarchical configuration structure that consolidates
//! all settings while maintaining separation of concerns through submodules.

use serde::{Deserialize, Serialize};
use std::path::Path;

// Re-export submodules
pub mod boundary;
pub mod grid;
pub mod medium;
pub mod output;
pub mod performance;
pub mod simulation;
pub mod solver;
pub mod source;
pub mod validation;

// Re-export types
pub use boundary::BoundaryParameters;
pub use grid::GridParameters;
pub use medium::MediumParameters;
pub use output::OutputParameters;
pub use performance::PerformanceParameters;
pub use simulation::SimulationParameters;
pub use solver::SolverParameters;
pub use source::SourceParameters;
pub use validation::ValidationParameters;

/// Master configuration structure - Single Source of Truth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Configuration {
    /// Simulation parameters
    pub simulation: SimulationParameters,
    /// Grid discretization
    pub grid: GridParameters,
    /// Medium properties
    pub medium: MediumParameters,
    /// Source configuration
    pub source: SourceParameters,
    /// Boundary conditions
    pub boundary: BoundaryParameters,
    /// Solver settings
    pub solver: SolverParameters,
    /// Output control
    pub output: OutputParameters,
    /// Performance tuning
    pub performance: PerformanceParameters,
    /// Validation settings
    pub validation: ValidationParameters,
}

impl Configuration {
    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> crate::error::KwaversResult<Self> {
        let contents = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            crate::error::ConfigError::FileNotFound {
                path: path.as_ref().display().to_string(),
            }
        })?;

        toml::from_str(&contents).map_err(|e| {
            crate::error::ConfigError::ParseError {
                line: 0, // toml error doesn't provide line info directly
                message: e.to_string(),
            }
            .into()
        })
    }

    /// Save configuration to TOML file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> crate::error::KwaversResult<()> {
        let contents =
            toml::to_string_pretty(self).map_err(|e| crate::error::ConfigError::ParseError {
                line: 0,
                message: e.to_string(),
            })?;

        std::fs::write(path, contents).map_err(|e| {
            crate::error::ConfigError::FileNotFound {
                path: "config file".to_string(),
            }
            .into()
        })
    }

    /// Validate configuration for consistency
    pub fn validate(&self) -> crate::error::KwaversResult<()> {
        // Validate each component
        self.simulation.validate()?;
        self.grid.validate()?;
        self.medium.validate()?;
        self.source.validate()?;
        self.boundary.validate()?;
        self.solver.validate()?;
        self.output.validate()?;
        self.performance.validate()?;
        self.validation.validate()?;

        // Cross-component validation
        self.validate_cross_dependencies()?;

        Ok(())
    }

    /// Validate cross-component dependencies
    fn validate_cross_dependencies(&self) -> crate::error::KwaversResult<()> {
        // CFL condition check
        if let Some(dt) = self.simulation.dt {
            let max_velocity = self.medium.sound_speed_max.unwrap_or(1500.0);
            let dx = self.grid.spacing[0];
            let cfl_actual = max_velocity * dt / dx;

            if cfl_actual > self.simulation.cfl {
                return Err(crate::error::ConfigError::InvalidValue {
                    parameter: "dt".to_string(),
                    value: format!("{}", dt),
                    constraint: format!(
                        "CFL condition violated: {} > {}",
                        cfl_actual, self.simulation.cfl
                    ),
                }
                .into());
            }
        }

        // Nyquist sampling check
        let min_wavelength =
            self.medium.sound_speed_min.unwrap_or(1000.0) / self.simulation.frequency;
        let min_ppw = min_wavelength / self.grid.spacing[0];

        if min_ppw < 2.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                parameter: "grid.spacing".to_string(),
                value: format!("{:?}", self.grid.spacing),
                constraint: format!(
                    "Nyquist criterion violated: {} points per wavelength < 2",
                    min_ppw
                ),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            simulation: SimulationParameters::default(),
            grid: GridParameters::default(),
            medium: MediumParameters::default(),
            source: SourceParameters::default(),
            boundary: BoundaryParameters::default(),
            solver: SolverParameters::default(),
            output: OutputParameters::default(),
            performance: PerformanceParameters::default(),
            validation: ValidationParameters::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configuration() {
        let config = Configuration::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cfl_validation() {
        let mut config = Configuration::default();
        config.simulation.dt = Some(1e-3); // Too large
        config.simulation.cfl = 0.1;
        config.medium.sound_speed_max = Some(1500.0);
        config.grid.spacing = [0.001, 0.001, 0.001];

        assert!(config.validate().is_err());
    }
}
