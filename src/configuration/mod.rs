//! Modular configuration system implementing SSOT principles
//!
//! This module provides a hierarchical configuration structure that consolidates
//! all settings while maintaining separation of concerns through submodules.

use serde::{Deserialize, Serialize};
use std::path::Path;

// Re-export submodules
pub mod boundary;
pub mod builder;
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
pub use builder::ConfigurationBuilder;
pub use grid::GridParameters;
pub use medium::MediumParameters;
pub use output::OutputParameters;
pub use performance::PerformanceParameters;
pub use simulation::SimulationParameters;
pub use solver::SolverParameters;
pub use source::SourceParameters;
pub use validation::ValidationParameters;

/// Master configuration structure - Single Source of Truth
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
        let contents = std::fs::read_to_string(path.as_ref()).map_err(|_e| {
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

        std::fs::write(path, contents).map_err(|_e| {
            crate::error::ConfigError::FileNotFound {
                path: "config file".to_string(),
            }
            .into()
        })
    }

    /// Validate configuration for consistency
    pub fn validate(&self) -> crate::error::KwaversResult<()> {
        let mut multi_error = crate::error::MultiError::new();

        // Validate each component - collect errors instead of returning early
        if let Err(e) = self.simulation.validate() {
            multi_error.add(e);
        }
        if let Err(e) = self.grid.validate() {
            multi_error.add(e);
        }
        if let Err(e) = self.medium.validate() {
            multi_error.add(e);
        }
        if let Err(e) = self.source.validate() {
            multi_error.add(e);
        }
        if let Err(e) = self.boundary.validate() {
            multi_error.add(e);
        }
        if let Err(e) = self.solver.validate() {
            multi_error.add(e);
        }
        if let Err(e) = self.output.validate() {
            multi_error.add(e);
        }
        if let Err(e) = self.performance.validate() {
            multi_error.add(e);
        }
        if let Err(e) = self.validation.validate() {
            multi_error.add(e);
        }

        // Cross-component validation
        if let Err(e) = self.validate_cross_dependencies() {
            multi_error.add(e);
        }

        multi_error.into_result()
    }

    /// Validate cross-component dependencies
    fn validate_cross_dependencies(&self) -> crate::error::KwaversResult<()> {
        let mut multi_error = crate::error::MultiError::new();

        // CFL condition check - require necessary values to be present
        if let Some(dt) = self.simulation.dt {
            if let Some(max_velocity) = self.medium.sound_speed_max {
                // Use minimum grid spacing for most restrictive CFL condition
                let min_spacing = self.grid.spacing[0]
                    .min(self.grid.spacing[1])
                    .min(self.grid.spacing[2]);
                let cfl_actual = max_velocity * dt / min_spacing;

                if cfl_actual > self.simulation.cfl {
                    multi_error.add(
                        crate::error::ConfigError::InvalidValue {
                            parameter: "dt".to_string(),
                            value: format!("{dt}"),
                            constraint: format!(
                                "CFL condition violated: {} > {} (max_velocity={}, min_spacing={})",
                                cfl_actual, self.simulation.cfl, max_velocity, min_spacing
                            ),
                        }
                        .into(),
                    );
                }
            } else {
                multi_error.add(
                    crate::error::ConfigError::MissingParameter {
                        parameter: "medium.sound_speed_max".to_string(),
                        section: "Required for CFL validation when dt is specified".to_string(),
                    }
                    .into(),
                );
            }
        }

        // Nyquist sampling check - require necessary values to be present
        if let Some(min_sound_speed) = self.medium.sound_speed_min {
            let min_wavelength = min_sound_speed / self.simulation.frequency;
            // Use maximum grid spacing for most restrictive Nyquist condition
            let max_spacing = self.grid.spacing[0]
                .max(self.grid.spacing[1])
                .max(self.grid.spacing[2]);
            let min_ppw = min_wavelength / max_spacing;

            if min_ppw < 2.0 {
                multi_error.add(crate::error::ConfigError::InvalidValue {
                    parameter: "grid.spacing".to_string(),
                    value: format!("{:?}", self.grid.spacing),
                    constraint: format!(
                        "Nyquist criterion violated: {min_ppw} points per wavelength < 2 (min_wavelength={}, max_spacing={})",
                        min_wavelength, max_spacing
                    ),
                }.into());
            }
        } else {
            multi_error.add(
                crate::error::ConfigError::MissingParameter {
                    parameter: "medium.sound_speed_min".to_string(),
                    section: "Required for Nyquist validation".to_string(),
                }
                .into(),
            );
        }

        multi_error.into_result()
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
