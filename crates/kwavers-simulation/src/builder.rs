//! Configuration builder for programmatic configuration creation
//!
//! Provides a fluent, type-safe API for building configurations without
//! dealing with nested public fields.

use super::{Configuration, SimulationParameters};
use kwavers_core::error::KwaversResult;
use kwavers_boundary::config::BoundaryParameters;
use kwavers_medium::config::DomainMediumParameters;
use kwavers_domain::source::config::DomainSourceParameters;
use crate::parameters::{OutputParameters, PerformanceParameters};
use kwavers_solver::config::SolverConfiguration;
use kwavers_solver::validation::ValidationParameters;

/// Builder for creating Configuration instances programmatically
#[derive(Debug, Default)]
pub struct ConfigurationBuilder {
    config: Configuration,
}

impl ConfigurationBuilder {
    /// Create a new configuration builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: Configuration::default(),
        }
    }

    /// Set simulation parameters
    #[must_use]
    pub fn simulation(mut self, params: SimulationParameters) -> Self {
        self.config.simulation = params;
        self
    }

    /// Set grid spacing
    #[must_use]
    pub fn grid_spacing(mut self, spacing: [f64; 3]) -> Self {
        self.config.grid.spacing = spacing;
        self
    }

    /// Set grid dimensions
    #[must_use]
    pub fn grid_dimensions(mut self, nx: usize, ny: usize, nz: usize) -> Self {
        self.config.grid.dimensions = [nx, ny, nz];
        self
    }

    /// Set medium properties
    #[must_use]
    pub fn medium(mut self, params: DomainMediumParameters) -> Self {
        self.config.medium = params;
        self
    }

    /// Set sound speed range
    #[must_use]
    pub fn sound_speed_range(mut self, min: f64, max: f64) -> Self {
        self.config.medium.sound_speed_min = Some(min);
        self.config.medium.sound_speed_max = Some(max);
        self
    }

    /// Set source configuration
    #[must_use]
    pub fn source(mut self, params: DomainSourceParameters) -> Self {
        self.config.source = params;
        self
    }

    /// Set boundary conditions
    #[must_use]
    pub fn boundary(mut self, params: BoundaryParameters) -> Self {
        self.config.boundary = params;
        self
    }

    /// Set solver parameters
    #[must_use]
    pub fn solver(mut self, params: SolverConfiguration) -> Self {
        self.config.solver = params;
        self
    }

    /// Set spatial order
    #[must_use]
    pub fn spatial_order(mut self, order: usize) -> Self {
        self.config.solver.spatial_order = order;
        self
    }

    /// Set output configuration
    #[must_use]
    pub fn output(mut self, params: OutputParameters) -> Self {
        self.config.output = params;
        self
    }

    /// Set performance tuning
    #[must_use]
    pub fn performance(mut self, params: PerformanceParameters) -> Self {
        self.config.performance = params;
        self
    }

    /// Set validation settings
    #[must_use]
    pub fn validation(mut self, params: ValidationParameters) -> Self {
        self.config.validation = params;
        self
    }

    /// Set time step
    #[must_use]
    pub fn time_step(mut self, dt: f64) -> Self {
        self.config.simulation.dt = Some(dt);
        self
    }

    /// Set simulation frequency
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn frequency(mut self, freq: f64) -> Self {
        self.config.simulation.frequency = freq;
        self
    }

    /// Set CFL number
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn cfl(mut self, cfl: f64) -> Self {
        self.config.simulation.cfl = cfl;
        self
    }

    /// Build and validate the configuration
    ///
    /// This method runs validation to ensure the configuration is complete
    /// and consistent before returning it.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn build(self) -> KwaversResult<Configuration> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation
    ///
    /// Returns the configuration without running validation. Use this if you
    /// need to perform custom validation or if the configuration is intentionally
    /// incomplete for testing purposes.
    #[must_use]
    pub fn build_unchecked(self) -> Configuration {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;

    #[test]
    fn test_configuration_builder_basic() {
        let config = ConfigurationBuilder::new()
            .grid_spacing([0.001, 0.001, 0.001])
            .grid_dimensions(100, 100, 100)
            .frequency(MHZ_TO_HZ)
            .cfl(0.5)
            .build_unchecked();

        assert_eq!(config.grid.spacing, [0.001, 0.001, 0.001]);
        assert_eq!(config.grid.dimensions, [100, 100, 100]);
        assert_eq!(config.simulation.frequency, MHZ_TO_HZ);
        assert_eq!(config.simulation.cfl, 0.5);
    }

    #[test]
    fn test_configuration_builder_validation() {
        // This should work with reasonable parameters
        // For 1 MHz frequency and 1000 m/s sound speed:
        // wavelength = 1000/1e6 = 0.001 m = 1 mm
        // For Nyquist criterion (2 points per wavelength):
        // max_spacing = wavelength/2 = 0.0005 m = 0.5 mm
        let result = ConfigurationBuilder::new()
            .grid_spacing([0.0005, 0.0005, 0.0005]) // 0.5 mm spacing
            .grid_dimensions(100, 100, 100) // Add required grid dimensions
            .frequency(MHZ_TO_HZ) // 1 MHz
            .sound_speed_range(1000.0, 1600.0) // 1000-1600 m/s
            .cfl(0.5)
            .time_step(1e-7)
            .build();

        // Debug validation errors if any
        match &result {
            Ok(_) => println!("Configuration validation passed"),
            Err(e) => println!("Configuration validation failed: {}", e),
        }

        assert!(
            result.is_ok(),
            "Configuration validation failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_configuration_builder_chaining() {
        let builder = ConfigurationBuilder::new()
            .frequency(MHZ_TO_HZ)
            .cfl(0.5)
            .grid_spacing([0.001, 0.001, 0.001])
            .sound_speed_range(1400.0, 1600.0);

        let config = builder.build_unchecked();
        assert_eq!(config.simulation.frequency, MHZ_TO_HZ);
        assert_eq!(config.simulation.cfl, 0.5);
        assert_eq!(config.grid.spacing, [0.001, 0.001, 0.001]);
    }
}
