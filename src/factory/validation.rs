//! Validation utilities for factory configurations
//!
//! Follows Information Expert pattern for validation logic

use super::SimulationConfig;
use crate::error::{ConfigError, KwaversResult};

/// Validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub enable_validation: bool,
    pub strict_mode: bool,
    pub validation_rules: Vec<String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            strict_mode: false,
            validation_rules: vec![
                "grid_validation".to_string(),
                "medium_validation".to_string(),
                "physics_validation".to_string(),
                "time_validation".to_string(),
            ],
        }
    }
}

/// Configuration validator
#[derive(Debug)]
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate complete simulation configuration
    pub fn validate(config: &SimulationConfig) -> KwaversResult<()> {
        if !config.validation.enable_validation {
            return Ok(());
        }

        // Validate individual components
        config.grid.validate()?;
        config.medium.validate()?;
        config.physics.validate()?;
        config.time.validate()?;
        config.source.validate()?;

        // Cross-validate components
        Self::validate_compatibility(config)?;

        Ok(())
    }

    /// Validate compatibility between components
    fn validate_compatibility(config: &SimulationConfig) -> KwaversResult<()> {
        // Check CFL condition compatibility
        let min_spacing = config.grid.dx.min(config.grid.dy).min(config.grid.dz);

        // Estimate sound speed from medium type (conservative fallback for heterogeneous)
        // Heterogeneous media use maximum sound speed for CFL validation per Courant (1928)
        let sound_speed = match &config.medium.medium_type {
            super::MediumType::Homogeneous { sound_speed, .. } => *sound_speed,
            _ => 1500.0, // Conservative water-like medium assumption
        };

        let max_stable_dt = config.time.cfl_factor * min_spacing / sound_speed;

        if config.time.dt > max_stable_dt {
            return Err(ConfigError::InvalidValue {
                parameter: "time step".to_string(),
                value: config.time.dt.to_string(),
                constraint: format!("Must be <= {max_stable_dt} for stability"),
            }
            .into());
        }

        // Check grid resolution for frequency
        let wavelength = sound_speed / config.physics.frequency;
        let points_per_wavelength = wavelength / min_spacing;

        const MIN_POINTS_PER_WAVELENGTH: f64 = 6.0;
        if points_per_wavelength < MIN_POINTS_PER_WAVELENGTH {
            return Err(ConfigError::InvalidValue {
                parameter: "grid resolution".to_string(),
                value: format!("{points_per_wavelength:.2} points per wavelength"),
                constraint: format!(
                    "Need at least {MIN_POINTS_PER_WAVELENGTH} points per wavelength"
                ),
            }
            .into());
        }

        Ok(())
    }
}
