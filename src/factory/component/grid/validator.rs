//! Grid validation rules - Evidence-based constraints
//!
//! Implements comprehensive validation following senior engineering standards

use super::config::GridConfig;
use crate::error::{ConfigError, KwaversResult};
use crate::physics::constants::numerical::MIN_DX;

/// Specialized grid validator following Single Responsibility Principle
#[derive(Debug)]
pub struct GridValidator;

impl GridValidator {
    /// Validate grid configuration with evidence-based constraints
    pub fn validate(config: &GridConfig) -> KwaversResult<()> {
        Self::validate_dimensions(config)?;
        Self::validate_spacing(config)?;
        Self::validate_memory_requirements(config)?;
        Self::validate_numerical_stability(config)?;
        Ok(())
    }

    /// Validate grid dimensions are physically meaningful
    fn validate_dimensions(config: &GridConfig) -> KwaversResult<()> {
        if config.nx == 0 || config.ny == 0 || config.nz == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "grid_dimensions".to_string(),
                value: format!("({}, {}, {})", config.nx, config.ny, config.nz),
                constraint: "All dimensions must be positive".to_string(),
            }
            .into());
        }

        // Minimum grid size for meaningful simulation
        const MIN_DIMENSION: usize = 10;
        if config.nx < MIN_DIMENSION || config.ny < MIN_DIMENSION || config.nz < MIN_DIMENSION {
            return Err(ConfigError::InvalidValue {
                parameter: "grid_dimensions".to_string(),
                value: format!("({}, {}, {})", config.nx, config.ny, config.nz),
                constraint: format!(
                    "All dimensions must be >= {MIN_DIMENSION} for meaningful simulation"
                ),
            }
            .into());
        }

        Ok(())
    }

    /// Validate grid spacing meets numerical requirements
    fn validate_spacing(config: &GridConfig) -> KwaversResult<()> {
        if config.dx <= 0.0 || config.dy <= 0.0 || config.dz <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "grid_spacing".to_string(),
                value: format!("({:.2e}, {:.2e}, {:.2e})", config.dx, config.dy, config.dz),
                constraint: "All spacing values must be positive".to_string(),
            }
            .into());
        }

        // Check minimum spacing for numerical stability
        if config.dx < MIN_DX || config.dy < MIN_DX || config.dz < MIN_DX {
            return Err(ConfigError::InvalidValue {
                parameter: "grid_spacing".to_string(),
                value: format!("({:.2e}, {:.2e}, {:.2e})", config.dx, config.dy, config.dz),
                constraint: format!("Spacing must be >= {MIN_DX:.2e} for numerical stability"),
            }
            .into());
        }

        Ok(())
    }

    /// Validate memory requirements are reasonable
    fn validate_memory_requirements(config: &GridConfig) -> KwaversResult<()> {
        const MAX_GRID_POINTS: usize = 100_000_000;
        const MAX_MEMORY_GB: usize = 8;

        let total_points = config.total_points();
        if total_points > MAX_GRID_POINTS {
            return Err(ConfigError::InvalidValue {
                parameter: "grid_size".to_string(),
                value: total_points.to_string(),
                constraint: format!("Total grid points must be <= {MAX_GRID_POINTS}"),
            }
            .into());
        }

        let memory_gb = config.memory_estimate() / (1024 * 1024 * 1024);
        if memory_gb > MAX_MEMORY_GB {
            return Err(ConfigError::InvalidValue {
                parameter: "memory_usage".to_string(),
                value: format!("{memory_gb}GB"),
                constraint: format!("Memory usage must be <= {MAX_MEMORY_GB}GB"),
            }
            .into());
        }

        Ok(())
    }

    /// Validate numerical stability requirements
    fn validate_numerical_stability(config: &GridConfig) -> KwaversResult<()> {
        // Check aspect ratios for numerical accuracy
        let max_aspect_ratio = 10.0;

        let dx_dy_ratio = config.dx / config.dy;
        if dx_dy_ratio > max_aspect_ratio || dx_dy_ratio < 1.0 / max_aspect_ratio {
            return Err(ConfigError::InvalidValue {
                parameter: "grid_aspect_ratio".to_string(),
                value: format!("{dx_dy_ratio:.2}"),
                constraint: format!(
                    "dx/dy ratio must be within 1/{max_aspect_ratio} to {max_aspect_ratio}"
                ),
            }
            .into());
        }

        Ok(())
    }
}
