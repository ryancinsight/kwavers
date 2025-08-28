//! Grid factory for creating computational grids
//!
//! Follows Information Expert principle - knows how to create and validate grids

use crate::constants;
use crate::error::{ConfigError, KwaversResult};
use crate::grid::Grid;

/// Grid configuration
#[derive(Debug, Clone)]
pub struct GridConfig {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl GridConfig {
    /// Validate grid configuration
    /// Follows Information Expert principle - knows how to validate itself
    pub fn validate(&self) -> KwaversResult<()> {
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "grid dimensions".to_string(),
                value: format!("({}, {}, {})", self.nx, self.ny, self.nz),
                constraint: "All dimensions must be positive".to_string(),
            }
            .into());
        }

        if self.dx <= 0.0 || self.dy <= 0.0 || self.dz <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "grid spacing".to_string(),
                value: format!("({}, {}, {})", self.dx, self.dy, self.dz),
                constraint: "All spacing values must be positive".to_string(),
            }
            .into());
        }

        // Check for reasonable grid size to prevent memory issues
        let total_points = self.nx * self.ny * self.nz;
        const MAX_GRID_POINTS: usize = 100_000_000; // Named constant instead of magic number
        if total_points > MAX_GRID_POINTS {
            return Err(ConfigError::InvalidValue {
                parameter: "grid size".to_string(),
                value: total_points.to_string(),
                constraint: format!("Total grid points must be <= {}", MAX_GRID_POINTS),
            }
            .into());
        }

        // Check minimum grid spacing
        if self.dx < constants::stability::MIN_DX
            || self.dy < constants::stability::MIN_DX
            || self.dz < constants::stability::MIN_DX
        {
            return Err(ConfigError::InvalidValue {
                parameter: "grid spacing".to_string(),
                value: format!("({}, {}, {})", self.dx, self.dy, self.dz),
                constraint: format!("Spacing must be >= {}", constants::stability::MIN_DX),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            nx: 128,
            ny: 128,
            nz: 128,
            dx: constants::grid::DEFAULT_GRID_SPACING,
            dy: constants::grid::DEFAULT_GRID_SPACING,
            dz: constants::grid::DEFAULT_GRID_SPACING,
        }
    }
}

/// Factory for creating grids
#[derive(Debug)]
pub struct GridFactory;

impl GridFactory {
    /// Create a grid from configuration
    pub fn create_grid(config: &GridConfig) -> KwaversResult<Grid> {
        config.validate()?;

        Ok(Grid::new(
            config.nx, config.ny, config.nz, config.dx, config.dy, config.dz,
        ))
    }

    /// Create a uniform grid with equal spacing
    pub fn create_uniform(size: usize, spacing: f64) -> KwaversResult<Grid> {
        let config = GridConfig {
            nx: size,
            ny: size,
            nz: size,
            dx: spacing,
            dy: spacing,
            dz: spacing,
        };
        Self::create_grid(&config)
    }

    /// Create a grid optimized for a given frequency
    pub fn create_for_frequency(
        frequency: f64,
        sound_speed: f64,
        points_per_wavelength: usize,
    ) -> KwaversResult<Grid> {
        let wavelength = sound_speed / frequency;
        let spacing = wavelength / points_per_wavelength as f64;

        // Calculate reasonable grid size based on wavelength
        let size = ((wavelength * 10.0) / spacing) as usize;
        let size = size.max(constants::grid::MIN_GRID_POINTS);

        let config = GridConfig {
            nx: size,
            ny: size,
            nz: size,
            dx: spacing,
            dy: spacing,
            dz: spacing,
        };
        Self::create_grid(&config)
    }
}
