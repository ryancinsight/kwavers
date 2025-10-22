//! Grid factory convenience wrapper
//!
//! Re-exports hierarchical implementation with additional convenience methods

pub use super::component::grid::{GridConfig, GridCreator, GridFactory, GridValidator};

// Additional convenience methods for the factory
use crate::error::KwaversResult;
use crate::grid::Grid;

impl GridFactory {
    /// Create a uniform grid with equal spacing
    /// Additional convenience method not in hierarchical version
    pub fn create_uniform(size: usize, spacing: f64) -> KwaversResult<Grid> {
        let config = GridConfig::new(size, size, size, spacing, spacing, spacing);
        Self::create_grid(&config)
    }

    /// Create a grid optimized for a given frequency
    /// Domain-specific factory method following Information Expert
    pub fn create_for_frequency(
        frequency: f64,
        sound_speed: f64,
        points_per_wavelength: usize,
    ) -> KwaversResult<Grid> {
        let wavelength = sound_speed / frequency;
        let spacing = wavelength / points_per_wavelength as f64;

        // Calculate reasonable grid size based on wavelength
        let size = ((wavelength * 10.0) / spacing) as usize;
        let size = size.max(16); // Minimum 16 points per dimension

        let config = GridConfig::new(size, size, size, spacing, spacing, spacing);
        Self::create_grid(&config)
    }
}
