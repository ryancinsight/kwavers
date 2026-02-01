// src/physics/optics/map_builder.rs
//! Optical Property Map Builder - Physics Analysis Utilities
//!
//! **ARCHITECTURAL NOTE**: Core types (Region, Layer, OpticalPropertyMap, OpticalPropertyMapBuilder)
//! have been moved to `domain::medium::optical_map` for clean architecture compliance.
//!
//! This module now contains only physics-specific analysis utilities and re-exports domain types
//! for backwards compatibility.
//!
//! # Architecture
//!
//! - Domain layer (`domain::medium::optical_map`) defines core types: Region, Layer, OpticalPropertyMap, Builder
//! - Physics layer (this module) provides analysis tools and physics-specific utilities
//! - Clinical layer uses domain types directly for clean architecture compliance
//!
//! # Migration Guide
//!
//! **Old (deprecated)**:
//! ```rust,ignore
//! use crate::physics::optics::map_builder::{OpticalPropertyMap, Region};
//! ```
//!
//! **New (recommended)**:
//! ```rust,ignore
//! use crate::domain::medium::optical_map::{OpticalPropertyMap, Region};
//! ```

use ndarray::Array3;

// Re-export domain types for backwards compatibility
pub use crate::domain::medium::optical_map::{
    Layer, OpticalPropertyMap, OpticalPropertyMapBuilder, Region,
};

/// Statistical summary of optical property distribution
///
/// Provides basic statistics for analyzing optical property maps,
/// useful for validation and quality assurance.
#[derive(Debug, Clone)]
pub struct PropertyStats {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
}

impl PropertyStats {
    /// Compute statistics from an array
    pub fn from_array(array: &Array3<f64>) -> Self {
        let values: Vec<f64> = array.iter().copied().collect();
        Self::from_values(&values)
    }

    /// Compute statistics from a slice
    pub fn from_values(values: &[f64]) -> Self {
        let n = values.len() as f64;

        let mean = values.iter().sum::<f64>() / n;

        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        Self {
            mean,
            std_dev,
            min,
            max,
        }
    }
}

impl OpticalPropertyMap {
    /// Compute statistics for absorption coefficient distribution
    pub fn absorption_stats(&self) -> PropertyStats {
        PropertyStats::from_array(&self.mu_a)
    }

    /// Compute statistics for scattering coefficient distribution
    pub fn scattering_stats(&self) -> PropertyStats {
        PropertyStats::from_array(&self.mu_s_prime)
    }

    /// Compute statistics for refractive index distribution
    pub fn refractive_index_stats(&self) -> PropertyStats {
        PropertyStats::from_array(&self.refractive_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::GridDimensions;
    use crate::domain::medium::properties::OpticalPropertyData;

    #[test]
    fn test_property_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = PropertyStats::from_values(&values);

        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_optical_map_stats() {
        let dims = GridDimensions {
            nx: 10,
            ny: 10,
            nz: 10,
            dx: 0.001,
            dy: 0.001,
            dz: 0.001,
        };

        let props = OpticalPropertyData::soft_tissue();
        let map = OpticalPropertyMap::homogeneous(&props, dims);

        let absorption_stats = map.absorption_stats();
        assert!((absorption_stats.mean - props.absorption_coefficient).abs() < 1e-10);
        assert!((absorption_stats.std_dev).abs() < 1e-10); // Should be zero for homogeneous
    }

    #[test]
    fn test_domain_types_reexport() {
        // Verify we can use domain types through physics re-export
        let region = Region::sphere([0.0, 0.0, 0.0], 1.0);
        assert!(region.contains([0.0, 0.0, 0.0]));
    }
}
