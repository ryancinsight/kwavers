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
//! use crate::optics::map_builder::{OpticalPropertyMap, Region};
//! ```
//!
//! **New (recommended)**:
//! ```rust,ignore
//! use kwavers_domain::medium::optical_map::{OpticalPropertyMap, Region};
//! ```

use ndarray::Array3;

// Re-export domain types for backwards compatibility
pub use kwavers_domain::medium::optical_map::{
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
    #[must_use]
    pub fn from_array(array: &Array3<f64>) -> Self {
        let values: Vec<f64> = array.iter().copied().collect();
        Self::from_values(&values)
    }

    /// Compute statistics from a slice.
    ///
    /// Returns all-zero stats when `values` is empty.
    #[must_use]
    pub fn from_values(values: &[f64]) -> Self {
        let n = values.len();
        if n == 0 {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }
        let n_f = n as f64;
        let mean = values.iter().sum::<f64>() / n_f;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_f;
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

/// Physics-layer statistical analysis over a domain [`OpticalPropertyMap`].
///
/// Implemented as an extension trait because the map type now lives in the
/// `kwavers-domain` crate (ADR 011); inherent `impl`s cannot cross crate
/// boundaries. Bring this trait into scope to call `*_stats()` on a map.
pub trait OpticalPropertyMapAnalysis {
    /// Statistics for the absorption-coefficient distribution.
    fn absorption_stats(&self) -> PropertyStats;
    /// Statistics for the reduced-scattering-coefficient distribution.
    fn scattering_stats(&self) -> PropertyStats;
    /// Statistics for the refractive-index distribution.
    fn refractive_index_stats(&self) -> PropertyStats;
}

impl OpticalPropertyMapAnalysis for OpticalPropertyMap {
    fn absorption_stats(&self) -> PropertyStats {
        PropertyStats::from_array(&self.mu_a)
    }

    fn scattering_stats(&self) -> PropertyStats {
        PropertyStats::from_array(&self.mu_s_prime)
    }

    fn refractive_index_stats(&self) -> PropertyStats {
        PropertyStats::from_array(&self.refractive_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_grid::GridDimensions;
    use kwavers_domain::medium::properties::OpticalPropertyData;

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
