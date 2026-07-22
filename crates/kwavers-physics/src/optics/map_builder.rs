//! Optical-property map analysis.
//!
//! `kwavers-medium` owns the validated spatial aggregate and its builder. This
//! module adds only physics-facing distribution statistics.

use kwavers_medium::optical_map::OpticalPropertyMap;

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
    /// Compute statistics from a slice.
    ///
    /// Returns all-zero stats when `values` is empty.
    #[must_use]
    pub fn from_values(values: &[f64]) -> Self {
        Self::from_samples(values.iter().copied())
    }

    /// Compute statistics from a stream without allocating an intermediate array.
    #[must_use]
    pub fn from_samples(values: impl IntoIterator<Item = f64>) -> Self {
        let mut count = 0.0;
        let mut mean = 0.0;
        let mut squared_deviation = 0.0;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for value in values {
            count += 1.0;
            let delta = value - mean;
            mean += delta / count;
            squared_deviation += delta * (value - mean);
            min = min.min(value);
            max = max.max(value);
        }

        if count == 0.0 {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }
        Self {
            mean,
            std_dev: (squared_deviation / count).sqrt(),
            min,
            max,
        }
    }
}

/// Physics-layer statistical analysis over a domain [`OpticalPropertyMap`].
///
/// Implemented as an extension trait because the map type now lives in the
/// `kwavers-medium` crate; inherent `impl`s cannot cross crate
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
        PropertyStats::from_samples(
            self.properties()
                .iter()
                .map(|properties| properties.absorption_coefficient()),
        )
    }

    fn scattering_stats(&self) -> PropertyStats {
        PropertyStats::from_samples(
            self.properties()
                .iter()
                .map(|properties| properties.reduced_scattering_coefficient()),
        )
    }

    fn refractive_index_stats(&self) -> PropertyStats {
        PropertyStats::from_samples(
            self.properties()
                .iter()
                .map(|properties| properties.refractive_index()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_grid::GridDimensions;
    use kwavers_medium::optical_map::Region;
    use kwavers_medium::properties::OpticalPropertyData;

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

        let props = OpticalPropertyData::new(2.0, 80.0, 0.72, 1.41).unwrap();
        let map = OpticalPropertyMap::homogeneous(&props, dims);

        let absorption_stats = map.absorption_stats();
        assert!((absorption_stats.mean - props.absorption_coefficient()).abs() < 1e-10);
        assert!((absorption_stats.std_dev).abs() < 1e-10); // Should be zero for homogeneous

        let scattering_stats = map.scattering_stats();
        assert_eq!(
            scattering_stats.mean,
            props.reduced_scattering_coefficient()
        );
        assert_eq!(scattering_stats.std_dev, 0.0);
    }

    #[test]
    fn test_domain_types_reexport() {
        // Verify we can use domain types through physics re-export
        let region = Region::sphere([0.0, 0.0, 0.0], 1.0);
        assert!(region.contains([0.0, 0.0, 0.0]));
    }
}
