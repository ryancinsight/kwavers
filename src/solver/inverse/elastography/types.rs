//! Elastography Result Types
//!
//! Domain types for elastography inversion results, including elasticity maps
//! and nonlinear parameter characterization.

use crate::domain::imaging::ultrasound::elastography::{ElasticityMap, NonlinearParameterMap};
use ndarray::Array3;

/// Extension trait for `ElasticityMap` statistics
pub trait ElasticityMapExt {
    /// Get elasticity statistics (min, max, mean)
    fn statistics(&self) -> (f64, f64, f64);
}

impl ElasticityMapExt for ElasticityMap {
    fn statistics(&self) -> (f64, f64, f64) {
        let min = self
            .youngs_modulus
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .youngs_modulus
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = self.youngs_modulus.mean().unwrap_or(0.0);
        (min, max, mean)
    }
}

/// Extension trait for `NonlinearParameterMap` statistics
pub trait NonlinearParameterMapExt {
    /// Get nonlinearity statistics (min, max, mean)
    fn nonlinearity_statistics(&self) -> (f64, f64, f64);

    /// Get estimation quality statistics
    fn quality_statistics(&self) -> (f64, f64, f64);
}

impl NonlinearParameterMapExt for NonlinearParameterMap {
    fn nonlinearity_statistics(&self) -> (f64, f64, f64) {
        let min = self
            .nonlinearity_parameter
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .nonlinearity_parameter
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = self.nonlinearity_parameter.mean().unwrap_or(0.0);
        (min, max, mean)
    }

    fn quality_statistics(&self) -> (f64, f64, f64) {
        let min = self
            .estimation_quality
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .estimation_quality
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = self.estimation_quality.mean().unwrap_or(0.0);
        (min, max, mean)
    }
}

/// Helper function to create `ElasticityMap` from shear wave speed field
///
/// Converts shear wave speed measurements to elasticity parameters using
/// established constitutive relations for incompressible isotropic materials.
///
/// # Physics
///
/// For incompressible materials (Poisson's ratio ν ≈ 0.5):
/// - Shear modulus: μ = ρ cs²
/// - Young's modulus: E = 3μ (for ν = 0.5)
/// - Bulk modulus: K → ∞
///
/// # Arguments
///
/// * `shear_wave_speed` - 3D array of shear wave speeds (m/s)
/// * `density` - Tissue density (kg/m³)
///
/// # References
///
/// - Fung, Y.C. (1993). "Biomechanics: Mechanical Properties of Living Tissues"
/// - Duck, F.A. (1990). "Physical Properties of Tissue"
pub fn elasticity_map_from_speed(shear_wave_speed: Array3<f64>, density: f64) -> ElasticityMap {
    ElasticityMap::from_shear_wave_speed(shear_wave_speed, density)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_elasticity_map_from_speed() {
        let speed = Array3::from_elem((10, 10, 10), 3.0); // 3 m/s
        let density = 1000.0; // kg/m³

        let map = elasticity_map_from_speed(speed, density);

        // Check physics: μ = ρcs²
        let expected_shear_modulus = density * 3.0 * 3.0; // 9000 Pa
        let expected_youngs_modulus = 3.0 * expected_shear_modulus; // 27000 Pa

        assert!((map.shear_modulus[[5, 5, 5]] - expected_shear_modulus).abs() < 1e-6);
        assert!((map.youngs_modulus[[5, 5, 5]] - expected_youngs_modulus).abs() < 1e-6);
    }

    #[test]
    fn test_elasticity_statistics() {
        let mut speed = Array3::from_elem((10, 10, 10), 3.0);
        speed[[5, 5, 5]] = 5.0; // Higher stiffness region

        let map = elasticity_map_from_speed(speed, 1000.0);
        let (min, max, mean) = map.statistics();

        assert!(min < max, "Minimum should be less than maximum");
        assert!(
            mean > min && mean < max,
            "Mean should be between min and max"
        );
    }

    #[test]
    fn test_nonlinear_parameter_statistics() {
        let nonlinearity_parameter = Array3::from_elem((10, 10, 10), 5.0);
        let elastic_constants = vec![
            Array3::from_elem((10, 10, 10), 1.0),
            Array3::from_elem((10, 10, 10), 2.0),
            Array3::from_elem((10, 10, 10), 3.0),
            Array3::from_elem((10, 10, 10), 4.0),
        ];
        let nonlinearity_uncertainty = Array3::from_elem((10, 10, 10), 0.5);
        let estimation_quality = Array3::from_elem((10, 10, 10), 0.9);

        let map = NonlinearParameterMap {
            nonlinearity_parameter,
            elastic_constants,
            nonlinearity_uncertainty,
            estimation_quality,
        };

        let (min, max, mean) = map.nonlinearity_statistics();
        assert!((mean - 5.0).abs() < 1e-6, "Mean should be 5.0");
        assert!((min - 5.0).abs() < 1e-6, "Min should be 5.0");
        assert!((max - 5.0).abs() < 1e-6, "Max should be 5.0");

        let (q_min, q_max, q_mean) = map.quality_statistics();
        assert!((q_mean - 0.9).abs() < 1e-6, "Quality mean should be 0.9");
        assert!((q_min - 0.9).abs() < 1e-6, "Quality min should be 0.9");
        assert!((q_max - 0.9).abs() < 1e-6, "Quality max should be 0.9");
    }
}
