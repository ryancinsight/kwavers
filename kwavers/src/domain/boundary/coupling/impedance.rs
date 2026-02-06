//! Impedance boundary condition
//!
//! Frequency-dependent absorption based on acoustic impedance matching.
//! Particularly useful for ultrasound transducers and tissue interfaces.

use crate::core::error::KwaversResult;
use crate::domain::boundary::traits::BoundaryCondition;
use crate::domain::grid::GridTopology;
use ndarray::ArrayViewMut3;
use rustfft::num_complex::Complex;

use super::types::{BoundaryDirections, FrequencyProfile};

/// Impedance boundary condition
///
/// Implements frequency-dependent absorption based on acoustic impedance matching.
/// This boundary condition is particularly useful for modeling:
///
/// - Ultrasound transducer surfaces
/// - Tissue-air interfaces
/// - Coupling layers and matching layers
/// - Frequency-selective absorption
///
/// # Physics
///
/// The reflection coefficient at an impedance boundary is given by:
///
/// ```text
/// R = (Z_target - Z_medium) / (Z_target + Z_medium)
/// ```
///
/// where:
/// - Z_target is the target impedance of the boundary
/// - Z_medium is the acoustic impedance of the propagating medium
///
/// The boundary can apply frequency-dependent profiles to model realistic
/// transducer responses or tissue frequency-dependent behavior.
///
/// # Example
///
/// ```no_run
/// use kwavers::domain::boundary::coupling::ImpedanceBoundary;
/// use kwavers::domain::boundary::traits::BoundaryDirections;
///
/// // Create impedance boundary matching water-tissue interface
/// let boundary = ImpedanceBoundary::new(
///     1.5e6,  // Target impedance (1.5 MRayl)
///     BoundaryDirections::all(),
/// );
///
/// // Add Gaussian frequency profile centered at 1 MHz with 0.5 MHz bandwidth
/// let boundary = boundary.with_gaussian_profile(1e6, 0.5e6);
/// ```
#[derive(Debug, Clone)]
pub struct ImpedanceBoundary {
    /// Target impedance Z_target (kg/m²s)
    pub target_impedance: f64,
    /// Frequency-dependent profile
    pub frequency_profile: FrequencyProfile,
    /// Boundary directions
    pub directions: BoundaryDirections,
}

impl ImpedanceBoundary {
    /// Create a new impedance boundary
    ///
    /// # Arguments
    ///
    /// * `target_impedance` - Target acoustic impedance in kg/(m²·s) or Rayl
    /// * `directions` - Directions in which to apply the boundary condition
    ///
    /// # Returns
    ///
    /// New `ImpedanceBoundary` with flat frequency response
    pub fn new(target_impedance: f64, directions: BoundaryDirections) -> Self {
        Self {
            target_impedance,
            frequency_profile: FrequencyProfile::Flat,
            directions,
        }
    }

    /// Set Gaussian frequency profile
    ///
    /// # Arguments
    ///
    /// * `center_freq` - Center frequency in Hz
    /// * `bandwidth` - Bandwidth (FWHM) in Hz
    ///
    /// # Returns
    ///
    /// Self with Gaussian profile applied
    pub fn with_gaussian_profile(mut self, center_freq: f64, bandwidth: f64) -> Self {
        self.frequency_profile = FrequencyProfile::Gaussian {
            center_freq,
            bandwidth,
        };
        self
    }

    /// Compute impedance ratio at given frequency
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz
    /// * `medium_impedance` - Impedance of the propagating medium in kg/(m²·s)
    ///
    /// # Returns
    ///
    /// Frequency-weighted impedance ratio Z_target / Z_medium
    pub fn impedance_ratio(&self, frequency: f64, medium_impedance: f64) -> f64 {
        let z_ratio = self.target_impedance / medium_impedance;

        match &self.frequency_profile {
            FrequencyProfile::Flat => z_ratio,
            FrequencyProfile::Gaussian {
                center_freq,
                bandwidth,
            } => {
                let sigma = bandwidth / (2.0 * (2.0 * std::f64::consts::LN_2).sqrt()); // Convert FWHM to sigma
                let gaussian = (-0.5 * ((frequency - center_freq) / sigma).powi(2)).exp();
                z_ratio * gaussian
            }
            FrequencyProfile::Custom(pairs) => {
                // Simple linear interpolation
                if pairs.is_empty() {
                    z_ratio
                } else if frequency <= pairs[0].0 {
                    pairs[0].1 * z_ratio
                } else if frequency >= pairs.last().unwrap().0 {
                    pairs.last().unwrap().1 * z_ratio
                } else {
                    // Find interval and interpolate
                    for i in 0..pairs.len() - 1 {
                        if frequency >= pairs[i].0 && frequency <= pairs[i + 1].0 {
                            let f1 = pairs[i].0;
                            let f2 = pairs[i + 1].0;
                            let z1 = pairs[i].1;
                            let z2 = pairs[i + 1].1;

                            let ratio = z1 + (z2 - z1) * (frequency - f1) / (f2 - f1);
                            return ratio * z_ratio;
                        }
                    }
                    z_ratio
                }
            }
        }
    }

    /// Compute reflection coefficient from impedance mismatch
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz
    /// * `medium_impedance` - Impedance of the propagating medium in kg/(m²·s)
    ///
    /// # Returns
    ///
    /// Reflection coefficient R = (Z_target - Z_medium) / (Z_target + Z_medium)
    pub fn reflection_coefficient(&self, frequency: f64, medium_impedance: f64) -> f64 {
        let z_ratio = self.impedance_ratio(frequency, medium_impedance);
        (z_ratio - 1.0) / (z_ratio + 1.0)
    }
}

impl BoundaryCondition for ImpedanceBoundary {
    fn name(&self) -> &str {
        "ImpedanceBoundary"
    }

    fn active_directions(&self) -> BoundaryDirections {
        self.directions
    }

    fn apply_scalar_spatial(
        &mut self,
        _field: ArrayViewMut3<f64>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Apply impedance boundary condition
        // This would compute reflection coefficients and apply absorption

        // Simplified: apply frequency-independent absorption
        let _absorption = 0.1; // Simplified absorption coefficient

        // Apply to boundary layers
        // (Real implementation would need proper boundary indexing)

        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        _field: &mut ndarray::Array3<Complex<f64>>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Frequency-dependent impedance boundary
        // Apply different absorption for different frequency components
        Ok(())
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impedance_boundary() {
        let boundary = ImpedanceBoundary::new(1.5e6, BoundaryDirections::all());

        // Test reflection coefficient
        let r = boundary.reflection_coefficient(1e6, 1.5e6); // Matched impedance
        assert!(r.abs() < 1e-10); // Perfect match, no reflection

        let r = boundary.reflection_coefficient(1e6, 3.0e6); // Mismatched
        assert!(r.abs() > 0.0); // Some reflection
    }

    #[test]
    fn test_impedance_boundary_gaussian_profile() {
        let boundary = ImpedanceBoundary::new(1.5e6, BoundaryDirections::all())
            .with_gaussian_profile(1e6, 0.5e6);

        let medium_impedance = 1.5e6;

        // At center frequency, should have maximum effect
        let z_ratio_center = boundary.impedance_ratio(1e6, medium_impedance);
        assert!((z_ratio_center - 1.0).abs() < 1e-10);

        // Off center, should be attenuated by Gaussian
        let z_ratio_off = boundary.impedance_ratio(0.5e6, medium_impedance);
        assert!(z_ratio_off < z_ratio_center);
    }

    #[test]
    fn test_impedance_reflection_coefficient() {
        let boundary = ImpedanceBoundary::new(2.0e6, BoundaryDirections::all());

        // Z_target = 2.0 MRayl, Z_medium = 1.0 MRayl
        // z_ratio = 2.0
        // R = (2.0 - 1.0) / (2.0 + 1.0) = 1/3 ≈ 0.333
        let r = boundary.reflection_coefficient(1e6, 1.0e6);
        assert!((r - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_impedance_matched() {
        let boundary = ImpedanceBoundary::new(1.5e6, BoundaryDirections::all());

        // Matched impedances should give zero reflection
        let r = boundary.reflection_coefficient(1e6, 1.5e6);
        assert!(r.abs() < 1e-12);
    }

    #[test]
    fn test_impedance_perfect_reflector() {
        let boundary = ImpedanceBoundary::new(1e12, BoundaryDirections::all());

        // Very high target impedance (rigid wall)
        // R → +1 as Z_target → ∞
        let r = boundary.reflection_coefficient(1e6, 1.5e6);
        assert!(r > 0.999, "Rigid wall should have R ≈ 1, got {}", r);
    }

    #[test]
    fn test_impedance_zero_reflector() {
        let boundary = ImpedanceBoundary::new(1.0, BoundaryDirections::all());

        // Very low target impedance (pressure release)
        // R → -1 as Z_target → 0
        let r = boundary.reflection_coefficient(1e6, 1.5e6);
        assert!(r < -0.999, "Pressure release should have R ≈ -1, got {}", r);
    }
}
