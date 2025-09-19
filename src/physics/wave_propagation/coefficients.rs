//! Propagation coefficients and calculation results
//!
//! This module provides structures for storing and manipulating wave propagation
//! calculation results, including reflection and transmission coefficients.

/// Result of wave propagation calculations
#[derive(Debug, Clone)]
pub struct PropagationCoefficients {
    /// Reflection amplitude coefficient
    pub reflection_amplitude: f64,
    /// Transmission amplitude coefficient  
    pub transmission_amplitude: f64,
    /// Reflection phase shift [radians]
    pub reflection_phase: f64,
    /// Transmission phase shift [radians]
    pub transmission_phase: f64,
    /// Whether total internal reflection occurs
    pub total_internal_reflection: bool,
    /// Incident angle [radians]
    pub incident_angle: f64,
    /// Transmitted angle [radians] (None for total internal reflection)
    pub transmitted_angle: Option<f64>,
}

impl PropagationCoefficients {
    /// Calculate reflectance (power reflection coefficient)
    #[must_use]
    pub fn reflectance(&self) -> f64 {
        self.reflection_amplitude.powi(2)
    }

    /// Calculate transmittance (power transmission coefficient)
    #[must_use]
    pub fn transmittance(&self) -> f64 {
        if self.total_internal_reflection {
            0.0
        } else {
            self.transmission_amplitude.powi(2)
        }
    }

    /// Verify energy conservation (R + T = 1 for lossless interface)
    #[must_use]
    pub fn energy_conservation_error(&self) -> f64 {
        let total = self.reflectance() + self.transmittance();
        (total - 1.0).abs()
    }

    /// Get the complex reflection coefficient
    #[must_use]
    pub fn complex_reflection_coefficient(&self) -> (f64, f64) {
        let real = self.reflection_amplitude * self.reflection_phase.cos();
        let imag = self.reflection_amplitude * self.reflection_phase.sin();
        (real, imag)
    }

    /// Get the complex transmission coefficient
    #[must_use]
    pub fn complex_transmission_coefficient(&self) -> (f64, f64) {
        let real = self.transmission_amplitude * self.transmission_phase.cos();
        let imag = self.transmission_amplitude * self.transmission_phase.sin();
        (real, imag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_reflectance_transmittance() {
        let coeffs = PropagationCoefficients {
            reflection_amplitude: 0.6,
            transmission_amplitude: 0.8,
            reflection_phase: 0.0,
            transmission_phase: 0.0,
            total_internal_reflection: false,
            incident_angle: 0.0,
            transmitted_angle: Some(0.0),
        };

        assert_relative_eq!(coeffs.reflectance(), 0.36, epsilon = 1e-10);
        assert_relative_eq!(coeffs.transmittance(), 0.64, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_conservation() {
        let coeffs = PropagationCoefficients {
            reflection_amplitude: 0.6,
            transmission_amplitude: 0.8,
            reflection_phase: 0.0,
            transmission_phase: 0.0,
            total_internal_reflection: false,
            incident_angle: 0.0,
            transmitted_angle: Some(0.0),
        };

        let error = coeffs.energy_conservation_error();
        assert!(error < 1e-10, "Energy conservation error: {}", error);
    }

    #[test]
    fn test_total_internal_reflection() {
        let coeffs = PropagationCoefficients {
            reflection_amplitude: 1.0,
            transmission_amplitude: 0.0,
            reflection_phase: 0.0,
            transmission_phase: 0.0,
            total_internal_reflection: true,
            incident_angle: std::f64::consts::PI / 3.0,
            transmitted_angle: None,
        };

        assert_relative_eq!(coeffs.reflectance(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs.transmittance(), 0.0, epsilon = 1e-10);
        assert!(coeffs.total_internal_reflection);
        assert!(coeffs.transmitted_angle.is_none());
    }
}
