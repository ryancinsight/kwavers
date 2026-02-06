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
    /// Reflection phase shift \[radians\]
    pub reflection_phase: f64,
    /// Transmission phase shift \[radians\]
    pub transmission_phase: f64,
    /// Whether total internal reflection occurs
    pub total_internal_reflection: bool,
    /// Incident angle \[radians\]
    pub incident_angle: f64,
    /// Transmitted angle \[radians\] (None for total internal reflection)
    pub transmitted_angle: Option<f64>,
    /// Acoustic impedance of medium 1 (optional, for energy conservation)
    pub impedance1: Option<f64>,
    /// Acoustic impedance of medium 2 (optional, for energy conservation)
    pub impedance2: Option<f64>,
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

    /// Verify energy conservation for acoustic waves
    ///
    /// For acoustic waves at oblique incidence, energy conservation requires:
    /// R + T_intensity = 1
    ///
    /// Where:
    /// - R = reflectance (|r|²)
    /// - T_intensity = |t|² × (Z₁/Z₂) × (cos θ_t / cos θ_i)
    /// - Z₁, Z₂ = acoustic impedances
    /// - θ_i, θ_t = incident and transmitted angles
    ///
    /// **Literature**: Hamilton & Blackstock (1998) "Nonlinear Acoustics", Chapter 3
    /// **Equation**: R + T × (ρ₁c₁ cos θ_t)/(ρ₂c₂ cos θ_i) = 1
    ///
    /// For normal incidence (θ = 0) or when impedances are not provided,
    /// falls back to simple R + T = 1 check (appropriate for optical waves).
    #[must_use]
    pub fn energy_conservation_error(&self) -> f64 {
        let r = self.reflectance();
        let t = self.transmittance();

        // If impedances are provided and we have transmitted angle, use full formula
        if let (Some(z1), Some(z2), Some(theta_t)) =
            (self.impedance1, self.impedance2, self.transmitted_angle)
        {
            let theta_i = self.incident_angle;
            let cos_i = theta_i.cos();
            let cos_t = theta_t.cos();

            // Avoid division by zero
            if cos_i.abs() < 1e-15 || z2.abs() < 1e-15 {
                return (r + t - 1.0).abs();
            }

            // Energy conservation with intensity correction for acoustic waves
            // T_intensity = T_amplitude * (Z1/Z2) * (cos_t/cos_i)
            let intensity_ratio = (z1 / z2) * (cos_t / cos_i);
            let t_intensity = t * intensity_ratio;
            let total = r + t_intensity;
            (total - 1.0).abs()
        } else {
            // Fallback to simple energy conservation for optical waves or missing data
            (r + t - 1.0).abs()
        }
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
            impedance1: None,
            impedance2: None,
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
            impedance1: None,
            impedance2: None,
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
            impedance1: None,
            impedance2: None,
        };

        assert_relative_eq!(coeffs.reflectance(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs.transmittance(), 0.0, epsilon = 1e-10);
        assert!(coeffs.total_internal_reflection);
        assert!(coeffs.transmitted_angle.is_none());
    }
}
