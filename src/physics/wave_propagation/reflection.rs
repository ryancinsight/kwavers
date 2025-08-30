//! Reflection calculations for wave propagation
//!
//! Implements reflection coefficients for acoustic and optical waves.

use std::f64::consts::PI;

/// Reflection calculator
#[derive(Debug))]
pub struct ReflectionCalculator {
    /// Impedance of medium 1
    z1: f64,
    /// Impedance of medium 2
    z2: f64,
}

impl ReflectionCalculator {
    /// Create new reflection calculator
    pub fn new(impedance1: f64, impedance2: f64) -> Self {
        Self {
            z1: impedance1,
            z2: impedance2,
        }
    }

    /// Calculate reflection coefficient at normal incidence
    pub fn normal_incidence(&self) -> f64 {
        (self.z2 - self.z1) / (self.z2 + self.z1)
    }

    /// Calculate transmission coefficient at normal incidence
    pub fn transmission_coefficient(&self) -> f64 {
        2.0 * self.z2 / (self.z2 + self.z1)
    }
}

/// Reflection coefficients
#[derive(Debug, Clone))]
pub struct ReflectionCoefficients {
    /// Amplitude reflection coefficient
    pub amplitude: f64,
    /// Phase shift upon reflection [radians]
    pub phase: f64,
    /// Energy reflection coefficient (R = |r|²)
    pub energy: f64,
    /// Transmission coefficient
    pub transmission: f64,
}

impl ReflectionCoefficients {
    /// Calculate acoustic reflection coefficients
    pub fn acoustic(z1: f64, z2: f64, incident_angle: f64, transmitted_angle: f64) -> Self {
        // Acoustic reflection coefficient (pressure)
        let cos_i = incident_angle.cos();
        let cos_t = transmitted_angle.cos();

        let r = (z2 * cos_i - z1 * cos_t) / (z2 * cos_i + z1 * cos_t);
        let t = 2.0 * z2 * cos_i / (z2 * cos_i + z1 * cos_t);

        // Phase shift: 0 if r > 0, π if r < 0
        let phase = if r < 0.0 { PI } else { 0.0 };

        Self {
            amplitude: r.abs(),
            phase,
            energy: r * r,
            transmission: t,
        }
    }

    /// Calculate optical reflection coefficients (Fresnel)
    pub fn optical_s_polarized(
        n1: f64,
        n2: f64,
        incident_angle: f64,
        transmitted_angle: f64,
    ) -> Self {
        let cos_i = incident_angle.cos();
        let cos_t = transmitted_angle.cos();

        let r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
        let t_s = 2.0 * n1 * cos_i / (n1 * cos_i + n2 * cos_t);

        let phase = if r_s < 0.0 { PI } else { 0.0 };

        Self {
            amplitude: r_s.abs(),
            phase,
            energy: r_s * r_s,
            transmission: t_s,
        }
    }

    /// Calculate optical reflection coefficients for p-polarized light
    pub fn optical_p_polarized(
        n1: f64,
        n2: f64,
        incident_angle: f64,
        transmitted_angle: f64,
    ) -> Self {
        let cos_i = incident_angle.cos();
        let cos_t = transmitted_angle.cos();

        let r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);
        let t_p = 2.0 * n1 * cos_i / (n2 * cos_i + n1 * cos_t);

        let phase = if r_p < 0.0 { PI } else { 0.0 };

        Self {
            amplitude: r_p.abs(),
            phase,
            energy: r_p * r_p,
            transmission: t_p,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normal_incidence() {
        let calc = ReflectionCalculator::new(1000.0, 2000.0);
        let r = calc.normal_incidence();
        assert_relative_eq!(r, 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_conservation() {
        let coeff = ReflectionCoefficients::acoustic(1000.0, 2000.0, 0.0, 0.0);
        // R + T should equal 1 for energy conservation (with impedance correction)
        let z_ratio = 1000.0 / 2000.0;
        assert_relative_eq!(
            coeff.energy + coeff.transmission * coeff.transmission * z_ratio,
            1.0,
            epsilon = 1e-10
        );
    }
}
