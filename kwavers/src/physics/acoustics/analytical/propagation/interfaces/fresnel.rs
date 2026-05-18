//! Fresnel equations for electromagnetic wave reflection and transmission
//!
//! Implements the Fresnel equations for calculating reflection and transmission
//! coefficients at dielectric interfaces.

use super::AnalyticalPolarization;
use crate::core::error::KwaversResult;
use std::f64::consts::PI;

/// Fresnel coefficients for reflection and transmission
#[derive(Debug, Clone)]
pub struct FresnelCoefficients {
    /// Reflection amplitude coefficient
    pub reflection_amplitude: f64,
    /// Transmission amplitude coefficient  
    pub transmission_amplitude: f64,
    /// Reflection phase shift \[radians\]
    pub reflection_phase: f64,
    /// Transmission phase shift \[radians\]
    pub transmission_phase: f64,
    /// Reflection coefficient for S-polarization
    pub rs: f64,
    /// Transmission coefficient for S-polarization
    pub ts: f64,
    /// Reflection coefficient for P-polarization
    pub rp: f64,
    /// Transmission coefficient for P-polarization
    pub tp: f64,
}

/// Fresnel equation calculator
#[derive(Debug)]
pub struct FresnelCalculator {
    /// Refractive index of medium 1
    n1: f64,
    /// Refractive index of medium 2
    n2: f64,
}

impl FresnelCalculator {
    /// Create a new Fresnel calculator
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(n1: f64, n2: f64) -> Self {
        Self { n1, n2 }
    }

    /// Calculate Fresnel coefficients for given angles and polarization
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn calculate(
        &self,
        incident_angle: f64,
        transmitted_angle: f64,
        polarization: AnalyticalPolarization,
    ) -> KwaversResult<FresnelCoefficients> {
        // Calculate S-polarization (TE) coefficients
        let (rs, ts) = self.calculate_s_polarization(incident_angle, transmitted_angle)?;

        // Calculate P-polarization (TM) coefficients
        let (rp, tp) = self.calculate_p_polarization(incident_angle, transmitted_angle)?;

        // Combine based on polarization state
        let (r, t, r_phase, t_phase) = match polarization {
            AnalyticalPolarization::TransverseElectric => (rs, ts, 0.0, 0.0),
            AnalyticalPolarization::TransverseMagnetic => (rp, tp, 0.0, 0.0),
            AnalyticalPolarization::Unpolarized => {
                // Average of S and P polarizations for unpolarized light
                let r_avg = (rs.mul_add(rs, rp * rp) / 2.0).sqrt();
                let t_avg = (ts.mul_add(ts, tp * tp) / 2.0).sqrt();
                (r_avg, t_avg, 0.0, 0.0)
            }
            AnalyticalPolarization::Circular | AnalyticalPolarization::Elliptical => {
                // For circular/elliptical, use equal weighting
                let r_avg = (rs.mul_add(rs, rp * rp) / 2.0).sqrt();
                let t_avg = (ts.mul_add(ts, tp * tp) / 2.0).sqrt();
                (r_avg, t_avg, 0.0, 0.0)
            }
        };

        // Calculate phase shifts
        let r_phase = if r < 0.0 { PI } else { r_phase };
        // t_phase already calculated, transmission typically has no phase shift

        Ok(FresnelCoefficients {
            reflection_amplitude: r.abs(),
            transmission_amplitude: t.abs(),
            reflection_phase: r_phase,
            transmission_phase: t_phase,
            rs,
            ts,
            rp,
            tp,
        })
    }

    /// Calculate S-polarization (TE) Fresnel coefficients
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn calculate_s_polarization(
        &self,
        incident_angle: f64,
        transmitted_angle: f64,
    ) -> KwaversResult<(f64, f64)> {
        let cos_i = incident_angle.cos();
        let cos_t = transmitted_angle.cos();

        // Fresnel equations for S-polarization
        // rs = (n1*cos(θi) - n2*cos(θt)) / (n1*cos(θi) + n2*cos(θt))
        let rs =
            self.n1.mul_add(cos_i, -(self.n2 * cos_t)) / self.n1.mul_add(cos_i, self.n2 * cos_t);

        // ts = 2*n1*cos(θi) / (n1*cos(θi) + n2*cos(θt))
        let ts = (2.0 * self.n1 * cos_i) / self.n1.mul_add(cos_i, self.n2 * cos_t);

        Ok((rs, ts))
    }

    /// Calculate P-polarization (TM) Fresnel coefficients
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn calculate_p_polarization(
        &self,
        incident_angle: f64,
        transmitted_angle: f64,
    ) -> KwaversResult<(f64, f64)> {
        let cos_i = incident_angle.cos();
        let cos_t = transmitted_angle.cos();

        // Fresnel equations for P-polarization
        // rp = (n2*cos(θi) - n1*cos(θt)) / (n2*cos(θi) + n1*cos(θt))
        let rp =
            self.n2.mul_add(cos_i, -(self.n1 * cos_t)) / self.n2.mul_add(cos_i, self.n1 * cos_t);

        // tp = 2*n1*cos(θi) / (n2*cos(θi) + n1*cos(θt))
        let tp = (2.0 * self.n1 * cos_i) / self.n2.mul_add(cos_i, self.n1 * cos_t);

        Ok((rp, tp))
    }

    /// Calculate reflectance (power reflection coefficient)
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn reflectance(
        &self,
        incident_angle: f64,
        polarization: AnalyticalPolarization,
    ) -> KwaversResult<f64> {
        // Calculate transmitted angle
        let sin_t = (self.n1 / self.n2) * incident_angle.sin();

        // Check for total internal reflection
        if sin_t > 1.0 {
            return Ok(1.0); // Total reflection
        }

        let transmitted_angle = sin_t.asin();
        let coeffs = self.calculate(incident_angle, transmitted_angle, polarization)?;

        match polarization {
            AnalyticalPolarization::TransverseElectric => Ok(coeffs.rs * coeffs.rs),
            AnalyticalPolarization::TransverseMagnetic => Ok(coeffs.rp * coeffs.rp),
            AnalyticalPolarization::Unpolarized => {
                Ok(coeffs.rs.mul_add(coeffs.rs, coeffs.rp * coeffs.rp) / 2.0)
            }
            _ => Ok(coeffs.reflection_amplitude * coeffs.reflection_amplitude),
        }
    }

    /// Calculate transmittance (power transmission coefficient)
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn transmittance(
        &self,
        incident_angle: f64,
        polarization: AnalyticalPolarization,
    ) -> KwaversResult<f64> {
        let reflectance = self.reflectance(incident_angle, polarization)?;
        Ok(1.0 - reflectance) // Energy conservation
    }

    /// Calculate Brewster's angle (no reflection for P-polarization)
    #[must_use]
    pub fn brewster_angle(&self) -> f64 {
        (self.n2 / self.n1).atan()
    }

    /// Calculate critical angle for total internal reflection
    #[must_use]
    pub fn critical_angle(&self) -> Option<f64> {
        if self.n1 > self.n2 {
            Some((self.n2 / self.n1).asin())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_incidence() {
        // Air to glass interface
        let calc = FresnelCalculator::new(1.0, 1.5);

        // At normal incidence
        let coeffs = calc
            .calculate(0.0, 0.0, AnalyticalPolarization::Unpolarized)
            .unwrap();

        // Expected reflection coefficient: (n2-n1)/(n2+n1) = 0.5/2.5 = 0.2
        let expected_r = ((1.5_f64 - 1.0) / (1.5 + 1.0)).abs();
        assert!((coeffs.reflection_amplitude - expected_r).abs() < 1e-10);

        // Reflectance should be R = r² = 0.04
        let reflectance = calc
            .reflectance(0.0, AnalyticalPolarization::Unpolarized)
            .unwrap();
        assert!((reflectance - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_brewster_angle() {
        // Air to glass
        let calc = FresnelCalculator::new(1.0, 1.5);

        let brewster = calc.brewster_angle();
        let expected = (1.5_f64).atan(); // ~56.3°
        assert!((brewster - expected).abs() < 1e-10);

        // At Brewster's angle, P-polarization should have zero reflection
        let sin_t = (1.0 / 1.5) * brewster.sin();
        let transmitted = sin_t.asin();
        let coeffs = calc
            .calculate(
                brewster,
                transmitted,
                AnalyticalPolarization::TransverseMagnetic,
            )
            .unwrap();
        assert!(coeffs.rp.abs() < 1e-10);
    }

    #[test]
    fn test_total_internal_reflection() {
        // Glass to air (n1 > n2)
        let calc = FresnelCalculator::new(1.5, 1.0);

        // Critical angle should be arcsin(1/1.5) ≈ 41.8°
        let critical = calc.critical_angle().unwrap();
        let expected = (1.0_f64 / 1.5).asin();
        assert!((critical - expected).abs() < 1e-10);

        // Above critical angle, reflectance should be 1
        let reflectance = calc
            .reflectance(critical + 0.1, AnalyticalPolarization::Unpolarized)
            .unwrap();
        assert!((reflectance - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_energy_conservation() {
        let calc = FresnelCalculator::new(1.0, 1.5);

        for angle in [0.0, PI / 6.0, PI / 4.0, PI / 3.0] {
            for pol in [
                AnalyticalPolarization::TransverseElectric,
                AnalyticalPolarization::TransverseMagnetic,
                AnalyticalPolarization::Unpolarized,
            ] {
                let r = calc.reflectance(angle, pol).unwrap();
                let t = calc.transmittance(angle, pol).unwrap();

                // R + T should equal 1 (energy conservation)
                assert!(
                    (r + t - 1.0).abs() < 1e-10,
                    "Energy not conserved at angle {} for {:?}: R={}, T={}, sum={}",
                    angle,
                    pol,
                    r,
                    t,
                    r + t
                );
            }
        }
    }
}
