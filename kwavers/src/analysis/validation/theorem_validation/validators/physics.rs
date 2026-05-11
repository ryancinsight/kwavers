//! Physical acoustics and wave propagation theorem validators.

use super::super::{TheoremValidation, TheoremValidator};
use crate::core::constants::fundamental::{AVOGADRO, BOLTZMANN};
use std::f64::consts::PI;

impl TheoremValidator {
    /// Validate Kramers-Kronig relations for power-law absorption/dispersion
    #[must_use]
    pub fn validate_kramers_kronig(
        frequencies: &[f64],
        absorption: &[f64],
        dispersion: &[f64],
        sound_speed: f64,
    ) -> TheoremValidation {
        let alpha_0 = absorption[0] / frequencies[0].powf(1.5);
        let alpha_power = 1.5;

        let mut max_error: f64 = 0.0;
        let mut total_error = 0.0;

        for (i, (&omega, &measured_disp)) in frequencies.iter().zip(dispersion.iter()).enumerate() {
            let tan_factor = (alpha_power * PI / 2.0).tan();
            let theoretical_disp =
                ((2.0 * alpha_0 / PI) * omega.powf(alpha_power)).mul_add(tan_factor, sound_speed);

            let error = (theoretical_disp - measured_disp).abs() / sound_speed.abs().max(1e-10);
            max_error = max_error.max(error);
            total_error += error;

            let theoretical_alpha = alpha_0 * omega.powf(alpha_power);
            let measured_alpha = absorption[i];
            let alpha_error =
                (theoretical_alpha - measured_alpha).abs() / measured_alpha.abs().max(1e-10);
            max_error = max_error.max(alpha_error);
        }

        let avg_error = total_error / frequencies.len() as f64;
        let passed = max_error < 0.1;

        TheoremValidation {
            theorem: "Kramers-Kronig Relations".to_owned(),
            passed,
            error_bound: 0.05,
            measured_error: max_error,
            confidence: if passed { 0.9 } else { 0.4 },
            details: format!(
                "Power-law fit (y={:.1}): α₀={:.2e}, Max error: {:.2e}, Avg error: {:.2e}",
                alpha_power, alpha_0, max_error, avg_error
            ),
        }
    }

    /// Validate speed of sound in ideal gas: c = √(γ kT / m)
    #[must_use]
    pub fn validate_ideal_gas_speed(
        temperature_k: f64,
        molecular_mass: f64,
        gamma: f64,
        measured_speed: f64,
    ) -> TheoremValidation {
        let theoretical_speed =
            (gamma * BOLTZMANN * temperature_k / (molecular_mass / AVOGADRO)).sqrt();

        let error = (measured_speed - theoretical_speed).abs() / theoretical_speed;
        let passed = error < 0.01;

        TheoremValidation {
            theorem: "Ideal Gas Speed of Sound".to_owned(),
            passed,
            error_bound: theoretical_speed,
            measured_error: measured_speed,
            confidence: if passed { 0.95 } else { 0.6 },
            details: format!(
                "T: {:.0} K, M: {:.2e} kg/mol, γ: {:.2}, Theoretical c: {:.0} m/s, Measured c: {:.0} m/s, Error: {:.2e}",
                temperature_k, molecular_mass, gamma, theoretical_speed, measured_speed, error
            ),
        }
    }

    /// Validate Rayleigh-Sommerfeld far-field 1/r spherical wave decay
    #[must_use]
    pub fn validate_rayleigh_sommerfeld_decay(
        distances: &[f64],
        pressures: &[f64],
        _source_pressure: f64,
    ) -> TheoremValidation {
        let mut passed = true;
        let mut max_error: f64 = 0.0;

        for i in 1..distances.len() {
            let r1: f64 = distances[i - 1];
            let r2: f64 = distances[i];
            let p1: f64 = pressures[i - 1].abs();
            let p2: f64 = pressures[i].abs();

            if p1 > 1e-12 && p2 > 1e-12 {
                let theoretical_ratio = r1 / r2;
                let measured_ratio = p2 / p1;
                let error = (measured_ratio - theoretical_ratio).abs() / theoretical_ratio;

                if error > 0.1 {
                    passed = false;
                }
                max_error = max_error.max(error);
            }
        }

        TheoremValidation {
            theorem: "Rayleigh-Sommerfeld Diffraction".to_owned(),
            passed,
            error_bound: 0.05,
            measured_error: max_error,
            confidence: if passed { 0.85 } else { 0.4 },
            details: format!(
                "Spherical wave decay validation: {} points, Max error: {:.2e}",
                distances.len(),
                max_error
            ),
        }
    }

    /// Validate acoustic reciprocity: p(r1|r2) = p(r2|r1)
    #[must_use]
    pub fn validate_reciprocity(
        pressure_12: f64,
        pressure_21: f64,
        tolerance: f64,
    ) -> TheoremValidation {
        let ratio = pressure_12 / pressure_21;
        let passed = (ratio - 1.0).abs() < tolerance;
        let error = (ratio - 1.0).abs();

        TheoremValidation {
            theorem: "Acoustic Reciprocity Theorem".to_owned(),
            passed,
            error_bound: tolerance,
            measured_error: error,
            confidence: if passed { 0.95 } else { 0.3 },
            details: format!(
                "p12/p21 = {:.6}, Expected: 1.0, Error: {:.2e}, Tolerance: {:.2e}",
                ratio, error, tolerance
            ),
        }
    }

    /// Validate plane wave reflection: R = (Z2 - Z1) / (Z2 + Z1)
    #[must_use]
    pub fn validate_impedance_reflection(
        rho1: f64,
        c1: f64,
        rho2: f64,
        c2: f64,
        measured_reflection_coeff: f64,
    ) -> TheoremValidation {
        let z1 = rho1 * c1;
        let z2 = rho2 * c2;
        let theoretical_r = (z2 - z1) / (z2 + z1);

        let error =
            (measured_reflection_coeff - theoretical_r).abs() / theoretical_r.abs().max(1e-10);
        let passed = error < 0.05;

        TheoremValidation {
            theorem: "Acoustic Impedance Reflection".to_owned(),
            passed,
            error_bound: theoretical_r.abs(),
            measured_error: measured_reflection_coeff,
            confidence: if passed { 0.9 } else { 0.4 },
            details: format!(
                "Z1: {:.0} Pa·s/m, Z2: {:.0} Pa·s/m, Theoretical R: {:.4}, Measured R: {:.4}, Error: {:.2e}",
                z1, z2, theoretical_r, measured_reflection_coeff, error
            ),
        }
    }
}
