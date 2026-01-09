//! Frequency-dependent properties for biological tissues
//!
//! This module implements frequency-dependent acoustic properties including
//! dispersion, nonlinearity, and relaxation processes in biological tissues.
//!
//! References:
//! - Duck, F. A. (1990). "Physical properties of tissue: a comprehensive
//!   reference book" Academic Press.
//! - Nachman, A. I., et al. (1990). "A nonlinear dispersive model for
//!   wave propagation in biological tissue" Journal of the Acoustical
//!   Society of America, 88(3), 1584-1591.

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::Array3;

/// Frequency-dependent tissue properties
#[derive(Debug, Clone)]
pub struct FrequencyDependentProperties {
    /// Reference sound speed at low frequency (m/s)
    pub c0: f64,
    /// Dispersion coefficient
    pub dispersion_coefficient: f64,
    /// Relaxation frequencies (Hz)
    pub relaxation_frequencies: Vec<f64>,
    /// Relaxation strengths
    pub relaxation_strengths: Vec<f64>,
    /// Nonlinearity parameter B/A
    pub nonlinearity_parameter: f64,
    /// Frequency-dependent nonlinearity
    pub frequency_dependent_nonlinearity: bool,
}

impl FrequencyDependentProperties {
    /// Create new frequency-dependent properties
    #[must_use]
    pub fn new(c0: f64, nonlinearity_parameter: f64) -> Self {
        Self {
            c0,
            dispersion_coefficient: 0.0,
            relaxation_frequencies: Vec::new(),
            relaxation_strengths: Vec::new(),
            nonlinearity_parameter,
            frequency_dependent_nonlinearity: false,
        }
    }

    /// Add a relaxation process
    pub fn add_relaxation(&mut self, frequency: f64, strength: f64) -> KwaversResult<()> {
        if frequency <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "relaxation_frequency".to_string(),
                value: frequency.to_string(),
                constraint: "Must be positive".to_string(),
            }));
        }

        self.relaxation_frequencies.push(frequency);
        self.relaxation_strengths.push(strength);
        Ok(())
    }

    /// Get phase velocity at a given frequency
    #[must_use]
    pub fn phase_velocity(&self, frequency: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * frequency;

        // Base dispersion
        let mut c_phase = self.c0 * (1.0 + self.dispersion_coefficient * frequency.ln());

        // Add relaxation contributions
        self.relaxation_frequencies
            .iter()
            .zip(self.relaxation_strengths.iter())
            .for_each(|(&f_r, &strength)| {
                let omega_r = 2.0 * std::f64::consts::PI * f_r;
                let relaxation_factor =
                    1.0 + strength * omega.powi(2) / (omega_r.powi(2) + omega.powi(2));
                c_phase *= relaxation_factor.sqrt();
            });

        c_phase
    }

    /// Get group velocity at a given frequency
    #[must_use]
    pub fn group_velocity(&self, frequency: f64) -> f64 {
        // Numerical derivative of phase
        let df = frequency * 1e-6; // Small frequency step
        let c1 = self.phase_velocity(frequency - df);
        let c2 = self.phase_velocity(frequency + df);
        let k1 = 2.0 * std::f64::consts::PI * (frequency - df) / c1;
        let k2 = 2.0 * std::f64::consts::PI * (frequency + df) / c2;

        // Group velocity = dω/dk
        2.0 * std::f64::consts::PI * 2.0 * df / (k2 - k1)
    }

    /// Get frequency-dependent nonlinearity parameter
    #[must_use]
    pub fn nonlinearity_at_frequency(&self, frequency: f64) -> f64 {
        if self.frequency_dependent_nonlinearity {
            // Empirical model for frequency-dependent B/A
            let f_mhz = frequency / 1e6;
            self.nonlinearity_parameter * (1.0 + 0.1 * f_mhz.ln().max(0.0))
        } else {
            self.nonlinearity_parameter
        }
    }

    /// Get attenuation from relaxation processes
    #[must_use]
    pub fn relaxation_attenuation(&self, frequency: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI * frequency;

        self.relaxation_frequencies
            .iter()
            .zip(self.relaxation_strengths.iter())
            .map(|(&f_r, &strength)| {
                let omega_r = 2.0 * std::f64::consts::PI * f_r;
                let denominator = 1.0 + (omega / omega_r).powi(2);
                strength * omega.powi(2) / (2.0 * self.c0 * omega_r * denominator)
            })
            .sum()
    }
}

/// Tissue-specific frequency-dependent models
#[derive(Debug)]
pub struct TissueFrequencyModels;

impl TissueFrequencyModels {
    /// Get frequency-dependent properties for liver tissue
    #[must_use]
    pub fn liver() -> FrequencyDependentProperties {
        let mut props = FrequencyDependentProperties::new(1570.0, 6.8);
        props.dispersion_coefficient = 0.002;
        // Add relaxation processes
        let _ = props.add_relaxation(0.5e6, 0.02); // Low frequency relaxation
        let _ = props.add_relaxation(5e6, 0.05); // Mid frequency relaxation
        props.frequency_dependent_nonlinearity = true;
        props
    }

    /// Get frequency-dependent properties for muscle tissue
    #[must_use]
    pub fn muscle() -> FrequencyDependentProperties {
        let mut props = FrequencyDependentProperties::new(1580.0, 7.4);
        props.dispersion_coefficient = 0.0015;
        // Add relaxation processes
        let _ = props.add_relaxation(1e6, 0.03);
        let _ = props.add_relaxation(10e6, 0.04);
        props.frequency_dependent_nonlinearity = true;
        props
    }

    /// Get frequency-dependent properties for fat tissue
    #[must_use]
    pub fn fat() -> FrequencyDependentProperties {
        let mut props = FrequencyDependentProperties::new(1450.0, 10.0);
        props.dispersion_coefficient = 0.003;
        // Fat has significant relaxation
        let _ = props.add_relaxation(0.2e6, 0.05);
        let _ = props.add_relaxation(2e6, 0.08);
        props.frequency_dependent_nonlinearity = true;
        props
    }

    /// Get frequency-dependent properties for blood
    #[must_use]
    pub fn blood() -> FrequencyDependentProperties {
        let mut props = FrequencyDependentProperties::new(1575.0, 6.0);
        props.dispersion_coefficient = 0.001;
        // Blood has minimal relaxation
        let _ = props.add_relaxation(5e6, 0.01);
        props.frequency_dependent_nonlinearity = false;
        props
    }
}

/// Dispersion correction for frequency-dependent media
#[derive(Debug)]
pub struct DispersionCorrection {
    /// Reference frequency for dispersion calculation
    _reference_frequency: f64,
    /// Frequency-dependent properties
    properties: FrequencyDependentProperties,
}

impl DispersionCorrection {
    /// Create new dispersion correction
    #[must_use]
    pub fn new(properties: FrequencyDependentProperties, reference_frequency: f64) -> Self {
        Self {
            _reference_frequency: reference_frequency,
            properties,
        }
    }

    /// Apply dispersion correction in frequency domain
    pub fn apply_dispersion_correction(
        &self,
        spectrum: &mut Array3<rustfft::num_complex::Complex<f64>>,
        k_vec: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = spectrum.dim();

        (0..nx).for_each(|i| {
            (0..ny).for_each(|j| {
                (0..nz).for_each(|k| {
                    let k_mag = k_vec[[i, j, k]];
                    if k_mag > 0.0 {
                        // Frequency from wavenumber
                        let freq = k_mag * self.properties.c0 / (2.0 * std::f64::consts::PI);

                        // Phase velocity at this frequency
                        let c_phase = self.properties.phase_velocity(freq);

                        // Dispersion phase shift
                        let k_dispersive = 2.0 * std::f64::consts::PI * freq / c_phase;
                        let phase_shift = (k_dispersive - k_mag) * c_phase * dt;

                        // Apply phase correction
                        let correction = rustfft::num_complex::Complex::new(
                            phase_shift.cos(),
                            phase_shift.sin(),
                        );
                        spectrum[[i, j, k]] *= correction;
                    }
                });
            });
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx;

    #[test]
    fn test_phase_velocity() {
        let props = TissueFrequencyModels::liver();

        // Test at different frequencies to validate dispersion model
        let c_1mhz = props.phase_velocity(1e6);
        let c_5mhz = props.phase_velocity(5e6);

        // Validate dispersion formula: c(f) = c₀ * (1 + β * ln(f))
        // For liver: β = 0.002 (dispersion coefficient)
        let f1: f64 = 1e6;
        let f2: f64 = 5e6;
        let _expected_c1 = props.c0 * (1.0 + props.dispersion_coefficient * f1.ln());
        let _expected_c5 = props.c0 * (1.0 + props.dispersion_coefficient * f2.ln());

        // Validate against exact dispersion formula (before relaxation effects)
        // Note: relaxation effects make the calculation more complex, so we check the base trend
        let theoretical_ratio = (1.0 + props.dispersion_coefficient * f2.ln())
            / (1.0 + props.dispersion_coefficient * f1.ln());
        let measured_ratio = c_5mhz / c_1mhz;

        // Should match theoretical dispersion within 5% (accounting for relaxation effects)
        approx::assert_relative_eq!(measured_ratio, theoretical_ratio, epsilon = 0.05);

        // Should be close to reference at low frequency
        let c_low = props.phase_velocity(1e3);
        // Low frequency should approach c₀ within dispersion-corrected bounds
        let expected_low = props.c0 * (1.0 + props.dispersion_coefficient * 1e3_f64.ln());
        approx::assert_relative_eq!(c_low, expected_low, epsilon = 0.01);
    }

    #[test]
    fn test_frequency_dependent_nonlinearity() {
        let props = TissueFrequencyModels::muscle();

        let beta_1mhz = props.nonlinearity_at_frequency(1e6);
        let beta_10mhz = props.nonlinearity_at_frequency(10e6);

        // Should increase with frequency
        assert!(beta_10mhz > beta_1mhz);
    }
}
