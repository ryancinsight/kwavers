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

use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::constants::tissue_acoustics::{
    SOUND_SPEED_BLOOD, SOUND_SPEED_FAT, SOUND_SPEED_LIVER, SOUND_SPEED_MUSCLE,
};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_math::fft::Complex64;
use leto::Array3;

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn add_relaxation(&mut self, frequency: f64, strength: f64) -> KwaversResult<()> {
        if frequency <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "relaxation_frequency".to_owned(),
                value: frequency.to_string(),
                constraint: "Must be positive".to_owned(),
            }));
        }

        self.relaxation_frequencies.push(frequency);
        self.relaxation_strengths.push(strength);
        Ok(())
    }

    /// Get phase velocity at a given frequency
    #[must_use]
    pub fn phase_velocity(&self, frequency: f64) -> f64 {
        let omega = TWO_PI * frequency;

        // Base dispersion
        let mut c_phase = self.c0 * self.dispersion_coefficient.mul_add(frequency.ln(), 1.0);

        // Add relaxation contributions
        self.relaxation_frequencies
            .iter()
            .zip(self.relaxation_strengths.iter())
            .for_each(|(&f_r, &strength)| {
                let omega_r = TWO_PI * f_r;
                let relaxation_factor =
                    1.0 + strength * omega.powi(2) / omega.mul_add(omega, omega_r.powi(2));
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
        let k1 = TWO_PI * (frequency - df) / c1;
        let k2 = TWO_PI * (frequency + df) / c2;

        // Group velocity = dω/dk
        TWO_PI * 2.0 * df / (k2 - k1)
    }

    /// Get frequency-dependent nonlinearity parameter
    #[must_use]
    pub fn nonlinearity_at_frequency(&self, frequency: f64) -> f64 {
        if self.frequency_dependent_nonlinearity {
            // Empirical model for frequency-dependent B/A
            let f_mhz = frequency / MHZ_TO_HZ;
            self.nonlinearity_parameter * 0.1f64.mul_add(f_mhz.ln().max(0.0), 1.0)
        } else {
            self.nonlinearity_parameter
        }
    }

    /// Get attenuation from relaxation processes
    #[must_use]
    pub fn relaxation_attenuation(&self, frequency: f64) -> f64 {
        let omega = TWO_PI * frequency;

        self.relaxation_frequencies
            .iter()
            .zip(self.relaxation_strengths.iter())
            .map(|(&f_r, &strength)| {
                let omega_r = TWO_PI * f_r;
                let denominator = (omega / omega_r).mul_add(omega / omega_r, 1.0);
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
        let mut props = FrequencyDependentProperties::new(SOUND_SPEED_LIVER, 6.8);
        props.dispersion_coefficient = 0.002;
        // Add relaxation processes
        let _ = props.add_relaxation(0.5 * MHZ_TO_HZ, 0.02); // Low frequency relaxation
        let _ = props.add_relaxation(5.0 * MHZ_TO_HZ, 0.05); // Mid frequency relaxation
        props.frequency_dependent_nonlinearity = true;
        props
    }

    /// Get frequency-dependent properties for muscle tissue
    #[must_use]
    pub fn muscle() -> FrequencyDependentProperties {
        let mut props = FrequencyDependentProperties::new(SOUND_SPEED_MUSCLE, 7.4);
        props.dispersion_coefficient = 0.0015;
        // Add relaxation processes
        let _ = props.add_relaxation(MHZ_TO_HZ, 0.03); // 1 MHz relaxation frequency
        let _ = props.add_relaxation(10.0 * MHZ_TO_HZ, 0.04);
        props.frequency_dependent_nonlinearity = true;
        props
    }

    /// Get frequency-dependent properties for fat tissue
    #[must_use]
    pub fn fat() -> FrequencyDependentProperties {
        let mut props = FrequencyDependentProperties::new(SOUND_SPEED_FAT, 10.0);
        props.dispersion_coefficient = 0.003;
        // Fat has significant relaxation
        let _ = props.add_relaxation(0.2 * MHZ_TO_HZ, 0.05);
        let _ = props.add_relaxation(2.0 * MHZ_TO_HZ, 0.08);
        props.frequency_dependent_nonlinearity = true;
        props
    }

    /// Get frequency-dependent properties for blood
    #[must_use]
    pub fn blood() -> FrequencyDependentProperties {
        let mut props = FrequencyDependentProperties::new(SOUND_SPEED_BLOOD, 6.0);
        props.dispersion_coefficient = 0.001;
        // Blood has minimal relaxation
        let _ = props.add_relaxation(5.0 * MHZ_TO_HZ, 0.01);
        props.frequency_dependent_nonlinearity = false;
        props
    }
}

/// Dispersion correction for frequency-dependent media
#[derive(Debug)]
pub struct FreqDispersionCorrection {
    /// Reference frequency for dispersion calculation
    _reference_frequency: f64,
    /// Frequency-dependent properties
    properties: FrequencyDependentProperties,
}

impl FreqDispersionCorrection {
    /// Create new dispersion correction
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(properties: FrequencyDependentProperties, reference_frequency: f64) -> Self {
        Self {
            _reference_frequency: reference_frequency,
            properties,
        }
    }

    /// Apply dispersion correction in frequency domain
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_dispersion_correction(
        &self,
        spectrum: &mut Array3<Complex64>,
        k_vec: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        crate::parallel::zip_mut_ref(spectrum, k_vec, |s, &k_mag| {
            if k_mag > 0.0 {
                // Frequency from wavenumber
                let freq = k_mag * self.properties.c0 / (TWO_PI);
                // Phase velocity at this frequency
                let c_phase = self.properties.phase_velocity(freq);
                // Dispersion phase shift
                let k_dispersive = TWO_PI * freq / c_phase;
                let phase_shift = (k_dispersive - k_mag) * c_phase * dt;
                // Apply phase correction
                let correction = Complex64::new(phase_shift.cos(), phase_shift.sin());
                *s *= correction;
            }
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia;

    #[test]
    fn test_phase_velocity() {
        let props = TissueFrequencyModels::liver();

        // Test at different frequencies to validate dispersion model
        let c_1mhz = props.phase_velocity(MHZ_TO_HZ); // 1 MHz
        let c_5mhz = props.phase_velocity(5.0 * MHZ_TO_HZ); // 5 MHz

        // Validate dispersion formula: c(f) = c₀ * (1 + β * ln(f))
        // For liver: β = 0.002 (dispersion coefficient)
        let f1: f64 = MHZ_TO_HZ; // 1 MHz
        let f2: f64 = 5.0 * MHZ_TO_HZ; // 5 MHz
        let _expected_c1 = props.c0 * (1.0 + props.dispersion_coefficient * f1.ln());
        let _expected_c5 = props.c0 * (1.0 + props.dispersion_coefficient * f2.ln());

        // Validate against exact dispersion formula (before relaxation effects)
        // Note: relaxation effects make the calculation more complex, so we check the base trend
        let theoretical_ratio = (1.0 + props.dispersion_coefficient * f2.ln())
            / (1.0 + props.dispersion_coefficient * f1.ln());
        let measured_ratio = c_5mhz / c_1mhz;

        // Should match theoretical dispersion within 5% (accounting for relaxation effects)
        eunomia::assert_relative_eq!(measured_ratio, theoretical_ratio, epsilon = 0.05);

        // Should be close to reference at low frequency
        let c_low = props.phase_velocity(1e3);
        // Low frequency should approach c₀ within dispersion-corrected bounds
        let expected_low = props.c0 * (1.0 + props.dispersion_coefficient * 1e3_f64.ln());
        eunomia::assert_relative_eq!(c_low, expected_low, epsilon = 0.01);
    }

    #[test]
    fn test_frequency_dependent_nonlinearity() {
        let props = TissueFrequencyModels::muscle();

        let beta_1mhz = props.nonlinearity_at_frequency(MHZ_TO_HZ); // 1 MHz
        let beta_10mhz = props.nonlinearity_at_frequency(10.0 * MHZ_TO_HZ); // 10 MHz

        // Should increase with frequency
        assert!(beta_10mhz > beta_1mhz);
    }
}
