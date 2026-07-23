//! Frequency Response Module
//!
//! Models the frequency-dependent behavior of transducers including
//! bandwidth, sensitivity, and impedance characteristics.

use aequitas::systems::si::quantities::{Frequency, Length, Velocity};
use eunomia::Complex64;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};
use leto::Array1;

fn linspace(start: Frequency, end: Frequency, num_points: usize) -> Array1<Frequency> {
    if num_points == 0 {
        return Array1::zeros([0]);
    }
    if num_points == 1 {
        return Array1::from_vec([1], vec![start]).expect("linspace shape must match");
    }
    let start_base = *start.as_base();
    let step = (*end.as_base() - start_base) / (num_points - 1) as f64;
    Array1::from_vec(
        [num_points],
        (0..num_points)
            .map(|i| Frequency::from_base(start_base + step * i as f64))
            .collect(),
    )
    .expect("linspace shape must match")
}

/// Frequency response characteristics of a transducer
///
/// Based on KLM model and Mason equivalent circuit
/// References:
/// - Krimholtz et al. (1970): "New equivalent circuits for elementary piezoelectric transducers"
/// - Mason (1948): "Electromechanical Transducers and Wave Filters"
#[derive(Debug, Clone)]
pub struct FrequencyResponse {
    /// Center frequency (Hz)
    pub center_frequency: Frequency,
    /// -3 dB bandwidth (Hz)
    pub bandwidth_3db: Frequency,
    /// -6 dB bandwidth (Hz)
    pub bandwidth_6db: Frequency,
    /// Fractional bandwidth (%)
    pub fractional_bandwidth: f64,
    /// Quality factor
    pub quality_factor: f64,
    /// Frequency vector (Hz)
    pub frequencies: Array1<Frequency>,
    /// Magnitude response (normalized)
    pub magnitude: Array1<f64>,
    /// Phase response (radians)
    pub phase: Array1<f64>,
    /// Electrical impedance (complex)
    pub impedance: Array1<Complex64>,
}

impl FrequencyResponse {
    /// Calculate frequency response using KLM model
    ///
    /// # Arguments
    /// * `center_freq` - Center frequency in Hz
    /// * `coupling` - Electromechanical coupling coefficient
    /// * `mechanical_q` - Mechanical quality factor
    /// * `electrical_q` - Electrical quality factor
    /// * `num_points` - Number of frequency points
    /// # Errors
    /// - Returns `KwaversError::Config` if the precondition for a Config-class constraint is violated.
    ///
    pub fn from_klm_model(
        center_freq: Frequency,
        coupling: f64,
        mechanical_q: f64,
        electrical_q: f64,
        num_points: usize,
    ) -> KwaversResult<Self> {
        if center_freq <= Frequency::from_base(0.0) {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "center_frequency".to_owned(),
                value: center_freq.as_base().to_string(),
                constraint: "Center frequency must be positive".to_owned(),
            }));
        }

        if num_points < 2 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "num_points".to_owned(),
                value: num_points.to_string(),
                constraint: "At least two frequency samples are required".to_owned(),
            }));
        }

        // Create frequency vector (0.5 to 1.5 times center frequency)
        let freq_min = center_freq * 0.5;
        let freq_max = center_freq * 1.5;
        let frequencies = linspace(freq_min, freq_max, num_points);

        // Calculate normalized frequency
        let mut magnitude = Array1::zeros([num_points]);
        let mut phase = Array1::zeros([num_points]);
        let mut impedance = Array1::zeros([num_points]);

        // Effective bandwidth factor
        let _bandwidth_factor = coupling.powi(2) / mechanical_q.sqrt();

        for (i, &freq) in frequencies.iter().enumerate() {
            let normalized_freq = *freq.as_base() / *center_freq.as_base();
            let delta = normalized_freq - 1.0;

            // Mason model response
            let mechanical_term = Complex64::new(1.0, 2.0 * delta * mechanical_q);
            let electrical_term = Complex64::new(1.0, 2.0 * delta * electrical_q);

            // Combined response
            let response: Complex64 =
                Complex64::new(coupling.powi(2), 0.0) / (mechanical_term * electrical_term);

            magnitude[i] = response.norm();
            phase[i] = response.arg();

            // Electrical impedance: Z = Z₀ × electrical transfer function
            // Per Kinsler et al. (2000) "Fundamentals of Acoustics" Ch. 10
            // Nominal 50Ω reference impedance standard for RF systems
            let z0 = 50.0; // Nominal impedance
            impedance[i] = z0 * electrical_term;
        }

        // Normalize magnitude
        let max_mag = magnitude.iter().fold(0.0_f64, |a: f64, &b| a.max(b));
        if max_mag > 0.0 {
            for value in magnitude.iter_mut() {
                *value /= max_mag;
            }
        }

        // Find bandwidth points
        let (bandwidth_3db, bandwidth_6db) = Self::find_bandwidth_points(&frequencies, &magnitude);
        let fractional_bandwidth = 100.0 * *bandwidth_3db.as_base() / *center_freq.as_base();
        let quality_factor = *center_freq.as_base() / *bandwidth_3db.as_base();

        Ok(Self {
            center_frequency: center_freq,
            bandwidth_3db,
            bandwidth_6db,
            fractional_bandwidth,
            quality_factor,
            frequencies,
            magnitude,
            phase,
            impedance,
        })
    }

    /// Find -3dB and -6dB bandwidth points
    fn find_bandwidth_points(
        frequencies: &Array1<Frequency>,
        magnitude: &Array1<f64>,
    ) -> (Frequency, Frequency) {
        let threshold_3db = 1.0 / 2.0_f64.sqrt(); // -3 dB
        let threshold_6db = 0.5; // -6 dB

        let mut freq_low_3db = frequencies[0];
        let mut freq_high_3db = frequencies[frequencies.len() - 1];
        let mut freq_low_6db = frequencies[0];
        let mut freq_high_6db = frequencies[frequencies.len() - 1];

        // Find max index
        let max_idx = magnitude
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map_or(0, |(i, _)| i);

        // Search for -3dB points
        for i in (0..max_idx).rev() {
            if magnitude[i] < threshold_3db {
                freq_low_3db = frequencies[i];
                break;
            }
        }

        for i in max_idx..frequencies.len() {
            if magnitude[i] < threshold_3db {
                freq_high_3db = frequencies[i];
                break;
            }
        }

        // Search for -6dB points
        for i in (0..max_idx).rev() {
            if magnitude[i] < threshold_6db {
                freq_low_6db = frequencies[i];
                break;
            }
        }

        for i in max_idx..frequencies.len() {
            if magnitude[i] < threshold_6db {
                freq_high_6db = frequencies[i];
                break;
            }
        }

        let bandwidth_3db = freq_high_3db - freq_low_3db;
        let bandwidth_6db = freq_high_6db - freq_low_6db;

        (bandwidth_3db, bandwidth_6db)
    }

    /// Calculate pulse response characteristics
    ///
    /// Returns (pulse length, axial resolution) in meters
    #[must_use]
    pub fn pulse_characteristics(&self, sound_speed: Velocity) -> (Length, Length) {
        // Pulse length ≈ (cycles * wavelength)
        // For typical transducers: 2-3 cycles
        let cycles = 2.5;
        let wavelength = sound_speed / self.center_frequency;
        let pulse_length = wavelength * cycles;

        // Axial resolution ≈ pulse_length / 2
        let axial_resolution = pulse_length / 2.0;

        (pulse_length, axial_resolution)
    }

    /// Calculate sensitivity roll-off at a given frequency
    #[must_use]
    pub fn sensitivity_at_frequency(&self, frequency: Frequency) -> f64 {
        // Linear interpolation in the magnitude response
        let idx = self
            .frequencies
            .iter()
            .position(|&f| f >= frequency)
            .unwrap_or(self.frequencies.len() - 1);

        if idx == 0 {
            self.magnitude[0]
        } else if idx >= self.frequencies.len() {
            self.magnitude[self.frequencies.len() - 1]
        } else {
            // Linear interpolation
            let f1 = *self.frequencies[idx - 1].as_base();
            let f2 = *self.frequencies[idx].as_base();
            let m1 = self.magnitude[idx - 1];
            let m2 = self.magnitude[idx];

            let t = (*frequency.as_base() - f1) / (f2 - f1);
            m1.mul_add(1.0 - t, m2 * t)
        }
    }

    /// Check if response meets bandwidth requirements
    #[must_use]
    pub fn validate_bandwidth(&self, min_fractional_bw: f64) -> bool {
        self.fractional_bandwidth >= min_fractional_bw
    }

    /// Calculate insertion loss at center frequency
    #[must_use]
    pub fn insertion_loss(&self) -> f64 {
        // Insertion loss from impedance mismatch: IL = -10log₁₀(1-|Γ|²)
        // Reflection coefficient Γ = (Z-Z₀)/(Z+Z₀)
        // Per IEEE Std 177: "Standard Definitions and Methods of Measurement"
        let z0 = 50.0; // Reference impedance
        let center_idx = self.frequencies.len() / 2;
        let z = self.impedance[center_idx];

        let reflection_coeff = (z - z0) / (z + z0);
        let transmission: f64 = reflection_coeff
            .norm()
            .mul_add(-reflection_coeff.norm(), 1.0);

        -10.0 * transmission.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::units::{Hertz, Megahertz, MeterPerSecond};

    #[test]
    fn klm_response_keeps_frequency_metrics_typed() {
        let response = FrequencyResponse::from_klm_model(
            Frequency::from_unit::<Megahertz>(3.0),
            0.75,
            80.0,
            50.0,
            201,
        )
        .expect("valid KLM parameters");

        assert!(response.center_frequency > Frequency::from_unit::<Hertz>(0.0));
        assert!(response.bandwidth_3db > Frequency::from_unit::<Hertz>(0.0));
        assert_eq!(response.frequencies.len(), 201);
        assert_eq!(response.frequencies[100], response.center_frequency);
    }

    #[test]
    fn pulse_characteristics_return_lengths() {
        let response = FrequencyResponse::from_klm_model(
            Frequency::from_unit::<Megahertz>(3.0),
            0.75,
            80.0,
            50.0,
            201,
        )
        .expect("valid KLM parameters");
        let (pulse_length, axial_resolution) =
            response.pulse_characteristics(Velocity::from_unit::<MeterPerSecond>(1500.0));

        assert!(*pulse_length.as_base() > 0.0);
        assert_eq!(*axial_resolution.as_base(), *pulse_length.as_base() / 2.0);
    }

    #[test]
    fn klm_response_rejects_insufficient_frequency_samples() {
        let error = FrequencyResponse::from_klm_model(
            Frequency::from_unit::<Megahertz>(3.0),
            0.75,
            80.0,
            50.0,
            1,
        )
        .expect_err("one sample cannot define bandwidth");

        assert!(error.to_string().contains("num_points"));
    }
}
