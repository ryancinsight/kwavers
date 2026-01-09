#![deny(missing_docs)]
//! Narrowband steering helpers for array processing.
//!
//! # Field jargon
//! For narrowband processing at frequency `f` (Hz), the steering vector for a candidate
//! point `p` (near-field) is commonly modeled as
//!
//! `a_i(p; f) = exp(-j 2π f τ_i(p))`
//!
//! where `τ_i(p) = ||x_i - p|| / c` is the propagation delay (time-of-flight, TOF) from
//! the source at `p` to sensor `i`, with speed of sound `c`.
//!
//! This representation is the standard bridge from geometric TOF to complex phase steering,
//! used by MVDR/Capon spatial spectra, MUSIC pseudospectra, and related narrowband methods.
//!
//! # Notes / invariants
//! - `frequency_hz` must be finite and > 0.
//! - `sound_speed` must be finite and > 0.
//! - All coordinates must be finite.
//! - Steering vectors are returned with unit magnitude (no spherical spreading / amplitude term).
//!   If you need amplitude modeling (e.g., `1/r`), that is a separate physical model.
//!
//! # Why this exists
//! `SteeringVectorMethod::SphericalWave` in `sensor::beamforming::steering` currently bakes in an
//! amplitude term (`1/r`) and uses `exp(+j k r)` sign convention. For narrowband adaptive beamforming
//! and localization scoring, the most common convention is *unit-norm phase-only* steering with
//! `exp(-j 2π f τ)`.
//!
//! This file provides explicit, jargon-aligned helpers to avoid silent convention mismatches.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array1;
use num_complex::Complex64;

/// Newtype representing a narrowband steering vector (complex phasors).
///
/// This is intentionally a very small wrapper around `Array1<Complex64>` to keep the API explicit.
#[derive(Debug, Clone)]
pub struct NarrowbandSteeringVector(pub Array1<Complex64>);

impl NarrowbandSteeringVector {
    /// Borrow the underlying phasor vector.
    #[must_use]
    pub fn as_array(&self) -> &Array1<Complex64> {
        &self.0
    }

    /// Consume and return the underlying phasor vector.
    #[must_use]
    pub fn into_array(self) -> Array1<Complex64> {
        self.0
    }
}

/// Narrowband steering helper.
///
/// Provides fast, explicit construction of phase-only steering vectors for near-field points.
#[derive(Debug, Clone)]
pub struct NarrowbandSteering {
    sensor_positions_m: Vec<[f64; 3]>,
    sound_speed_m_per_s: f64,
}

impl NarrowbandSteering {
    /// Create a narrowband steering helper for a fixed array geometry.
    ///
    /// # Errors
    /// Returns an error if sensor positions are empty, non-finite, or if `sound_speed_m_per_s <= 0`.
    pub fn new(sensor_positions_m: Vec<[f64; 3]>, sound_speed_m_per_s: f64) -> KwaversResult<Self> {
        if sensor_positions_m.is_empty() {
            return Err(KwaversError::InvalidInput(
                "NarrowbandSteering::new: sensor_positions_m must be non-empty".to_string(),
            ));
        }
        if !sound_speed_m_per_s.is_finite() || sound_speed_m_per_s <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "NarrowbandSteering::new: sound_speed_m_per_s must be finite and > 0".to_string(),
            ));
        }
        if sensor_positions_m
            .iter()
            .any(|p| p.iter().any(|v| !v.is_finite()))
        {
            return Err(KwaversError::InvalidInput(
                "NarrowbandSteering::new: sensor_positions_m must be finite".to_string(),
            ));
        }

        Ok(Self {
            sensor_positions_m,
            sound_speed_m_per_s,
        })
    }

    /// Number of sensors / elements.
    #[must_use]
    pub fn num_sensors(&self) -> usize {
        self.sensor_positions_m.len()
    }

    /// Speed of sound used for TOF calculations (m/s).
    #[must_use]
    pub fn sound_speed_m_per_s(&self) -> f64 {
        self.sound_speed_m_per_s
    }

    /// Compute absolute propagation delays `τ_i(p)` (seconds) for a candidate point.
    ///
    /// # Errors
    /// Returns an error if `candidate_m` contains non-finite values.
    pub fn propagation_delays_s(&self, candidate_m: [f64; 3]) -> KwaversResult<Vec<f64>> {
        if candidate_m.iter().any(|v| !v.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "NarrowbandSteering::propagation_delays_s: candidate_m must be finite".to_string(),
            ));
        }

        let c = self.sound_speed_m_per_s;
        Ok(self
            .sensor_positions_m
            .iter()
            .map(|&x_i| crate::domain::sensor::math::distance3(x_i, candidate_m) / c)
            .collect())
    }

    /// Compute a phase-only narrowband steering vector for a near-field candidate point:
    ///
    /// `a_i = exp(-j 2π f τ_i(p))`.
    ///
    /// # Errors
    /// Returns an error if `frequency_hz <= 0`, non-finite values are provided, or the candidate
    /// is non-finite.
    pub fn steering_vector_point(
        &self,
        candidate_m: [f64; 3],
        frequency_hz: f64,
    ) -> KwaversResult<NarrowbandSteeringVector> {
        validate_frequency_hz(frequency_hz)?;

        let delays_s = self.propagation_delays_s(candidate_m)?;
        Ok(steering_from_delays_s(&delays_s, frequency_hz))
    }
}

/// Construct `exp(-j 2π f τ)` from precomputed propagation delays.
///
/// This is the core primitive used by narrowband Capon/MVDR spectra and subspace methods.
///
/// # Panics
/// This function does not panic. All computations are safe.
///
/// # Notes
/// - The output has unit magnitude phasors.
/// - This applies a *negative* sign convention consistent with most array processing texts for
///   steering to compensate propagation delay.
#[must_use]
pub fn steering_from_delays_s(delays_s: &[f64], frequency_hz: f64) -> NarrowbandSteeringVector {
    // Caller is expected to validate `frequency_hz` and `delays_s` finiteness where needed.
    // We stay total here and propagate non-finite through to output if the caller violates invariants.
    let omega = -2.0 * std::f64::consts::PI * frequency_hz;
    let mut a = Array1::<Complex64>::zeros(delays_s.len());
    for (i, &tau) in delays_s.iter().enumerate() {
        let phase = omega * tau;
        a[i] = Complex64::new(0.0, phase).exp();
    }
    NarrowbandSteeringVector(a)
}

fn validate_frequency_hz(frequency_hz: f64) -> KwaversResult<()> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "Narrowband steering: frequency_hz must be finite and > 0".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steering_from_delays_has_unit_magnitude() {
        let delays = vec![0.0, 1e-6, 2e-6];
        let f = 1e6;
        let a = steering_from_delays_s(&delays, f).into_array();

        for v in a.iter() {
            let mag = v.norm();
            assert!(
                (mag - 1.0).abs() < 1e-12,
                "expected unit magnitude, got {mag}"
            );
        }
    }

    #[test]
    fn point_steering_is_deterministic() {
        let positions = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]];
        let steering = NarrowbandSteering::new(positions, 1500.0).expect("steering init");
        let p = [0.0, 0.0, 0.02];
        let f = 1e6;

        let a1 = steering
            .steering_vector_point(p, f)
            .expect("a1")
            .into_array();
        let a2 = steering
            .steering_vector_point(p, f)
            .expect("a2")
            .into_array();

        assert_eq!(a1.len(), 2);
        assert_eq!(a2.len(), 2);

        for i in 0..2 {
            assert!((a1[i] - a2[i]).norm() < 1e-15);
        }
    }

    #[test]
    fn invalid_frequency_is_rejected() {
        let positions = vec![[0.0, 0.0, 0.0]];
        let steering = NarrowbandSteering::new(positions, 1500.0).expect("steering init");

        let err = steering
            .steering_vector_point([0.0, 0.0, 0.0], 0.0)
            .expect_err("should error");
        assert!(err.to_string().contains("frequency_hz"));
    }

    #[test]
    fn invalid_candidate_is_rejected() {
        let positions = vec![[0.0, 0.0, 0.0]];
        let steering = NarrowbandSteering::new(positions, 1500.0).expect("steering init");

        let err = steering
            .steering_vector_point([f64::NAN, 0.0, 0.0], 1e6)
            .expect_err("should error");
        assert!(err.to_string().contains("candidate"));
    }
}
