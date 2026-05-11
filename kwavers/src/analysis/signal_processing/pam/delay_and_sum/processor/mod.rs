//! `DelayAndSumPAM` — time-domain DAS beamformer for passive acoustic mapping.
//!
//! ## Algorithm (Gyöngy & Coussios 2010)
//!
//! ```text
//! For each candidate source position r_s:
//!   τᵢ = ||r_s − rᵢ|| / c        (propagation delay to sensor i)
//!   sᵢ'(t) = sᵢ(t + τᵢ)          (time-shifted signal)
//!   P(r_s, t) = Σᵢ wᵢ · sᵢ'(t)   (coherent sum with apodization wᵢ)
//!   I(r_s) = ∫ |P|² dt            (intensity)
//! ```
//!
//! Partitioned by responsibility:
//! - `beamform` — input validation, delay computation, DAS accumulation.
//! - `detection` — threshold detection, event construction, frequency estimation.
//!
//! ## Theorem
//!
//! For a single impulsive cavitation source at position `r0`, sampled without
//! noise on an aperture with sensor positions `ri`, the DAS intensity is
//! maximized at `r0` among candidate grid points whose fractional propagation
//! delays differ from the true delay vector. At `r0`, every delayed sensor
//! trace contributes the impulse to the same sample, so the coherent sum has
//! amplitude `Σᵢ wᵢ`. At any other candidate, the missing positive cross terms
//! reduce the squared sum. This is the discrete passive-acoustic analogue of
//! matched filtering with a Green-function steering vector.

mod beamform;
mod detection;

use super::types::DelayAndSumConfig;
use crate::core::error::{KwaversError, KwaversResult};

/// Delay-and-Sum PAM processor.
#[derive(Debug)]
pub struct DelayAndSumPAM {
    pub(super) config: DelayAndSumConfig,
    pub(super) sensor_positions: Vec<[f64; 3]>,
    pub(super) num_sensors: usize,
}

impl DelayAndSumPAM {
    /// Create a new DAS PAM processor.
    ///
    /// Requires at least 3 sensors for 3-D localization.
    ///
    /// # Errors
    /// Returns `Err` if fewer than 3 sensors are provided or if `sound_speed`
    /// or `sampling_frequency` are non-positive.
    pub fn new(sensor_positions: Vec<[f64; 3]>, config: DelayAndSumConfig) -> KwaversResult<Self> {
        let num_sensors = sensor_positions.len();
        if num_sensors < 3 {
            return Err(KwaversError::InvalidInput(
                "Need at least 3 sensors for PAM".to_owned(),
            ));
        }
        if config.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sound speed must be positive".to_owned(),
            ));
        }
        if config.sampling_frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sampling frequency must be positive".to_owned(),
            ));
        }
        Ok(Self {
            config,
            sensor_positions,
            num_sensors,
        })
    }
}
