//! Ultrafast Ultrasound Transmission Sequence Scheduling
//!
//! This module provides tools for scheduling transmission events in ultrafast
//! ultrasound imaging, including interleaved angle sequences, PRF management,
//! and multi-zone depth scheduling.
//!
//! # Mathematical Foundation
//!
//! ## PRF Limit (Tanter & Fink 2014, §II.A)
//!
//! **Theorem**: For unambiguous receive from depth z_max, the pulse repetition
//! frequency is bounded by:
//! ```text
//!   PRF_max = c / (2 · z_max)     (Hz)
//! ```
//! **Proof**: The round-trip travel time is T_rt = 2z_max/c.  New pulses fired
//! before T_rt returns cause range-ambiguous echoes (Doppler aliasing and ghost
//! targets).  Hence T_PRI ≥ T_rt → PRF ≤ c/(2z_max). □
//!
//! ## Compounded Frame Rate
//!
//! With N_ang angles compounded, each complete frame requires N_ang transmissions:
//! ```text
//!   f_frame = PRF / N_ang     (frames/s)
//! ```
//!
//! ## Interleaved Angle Scheduling
//!
//! To minimise motion artefacts in coherent compounding, angles should be spread
//! evenly in time (interleaved) rather than transmitted in sequential order.
//! For N angles, the interleaved index order (Montaldo et al. 2009):
//! ```text
//!   k_interleaved = [0, N/2, 1, N/2+1, 2, N/2+2, …]
//! ```
//! This ensures maximum angular separation between consecutive transmissions.
//!
//! ## Flash Sequence (Flat Wavefront)
//!
//! A single unfocused flat (0° plane wave) transmission — the "flash" — provides
//! maximum frame rate at the cost of image quality.  Frame rate equals PRF.
//!
//! # References
//!
//! - Tanter, M., & Fink, M. (2014). *IEEE TUFFC*, 61(1), 102–119.
//! - Montaldo, G., et al. (2009). *IEEE TUFFC*, 56(3), 489–506.

use aequitas::systems::si::{
    quantities::{Angle, Frequency, Length, Time, Velocity},
    units::{Hertz, Meter, MeterPerSecond, Radian, Second},
};
use kwavers_core::error::{KwaversError, KwaversResult};

#[cfg(test)]
mod tests;

/// A single scheduled transmission event.
#[derive(Debug, Clone, PartialEq)]
pub struct TransmissionEvent {
    /// Index within the sequence (0-based)
    pub event_index: usize,
    /// Time of this transmission, measured from sequence start.
    pub t_start: Time<f64>,
    /// Tilt angle — zero for diverging-wave events.
    pub tilt_angle: Angle<f64>,
    /// Transmitting element index (None = all elements fire together)
    pub element_index: Option<usize>,
}

/// Transmission sequence schedule — ordered list of firing events.
#[derive(Debug, Clone)]
pub struct TransmissionSchedule {
    /// Ordered transmission events
    pub events: Vec<TransmissionEvent>,
    /// Pulse repetition frequency.
    pub prf: Frequency<f64>,
    /// Total sequence duration.
    pub total_duration: Time<f64>,
    /// Compound frame rate = PRF / n_angles.
    pub frame_rate: Frequency<f64>,
}

impl TransmissionSchedule {
    /// Number of transmission events.
    #[must_use]
    pub fn n_events(&self) -> usize {
        self.events.len()
    }
}

/// Transmission sequence scheduler.
#[derive(Debug, Clone)]
pub struct TransmissionSequencer {
    /// Speed of sound in the imaging medium.
    pub sound_speed: Velocity<f64>,
    /// Maximum imaging depth.
    pub max_depth: Length<f64>,
    /// Override PRF limit (None = use theoretical maximum)
    pub prf_override: Option<Frequency<f64>>,
}

impl TransmissionSequencer {
    /// Create a new sequencer.
    ///
    /// # Arguments
    /// * `sound_speed` - Speed of sound in the medium.
    /// * `max_depth`   - Maximum imaging depth.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(sound_speed: Velocity<f64>, max_depth: Length<f64>) -> KwaversResult<Self> {
        let sound_speed_m_s = sound_speed.in_unit::<MeterPerSecond>();
        let max_depth_m = max_depth.in_unit::<Meter>();
        if !sound_speed_m_s.is_finite() || sound_speed_m_s <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "sound speed must be finite and > 0 m/s, got {sound_speed_m_s}"
            )));
        }
        if !max_depth_m.is_finite() || max_depth_m <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "maximum depth must be finite and > 0 m, got {max_depth_m}"
            )));
        }
        Ok(Self {
            sound_speed,
            max_depth,
            prf_override: None,
        })
    }

    /// Set a specific PRF (must be ≤ PRF_max).
    ///
    /// Returns `Err` if the requested PRF exceeds the maximum allowed.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn with_prf(mut self, prf: Frequency<f64>) -> KwaversResult<Self> {
        let prf_max = self.max_prf();
        let prf_hz = prf.in_unit::<Hertz>();
        let prf_max_hz = prf_max.in_unit::<Hertz>();
        if !prf_hz.is_finite() || prf_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "PRF must be finite and > 0 Hz, got {prf_hz}"
            )));
        }
        if prf_hz > prf_max_hz * (1.0 + 1e-9) {
            return Err(KwaversError::InvalidInput(format!(
                "requested PRF {prf_hz:.0} Hz exceeds PRF_max = {prf_max_hz:.0} Hz \
                 for depth {:.1} mm",
                self.max_depth.in_unit::<Meter>() * 1e3
            )));
        }
        self.prf_override = Some(prf);
        Ok(self)
    }

    /// Maximum unambiguous PRF for the configured depth and sound speed.
    ///
    /// ```text
    ///   PRF_max = c / (2 · z_max)
    /// ```
    #[must_use]
    pub fn max_prf(&self) -> Frequency<f64> {
        Frequency::from_unit::<Hertz>(
            self.sound_speed.in_unit::<MeterPerSecond>()
                / (2.0 * self.max_depth.in_unit::<Meter>()),
        )
    }

    /// Effective PRF: override if set, otherwise PRF_max.
    #[must_use]
    pub fn effective_prf(&self) -> Frequency<f64> {
        self.prf_override.unwrap_or_else(|| self.max_prf())
    }

    /// Compound frame rate for N_ang angles.
    ///
    /// ```text
    ///   f_frame = PRF / N_ang
    /// ```
    pub fn frame_rate(&self, n_angles: usize) -> KwaversResult<Frequency<f64>> {
        if n_angles == 0 {
            return Err(KwaversError::InvalidInput(
                "n_angles must be > 0".to_owned(),
            ));
        }
        Ok(Frequency::from_unit::<Hertz>(
            self.effective_prf().in_unit::<Hertz>() / n_angles as f64,
        ))
    }

    /// Build a sequential plane-wave angle schedule (no interleaving).
    ///
    /// Transmits angles `[θ₀, θ₁, …, θ_{n-1}]` in order, one per PRI.
    ///
    /// # Arguments
    /// * `tilt_angles` - Slice of tilt angles.
    pub fn sequential_schedule(
        &self,
        tilt_angles: &[Angle<f64>],
    ) -> KwaversResult<TransmissionSchedule> {
        let prf = self.effective_prf();
        let pri = Time::from_unit::<Second>(1.0 / prf.in_unit::<Hertz>());
        let n = tilt_angles.len();
        let frame_rate = self.frame_rate(n)?;

        let events: Vec<TransmissionEvent> = tilt_angles
            .iter()
            .enumerate()
            .map(|(k, &theta)| TransmissionEvent {
                event_index: k,
                t_start: Time::from_unit::<Second>(k as f64 * pri.in_unit::<Second>()),
                tilt_angle: theta,
                element_index: None,
            })
            .collect();

        Ok(TransmissionSchedule {
            total_duration: Time::from_unit::<Second>(n as f64 * pri.in_unit::<Second>()),
            frame_rate,
            events,
            prf,
        })
    }

    /// Build an interleaved plane-wave angle schedule (Montaldo et al. 2009).
    ///
    /// ## Algorithm: Interleaved Angle Ordering
    ///
    /// For N angles sorted by index [0, 1, …, N−1], the interleaved order ensures
    /// maximum angular separation between consecutive transmissions:
    /// ```text
    ///   k_interleaved[2m]   = m            for m = 0, …, ⌊N/2⌋ − 1
    ///   k_interleaved[2m+1] = ⌊N/2⌋ + m   for m = 0, …, ⌈N/2⌉ − 1
    /// ```
    /// For N=11: [0, 5, 1, 6, 2, 7, 3, 8, 4, 9, 10] (zipper interleave).
    ///
    /// This minimises the time between angles θ_k and θ_{k+N/2}, ensuring the
    /// coherently compounded frame is phase-consistent across the full aperture
    /// even with tissue motion (Montaldo et al. 2009, §II.D).
    ///
    /// # Arguments
    /// * `tilt_angles` - Slice of tilt angles (will be reordered).
    pub fn interleaved_schedule(
        &self,
        tilt_angles: &[Angle<f64>],
    ) -> KwaversResult<TransmissionSchedule> {
        let n = tilt_angles.len();
        let frame_rate = self.frame_rate(n)?;
        let half = n / 2;

        // Build interleaved index order: [0, half, 1, half+1, 2, half+2, …]
        let mut order = Vec::with_capacity(n);
        let (lo_count, hi_count) = (half, n - half);
        let max_pairs = lo_count.min(hi_count);
        for m in 0..max_pairs {
            order.push(m);
            order.push(half + m);
        }
        // Append any remainder (if N is odd, the last element has no pair)
        for m in max_pairs..lo_count {
            order.push(m);
        }
        for m in max_pairs..hi_count {
            order.push(half + m);
        }

        let prf = self.effective_prf();
        let pri = Time::from_unit::<Second>(1.0 / prf.in_unit::<Hertz>());

        let events: Vec<TransmissionEvent> = order
            .iter()
            .enumerate()
            .map(|(firing, &original_idx)| TransmissionEvent {
                event_index: firing,
                t_start: Time::from_unit::<Second>(firing as f64 * pri.in_unit::<Second>()),
                tilt_angle: tilt_angles[original_idx],
                element_index: None,
            })
            .collect();

        Ok(TransmissionSchedule {
            total_duration: Time::from_unit::<Second>(n as f64 * pri.in_unit::<Second>()),
            frame_rate,
            events,
            prf,
        })
    }

    /// Build a single-element STA (Synthetic Transmit Aperture) schedule.
    ///
    /// Each element fires in turn; all elements receive each transmission.
    /// After all N firings, a full STA frame is available.
    ///
    /// # Arguments
    /// * `n_elements` - Number of transducer elements
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn sta_schedule(&self, n_elements: usize) -> KwaversResult<TransmissionSchedule> {
        if n_elements == 0 {
            return Err(KwaversError::InvalidInput(
                "n_elements must be > 0".to_owned(),
            ));
        }
        let prf = self.effective_prf();
        let pri = Time::from_unit::<Second>(1.0 / prf.in_unit::<Hertz>());
        let frame_rate = self.frame_rate(n_elements)?;

        let events: Vec<TransmissionEvent> = (0..n_elements)
            .map(|k| TransmissionEvent {
                event_index: k,
                t_start: Time::from_unit::<Second>(k as f64 * pri.in_unit::<Second>()),
                tilt_angle: Angle::from_unit::<Radian>(0.0),
                element_index: Some(k),
            })
            .collect();

        Ok(TransmissionSchedule {
            total_duration: Time::from_unit::<Second>(n_elements as f64 * pri.in_unit::<Second>()),
            frame_rate,
            events,
            prf,
        })
    }

    /// Flash sequence: single unfocused plane wave (θ=0°), maximum frame rate.
    pub fn flash_schedule(&self) -> KwaversResult<TransmissionSchedule> {
        self.sequential_schedule(&[Angle::from_unit::<Radian>(0.0)])
    }
}
