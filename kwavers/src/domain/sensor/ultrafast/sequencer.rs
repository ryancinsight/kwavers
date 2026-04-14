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

use crate::core::error::{KwaversError, KwaversResult};

/// A single scheduled transmission event.
#[derive(Debug, Clone, PartialEq)]
pub struct TransmissionEvent {
    /// Index within the sequence (0-based)
    pub event_index: usize,
    /// Time of this transmission (s), measured from sequence start
    pub t_start: f64,
    /// Tilt angle (radians) — 0.0 for diverging-wave events
    pub tilt_angle: f64,
    /// Transmitting element index (None = all elements fire together)
    pub element_index: Option<usize>,
}

/// Transmission sequence schedule — ordered list of firing events.
#[derive(Debug, Clone)]
pub struct TransmissionSchedule {
    /// Ordered transmission events
    pub events: Vec<TransmissionEvent>,
    /// Pulse repetition frequency (Hz)
    pub prf: f64,
    /// Total sequence duration (s)
    pub total_duration: f64,
    /// Compound frame rate (Hz) = PRF / n_angles
    pub frame_rate: f64,
}

impl TransmissionSchedule {
    /// Number of transmission events.
    pub fn n_events(&self) -> usize {
        self.events.len()
    }
}

/// Transmission sequence scheduler.
#[derive(Debug, Clone)]
pub struct TransmissionSequencer {
    /// Speed of sound (m/s)
    pub sound_speed: f64,
    /// Maximum imaging depth (m)
    pub max_depth: f64,
    /// Override PRF limit (None = use theoretical maximum)
    pub prf_override: Option<f64>,
}

impl TransmissionSequencer {
    /// Create a new sequencer.
    ///
    /// # Arguments
    /// * `sound_speed` - Speed of sound in the medium (m/s)
    /// * `max_depth`   - Maximum imaging depth (m)
    pub fn new(sound_speed: f64, max_depth: f64) -> Self {
        Self {
            sound_speed,
            max_depth,
            prf_override: None,
        }
    }

    /// Set a specific PRF (must be ≤ PRF_max).
    ///
    /// Returns `Err` if the requested PRF exceeds the maximum allowed.
    pub fn with_prf(mut self, prf: f64) -> KwaversResult<Self> {
        let prf_max = self.max_prf();
        if prf > prf_max * (1.0 + 1e-9) {
            return Err(KwaversError::InvalidInput(format!(
                "Requested PRF {prf:.0} Hz exceeds PRF_max = {prf_max:.0} Hz \
                 for depth {:.1} mm",
                self.max_depth * 1e3
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
    pub fn max_prf(&self) -> f64 {
        self.sound_speed / (2.0 * self.max_depth)
    }

    /// Effective PRF: override if set, otherwise PRF_max.
    pub fn effective_prf(&self) -> f64 {
        self.prf_override.unwrap_or_else(|| self.max_prf())
    }

    /// Compound frame rate for N_ang angles.
    ///
    /// ```text
    ///   f_frame = PRF / N_ang
    /// ```
    pub fn frame_rate(&self, n_angles: usize) -> f64 {
        self.effective_prf() / n_angles as f64
    }

    /// Build a sequential plane-wave angle schedule (no interleaving).
    ///
    /// Transmits angles `[θ₀, θ₁, …, θ_{n-1}]` in order, one per PRI.
    ///
    /// # Arguments
    /// * `tilt_angles` - Slice of tilt angles in radians
    pub fn sequential_schedule(&self, tilt_angles: &[f64]) -> TransmissionSchedule {
        let prf = self.effective_prf();
        let pri = 1.0 / prf;
        let n = tilt_angles.len();

        let events: Vec<TransmissionEvent> = tilt_angles
            .iter()
            .enumerate()
            .map(|(k, &theta)| TransmissionEvent {
                event_index: k,
                t_start: k as f64 * pri,
                tilt_angle: theta,
                element_index: None,
            })
            .collect();

        TransmissionSchedule {
            total_duration: n as f64 * pri,
            frame_rate: prf / n as f64,
            events,
            prf,
        }
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
    /// * `tilt_angles` - Slice of tilt angles in radians (will be reordered)
    pub fn interleaved_schedule(&self, tilt_angles: &[f64]) -> TransmissionSchedule {
        let n = tilt_angles.len();
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
        let pri = 1.0 / prf;

        let events: Vec<TransmissionEvent> = order
            .iter()
            .enumerate()
            .map(|(firing, &original_idx)| TransmissionEvent {
                event_index: firing,
                t_start: firing as f64 * pri,
                tilt_angle: tilt_angles[original_idx],
                element_index: None,
            })
            .collect();

        TransmissionSchedule {
            total_duration: n as f64 * pri,
            frame_rate: prf / n as f64,
            events,
            prf,
        }
    }

    /// Build a single-element STA (Synthetic Transmit Aperture) schedule.
    ///
    /// Each element fires in turn; all elements receive each transmission.
    /// After all N firings, a full STA frame is available.
    ///
    /// # Arguments
    /// * `n_elements` - Number of transducer elements
    pub fn sta_schedule(&self, n_elements: usize) -> KwaversResult<TransmissionSchedule> {
        if n_elements == 0 {
            return Err(KwaversError::InvalidInput(
                "n_elements must be > 0".to_string(),
            ));
        }
        let prf = self.effective_prf();
        let pri = 1.0 / prf;

        let events: Vec<TransmissionEvent> = (0..n_elements)
            .map(|k| TransmissionEvent {
                event_index: k,
                t_start: k as f64 * pri,
                tilt_angle: 0.0,
                element_index: Some(k),
            })
            .collect();

        Ok(TransmissionSchedule {
            total_duration: n_elements as f64 * pri,
            frame_rate: prf / n_elements as f64,
            events,
            prf,
        })
    }

    /// Flash sequence: single unfocused plane wave (θ=0°), maximum frame rate.
    pub fn flash_schedule(&self) -> TransmissionSchedule {
        self.sequential_schedule(&[0.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sequencer_40mm() -> TransmissionSequencer {
        TransmissionSequencer::new(1540.0, 0.040)
    }

    /// PRF_max = c / (2·z_max) = 1540 / 0.080 = 19 250 Hz.
    #[test]
    fn test_max_prf() {
        let seq = sequencer_40mm();
        let expected = 1540.0 / 0.080;
        assert!(
            (seq.max_prf() - expected).abs() / expected < 1e-10,
            "PRF_max mismatch: {:.1} Hz expected {:.1} Hz",
            seq.max_prf(),
            expected
        );
    }

    /// Setting PRF above PRF_max must return an error.
    #[test]
    fn test_prf_exceeds_max_returns_error() {
        let seq = sequencer_40mm();
        let prf_over = seq.max_prf() * 2.0;
        let result = seq.with_prf(prf_over);
        assert!(result.is_err(), "PRF exceeding max must return Err");
    }

    /// Setting PRF ≤ PRF_max must succeed.
    #[test]
    fn test_prf_at_max_is_valid() {
        let seq = sequencer_40mm();
        let prf_max = seq.max_prf();
        let result = seq.with_prf(prf_max);
        assert!(result.is_ok(), "PRF = PRF_max must succeed");
    }

    /// Sequential schedule has correct event count, timing, and PRF.
    #[test]
    fn test_sequential_schedule_timing() {
        let seq = sequencer_40mm();
        let angles: Vec<f64> = (-2..=2).map(|i| i as f64 * PI / 180.0).collect(); // 5 angles
        let sched = seq.sequential_schedule(&angles);

        assert_eq!(sched.n_events(), 5);
        let pri = 1.0 / seq.max_prf();
        for (k, ev) in sched.events.iter().enumerate() {
            let expected_t = k as f64 * pri;
            assert!(
                (ev.t_start - expected_t).abs() < 1e-15,
                "Event {k} t_start mismatch"
            );
        }
        assert!((sched.frame_rate - seq.max_prf() / 5.0).abs() < 1e-6);
    }

    /// Flash schedule has exactly one event at t=0 with θ=0.
    #[test]
    fn test_flash_schedule() {
        let seq = sequencer_40mm();
        let sched = seq.flash_schedule();
        assert_eq!(sched.n_events(), 1);
        assert_eq!(sched.events[0].event_index, 0);
        assert!(sched.events[0].t_start.abs() < 1e-15);
        assert!(sched.events[0].tilt_angle.abs() < 1e-15);
        assert!((sched.frame_rate - seq.max_prf()).abs() < 1e-6);
    }

    /// Interleaved schedule preserves all angles (as a multiset).
    #[test]
    fn test_interleaved_schedule_all_angles_present() {
        let seq = sequencer_40mm();
        let angles: Vec<f64> = (-5..=5).map(|i| i as f64 * PI / 180.0).collect(); // 11 angles
        let sched = seq.interleaved_schedule(&angles);

        assert_eq!(sched.n_events(), 11);

        // Collect fired angles and sort
        let mut fired: Vec<i64> = sched
            .events
            .iter()
            .map(|ev| (ev.tilt_angle * 180.0 / PI).round() as i64)
            .collect();
        fired.sort();
        let mut expected: Vec<i64> = (-5..=5).collect();
        expected.sort();
        assert_eq!(
            fired, expected,
            "All 11 angles must appear in interleaved schedule"
        );
    }

    /// Interleaved schedule: first two events should have the maximum angular separation.
    ///
    /// For 11 angles [-5°,…,5°] the interleaved order is [0, 5, 1, 6, …],
    /// i.e., angles -5° and 0° at events 0 and 1.
    #[test]
    fn test_interleaved_max_separation_first_pair() {
        let seq = sequencer_40mm();
        let n = 11usize;
        let angles: Vec<f64> = (0..n).map(|i| i as f64 * PI / 180.0).collect(); // [0°, 1°, …, 10°]
        let sched = seq.interleaved_schedule(&angles);

        // Expected: event 0 → angle 0°, event 1 → angle 5° (half=5)
        let a0 = sched.events[0].tilt_angle;
        let a1 = sched.events[1].tilt_angle;
        let sep = (a1 - a0).abs();
        let expected_sep = 5.0 * PI / 180.0;
        assert!(
            (sep - expected_sep).abs() < 1e-12,
            "First pair separation {:.1}° ≠ 5°",
            sep * 180.0 / PI
        );
    }

    /// STA schedule: each event fires a different element, tilt_angle = 0.
    #[test]
    fn test_sta_schedule_element_ordering() {
        let seq = sequencer_40mm();
        let sched = seq.sta_schedule(8).unwrap();
        assert_eq!(sched.n_events(), 8);
        for (k, ev) in sched.events.iter().enumerate() {
            assert_eq!(ev.element_index, Some(k));
            assert!(ev.tilt_angle.abs() < 1e-15);
        }
    }

    /// STA schedule with 0 elements returns error.
    #[test]
    fn test_sta_zero_elements_errors() {
        let seq = sequencer_40mm();
        assert!(seq.sta_schedule(0).is_err());
    }

    /// Frame rate for N compounding events = PRF / N.
    #[test]
    fn test_frame_rate_formula() {
        let seq = sequencer_40mm();
        let n = 11usize;
        let expected = seq.max_prf() / n as f64;
        assert!(
            (seq.frame_rate(n) - expected).abs() / expected < 1e-10,
            "Frame rate {:.1} Hz ≠ {:.1} Hz",
            seq.frame_rate(n),
            expected
        );
    }
}
