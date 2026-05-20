//! Safety limiting for power modulation

use super::constants::{MAX_AMPLITUDE_RATE, MECHANICAL_INDEX_LIMIT};
use crate::physics::acoustics::analysis::calculate_mechanical_index;

const PASCALS_PER_MEGAPASCAL: f64 = 1.0e6;
const HERTZ_PER_MEGAHERTZ: f64 = 1.0e6;

/// Safety limiter for preventing excessive power output
#[derive(Debug, Clone)]
pub struct SafetyLimiter {
    max_amplitude: f64,
    max_rate: f64,
    last_output: f64,
    mechanical_index_limit: f64,
}

impl SafetyLimiter {
    /// Create new safety limiter
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_amplitude: 1.0,
            max_rate: MAX_AMPLITUDE_RATE,
            last_output: 0.0,
            mechanical_index_limit: MECHANICAL_INDEX_LIMIT,
        }
    }

    /// Apply safety limiting to amplitude
    pub fn limit(&mut self, amplitude: f64) -> f64 {
        // Clamp to maximum amplitude
        let clamped = amplitude.clamp(0.0, self.max_amplitude);

        // Apply rate limiting (assuming 1ms timestep if not specified)
        // For instantaneous changes (like after reset), allow full change
        let max_change = if self.last_output == 0.0 && amplitude > 0.0 {
            // Allow initial ramp-up without rate limiting
            amplitude
        } else {
            self.max_rate * 0.001 // 1ms timestep default
        };

        let limited = if (clamped - self.last_output).abs() > max_change {
            if clamped > self.last_output {
                self.last_output + max_change
            } else {
                self.last_output - max_change
            }
        } else {
            clamped
        };

        self.last_output = limited;
        limited
    }

    /// Check mechanical index safety.
    ///
    /// The calculation delegates to the canonical acoustic pressure-analysis
    /// contract `MI = |p_r|_MPa / sqrt(f_MHz)`. Invalid frequency or pressure
    /// domains fail closed because this limiter is a safety boundary.
    #[must_use]
    pub fn check_mechanical_index(&self, pressure_mpa: f64, frequency_mhz: f64) -> bool {
        if !pressure_mpa.is_finite() || !frequency_mhz.is_finite() || frequency_mhz <= 0.0 {
            return false;
        }

        let mi = calculate_mechanical_index(
            pressure_mpa * PASCALS_PER_MEGAPASCAL,
            frequency_mhz * HERTZ_PER_MEGAHERTZ,
        );

        mi <= self.mechanical_index_limit
    }

    /// Set maximum amplitude limit
    pub fn set_max_amplitude(&mut self, max_amplitude: f64) {
        self.max_amplitude = max_amplitude.clamp(0.0, 1.0);
    }

    /// Set maximum rate of change
    pub fn set_max_rate(&mut self, max_rate: f64) {
        self.max_rate = max_rate.abs();
    }

    /// Reset limiter state
    pub fn reset(&mut self) {
        self.last_output = 0.0;
    }
}

impl Default for SafetyLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::constants::MECHANICAL_INDEX_LIMIT;
    use super::*;

    /// check_mechanical_index: MI = pressure / sqrt(frequency).
    /// For P=1 MPa, f=1 MHz: MI=1.0 ≤ MECHANICAL_INDEX_LIMIT (1.9) → true.
    #[test]
    fn mechanical_index_safe_when_below_limit() {
        let lim = SafetyLimiter::new();
        assert!(
            lim.check_mechanical_index(1.0, 1.0),
            "MI=1.0 must be within limit={MECHANICAL_INDEX_LIMIT}"
        );
    }

    /// For P=2 MPa, f=0.5 MHz: MI=2/sqrt(0.5)=2.828 > 1.9 → false.
    #[test]
    fn mechanical_index_unsafe_above_limit() {
        let lim = SafetyLimiter::new();
        let mi = 2.0_f64 / 0.5_f64.sqrt();
        assert!(
            mi > MECHANICAL_INDEX_LIMIT,
            "computed MI={mi:.3} must exceed limit for this test to be meaningful"
        );
        assert!(
            !lim.check_mechanical_index(2.0, 0.5),
            "MI={mi:.3} must exceed limit={MECHANICAL_INDEX_LIMIT}"
        );
    }

    /// Rarefactional pressure sign is not a safety bypass; MI uses magnitude.
    #[test]
    fn mechanical_index_uses_pressure_magnitude() {
        let lim = SafetyLimiter::new();

        assert!(lim.check_mechanical_index(-1.0, 1.0));
        assert!(!lim.check_mechanical_index(-2.0, 0.5));
    }

    /// Invalid pressure/frequency domains fail closed at the limiter boundary.
    #[test]
    fn mechanical_index_rejects_invalid_domains() {
        let lim = SafetyLimiter::new();

        assert!(!lim.check_mechanical_index(f64::NAN, 1.0));
        assert!(!lim.check_mechanical_index(f64::INFINITY, 1.0));
        assert!(!lim.check_mechanical_index(1.0, 0.0));
        assert!(!lim.check_mechanical_index(1.0, -1.0));
        assert!(!lim.check_mechanical_index(1.0, f64::NAN));
        assert!(!lim.check_mechanical_index(1.0, f64::INFINITY));
    }

    /// reset zeroes last_output; subsequent initial limit call is not rate-limited.
    #[test]
    fn safety_limiter_reset_allows_ramp() {
        let mut lim = SafetyLimiter::new();
        lim.limit(0.5); // set last_output to 0.5
        lim.reset(); // last_output → 0
                     // After reset, initial ramp-up is allowed (max_change = amplitude)
        let out = lim.limit(0.8);
        assert!(
            (out - 0.8).abs() < 1e-14,
            "first limit after reset must allow full amplitude; got {out}"
        );
    }

    /// limit clamps amplitude above max_amplitude=1.0.
    #[test]
    fn safety_limiter_clamps_above_max() {
        let mut lim = SafetyLimiter::new();
        let out = lim.limit(1.5); // > max_amplitude=1.0
        assert!(
            out <= 1.0,
            "output must not exceed max_amplitude=1.0; got {out}"
        );
    }
}
